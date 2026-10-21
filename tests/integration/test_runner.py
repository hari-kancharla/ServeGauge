import asyncio
import json
import ssl
from pathlib import Path
from typing import Any

import httpx
import pytest

from servegauge.core.models import EndpointSpec
from servegauge.mock.server import create_app
from servegauge.runner.plan import RunPlan
from servegauge.runner.scheduler import run_plan
from servegauge.storage.raw_store import RawStore


@pytest.fixture(autouse=True)
def budget_config(tmp_path: Path) -> None:
    directory = tmp_path / "configs"
    directory.mkdir()
    (directory / "budgets.yaml").write_bytes(
        Path("tests/fixtures/simulated-budgets.yaml").read_bytes()
    )


def plan() -> RunPlan:
    return RunPlan(
        plan_id="test",
        profile="gpt-oss-120b",
        level="smoke",
        seed=1,
        reference={"tier": "MOCK"},
        max_usd=0,
        concurrency_per_endpoint=1,
        requests_per_minute=60000,
        endpoints=[
            EndpointSpec(
                id="mock:one",
                kind="mock",
                base_url="http://127.0.0.1:8011/v1",
                model="mock",
                claimed_profile="gpt-oss-120b",
            )
        ],
        suites=["S0"],
        overrides={"S0": {"items": 6}},
    )


async def test_asgi_capture_resume_and_no_duplicate_ids(tmp_path: Path, monkeypatch) -> None:
    from servegauge.runner import scheduler

    transport = httpx.ASGITransport(app=create_app())
    seen = 0

    def crash_after_two(record: Any) -> None:
        nonlocal seen
        seen += 1
        if seen == 2:
            raise RuntimeError("simulated_interruption")

    with pytest.raises(ExceptionGroup):
        await run_plan(
            plan(),
            root=tmp_path,
            run_id="resume-a",
            transports={"mock:one": transport},
            after_capture=crash_after_two,
        )
    assert len(list(RawStore(tmp_path / "data/raw").exchanges())) == 2
    original = list(RawStore(tmp_path / "data/raw").exchanges())
    build_manifest = scheduler.build_manifest

    def revised_manifest(*args, **kwargs):
        value = build_manifest(*args, **kwargs)
        value.update(git_sha="resumed-revision", source_tree_hash="resumed-source")
        return value

    monkeypatch.setattr(scheduler, "build_manifest", revised_manifest)
    result = await run_plan(
        plan(), root=tmp_path, run_id="resume-a", resume=True, transports={"mock:one": transport}
    )
    records = list(RawStore(tmp_path / "data/raw").exchanges())
    assert result.completed == 6 and result.skipped == 2
    assert len(records) == len({record.exchange_id for record in records}) == 6
    assert all(record.cost_usd == 0 for record in records)
    assert all(record.attempts[0].status == 200 for record in records)
    assert records[:2] == original
    assert all(
        r.attempts[0].metadata["execution_source"]["git_sha"] == "resumed-revision"
        for r in records[2:]
    )
    assert all(
        r.attempts[0].metadata["execution_source"]["source_identity"]["scope"]
        == "loaded-package-python-v1"
        for r in records
    )
    from servegauge.core.ids import safe_filename

    manifest = json.loads(
        (tmp_path / "data/runs" / safe_filename("resume-a") / "manifest.json").read_text()
    )
    assert manifest["git_sha"] != "resumed-revision"


async def test_harness_failure_locations_exclude_exception_messages(tmp_path):
    from servegauge.core.ids import safe_filename

    def fail(request):
        raise RuntimeError("do-not-log-private-request-content")

    with pytest.raises(ExceptionGroup):
        await run_plan(
            plan(),
            root=tmp_path,
            run_id="safe-failure",
            transports={"mock:one": httpx.MockTransport(fail)},
        )
    text = (tmp_path / "data/runs" / safe_filename("safe-failure") / "failure.json").read_text()
    assert "do-not-log-private-request-content" not in text
    value = json.loads(text)
    assert value["exceptions"][0]["type"] == "RuntimeError"
    assert value["exceptions"][0]["frames"][-1]["function"] == "fail"


async def test_malformed_usage_shape_is_still_journaled_and_captured(tmp_path):
    current = plan()
    current.overrides["S0"]["items"] = 1
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": "invalid-shape",
            },
        )
    )
    result = await run_plan(
        current,
        root=tmp_path,
        run_id="malformed-usage",
        transports={current.endpoints[0].id: transport},
    )
    record = next(RawStore(tmp_path / "data/raw").exchanges(run_id=result.run_id))
    assert record.usage is None and record.response["usage"] == "invalid-shape"
    assert record.metadata["invalid_usage_shape"] and record.attempts[0].response


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("envelope", ["finish_error", "root_error"])
async def test_http_200_completion_error_is_captured_and_charged(tmp_path, streaming, envelope):
    current = plan()
    current.overrides["S0"]["items"] = 1
    endpoint = current.endpoints[0]
    endpoint.kind = "openai_compat"
    endpoint.metadata = {
        "max_prompt_tokens": 1000,
        "pricing": {"prompt": "0.000001", "completion": "0"},
    }
    current.max_usd = 0.1
    value = {"choices": [{"message": {"content": None}, "finish_reason": "error"}]}
    if envelope == "root_error":
        value = {"error": {"message": "upstream completion failed"}}
    sent = 0

    def handler(request):
        nonlocal sent
        sent += 1
        if streaming:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=f"data: {json.dumps(value)}\n\ndata: [DONE]\n\n".encode(),
            )
        return httpx.Response(200, json=value)

    result = await run_plan(
        current,
        root=tmp_path,
        run_id="provider-error",
        approved=True,
        transports={endpoint.id: httpx.MockTransport(handler)},
    )
    record = next(RawStore(tmp_path / "data/raw").exchanges(run_id=result.run_id))
    assert sent == 1 and result.errors == 1
    assert record.status == 200 and record.error == "provider_completion_error"
    assert record.attempts[0].error == "provider_completion_error"
    assert record.cost_usd == 0.001 and float(result.spent_usd) == 0.001
    if envelope == "root_error":
        assert record.response["error"]["message"] == "upstream completion failed"


@pytest.mark.parametrize("streaming", [False, True])
async def test_total_request_deadline_preserves_partial_capture_and_charge(tmp_path, streaming):
    current = plan()
    current.overrides["S0"]["items"] = 1
    current.timeout_seconds, current.retries, current.max_usd = 0.01, 0, 0.1
    endpoint = current.endpoints[0]
    endpoint.kind = "openai_compat"
    endpoint.metadata = {
        "max_prompt_tokens": 1000,
        "pricing": {"prompt": "0.000001", "completion": "0"},
    }

    class NeverFinishes(httpx.AsyncByteStream):
        async def __aiter__(self):
            if streaming:
                yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            else:
                yield b'{"choices":['
            await asyncio.Event().wait()

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream" if streaming else "application/json"},
            stream=NeverFinishes(),
        )
    )
    result = await asyncio.wait_for(
        run_plan(
            current,
            root=tmp_path,
            run_id="total-deadline",
            approved=True,
            transports={endpoint.id: transport},
        ),
        timeout=0.5,
    )
    record = next(RawStore(tmp_path / "data/raw").exchanges(run_id=result.run_id))
    assert result.errors == 1 and record.error == "transport_timeout"
    assert len(record.attempts) == 1 and record.cost_usd == 0.001
    assert record.metadata["timeout_scope"] == "total_request"
    if streaming:
        assert record.sse_events and record.metadata["incomplete"]
        assert record.response["choices"][0]["message"]["content"] == "partial"
    else:
        assert record.response_text == '{"choices":['


@pytest.mark.parametrize("phase", ["headers", "json", "sse"])
@pytest.mark.parametrize("retries", [0, 1])
async def test_unwrapped_tls_failure_is_captured_retried_and_charged(tmp_path, phase, retries):
    from servegauge.core.ids import safe_filename

    current = plan()
    current.overrides["S0"]["items"] = 1
    current.retries, current.max_usd = retries, 0.1
    current.retry_base_seconds = current.retry_max_seconds = 0
    endpoint = current.endpoints[0]
    endpoint.kind = "openai_compat"
    endpoint.metadata = {
        "max_prompt_tokens": 1000,
        "pricing": {"prompt": "0.000001", "completion": "0"},
    }

    class BrokenTls(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield (
                b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
                if phase == "sse"
                else b'{"choices":['
            )
            raise ssl.SSLError("do-not-log-private-provider-details")

    def handler(request):
        if phase == "headers":
            raise ssl.SSLError("do-not-log-private-provider-details")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream" if phase == "sse" else "application/json"},
            stream=BrokenTls(),
        )

    result = await run_plan(
        current,
        root=tmp_path,
        run_id="tls-failure",
        approved=True,
        transports={endpoint.id: httpx.MockTransport(handler)},
    )
    record = next(RawStore(tmp_path / "data/raw").exchanges(run_id=result.run_id))
    assert result.errors == 1 and record.error == "transport_failure"
    assert len(record.attempts) == retries + 1
    assert record.cost_usd == 0.001 * (retries + 1)
    assert all(a.error == "transport_failure" for a in record.attempts)
    assert "do-not-log-private-provider-details" not in record.model_dump_json()
    assert not (tmp_path / "data/runs" / safe_filename(result.run_id) / "failure.json").exists()
    if phase == "sse":
        assert record.sse_events and record.metadata["incomplete"]
    elif phase == "json":
        assert record.response_text == '{"choices":['


async def test_replay_scope_cannot_reset_the_original_spending_limit(tmp_path: Path) -> None:
    from servegauge.runner.budget import BudgetExceeded

    current = plan().model_copy(deep=True)
    current.max_usd = 0.015
    current.overrides = {"S0": {"items": 1}}
    endpoint = current.endpoints[0]
    endpoint.kind = "openai_compat"
    endpoint.metadata = {
        "max_prompt_tokens": 1000,
        "pricing": {"prompt": ".00001", "completion": "0"},
    }
    sent = 0

    def handler(request):
        nonlocal sent
        sent += 1
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "OK"}}], "usage": {"cost": 0.01}}
        )

    transports = {endpoint.id: httpx.MockTransport(handler)}
    await run_plan(
        current,
        root=tmp_path,
        run_id="primary",
        budget_scope="shared",
        approved=True,
        transports=transports,
    )
    with pytest.raises(ExceptionGroup) as failure:
        await run_plan(
            current,
            root=tmp_path,
            run_id="replay",
            budget_scope="shared",
            approved=True,
            transports=transports,
        )
    assert any(isinstance(error, BudgetExceeded) for error in failure.value.exceptions)
    assert sent == 1
    with pytest.raises(ValueError, match="resume_config_mismatch"):
        await run_plan(
            current,
            root=tmp_path,
            run_id="primary",
            resume=True,
            budget_scope="different",
            approved=True,
            transports=transports,
        )


async def test_resume_rejects_config_changes_and_dry_run_opens_no_transport(tmp_path: Path) -> None:
    transport = httpx.ASGITransport(app=create_app())
    result = await run_plan(plan(), root=tmp_path, dry_run=True)
    assert result.completed == 0 and not (tmp_path / "data/raw").exists()
    await run_plan(plan(), root=tmp_path, run_id="a", transports={"mock:one": transport})
    changed = plan().model_copy(update={"seed": 8})
    with pytest.raises(ValueError, match="resume_config_mismatch"):
        await run_plan(
            changed, root=tmp_path, run_id="a", resume=True, transports={"mock:one": transport}
        )


async def test_retry_attempts_are_reserved_and_captured(tmp_path: Path) -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429, json={"error": "rate_limited"}, headers={"retry-after": "0.1"}
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    current = plan().model_copy(update={"overrides": {"S0": {"items": 1}}})
    result = await run_plan(
        current,
        root=tmp_path,
        transports={"mock:one": httpx.MockTransport(handler)},
        sleep=fake_sleep,
    )
    assert result.completed == 1 and sleeps
    record = next(RawStore(tmp_path / "data/raw").exchanges())
    assert [attempt.status for attempt in record.attempts] == [429, 200]
    assert record.attempts[0].response == {"error": "rate_limited"}


async def test_stream_capture_and_request_secrets_never_persist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-" + "privatevalue" * 4
    monkeypatch.setenv("TEST_API_KEY", secret)
    current = plan()
    current.endpoints[0].api_key_env = "TEST_API_KEY"
    current.endpoints[0].defaults = {"stream": True}
    await run_plan(
        current, root=tmp_path, transports={"mock:one": httpx.ASGITransport(app=create_app())}
    )
    for record in RawStore(tmp_path / "data/raw").exchanges():
        assert record.sse_events and record.sse_events[-1].data == "[DONE]"
        assert record.response and record.response["choices"]
        assert secret not in record.model_dump_json()
        assert record.request_headers["authorization"] == "[REDACTED]"


async def test_transport_failure_text_is_controlled_and_partial_body_retained(
    tmp_path: Path,
) -> None:
    secret = "sk-" + "opaquevalue" * 5

    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield ('{"error": "' + secret).encode()
            raise httpx.ReadError(secret)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=BrokenStream())

    async def fake_sleep(delay: float) -> None:
        pass

    current = plan().model_copy(update={"overrides": {"S0": {"items": 1}}})
    result = await run_plan(
        current,
        root=tmp_path,
        transports={"mock:one": httpx.MockTransport(handler)},
        sleep=fake_sleep,
    )
    record = next(RawStore(tmp_path / "data/raw").exchanges())
    assert result.errors == 1
    assert record.error == "transport_failure"
    assert secret not in record.model_dump_json()
    assert record.response_text and "[REDACTED]" in record.response_text
    assert len(record.attempts) == current.retries + 1


async def test_echoed_split_stream_credential_is_scrubbed_before_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-" + "opaquepart" * 4
    monkeypatch.setenv("TEST_API_KEY", secret)
    wire = (
        "".join(
            "data: " + json.dumps({"choices": [{"delta": {"content": secret[i : i + 7]}}]}) + "\n\n"
            for i in range(0, len(secret), 7)
        )
        + "data: [DONE]\n\n"
    )
    current = plan().model_copy(update={"overrides": {"S0": {"items": 1}}})
    current.endpoints[0].defaults = {"stream": True}
    current.endpoints[0].api_key_env = "TEST_API_KEY"
    result = await run_plan(
        current,
        root=tmp_path,
        transports={
            "mock:one": httpx.MockTransport(
                lambda request: httpx.Response(
                    200, text=wire, headers={"content-type": "text/event-stream"}
                )
            )
        },
    )
    record = next(RawStore(tmp_path / "data/raw").exchanges())
    assert result.errors == 1 and len(record.attempts) == 1
    assert record.error == "sensitive_response_redacted"
    assert all(e.data == "[REDACTED]" for e in record.sse_events or [])
    assert secret not in record.model_dump_json()
    for path in (tmp_path / "data").rglob("*.jsonl"):
        assert secret not in path.read_text()


async def test_live_execution_and_missing_auth_fail_before_dispatch(tmp_path: Path) -> None:
    from servegauge.core.errors import SafeError
    from servegauge.runner.budget import BudgetExceeded

    current = plan()
    current.endpoints[0].kind = "openai_compat"
    with pytest.raises(BudgetExceeded, match="explicit_yes"):
        await run_plan(current, root=tmp_path)
    current.endpoints[0].metadata = {
        "pricing": {"prompt": "0", "completion": "0"},
        "context_length": 1000,
    }
    current.endpoints[0].api_key_env = "SERVEGAUGE_NONEXISTENT_CREDENTIAL"
    with pytest.raises(SafeError, match="environment_missing"):
        await run_plan(current, root=tmp_path, approved=True)


async def test_proxy_uses_runner_and_rejects_other_urls_or_models(tmp_path: Path) -> None:
    from servegauge.runner.proxy import reserved_forwarder

    current = plan()
    forward = reserved_forwarder(
        current, root=tmp_path, transports={"mock:one": httpx.ASGITransport(app=create_app())}
    )
    body = {"model": "mock", "messages": [{"role": "user", "content": "hello"}]}
    result = await forward(current.endpoints[0].base_url, body)
    assert result["choices"]
    assert len(list(RawStore(tmp_path / "data/raw").exchanges())) == 1
    assert (tmp_path / "data/ledger/spend.jsonl").exists()
    with pytest.raises(ValueError, match="endpoint_or_model_mismatch"):
        await forward("https://other.example/v1", body)
    with pytest.raises(ValueError, match="endpoint_or_model_mismatch"):
        await forward(current.endpoints[0].base_url, {**body, "model": "wrong"})
    with pytest.raises(ValueError, match="exactly_one_endpoint"):
        reserved_forwarder(current.model_copy(update={"endpoints": current.endpoints * 2}))


async def test_budget_overrun_retains_full_exchange_before_halting(tmp_path: Path) -> None:
    current = plan().model_copy(update={"max_usd": 1, "overrides": {"S0": {"items": 1}}})
    endpoint = current.endpoints[0]
    endpoint.kind = "openai_compat"
    endpoint.metadata = {"pricing": {"prompt": "0", "completion": "0.01"}, "context_length": 1000}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"cost": 0.33, "prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    with pytest.raises(ExceptionGroup):
        await run_plan(
            current,
            root=tmp_path,
            approved=True,
            transports={endpoint.id: httpx.MockTransport(handler)},
        )
    record = next(RawStore(tmp_path / "data/raw").exchanges())
    assert record.error == "actual_exceeded_reservation"
    assert record.cost_usd == 0.33 and record.response
