"""Reserved operational collection, simulated actions and sealed raw-only replay."""

import json
import shutil
import time
from types import SimpleNamespace

import httpx
import pytest

from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.mock.server import create_app, stream_frames
from servegauge.runner.check import normalize_tests, run_check
from servegauge.runner.endpoints import resolve_endpoint
from servegauge.runner.estimate import estimate_plan
from servegauge.runner.execute import execute_plan
from servegauge.runner.plan import RunPlan
from servegauge.scoring.common import calls
from servegauge.scoring.routing import ROUTING_CHECK
from servegauge.storage.analysis import reconstruct
from servegauge.suites.s8_operations import HEALTH_CHECKS, OPERATIONAL_CHECKS
from servegauge.transport import http


def plan(root, *, items=8, agents=3):
    (root / "configs").mkdir()
    shutil.copyfile("configs/budgets.yaml", root / "configs/budgets.yaml")
    endpoint = resolve_endpoint("http://127.0.0.1:8011/v1", "mock", "gpt-oss-120b")
    return RunPlan(
        plan_id="operations",
        profile="gpt-oss-120b",
        level="smoke",
        seed=5011,
        reference={"tier": "MOCK"},
        endpoints=[endpoint],
        suites=["S8"],
        max_usd=0,
        requests_per_minute=600000,
        retries=0,
        concurrency_per_endpoint=1,
        overrides={"S8": {"items": items, "agent_tasks": agents}},
    )


async def test_operational_check_agents_raw_reconstruction_and_estimate(tmp_path):
    config = plan(tmp_path)
    estimate = estimate_plan(config)
    assert estimate["total_reserved_cost_usd"] == "0"
    assert estimate["cost_table"][0]["n_requests"] == 16 + 3 * 5
    assert normalize_tests("operations") == {"S8"}
    assert all(d["mde"] is None for d in estimate["mde_table"] if d["check"] not in HEALTH_CHECKS)
    assert all(d["mde"] is not None for d in estimate["mde_table"] if d["check"] in HEALTH_CHECKS)
    result = await run_check(
        config,
        root=tmp_path,
        approved=True,
        tests="operations",
        transports={config.endpoints[0].id: httpx.ASGITransport(app=create_app(seed=91))},
    )
    analysis = reconstruct(tmp_path / "data/raw", result.run_id)
    ops = {r.check_id: r for r in result.checks if r.check_id.startswith("S8.")}
    assert set(ops) == set(OPERATIONAL_CHECKS) | set(HEALTH_CHECKS) | {ROUTING_CHECK}
    assert ops[ROUTING_CHECK].verdict == "NOT_APPLICABLE"
    assert all(r.check_class == "report_only" and r.verdict != "FAIL" for r in ops.values())
    assert ops["S8.agent_task_success_rate"].estimate == 1
    assert ops["S8.agent_task_success_rate"].n == 3
    assert ops["S8.completion_tokens_per_action"].n == 3
    task_records = [r for r in analysis.records if r.item_id.startswith("ops-agent-")]
    assert len(task_records) == 9
    actions = sum(len(calls(r)) for r in task_records)
    assert actions == 9
    native_tokens = sum(r.usage["completion_tokens"] for r in task_records)
    assert ops["S8.completion_tokens_per_action"].estimate == pytest.approx(native_tokens / actions)
    assert len(ops["S8.completion_tokens_per_action"].evidence) == 9
    assert ops["S8.request_error_rate"].details["observed_denominator"] == 25
    assert ops["S8.request_error_rate"].estimate == 0
    assert ops["S8.request_error_rate"].ci_high > 0
    raw = tmp_path / "raw-only"
    shutil.copytree(tmp_path / "data/raw", raw)
    shutil.rmtree(tmp_path / "data/runs")
    rebuilt = reconstruct(raw, result.run_id)
    assert rebuilt.checks == analysis.checks
    assert [r for r in rebuilt.checks if r.suite_id == "S8"] == list(ops.values())
    receipt = {
        "scope": "reserved ASGI scripted tasks; no provider performance claim",
        "model_api_calls": 0,
        "spend_usd": 0,
        "static_requests": 16,
        "agent_tasks": 3,
        "agent_turn_requests": 9,
        "executed_actions": actions,
        "raw_only_equal": True,
        "cards": [r.model_dump(mode="json") for r in ops.values()],
    }
    (tmp_path / "operations-acceptance.json").write_text(json.dumps(receipt, indent=2) + "\n")


async def test_operational_transport_faults_preserve_request_attempt_and_empty_denominators(
    tmp_path,
):
    config = plan(tmp_path, items=6, agents=0)
    config.retries = 1
    config.retry_base_seconds = config.retry_max_seconds = 0
    delegate = httpx.ASGITransport(app=create_app(seed=93))
    retried = False

    async def respond(request):
        nonlocal retried
        if request.method == "POST":
            body = json.loads(request.content)
            prompt = str(body["messages"])
            if not body.get("stream"):
                if "timing case 5011." in prompt and not retried:
                    retried = True
                    return httpx.Response(503, json={"error": {"code": "unavailable"}})
                if "timing case 5012." in prompt:
                    raise httpx.ReadTimeout("controlled_timeout")
                if "timing case 5013." in prompt:
                    return httpx.Response(200, content=b"")
        return await delegate.handle_async_request(request)

    result = await execute_plan(
        config,
        root=tmp_path,
        approved=True,
        transports={config.endpoints[0].id: httpx.MockTransport(respond)},
    )
    analysis = reconstruct(tmp_path / "data/raw", result.run_id)
    by = {r.check_id: r for r in analysis.checks if r.suite_id == "S8"}
    assert len(analysis.records) == 12
    assert sum(len(r.attempts) for r in analysis.records) == 14
    assert by["S8.request_error_rate"].estimate == pytest.approx(2 / 12)
    assert by["S8.attempt_error_rate"].estimate == pytest.approx(4 / 14)
    assert by["S8.request_timeout_rate"].estimate == pytest.approx(1 / 12)
    assert by["S8.attempt_timeout_rate"].estimate == pytest.approx(2 / 14)
    assert by["S8.empty_response_rate"].estimate == pytest.approx(1 / 11)
    assert all(
        by[key].verdict == "INFO"
        for key in (
            "S8.request_error_rate",
            "S8.attempt_error_rate",
            "S8.request_timeout_rate",
            "S8.attempt_timeout_rate",
            "S8.empty_response_rate",
        )
    )
    raw = tmp_path / "raw-only"
    shutil.copytree(tmp_path / "data/raw", raw)
    shutil.rmtree(tmp_path / "data/runs")
    assert reconstruct(raw, result.run_id).checks == analysis.checks
    (tmp_path / "operations-fault-acceptance.json").write_text(
        json.dumps(
            {
                "scope": "reserved 503/timeout/empty controls; no provider reliability claim",
                "model_api_calls": 0,
                "spend_usd": 0,
                "requests": 12,
                "attempts": 14,
                "raw_only_equal": True,
                "cards": [r.model_dump(mode="json") for r in by.values()],
            },
            indent=2,
        )
        + "\n"
    )


class Clock:
    now = 1000.0

    def monotonic(self):
        return self.now


class ClockStream(httpx.AsyncByteStream):
    def __init__(self, clock, chunks):
        self.clock, self.chunks = clock, chunks

    async def __aiter__(self):
        for delay, frame in self.chunks:
            self.clock.now += delay / 1000
            yield frame


async def test_controlled_sse_first_body_first_output_and_throughput(tmp_path, monkeypatch):
    config = plan(tmp_path, agents=0)
    clock = Clock()
    monkeypatch.setattr(http, "time", SimpleNamespace(monotonic=clock.monotonic, time=time.time))
    backend = ScriptedBackend(92, FaultConfig())
    metadata = httpx.ASGITransport(app=create_app(seed=92))

    async def respond(request):
        if request.method == "GET":
            return await metadata.handle_async_request(request)
        body = json.loads(request.content)
        response = backend.respond(body)
        if not body.get("stream"):
            clock.now += 0.1
            return httpx.Response(200, json=response)
        frames = stream_frames(response, FaultConfig())
        output = response["choices"][0]["message"]["content"]
        if output == "one two three four":

            def event(delta, finish=None, **fields):
                return (
                    "data: "
                    + json.dumps(
                        {
                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                            **fields,
                        }
                    )
                    + "\n\n"
                ).encode()

            chunks = [
                (5, b": keepalive\n\n"),
                (5, event({"role": "assistant"})),
                (10, event({"content": "one two"})),
                (20, event({"content": " three four"})),
                (5, event({}, finish="stop", usage=response["usage"])),
                (5, b"data: [DONE]\n\n"),
            ]
        else:
            chunks = [(1, f) for f in frames]
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=ClockStream(clock, chunks)
        )

    result = await execute_plan(
        config,
        root=tmp_path,
        approved=True,
        transports={config.endpoints[0].id: httpx.MockTransport(respond)},
    )
    analysis = reconstruct(tmp_path / "data/raw", result.run_id)
    by = {r.check_id: r for r in analysis.checks if r.suite_id == "S8"}
    assert by["S8.ttft_ms"].estimate == pytest.approx(20)
    assert by["S8.first_body_ms"].estimate == pytest.approx(52.5)
    assert by["S8.output_tokens_per_second"].estimate == pytest.approx(64 / 1.2)
    assert by["S8.delivery_tokens_per_second"].estimate == pytest.approx(200)
    assert by["S8.ttft_ms"].n == 8
    assert by["S8.ttft_ms"].details["excluded_observations"] == 8
    assert by["S8.completion_tokens_per_action"].verdict == "INCONCLUSIVE"
    assert all(
        r.verdict == "INFO"
        for key, r in by.items()
        if key
        not in {
            "S8.completion_tokens_per_action",
            "S8.agent_task_success_rate",
            *HEALTH_CHECKS,
            ROUTING_CHECK,
        }
    )
    assert by[ROUTING_CHECK].verdict == "NOT_APPLICABLE"
    raw = tmp_path / "raw-only"
    shutil.copytree(tmp_path / "data/raw", raw)
    shutil.rmtree(tmp_path / "data/runs")
    assert reconstruct(raw, result.run_id).checks == analysis.checks
    (tmp_path / "operations-timing-acceptance.json").write_text(
        json.dumps(
            {
                "scope": "virtual client clock over reserved MockTransport; no server clock claim",
                "model_api_calls": 0,
                "spend_usd": 0,
                "requests": 16,
                "raw_only_equal": True,
                "cards": [r.model_dump(mode="json") for r in by.values()],
            },
            indent=2,
        )
        + "\n"
    )
