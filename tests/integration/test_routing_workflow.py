"""Provider mismatches, including failed preflight, use reservations and raw replay."""

import importlib
import json
import shutil

import httpx
import pytest

from servegauge.core.models import EndpointSpec, OpenRouterPin
from servegauge.diagnose.bundle import write_bundle
from servegauge.diagnose.repro import replay_bundle
from servegauge.diagnose.rules import diagnoses
from servegauge.runner.check import run_check
from servegauge.runner.execute import execute_plan
from servegauge.runner.plan import RunPlan
from servegauge.scoring.routing import ROUTING_CHECK, ROUTING_CONTRACT
from servegauge.storage.analysis import reconstruct


@pytest.mark.parametrize(
    ("entrypoint", "mode"),
    [
        ("check", "preflight"),
        ("check", "primary"),
        ("check", "clean"),
        ("check", "routing-only"),
        ("run", "preflight"),
        ("run", "primary"),
        ("run", "clean"),
        ("run-no-preflight", "primary"),
        ("run-capabilities", "preflight"),
    ],
)
async def test_reserved_routing_metadata_preflight_replay_and_raw_only(
    tmp_path, monkeypatch, entrypoint, mode
):
    (tmp_path / "configs").mkdir()
    shutil.copyfile("configs/budgets.yaml", tmp_path / "configs/budgets.yaml")
    endpoint = EndpointSpec(
        id="or:fixture@provider-a/fp4/us",
        kind="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="example/model",
        claimed_profile="gpt-oss-120b",
        openrouter_pin=OpenRouterPin(tag="provider-a/fp4/us"),
        labels={"provider_name": "Provider A"},
        metadata={
            "tag": "provider-a/fp4/us",
            "provider_name": "Provider A",
            "status": 0,
            "supported_parameters": [
                "max_tokens",
                "temperature",
                "top_p",
                "top_k",
                "seed",
                "stop",
                "tools",
                "tool_choice",
                "parallel_tool_calls",
                "response_format",
                "reasoning",
                "logprobs",
                "top_logprobs",
            ],
            "supports_tool_choice": {
                "none": True,
                "required": True,
                "auto": True,
                "function": True,
            },
            "max_completion_tokens": 8192,
            "context_length": 131072,
            "pricing": {"prompt": "0", "completion": "0"},
        },
    )
    plan = RunPlan(
        plan_id="routing-fixture",
        profile="gpt-oss-120b",
        level="core",
        seed=2031,
        reference={"tier": "BASELINE"},
        endpoints=[endpoint],
        suites=["S8"],
        max_usd=0,
        retries=0,
        requests_per_minute=600000,
        overrides={"S8": {"items": 400, "agent_tasks": 0}},
    )
    generations = {}
    posts = []

    def handler(request):
        if request.url.path.endswith("/key"):
            return httpx.Response(200, json={"data": {"limit": 10, "limit_remaining": 10}})
        if request.url.path.endswith("/generation"):
            identifier = request.url.params["id"]
            return httpx.Response(200, json={"data": generations[identifier]})
        assert request.method == "POST" and request.url.path.endswith("/chat/completions")
        body = json.loads(request.content)
        assert body["provider"]["only"] == [endpoint.openrouter_pin.tag]
        assert body["provider"]["allow_fallbacks"] is False
        posts.append(body)
        identifier = f"gen-fixture-{len(posts)}"
        primary = "Synthetic client timing case" in body["messages"][0]["content"]
        wrong = (mode == "primary" and primary) or (
            mode in {"preflight", "routing-only"} and not primary
        )
        generations[identifier] = {
            "id": identifier,
            "provider_name": "Provider B" if wrong else "Provider A",
            "total_cost": 0,
            "provider_responses": [
                {"provider_name": "Provider B"},
                {"provider_name": "Provider A"},
            ],
        }
        response = {
            "id": identifier,
            "model": endpoint.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "OK"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        if not body.get("stream"):
            return httpx.Response(200, json=response)
        frames = [
            {
                "id": identifier,
                "model": endpoint.model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "OK"}}],
            },
            {
                "id": identifier,
                "model": endpoint.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": response["usage"],
            },
        ]
        stream = "".join("data: " + json.dumps(f) + "\n\n" for f in frames) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=stream)

    transport = {endpoint.id: httpx.MockTransport(handler)}
    tests = ROUTING_CHECK if mode == "routing-only" else "operations"
    if entrypoint == "check":
        checked = await run_check(
            plan, root=tmp_path, approved=True, tests=tests, transports=transport
        )
    else:
        plan.suites = (
            ["S8"]
            if entrypoint == "run-no-preflight"
            else ["S0"]
            if entrypoint == "run-capabilities"
            else ["S0", "S8"]
        )
        checked = await execute_plan(plan, root=tmp_path, approved=True, transports=transport)
    source = reconstruct(tmp_path / "data/raw", checked.run_id)
    if entrypoint == "check":
        assert source.checks == checked.checks
    assert source.metadata["routing_contract"] == ROUTING_CONTRACT
    card = next(c for c in source.checks if c.check_id == ROUTING_CHECK)
    assert card.n > 0 and card.evidence and card.reference_estimate == 0
    assert not card.details["tag_verified"] and not card.details["fallback_qualified"]
    assert not any(c.verdict == "FAIL" for c in source.checks if c.check_id == ROUTING_CHECK)
    assert card.details["operational_alarm"] == (mode != "clean")
    replayed = None
    if mode == "clean":
        assert card.details["comparison_verdict"] == "PASS" and card.violations == 0
        assert not diagnoses(source.checks)
    else:
        assert card.details["comparison_verdict"] == "FAIL" and card.violations >= 2
        assert card.details["replay_confirmed"]
        if mode in {"preflight", "routing-only"}:
            assert any(c.check_id == "S0.answers" and c.verdict == "ERROR" for c in source.checks)
            assert not card.details["incomplete_trials"]
            assert card.details["unsent_units"] >= 1
        found = next(d for d in diagnoses(source.checks) if d.code == "ROUTING_INSTABILITY")
        assert found.failure_class == "ROUTING_OR_INFRA"
        bundle = write_bundle(source, found, root=tmp_path)

        def regeneration_forbidden(*args, **kwargs):
            raise AssertionError(
                "routing replay must use the original capability/primary inventory"
            )

        module = importlib.import_module("servegauge.runner.check")
        monkeypatch.setattr(module, "capability_items", regeneration_forbidden)
        monkeypatch.setattr(module, "operational_items", regeneration_forbidden)
        replay = await replay_bundle(
            str(bundle), root=tmp_path, approved=True, transports=transport
        )
        assert replay["reproduced"] and replay["budget_scope"] == checked.run_id
        assert replay["expected"] == {ROUTING_CHECK: "OPERATIONAL_ALARM"}
        replayed = reconstruct(tmp_path / "data/raw", replay["run_id"])
        assert [i.model_dump() for i in source.items] == [i.model_dump() for i in replayed.items]
    raw = tmp_path / "raw-only"
    shutil.copytree(tmp_path / "data/raw", raw)
    shutil.rmtree(tmp_path / "data/runs")
    for actual in (source, replayed):
        if actual is not None:
            rebuilt = reconstruct(raw, actual.manifest["run_id"])
            assert rebuilt.checks == actual.checks and rebuilt.samples == actual.samples
    (tmp_path / "routing-acceptance.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "entrypoint": entrypoint,
                "new_spend_usd": 0,
                "model_api_calls": 0,
                "capture_records_including_unsent": len(source.records)
                + (len(replayed.records) if replayed else 0),
                "sent_logical_requests": sum(
                    bool(r.attempts) for a in (source, replayed) if a for r in a.records
                ),
                "captured_attempts": sum(
                    len(r.attempts) for a in (source, replayed) if a for r in a.records
                ),
                "mock_transport_posts": len(posts),
                "raw_only_equal": True,
                "exact_inventory_replayed": replayed is not None,
                "card": card.model_dump(mode="json"),
            },
            indent=2,
        )
        + "\n"
    )
