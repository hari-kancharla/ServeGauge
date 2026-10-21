from pathlib import Path

import httpx

from servegauge.core.ids import safe_filename
from servegauge.mock.server import create_app
from servegauge.refs.pack import ReferenceRun, build_pack, save_pack
from servegauge.runner.endpoints import resolve_endpoint
from servegauge.runner.execute import execute_plan
from servegauge.runner.plan import RunPlan
from servegauge.storage.analysis import reconstruct
from servegauge.storage.scientific import score_run
from servegauge.suites.s0_capabilities import capability_items


async def test_round_preflights_then_scores_builds_R3_and_reconstructs_attachment(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/budgets.yaml").write_bytes(Path("configs/budgets.yaml").read_bytes())
    endpoints = [
        resolve_endpoint(
            f"http://127.0.0.1:{8011 + index}/v1", "mock", "gpt-oss-120b", identifier=str(index)
        )
        for index in range(4)
    ]
    plan = RunPlan(
        plan_id="round",
        profile="gpt-oss-120b",
        level="smoke",
        seed=42,
        reference={"tier": "R3"},
        suites=["S0", "S7"],
        endpoints=endpoints,
        max_usd=0,
        requests_per_minute=60000,
        overrides={"S7": {"items": 8}},
    )
    incoming = []
    transports = {}
    for index, endpoint in enumerate(endpoints):
        service = create_app(seed=index)

        @service.middleware("http")
        async def observed(request, call_next):
            body = await request.json()
            incoming.append(
                "accounting" if "Synthetic tokenizer case" in str(body) else "capability"
            )
            return await call_next(request)

        transports[endpoint.id] = httpx.ASGITransport(app=service)
    result = await execute_plan(plan, root=tmp_path, approved=True, transports=transports)
    assert incoming.index("accounting") == len(capability_items()) * 4
    analysis, found = score_run(tmp_path, result.run_id)
    assert not found and all(
        check.verdict == "INCONCLUSIVE" for check in analysis.checks if check.check_id == "B01"
    )
    directory = tmp_path / "data/runs" / safe_filename(result.run_id)
    source = ReferenceRun.model_validate_json((directory / "outcomes.json").read_bytes())
    pack = build_pack(source, "R3", seed=42)
    path = save_pack(pack, tmp_path / "refs")
    qualified, _ = score_run(tmp_path, result.run_id, reference=str(path))
    assert all(
        check.reference_tier == "R3"
        and check.verdict == "INCONCLUSIVE"
        and "noise exceeds" in check.reason
        for check in qualified.checks
        if check.check_id == "B01"
    )
    offline = reconstruct(tmp_path / "data/raw", result.run_id)
    assert [check.model_dump() for check in qualified.checks] == [
        check.model_dump() for check in offline.checks
    ]


async def test_final_spend_includes_confirmation_calls(tmp_path):
    import json
    from decimal import Decimal

    from servegauge.core.models import EndpointSpec
    from servegauge.runner.ledger import Ledger

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/budgets.yaml").write_bytes(
        Path("tests/fixtures/simulated-budgets.yaml").read_bytes()
    )
    endpoint = EndpointSpec(
        id="paid",
        kind="openai_compat",
        base_url="http://127.0.0.1/v1",
        model="example",
        claimed_profile="gpt-oss-120b",
        metadata={
            "context_length": 512,
            "max_prompt_tokens": 1000,
            "max_completion_tokens": 4096,
            "pricing": {"prompt": "0.000001", "completion": "0"},
        },
    )
    plan = RunPlan(
        plan_id="final-spend",
        profile="gpt-oss-120b",
        level="smoke",
        seed=42,
        reference={"tier": "R3"},
        suites=["S1"],
        endpoints=[endpoint],
        max_usd=0.1,
        requests_per_minute=60000,
        overrides={
            "S1": {
                "invariant_trials": 5,
                "diversity_prompts": 1,
                "diversity_repeats": 2,
                "effort_items": 1,
                "seed_trials": 1,
            }
        },
    )
    sent = 0

    def handler(request):
        nonlocal sent
        sent += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}
                ],
                "usage": {"cost": 0.0001},
            },
        )

    result = await execute_plan(
        plan, root=tmp_path, approved=True, transports={"paid": httpx.MockTransport(handler)}
    )
    directory = tmp_path / "data/runs" / safe_filename(result.run_id)
    metadata = json.loads((directory / "evaluation.json").read_text())
    assert any(metadata["replay_by_endpoint"]["paid"].values())
    ledger = Ledger(tmp_path)
    ledger.refresh()
    assert result.spent_usd == ledger.by_run[result.run_id] == Decimal("0.0001") * sent
    assert json.loads((directory / "result.json").read_text())["spent_usd"] == str(result.spent_usd)
