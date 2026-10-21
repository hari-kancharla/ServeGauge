"""Filtered, resumable window jobs must retain one reserved parent scope."""

import copy
import json
from pathlib import Path

import httpx
import pytest

from servegauge.core.ids import safe_filename
from servegauge.mock.faults import FaultConfig
from servegauge.mock.server import create_app
from servegauge.profiles import load_profile
from servegauge.runner.endpoints import resolve_endpoint
from servegauge.runner.execute import execute_plan
from servegauge.runner.plan import RunPlan
from servegauge.storage.analysis import reconstruct
from servegauge.storage.raw_store import RawStore
from servegauge.suites.s0_capabilities import capability_items
from servegauge.suites.s1_protocol import protocol_items
from servegauge.suites.s8_operations import operational_items


@pytest.fixture
def window_plan(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/budgets.yaml").write_bytes(Path("configs/budgets.yaml").read_bytes())
    endpoint = resolve_endpoint("http://127.0.0.1:8011/v1", "mock", "gpt-oss-120b")
    plan = RunPlan(
        plan_id="window-job",
        profile="gpt-oss-120b",
        level="smoke",
        seed=1305,
        endpoints=[endpoint],
        reference={"tier": "MOCK"},
        suites=["S0", "S1"],
        max_usd=0,
        requests_per_minute=60000,
    )
    probes = [
        i
        for i in protocol_items(
            "smoke",
            plan.seed,
            endpoint=endpoint,
            profile=load_profile(plan.profile),
            overrides={"invariant_trials": 4},
        )
        if i.expected.get("check_id") == "P01"
    ]
    assert len(probes) == 4
    return plan, probes


def manifests(root):
    return [json.loads(p.read_text()) for p in (root / "data/runs").glob("*/manifest.json")]


async def test_filtered_inventory_and_stable_resume_do_not_generate_or_send_other_items(
    tmp_path, window_plan
):
    plan, probes = window_plan
    before = copy.deepcopy(probes)
    app = create_app(seed=1305)
    sent = []

    @app.middleware("http")
    async def observed(request, call_next):
        sent.append(await request.json())
        return await call_next(request)

    transports = {plan.endpoints[0].id: httpx.ASGITransport(app=app)}
    first = await execute_plan(
        plan,
        root=tmp_path,
        run_id="window-job-1",
        approved=True,
        inventory=probes,
        budget_scope="window-parent",
        transports=transports,
    )
    assert len(sent) == 4 and probes == before
    original = reconstruct(tmp_path / "data/raw", first.run_id)
    primary = list(RawStore(tmp_path / "data/raw").exchanges(run_id=first.run_id))
    assert {r.item_id for r in primary} == {i.item_id for i in probes}
    assert all(m["budget_scope"] == "window-parent" for m in manifests(tmp_path))
    saved = (tmp_path / "data/raw" / safe_filename(first.run_id) / "analysis.json").read_bytes()
    again = await execute_plan(
        plan,
        root=tmp_path,
        run_id=first.run_id,
        approved=True,
        resume=True,
        inventory=probes,
        budget_scope="window-parent",
        transports=transports,
    )
    assert again.run_id == first.run_id and len(sent) == 4
    assert reconstruct(tmp_path / "data/raw", first.run_id).checks == original.checks
    assert (
        tmp_path / "data/raw" / safe_filename(first.run_id) / "analysis.json"
    ).read_bytes() == saved


@pytest.mark.parametrize("kind", ["empty", "duplicate", "wrong_suite"])
async def test_invalid_window_inventory_refuses_before_transport_or_raw_writes(
    tmp_path, window_plan, kind
):
    plan, probes = window_plan
    selected = [] if kind == "empty" else [probes[0], probes[0]]
    if kind == "wrong_suite":
        selected = [capability_items()[0]]
        plan.suites = ["S1"]
    with pytest.raises(ValueError, match="inventory"):
        await execute_plan(
            plan, root=tmp_path, approved=True, inventory=selected, budget_scope="window-parent"
        )
    assert not (tmp_path / "data/raw").exists()


async def test_parent_scope_covers_preflight_primary_and_real_confirmation(tmp_path, window_plan):
    plan, probes = window_plan
    transports = {
        plan.endpoints[0].id: httpx.ASGITransport(
            app=create_app(seed=1305, faults=FaultConfig(ignore_max_tokens=True))
        )
    }
    result = await execute_plan(
        plan,
        root=tmp_path,
        run_id="window-fault",
        approved=True,
        inventory=[*capability_items(), *probes],
        budget_scope="window-parent",
        transports=transports,
    )
    saved = manifests(tmp_path)
    assert any(m["run_id"].endswith("-capabilities") for m in saved)
    assert any("confirmation" in m["run_id"] for m in saved)
    assert all(m["budget_scope"] == "window-parent" for m in saved)
    assert any(
        c.check_id == "P01" and c.verdict == "FAIL"
        for c in reconstruct(tmp_path / "data/raw", result.run_id).checks
    )


async def test_parent_scope_covers_actual_agent_continuation_turns(tmp_path, window_plan):
    plan, _ = window_plan
    plan.suites = ["S0", "S8"]
    task = next(i for i in operational_items("smoke", plan.seed) if "agent_goal" in i.expected)
    transports = {plan.endpoints[0].id: httpx.ASGITransport(app=create_app(seed=1305))}
    await execute_plan(
        plan,
        root=tmp_path,
        run_id="window-task",
        approved=True,
        inventory=[*capability_items(), task],
        budget_scope="window-parent",
        transports=transports,
    )
    saved = manifests(tmp_path)
    assert any("-agent-" in m["run_id"] for m in saved)
    assert len(saved) > 2 and all(m["budget_scope"] == "window-parent" for m in saved)


async def test_resume_after_preflight_creates_primary_once_and_never_repeats_preflight(
    tmp_path, window_plan, monkeypatch
):
    import servegauge.runner.execute as execution

    plan, probes = window_plan
    service = create_app(seed=1305)
    sent = []

    @service.middleware("http")
    async def observed(request, call_next):
        sent.append(await request.json())
        return await call_next(request)

    transports = {plan.endpoints[0].id: httpx.ASGITransport(app=service)}
    ordinary = execution.run_plan

    async def interrupted(plan, **kwargs):
        if kwargs["run_id"] == "window-interrupted":
            raise RuntimeError("controlled interruption after preflight")
        return await ordinary(plan, **kwargs)

    monkeypatch.setattr(execution, "run_plan", interrupted)
    with pytest.raises(RuntimeError, match="controlled interruption"):
        await execute_plan(
            plan,
            root=tmp_path,
            run_id="window-interrupted",
            approved=True,
            inventory=[*capability_items(), *probes],
            budget_scope="window-parent",
            transports=transports,
        )
    assert len(sent) == len(capability_items())
    monkeypatch.setattr(execution, "run_plan", ordinary)
    result = await execute_plan(
        plan,
        root=tmp_path,
        run_id="window-interrupted",
        approved=True,
        resume=True,
        inventory=[*capability_items(), *probes],
        budget_scope="window-parent",
        transports=transports,
    )
    assert len(sent) == len(capability_items()) + len(probes)
    assert reconstruct(tmp_path / "data/raw", result.run_id).checks


async def test_missing_preflight_manifest_cannot_trigger_recollection_on_resume(
    tmp_path, window_plan
):
    plan, probes = window_plan
    service = create_app(seed=1305)
    sent = []

    @service.middleware("http")
    async def observed(request, call_next):
        sent.append(await request.json())
        return await call_next(request)

    transports = {plan.endpoints[0].id: httpx.ASGITransport(app=service)}
    result = await execute_plan(
        plan,
        root=tmp_path,
        run_id="window-missing-caps",
        approved=True,
        inventory=[*capability_items(), *probes],
        budget_scope="window-parent",
        transports=transports,
    )
    count = len(sent)
    manifest = (
        tmp_path / "data/runs" / safe_filename(result.run_id + "-capabilities") / "manifest.json"
    )
    manifest.rename(manifest.with_name("preserved-manifest.json"))
    with pytest.raises(ValueError, match="preflight"):
        await execute_plan(
            plan,
            root=tmp_path,
            run_id=result.run_id,
            approved=True,
            resume=True,
            inventory=[*capability_items(), *probes],
            budget_scope="window-parent",
            transports=transports,
        )
    assert len(sent) == count


async def test_two_mocked_paid_jobs_charge_one_parent_and_report_its_cumulative_spend(
    tmp_path, window_plan
):
    from decimal import Decimal

    from servegauge.runner.ledger import Ledger

    plan, probes = window_plan
    plan.endpoints[0].kind = "openai_compat"
    plan.endpoints[0].metadata["pricing"] = {"prompt": "0.00001", "completion": "0.00001"}
    plan.endpoints[0].metadata.update(
        {"context_length": 512, "max_prompt_tokens": 1000, "max_completion_tokens": 4096}
    )
    plan.max_usd = 0.1
    (tmp_path / "configs/budgets.yaml").write_bytes(
        Path("tests/fixtures/simulated-budgets.yaml").read_bytes()
    )
    sent = []

    def fake_provider(request):
        sent.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}
                ],
                "usage": {"cost": 0.0001},
            },
        )

    transports = {plan.endpoints[0].id: httpx.MockTransport(fake_provider)}
    for index in range(2):
        result = await execute_plan(
            plan,
            root=tmp_path,
            run_id=f"window-paid-{index}",
            approved=True,
            inventory=probes[:2],
            budget_scope="window-parent",
            transports=transports,
        )
        ledger = Ledger(tmp_path)
        ledger.refresh()
        assert result.spent_usd == ledger.by_run["window-parent"] == Decimal("0.0001") * len(sent)
        assert set(ledger.by_run) == {"window-parent"}
    assert len(sent) == 4
    assert all(m["budget_scope"] == "window-parent" for m in manifests(tmp_path))


async def test_custom_S0_inventory_is_scientific_and_does_not_use_legacy_transport_shortcut(
    tmp_path, window_plan
):
    plan, _ = window_plan
    plan.suites = ["S0"]
    plan.overrides["S0"] = {"items": 2}
    inventory = capability_items()
    result = await execute_plan(
        plan,
        root=tmp_path,
        run_id="window-capabilities",
        approved=True,
        inventory=inventory,
        budget_scope="window-parent",
        transports={plan.endpoints[0].id: httpx.ASGITransport(app=create_app())},
    )
    analysis = reconstruct(tmp_path / "data/raw", result.run_id)
    assert len(analysis.records) == len(inventory)
    assert any(c.check_id == "S0.answers" for c in analysis.checks)
    assert all(m["budget_scope"] == "window-parent" for m in manifests(tmp_path))
