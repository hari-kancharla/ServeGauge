"""Dispatch contract only; synthetic cards do not qualify a serving process."""

import importlib
import json
from types import SimpleNamespace

import pytest

from servegauge.core.hashing import content_hash
from servegauge.core.ids import safe_filename
from servegauge.core.models import CheckResult, Item
from servegauge.diagnose.rules import diagnoses
from servegauge.runner.plan import RunPlan
from servegauge.validation.capture import endpoint


def fixture(tmp_path, code, context):
    target = endpoint("mock:target", 101, clean=False)
    plan = RunPlan(
        plan_id="captured-replay",
        profile="gpt-oss-120b",
        level="core",
        seed=19,
        endpoints=[target, endpoint("mock:baseline", 102, clean=True)],
        suites=["S1", "S5", "S6", "S7", "S8"],
        reference={"tier": "BASELINE"},
        max_usd=0,
    )

    def card(check, suite):
        return CheckResult(
            check_id=check,
            suite_id=suite,
            check_class="comparative",
            run_id="source",
            endpoint_id=target.id,
            n=400,
            estimate=0.8,
            ci_low=0.75,
            ci_high=0.85,
            reference_estimate=1,
            reference_tier="BASELINE",
            delta=-0.2,
            delta_ci_low=-0.25,
            delta_ci_high=-0.15,
            margin=0.03,
            p_value=0.001,
            adjusted_p=0.003,
            verdict="FAIL",
            reason="fixture",
            evidence=["capture-" + check],
        )

    if code == "DIFFERENT_MODEL_LIKE":
        checks = [card("FP1", "S6"), card("FP2", "S6"), card("R01", "S5")]
        for check in checks[:2]:
            check.details["fingerprint_position"] = "different-model-band"
    elif code == "INFRA_UNRELIABLE":
        checks = [card("S8.health.requests", "S8")]
        checks[0].check_class, checks[0].verdict = "report_only", "INFO"
        checks[0].details["operational_alarm"] = True
    else:
        checks = [card("B01", "S7")]
    diagnosis = next(d for d in diagnoses(checks) if d.code == code)

    def item(identifier, suite):
        return Item(
            item_id=identifier,
            suite_id=suite,
            version="captured-v1",
            payload={"messages": [{"role": "user", "content": identifier}]},
            expected={"check_id": "P05", "samples": 1},
        )

    primary = [item("retained-root", "S1")]
    caps = [item("retained-preflight", "S0")]
    if context == "batch-primary-S0":
        primary = [*caps, *primary]
    capability_run = "preflight" if context == "separate" else None
    metadata = {
        "baseline_id": "mock:baseline",
        "tests": "S1,S5,S6,S7,S8",
        "capability_run_id": capability_run,
    }
    analysis = SimpleNamespace(
        plan=plan,
        items=primary,
        records=[],
        checks=checks,
        metadata=metadata,
        manifest={"budget_scope": "original-budget-scope"},
        reference_pack=None,
    )
    path = tmp_path / "data/raw" / safe_filename("preflight") / "capture.json"
    path.parent.mkdir(parents=True)
    captured = {"items": [i.model_dump(mode="json") for i in caps]}
    path.write_text(json.dumps({"value": captured, "hash": content_hash(captured)}))
    return analysis, diagnosis, primary, caps, path


def prepare(monkeypatch, analysis, diagnosis):
    module = importlib.import_module("servegauge.diagnose.repro")
    monkeypatch.setattr(
        module, "read_bundle", lambda *args: {"diagnosis": diagnosis.model_dump(mode="json")}
    )
    monkeypatch.setattr(module, "reconstruct", lambda *args: analysis)
    dispatched = []

    async def run(plan, **kwargs):
        dispatched.append(kwargs)
        return SimpleNamespace(checks=analysis.checks, run_id="replayed")

    monkeypatch.setattr(module, "run_check", run)
    return module, dispatched


@pytest.mark.parametrize(
    "code", ["DIFFERENT_MODEL_LIKE", "INFRA_UNRELIABLE", "ACCOUNTING_MISMATCH"]
)
@pytest.mark.parametrize("context", ["separate", "batch-primary-S0", "batch-no-preflight"])
async def test_derived_replay_uses_complete_captured_primary_and_preflight_inventories(
    tmp_path, monkeypatch, code, context
):
    analysis, diagnosis, primary, caps, _ = fixture(tmp_path, code, context)
    module, dispatched = prepare(monkeypatch, analysis, diagnosis)
    await module.replay_bundle("bundle", root=tmp_path, approved=True)
    assert len(dispatched) == 1
    request = dispatched[0]
    assert request["inventory"] is not None
    assert [i.model_dump(mode="json") for i in request["inventory"]] == [
        i.model_dump(mode="json") for i in primary
    ]
    assert request["capability_inventory"] is not None
    assert [i.model_dump(mode="json") for i in request["capability_inventory"]] == (
        [i.model_dump(mode="json") for i in caps] if context == "separate" else []
    )
    assert set(request["tests"].split(",")) == set(analysis.metadata["tests"].split(","))
    assert request["budget_scope"] == "original-budget-scope"


@pytest.mark.parametrize(
    "code", ["DIFFERENT_MODEL_LIKE", "INFRA_UNRELIABLE", "ACCOUNTING_MISMATCH"]
)
@pytest.mark.parametrize("fault", ["hash", "metadata-type"])
async def test_changed_or_unknown_derived_preflight_context_refuses_before_dispatch(
    tmp_path, monkeypatch, code, fault
):
    analysis, diagnosis, _, _, path = fixture(tmp_path, code, "separate")
    if fault == "hash":
        value = json.loads(path.read_text())
        value["value"]["items"][0]["payload"]["messages"][0]["content"] = "changed"
        path.write_text(json.dumps(value))
        reason = "scientific_context_hash_mismatch"
    else:
        analysis.metadata["capability_run_id"] = 13
        reason = "replay_requires_original_capability_inventory"
    module, dispatched = prepare(monkeypatch, analysis, diagnosis)
    with pytest.raises(ValueError, match=reason):
        await module.replay_bundle("bundle", root=tmp_path, approved=True)
    assert not dispatched
