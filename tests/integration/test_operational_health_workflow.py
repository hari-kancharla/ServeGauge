"""Calibrated S8 alarms use reserved matched traffic, saved packs and exact replay."""

import importlib
import json
import shutil

import httpx

from servegauge.diagnose.bundle import write_bundle
from servegauge.diagnose.repro import replay_bundle
from servegauge.diagnose.rules import diagnoses
from servegauge.mock.faults import FaultConfig
from servegauge.mock.server import create_app
from servegauge.refs.pack import build_pack, load_pack, save_pack
from servegauge.runner.check import run_check
from servegauge.runner.endpoints import resolve_endpoint
from servegauge.runner.evaluate import evaluate, reference_source
from servegauge.runner.plan import RunPlan
from servegauge.storage.analysis import reconstruct
from servegauge.suites.s8_operations import HEALTH_CHECKS
from servegauge.verdicts.config import load_policy


async def test_reserved_original_error_rate_alarm_pack_exact_inventory_replay_and_raw_only(
    tmp_path, monkeypatch
):
    (tmp_path / "configs").mkdir()
    shutil.copyfile("configs/budgets.yaml", tmp_path / "configs/budgets.yaml")
    target = resolve_endpoint(
        "http://127.0.0.1:8011/v1", "mock", "gpt-oss-120b", identifier="target"
    )
    baseline = resolve_endpoint(
        "http://127.0.0.1:8012/v1", "mock", "gpt-oss-120b", identifier="baseline"
    )
    plan = RunPlan(
        plan_id="operational-health",
        profile="gpt-oss-120b",
        level="core",
        seed=1931,
        reference={"tier": "BASELINE"},
        endpoints=[target, baseline],
        suites=["S8", "S1"],
        max_usd=0,
        retries=0,
        requests_per_minute=600000,
        overrides={"S8": {"items": 200, "agent_tasks": 3}, "S1": {"seed_trials": 4}},
    )

    def transports():
        return {
            target.id: httpx.ASGITransport(
                app=create_app(seed=1301, faults=FaultConfig(error_rate=0.05))
            ),
            baseline.id: httpx.ASGITransport(app=create_app(seed=1302)),
        }

    result = await run_check(
        plan,
        root=tmp_path,
        approved=True,
        tests="operations,P05",
        baseline_id=baseline.id,
        transports=transports(),
    )
    analysis = reconstruct(tmp_path / "data/raw", result.run_id)
    health = {c.check_id: c for c in result.checks if c.check_id in HEALTH_CHECKS}
    assert set(health) == set(HEALTH_CHECKS)
    assert health["S8.health.requests"].details["operational_alarm"]
    assert health["S8.health.attempts"].details["operational_alarm"]
    assert health["S8.health.requests"].n == 207
    assert health["S8.health.nonempty"].verdict == "INCONCLUSIVE"
    assert all(c.verdict != "FAIL" and c.check_class == "report_only" for c in health.values())
    diagnosis = next(d for d in diagnoses(result.checks) if d.code == "INFRA_UNRELIABLE")
    assert diagnosis.confidence == "low"
    assert set(diagnosis.check_ids) == {"S8.health.requests", "S8.health.attempts"}
    assert any(i.suite_id == "S1" for i in analysis.items)
    policy = load_policy()
    source = reference_source(
        analysis.plan,
        result.run_id,
        analysis.items,
        analysis.samples,
        policy,
        analysis.manifest,
        baseline_id=baseline.id,
    )
    pack = load_pack(
        str(save_pack(build_pack(source, "BASELINE", seed=plan.seed), tmp_path / "refs"))
    )
    packed = evaluate(
        analysis.plan,
        result.run_id,
        analysis.items,
        analysis.samples,
        policy,
        target_ids=[target.id],
        reference_pack=pack,
        operational_records=analysis.records,
    )
    assert [
        (c.check_id, c.details["operational_alarm"]) for c in packed if c.check_id in HEALTH_CHECKS
    ] == [(c.check_id, c.details["operational_alarm"]) for c in health.values()]
    bundle = write_bundle(analysis, diagnosis, root=tmp_path)

    def regeneration_forbidden(*args, **kwargs):
        raise AssertionError("derived operational replay must use the exact captured inventory")

    module = importlib.import_module("servegauge.runner.check")
    monkeypatch.setattr(module, "operational_items", regeneration_forbidden)
    monkeypatch.setattr(module, "protocol_items", regeneration_forbidden)
    replay = await replay_bundle(str(bundle), root=tmp_path, approved=True, transports=transports())
    assert replay["reproduced"] and replay["budget_scope"] == result.run_id
    assert set(replay["expected"].values()) == {"OPERATIONAL_ALARM"}
    repeated = reconstruct(tmp_path / "data/raw", replay["run_id"])
    assert [i.item_id for i in repeated.items] == [i.item_id for i in analysis.items]
    assert (
        next(d for d in diagnoses(repeated.checks) if d.code == diagnosis.code).confidence
        == "medium"
    )
    raw = tmp_path / "raw-only"
    shutil.copytree(tmp_path / "data/raw", raw)
    shutil.rmtree(tmp_path / "data/runs")
    shutil.copyfile("configs/suites.yaml", tmp_path / "configs/suites.yaml")
    monkeypatch.chdir(tmp_path)
    for actual in (analysis, repeated):
        rebuilt = reconstruct(raw, actual.manifest["run_id"])
        assert rebuilt.checks == actual.checks and rebuilt.samples == actual.samples
    (tmp_path / "operational-health-acceptance.json").write_text(
        json.dumps(
            {
                "scope": "original 5% errors, reserved matched baseline; scripted only",
                "model_api_calls": 0,
                "new_spend_usd": 0,
                "error_rate": 0.05,
                "n_root_procedures": 207,
                "captured_requests": len(analysis.records) + len(repeated.records),
                "captured_attempts": sum(
                    len(r.attempts) for a in (analysis, repeated) for r in a.records
                ),
                "baseline_pack_alarm_states_equal": True,
                "exact_source_inventory_replayed": True,
                "raw_only_checks_and_samples_equal": True,
                "shared_evidence_confidence": "medium",
                "cards": [c.model_dump(mode="json") for c in health.values()],
            },
            indent=2,
        )
        + "\n"
    )
