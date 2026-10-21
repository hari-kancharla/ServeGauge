"""The health lab captures complete attempts and bounded terminal agent failures."""

import pytest

from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.runner.plan import RunPlan
from servegauge.scoring.operational_health import health_samples
from servegauge.suites.s8_operations import HEALTH_CHECKS, operational_items
from servegauge.validation.capture import endpoint
from servegauge.validation.operational_health import procedure_captures, run_health_replicate


def test_health_lab_captures_attempt_response_and_terminal_agent_failures():
    target = endpoint("mock:lab", 91, clean=False)
    items = operational_items("smoke", 19, {"items": 2, "agent_tasks": 2})
    plan = RunPlan(
        plan_id="health-lab",
        profile=target.claimed_profile,
        level="smoke",
        seed=19,
        reference={"tier": "MOCK"},
        endpoints=[target],
        suites=["S8"],
        retries=0,
        max_usd=0,
    )
    healthy = procedure_captures(items, ScriptedBackend(91, FaultConfig()), target, "run")
    assert len(healthy) == 10
    assert all(
        r.attempts[0].response == r.response and r.attempts[0].metadata == r.metadata
        for r in healthy
    )
    samples = health_samples(plan, items, healthy, target.id, "run")
    assert all(
        s.status is None and len(s.values) == 4 and min(s.values.values()) == 1
        for s in samples.values()
    )
    failed = procedure_captures(
        items, ScriptedBackend(91, FaultConfig(error_rate=1)), target, "run"
    )
    assert len(failed) == 6 and all(r.status == 503 for r in failed)
    samples = health_samples(plan, items, failed, target.id, "run")
    assert samples["S8.health.requests"].status is None
    assert all(v == 0 for v in samples["S8.health.requests"].values.values())
    assert samples["S8.health.nonempty"].status == "INCONCLUSIVE"


def test_health_lab_uses_real_paired_decisions_and_mock_pack():
    clean = run_health_replicate(92713, FaultConfig())
    assert {c["check_id"] for c in clean["checks"]} == set(HEALTH_CHECKS)
    assert all(
        c["comparison_verdict"] == "PASS" and c["reference_tier"] == "MOCK" for c in clean["checks"]
    )
    assert not clean["diagnoses"]
    fault = run_health_replicate(92714, FaultConfig(error_rate=1))
    assert any(c["operational_alarm"] for c in fault["checks"])
    assert any(
        d["code"] == "INFRA_UNRELIABLE" and d["failure_class"] == "ROUTING_OR_INFRA"
        for d in fault["diagnoses"]
    )
    assert all(c["n"] == 212 for c in fault["checks"] if c["operational_alarm"])


def test_health_validation_hashes_budget_without_writing_it_or_accepting_worker_failure(
    tmp_path, monkeypatch
):
    import servegauge.validation.operational_health as lab

    (tmp_path / "budget").mkdir()
    budget = tmp_path / "budget/dev_spend.json"
    original = '{"dev_spend_usd":"0"}\n'
    budget.write_text(original)
    fault_bytes = (lab.ROOT / "configs/faults/errors.yaml").read_bytes()
    (tmp_path / "configs/faults").mkdir(parents=True)
    (tmp_path / "configs/faults/errors.yaml").write_bytes(fault_bytes)
    monkeypatch.setattr(lab, "ROOT", tmp_path)

    class UnavailablePool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            raise RuntimeError("controlled_worker_unavailable")

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(lab, "ProcessPoolExecutor", UnavailablePool)
    with pytest.raises(RuntimeError, match="controlled_worker_unavailable"):
        lab.health_validation(tmp_path / "report")
    assert budget.read_text() == original
    assert not (tmp_path / "report/health.json").exists()


def test_health_validation_cannot_overwrite_an_existing_study(tmp_path):
    from servegauge.validation.operational_health import health_validation

    path = tmp_path / "health.json"
    original = '{"preserved":"historical"}\n'
    path.write_text(original)
    with pytest.raises(ValueError, match="output_exists"):
        health_validation(tmp_path)
    assert path.read_text() == original


def test_fault_template_seed_cannot_override_an_independent_trial_seed():
    first = run_health_replicate(918273, FaultConfig(error_rate=0.05, seed=7))
    second = run_health_replicate(918274, FaultConfig(error_rate=0.05, seed=7))
    assert first["backend_seeds"]["mock:target"] == first["seed"] == 918273
    assert second["backend_seeds"]["mock:target"] == second["seed"] == 918274
    assert set(first["backend_seeds"].values()).isdisjoint(second["backend_seeds"].values())


def test_health_capturer_rejects_a_mislabeled_backend_seed():
    target = endpoint("mock:seed-proof", 91, clean=False)
    items = operational_items("smoke", 19, {"items": 2, "agent_tasks": 0})
    backend = ScriptedBackend(91, FaultConfig(error_rate=0.05, seed=7))
    with pytest.raises(ValueError, match="backend_seed_mismatch"):
        procedure_captures(items, backend, target, "run")
