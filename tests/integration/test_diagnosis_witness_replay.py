"""Normal reserved replay of a diagnosis that requires a passing control."""

import hashlib
import json
import shutil

import httpx
import pytest

from servegauge.diagnose.bundle import write_bundle
from servegauge.diagnose.repro import replay_bundle
from servegauge.diagnose.rules import diagnoses
from servegauge.mock.faults import FaultConfig
from servegauge.mock.server import create_app
from servegauge.runner.check import run_check
from servegauge.runner.endpoints import resolve_endpoint
from servegauge.runner.plan import RunPlan
from servegauge.storage.analysis import reconstruct


async def test_reserved_long_output_witness_replay_controls_privacy_and_raw_equality(tmp_path):
    (tmp_path / "configs").mkdir()
    shutil.copyfile("configs/budgets.yaml", tmp_path / "configs/budgets.yaml")
    target = resolve_endpoint(
        "http://127.0.0.1:8011/v1", "mock", "gpt-oss-120b", identifier="target"
    )
    baseline = resolve_endpoint(
        "http://127.0.0.1:8012/v1", "mock", "gpt-oss-120b", identifier="baseline"
    )
    plan = RunPlan(
        plan_id="witness-replay",
        profile="gpt-oss-120b",
        level="core",
        seed=7281,
        reference={"tier": "BASELINE"},
        endpoints=[target, baseline],
        suites=["S5"],
        max_usd=0,
        retries=0,
        requests_per_minute=600000,
        overrides={"S5": {"items": 1, "samples": 1, "long_items": 112, "lengths": [4000, 8000]}},
    )

    def transports(cut):
        return {
            target.id: httpx.ASGITransport(
                app=create_app(seed=121, faults=FaultConfig(truncate_output_after=cut))
            ),
            baseline.id: httpx.ASGITransport(app=create_app(seed=122)),
        }

    source = await run_check(
        plan,
        root=tmp_path,
        approved=True,
        tests="R03.4000,R03.8000",
        baseline_id=baseline.id,
        transports=transports(6000),
    )
    analysis = reconstruct(tmp_path / "data/raw", source.run_id)
    diagnosis = next(d for d in diagnoses(analysis.checks) if d.code == "LONG_OUTPUT_FAULT")
    assert diagnosis.check_ids == ["R03.4000", "R03.8000"]
    assert diagnosis.confidence == "low"
    path = write_bundle(analysis, diagnosis, root=tmp_path)
    bundle = json.loads((path / "bundle.json").read_text())
    assert bundle["replay_expectations"]["expected"] == {"R03.4000": "PASS", "R03.8000": "FAIL"}
    assert {c["check_id"] for c in bundle["expected_vs_observed"]} == set(diagnosis.check_ids)
    assert {r["exchange_id"] for r in bundle["exchanges"]} == set(diagnosis.evidence)

    reproduced = await replay_bundle(
        str(path), root=tmp_path, approved=True, transports=transports(6000)
    )
    assert reproduced["reproduced"] and reproduced["expected_code"] == "LONG_OUTPUT_FAULT"
    assert reproduced["budget_scope"] == source.run_id
    positive = reconstruct(tmp_path / "data/raw", reproduced["run_id"])
    assert {c.check_id: c for c in positive.checks}["R03.4000"].details[
        "bundle_replay_reproduced"
    ] == diagnosis.diagnosis_id
    assert (
        next(d for d in diagnoses(positive.checks) if d.code == diagnosis.code).confidence
        == "medium"
    )

    changed = await replay_bundle(
        str(path), root=tmp_path, approved=True, transports=transports(3000)
    )
    assert not changed["reproduced"]
    assert changed["observed"]["R03.8000"]["verdict"] == "FAIL"
    assert changed["observed"]["R03.4000"]["verdict"] == "FAIL"
    assert "LONG_OUTPUT_FAULT" not in changed["observed_codes"]
    negative = reconstruct(tmp_path / "data/raw", changed["run_id"])
    assert not any(c.details.get("bundle_replay_reproduced") for c in negative.checks)

    original_bytes = (path / "bundle.json").read_bytes()
    with pytest.raises(ValueError, match="unsupported_diagnosis"):
        write_bundle(analysis, diagnosis.model_copy(update={"confidence": "high"}), root=tmp_path)
    assert (path / "bundle.json").read_bytes() == original_bytes
    tampered = json.loads(original_bytes)
    tampered["replay_expectations"]["expected"]["R03.4000"] = "FAIL"
    (path / "bundle.json").write_text(json.dumps(tampered))
    dispatched = []

    def forbidden(request):
        dispatched.append(request.method)
        raise AssertionError("forged bundle must fail before any request")

    with pytest.raises(ValueError, match="bundle_expectations_not_supported"):
        await replay_bundle(
            str(path),
            root=tmp_path,
            approved=True,
            transports={
                target.id: httpx.MockTransport(forbidden),
                baseline.id: httpx.MockTransport(forbidden),
            },
        )
    assert not dispatched
    (path / "bundle.json").write_bytes(original_bytes)

    marker = "PRIVATE_WITNESS_TEST_PAYLOAD"
    analysis.plan.endpoints[0].labels[marker] = marker
    redacted = write_bundle(analysis, diagnosis, root=tmp_path, redact_content=True)
    assert all(marker not in p.read_text() for p in redacted.iterdir())
    raw = tmp_path / "raw-only"
    shutil.copytree(tmp_path / "data/raw", raw)
    shutil.rmtree(tmp_path / "data/runs")
    for actual in (analysis, positive, negative):
        rebuilt = reconstruct(raw, actual.manifest["run_id"])
        assert rebuilt.checks == actual.checks
        assert diagnoses(rebuilt.checks) == diagnoses(actual.checks)
    receipt = {
        "scope": "reserved scripted long-output fault/control replay; no provider validation",
        "source_run": source.run_id,
        "new_spend_usd": 0,
        "model_api_calls": 0,
        "check_ids": diagnosis.check_ids,
        "expected": reproduced["expected"],
        "same_witness_reproduced": True,
        "changed_control_rejected": True,
        "forged_expectations_dispatched_requests": len(dispatched),
        "redacted_marker_absent": True,
        "raw_only_checks_and_diagnoses_equal": True,
        "captured_requests": sum(len(a.records) for a in (analysis, positive, negative)),
        "captured_attempts": sum(
            len(r.attempts) for a in (analysis, positive, negative) for r in a.records
        ),
        "source_confidence": diagnosis.confidence,
        "replayed_confidence": "medium",
        "original_private_bundle_sha256": hashlib.sha256(original_bytes).hexdigest(),
    }
    (tmp_path / "witness-replay-acceptance.json").write_text(json.dumps(receipt, indent=2) + "\n")
