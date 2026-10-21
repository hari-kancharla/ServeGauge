"""A guarded drift hypothesis keeps sub-margin evidence and full replay controls."""

import copy

import pytest

from servegauge.core.models import CheckResult, EndpointSpec
from servegauge.diagnose.drift import DRIFT_CONTRACT, REQUIRED_CONTROLS, annotate_context
from servegauge.diagnose.replay_expectations import (
    expected_context,
    mark_reproduced,
    replay_matches,
)
from servegauge.diagnose.rules import diagnoses
from servegauge.verdicts.classes import check_class
from servegauge.verdicts.config import load_policy


def fixture():
    checks = []
    for name in REQUIRED_CONTROLS:
        invariant = check_class(name) == "invariant"
        checks.append(
            CheckResult(
                check_id=name,
                check_class=check_class(name),
                suite_id="S1" if name.startswith("P") else "S2",
                endpoint_id="target",
                run_id="run",
                n=1000,
                estimate=0 if invariant else 1,
                ci_low=0 if invariant else 0.99,
                ci_high=0.005 if invariant else 1,
                reference_estimate=0 if invariant else 1,
                reference_tier=None if invariant else "BASELINE",
                delta=None if invariant else 0,
                delta_ci_low=None if invariant else -0.01,
                delta_ci_high=None if invariant else 0.01,
                margin=0.01 if invariant else load_policy().margin(name),
                violations=0 if invariant else None,
                verdict="PASS",
                reason="qualified control",
                evidence=[name + "-evidence"],
            )
        )
    for method, delta, low, high, noise, lower in [
        ("FP1", 0.07, 0.05, 0.09, 0.02, 0.13),
        ("FP2", 0.016, 0.012, 0.020, 0.01, 0.025),
    ]:
        controls = [
            {
                "control_run_id": "ref-" + k,
                "control_endpoint_id": k,
                "model_identity": k,
                "capture_hash": "captured-" + k,
                "bands_separated": True,
                "different_model_band": [lower, lower + 0.1],
                "position": "between-bands",
            }
            for k in ["model-a", "model-b"]
        ]
        metric = {
            "n": 40,
            "delta": delta,
            "delta_ci_low": low,
            "delta_ci_high": high,
            "noise_band": noise,
            "position": "between-bands",
            "control_positions": controls,
            "equivalence_qualified": True,
        }
        checks.append(
            CheckResult(
                check_id=method,
                check_class="comparative",
                suite_id="S6",
                endpoint_id="target",
                run_id="run",
                n=40,
                estimate=delta,
                ci_low=low,
                ci_high=high,
                reference_estimate=0,
                reference_tier="MOCK",
                delta=delta,
                delta_ci_low=low,
                delta_ci_high=high,
                margin=0.03,
                p_value=0.001,
                adjusted_p=0.003,
                verdict="FAIL" if method == "FP1" else "INCONCLUSIVE",
                reason="observed interval; practical verdict unchanged",
                evidence=[method + "-evidence"],
                details={
                    "fingerprint_position": "between-bands",
                    "reference_pack_id": "unit-fixture-pack",
                    "equivalence_qualified": True,
                    "qualification": {"id": "unit-calibration-fixture", "binding": "unit-only"},
                    "fingerprint_metrics": {method: metric},
                },
            )
        )
    checks.append(
        CheckResult(
            check_id="R01",
            check_class="comparative",
            suite_id="S5",
            endpoint_id="target",
            run_id="run",
            n=10000,
            estimate=0.93,
            ci_low=0.924,
            ci_high=0.936,
            reference_estimate=0.95,
            reference_tier="BASELINE",
            delta=-0.02,
            delta_ci_low=-0.025,
            delta_ci_high=-0.015,
            margin=0.03,
            p_value=0.001,
            adjusted_p=0.003,
            verdict="PASS",
            reason="small consistent loss within practical margin",
            evidence=["accuracy-evidence"],
        )
    )
    endpoint = EndpointSpec(
        id="target",
        kind="mock",
        base_url="http://127.0.0.1:8011/v1",
        model="mock",
        claimed_profile="gpt-oss-120b",
        labels={"quantization": "fp4"},
    )
    annotate_context(checks, [endpoint], load_policy(), contract=DRIFT_CONTRACT)
    return checks, endpoint


def drift(checks):
    return [d for d in diagnoses(checks) if d.code == "QUANTIZATION_LIKE_DRIFT"]


def test_guarded_drift_includes_submargin_method_and_all_passing_prerequisites():
    checks, _ = fixture()
    found = drift(checks)
    assert len(found) == 1 and found[0].failure_class == "MODEL_BEHAVIOR"
    assert set(found[0].check_ids) == {*REQUIRED_CONTROLS, "FP1", "FP2", "R01"}
    assert found[0].confidence == "low" and "DISCLOSED" in found[0].summary
    assert "fp4" in found[0].summary and "not hardware" in found[0].summary
    assert any("mixed" in text for text in found[0].alternatives)
    second = next(c for c in checks if c.check_id == "FP2")
    assert second.verdict == "INCONCLUSIVE" and second.delta < second.margin


@pytest.mark.parametrize("name", REQUIRED_CONTROLS)
def test_missing_required_control_never_establishes_clean_api_context(name):
    checks, _ = fixture()
    assert not drift([c for c in checks if c.check_id != name])


@pytest.mark.parametrize("status", ["FAIL", "ERROR", "INCONCLUSIVE", "NOT_SUPPORTED"])
def test_uncertain_or_failed_required_control_abstains(status):
    checks, _ = fixture()
    next(c for c in checks if c.check_id == "P03").verdict = status
    assert not drift(checks)


@pytest.mark.parametrize(
    "fault",
    [
        "noise-boundary",
        "different-boundary",
        "overlap",
        "negative",
        "missing-qualification",
        "missing-provenance",
        "same-control",
        "bad-band",
        "nonnumeric-band",
        "wrong-position",
        "not-significant",
        "info",
        "cross-run",
        "duplicate",
    ],
)
def test_incomplete_or_contradictory_fingerprint_geometry_abstains(fault):
    checks, _ = fixture()
    second = next(c for c in checks if c.check_id == "FP2")
    metric = second.details["fingerprint_metrics"]["FP2"]
    if fault == "noise-boundary":
        second.delta_ci_low = metric["noise_band"]
    elif fault == "different-boundary":
        second.delta_ci_high = metric["control_positions"][0]["different_model_band"][0]
    elif fault == "overlap":
        metric["control_positions"][0]["bands_separated"] = False
    elif fault == "negative":
        second.delta, second.delta_ci_low, second.delta_ci_high = -0.016, -0.02, -0.012
    elif fault == "missing-qualification":
        second.details.pop("qualification")
    elif fault == "missing-provenance":
        metric["control_positions"][0].pop("capture_hash")
    elif fault == "same-control":
        metric["control_positions"][1]["model_identity"] = "model-a"
    elif fault == "bad-band":
        metric["control_positions"][0]["different_model_band"] = [0.4, 0.1]
    elif fault == "nonnumeric-band":
        metric["control_positions"][0]["different_model_band"] = ["invalid", 0.1]
    elif fault == "wrong-position":
        metric["control_positions"][0]["position"] = "different-model-band"
    elif fault == "not-significant":
        second.adjusted_p = 0.5
    elif fault == "info":
        second.verdict = "INFO"
        second.check_class = "report_only"
        second.details["signal_detected"] = True
    elif fault == "cross-run":
        second.run_id = "other"
    else:
        checks.append(second.model_copy(deep=True))
    assert not drift(checks)


@pytest.mark.parametrize(
    "fault",
    [
        "no-loss",
        "uncertain-direction",
        "margin-boundary",
        "large-loss",
        "no-reference",
        "duplicate-run",
    ],
)
def test_accuracy_context_is_small_directional_and_same_run(fault):
    checks, _ = fixture()
    accuracy = next(c for c in checks if c.check_id == "R01")
    if fault == "no-loss":
        accuracy.delta, accuracy.delta_ci_low, accuracy.delta_ci_high = 0, -0.01, 0.01
    elif fault == "uncertain-direction":
        accuracy.delta_ci_high = 0
    elif fault == "margin-boundary":
        accuracy.delta_ci_low = -accuracy.margin
    elif fault == "large-loss":
        accuracy.verdict = "FAIL"
        accuracy.delta, accuracy.delta_ci_low, accuracy.delta_ci_high = -0.2, -0.25, -0.15
    elif fault == "no-reference":
        accuracy.reference_tier = None
    else:
        accuracy.run_id = "other"
    assert not drift(checks)


def test_other_api_failure_and_operational_alarm_prevent_drift_attribution():
    checks, _ = fixture()
    extra = checks[0].model_copy(deep=True)
    extra.check_id, extra.verdict = "P13", "FAIL"
    assert not drift([*checks, extra])
    extra.check_id, extra.verdict, extra.check_class = "S8.health.requests", "INFO", "report_only"
    extra.details["operational_alarm"] = True
    assert not drift([*checks, extra])


def test_no_real_practical_failure_means_no_diagnosis_or_fabricated_fail():
    checks, _ = fixture()
    next(c for c in checks if c.check_id == "FP1").verdict = "INCONCLUSIVE"
    assert not drift(checks)


def test_full_replay_rechecks_numeric_signature_even_if_all_states_stay_equal():
    checks, _ = fixture()
    diagnosis = drift(checks)[0]
    context = expected_context(diagnosis, checks)
    assert context["expected"]["FP2"] == "INCONCLUSIVE"
    changed = copy.deepcopy(checks)
    c = next(c for c in changed if c.check_id == "R01")
    c.delta, c.delta_ci_low, c.delta_ci_high = 0, -0.01, 0.01
    assert {c.check_id: c.verdict for c in changed} == {c.check_id: c.verdict for c in checks}
    assert not replay_matches(changed, context, endpoint_id="target")
    assert not mark_reproduced(changed, context)
    assert mark_reproduced(checks, context)
    assert drift(checks)[0].confidence == "medium"


@pytest.mark.parametrize(
    "metadata,labels,status",
    [
        ({}, {}, "UNVERIFIED"),
        ({"quantization": None}, {}, "UNDISCLOSED"),
        ({}, {"quantization": "unknown"}, "UNDISCLOSED"),
        ({}, {"quantization": "bf16"}, "DISCLOSED"),
        ({"quantization": "fp4"}, {"quantization": "bf16"}, "UNVERIFIED"),
        ({}, {"quantization": "[REDACTED]"}, "UNVERIFIED"),
    ],
)
def test_disclosure_is_captured_context_not_a_failure_or_hardware_fact(metadata, labels, status):
    checks, endpoint = fixture()
    endpoint.metadata, endpoint.labels = metadata, labels
    annotate_context(checks, [endpoint], load_policy(), contract=DRIFT_CONTRACT)
    assert len(drift(checks)) == 1 and status in drift(checks)[0].summary
    context = next(c for c in checks if c.check_id == "FP1").details["drift_context"]
    assert context["disclosure"]["status"] == status
    assert context["disclosure"]["hardware_verified"] is False


def test_legacy_and_unknown_context_contracts_cannot_silently_change_interpretation():
    checks, endpoint = fixture()
    for check in checks:
        check.details.pop("drift_context", None)
    annotate_context(checks, [endpoint], load_policy())
    assert not drift(checks)
    with pytest.raises(ValueError, match="drift_contract"):
        annotate_context(checks, [endpoint], load_policy(), contract={"version": "unknown"})


@pytest.mark.parametrize(
    "fault",
    [
        "metrics-list",
        "missing-disclosure",
        "bad-disclosure",
        "metric-ci-conflict",
        "missing-reference-value",
        "reused-control-capture",
    ],
)
def test_malformed_or_inconsistent_bound_context_abstains_without_crashing(fault):
    checks, _ = fixture()
    second = next(c for c in checks if c.check_id == "FP2")
    if fault == "metrics-list":
        second.details["fingerprint_metrics"] = []
    elif fault == "missing-disclosure":
        second.details["drift_context"].pop("disclosure")
    elif fault == "bad-disclosure":
        for c in checks:
            if "drift_context" in c.details:
                c.details["drift_context"]["disclosure"] = None
    elif fault == "metric-ci-conflict":
        second.details["fingerprint_metrics"]["FP2"]["delta_ci_high"] = 0.1
    elif fault == "missing-reference-value":
        second.reference_estimate = None
    else:
        row = second.details["fingerprint_metrics"]["FP2"]["control_positions"]
        row[1]["capture_hash"] = row[0]["capture_hash"]
    assert not drift(checks)
