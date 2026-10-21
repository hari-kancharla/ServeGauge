import copy

import pytest

from servegauge.core.models import CheckResult
from servegauge.mock.faults import FaultConfig
from servegauge.suites.s6_fingerprint import fingerprint_items
from servegauge.validation.fingerprint_wire_capture import capture_run, outcome_signals
from servegauge.verdicts.config import load_policy


def checks(*, delta=0, low=-0.01, high=0.01, verdict="INCONCLUSIVE", log_signal=False):
    result = []
    for method in ("FP1", "FP2", "FP3"):
        metrics = {
            name: {
                "position": "same-model-band",
                "p_value": 1.0,
                "delta_ci_low": -0.01,
                "control_positions": [{"position": "same-model-band"}] * 2,
            }
            for name in (["FP3-mean", "FP3-dispersion"] if method == "FP3" else [method])
        }
        result.append(
            CheckResult(
                check_id=method,
                suite_id="S6",
                check_class="report_only" if method == "FP3" else "comparative",
                endpoint_id="target",
                run_id="run",
                n=400,
                verdict="INFO" if method == "FP3" else verdict,
                reason="fixture",
                reference_tier="MOCK",
                delta=delta,
                delta_ci_low=low,
                delta_ci_high=high,
                p_value=1,
                adjusted_p=1,
                margin=None if method == "FP3" else 0.03,
                details={
                    "noise_band": 0.01,
                    "fingerprint_metrics": metrics,
                    "signal_detected": log_signal,
                },
            )
        )
    return result


def test_text_equivalence_candidate_does_not_enable_or_mutate_a_real_verdict():
    rows = checks()
    before = copy.deepcopy(rows)
    signals = outcome_signals(rows, load_policy())
    assert signals["text_equivalence_candidate"] is True
    assert not signals["swap_signature"] and not signals["noise_signature"]
    assert rows == before and all(r.verdict != "PASS" for r in rows)
    rows[0].details["noise_band"] = 0.1
    assert outcome_signals(rows, load_policy())["text_equivalence_candidate"] is False


def test_noise_requires_the_corrected_dispersion_signal_and_positive_interval():
    rows = checks(log_signal=True)
    dispersion = rows[2].details["fingerprint_metrics"]["FP3-dispersion"]
    dispersion.update(
        p_value=0.001, delta_ci_low=0.04, control_positions=[{"position": "between-bands"}] * 2
    )
    result = outcome_signals(rows, load_policy())
    assert result["noise_signature"] is True and result["text_equivalence_candidate"] is True
    dispersion["p_value"] = 0.04
    assert outcome_signals(rows, load_policy())["noise_signature"] is False
    dispersion.update(p_value=0.001, delta_ci_low=-0.01)
    assert outcome_signals(rows, load_policy())["noise_signature"] is False


def test_swap_requires_both_text_failures_in_the_positive_different_band():
    rows = checks(delta=0.2, low=0.1, high=0.3, verdict="FAIL")
    for row in rows[:2]:
        row.details["fingerprint_metrics"][row.check_id]["position"] = (
            "different-model-band-or-beyond"
        )
    assert outcome_signals(rows, load_policy())["swap_signature"] is True
    rows[1].details["fingerprint_metrics"]["FP2"]["position"] = "below-same-model-band"
    assert outcome_signals(rows, load_policy())["swap_signature"] is False
    rows[1].verdict = "INCONCLUSIVE"
    assert outcome_signals(rows, load_policy())["swap_signature"] is False


def test_mixture_uses_actual_signals_and_an_individual_between_control():
    rows = checks(delta=0.1, low=0.06, high=0.14, verdict="FAIL", log_signal=True)
    rows[0].details["fingerprint_metrics"]["FP1"]["control_positions"] = [
        {"position": "between-bands"}
    ] * 2
    rows[2].details["fingerprint_metrics"]["FP3-mean"]["control_positions"] = [
        {"position": "beyond-different-model-band"},
        {"position": "between-bands"},
    ]
    assert outcome_signals(rows, load_policy())["mixture_signature"] is True
    rows[2].details["signal_detected"] = False
    assert outcome_signals(rows, load_policy())["mixture_signature"] is False


def test_missing_or_duplicate_checks_cannot_supply_success():
    assert not any(outcome_signals([], load_policy()).values())
    incomplete = checks(verdict="FAIL")[:2]
    result = outcome_signals(incomplete, load_policy())
    assert result["any_comparative_fail"] is True
    assert result["text_equivalence_candidate"] is False
    with pytest.raises(ValueError):
        outcome_signals(checks() + checks()[:1], load_policy())


def test_captured_target_identity_and_declaration_do_not_reveal_fault_truth():
    items = fingerprint_items(
        "core", 91, {"methods": ["FP1", "FP3"], "fp1_cells": 2, "fp1_samples": 2, "fp3_samples": 2}
    )
    first, a = capture_run(
        items, load_policy(), seeds=[101], source_id="opaque-run", healthy_jitter=0.3
    )
    second, b = capture_run(
        items,
        load_policy(),
        seeds=[101],
        source_id="opaque-run",
        healthy_jitter=0.3,
        fault=FaultConfig(logit_noise=0.3),
    )
    assert first.manifest == second.manifest
    assert first.endpoints == second.endpoints
    assert first.endpoints[0].clean is False
    assert a[0]["captures"] == b[0]["captures"] == 6
    assert first.fingerprints.keys() == second.fingerprints.keys()
    assert a[0]["record_hash"] != b[0]["record_hash"]
    _, repeated = capture_run(
        items, load_policy(), seeds=[101], source_id="opaque-run", healthy_jitter=0.3
    )
    assert repeated == a


def test_faulty_stimulus_cannot_be_declared_a_clean_reference():
    items = fingerprint_items("smoke", 92, {"fp1_cells": 2, "fp1_samples": 2, "fp3_samples": 2})
    with pytest.raises(ValueError):
        capture_run(
            items,
            load_policy(),
            seeds=[103],
            source_id="reference",
            healthy_jitter=0.3,
            fault=FaultConfig(logit_noise=0.3),
            reference=True,
        )
