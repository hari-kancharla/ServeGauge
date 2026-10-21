import pytest

from servegauge.core.models import CheckResult
from servegauge.verdicts.config import load_policy
from servegauge.verdicts.engine import Measurement, decide_family


def fingerprint(**overrides):
    return CheckResult.model_validate(
        {
            "check_id": "FP1",
            "check_class": "comparative",
            "suite_id": "S6",
            "endpoint_id": "target",
            "run_id": "run",
            "n": 40,
            "estimate": 0.2,
            "reference_estimate": 0.1,
            "delta": 0.1,
            "delta_ci_low": 0.08,
            "delta_ci_high": 0.12,
            "margin": 0.03,
            "p_value": 0.04,
            "reference_tier": "MOCK",
            "verdict": "INCONCLUSIVE",
            "reason": "awaiting family decision",
            "details": {"noise_band": 0.01, "equivalence_qualified": False},
            **overrides,
        }
    )


def test_fingerprint_joins_whole_endpoint_family_and_does_not_mutate_input():
    policy = load_policy()
    external = fingerprint()
    alone = decide_family([], policy, external=[external])[0]
    assert alone.verdict == "FAIL" and alone.adjusted_p == 0.04
    unavailable = Measurement(
        check_id="T02",
        suite_id="S2",
        endpoint_id="target",
        run_id="run",
        check_class="comparative",
        outcomes=[],
        seed=8,
    )
    combined = decide_family([unavailable], policy, external=[external])[1]
    assert combined.verdict == "INCONCLUSIVE" and combined.adjusted_p == 0.08
    assert external.adjusted_p is None and external.verdict == "INCONCLUSIVE"


def test_unqualified_equivalence_or_noisy_reference_never_passes():
    policy = load_policy()
    clean = fingerprint(delta=0, delta_ci_low=-0.001, delta_ci_high=0.001, p_value=1)
    result = decide_family([], policy, external=[clean])[0]
    assert result.verdict == "INCONCLUSIVE" and "qualification" in result.reason
    clean.details["equivalence_qualified"] = True
    assert decide_family([], policy, external=[clean])[0].verdict == "PASS"
    clean.details["noise_band"] = 0.1
    result = decide_family([], policy, external=[clean])[0]
    assert result.verdict == "INCONCLUSIVE" and "noise" in result.reason
    assert result.margin == 0.1


def test_logprob_diagnostic_gets_family_correction_without_quality_verdict():
    info = fingerprint(check_id="FP3", check_class="report_only", verdict="INFO", margin=None)
    info.details["family_test"] = True
    result = decide_family([], load_policy(), external=[info])[0]
    assert result.verdict == "INFO" and result.adjusted_p == 0.04
    assert result.details["signal_detected"] is True and result.margin is None


@pytest.mark.parametrize(
    "override", [{"endpoint_id": "other"}, {"verdict": "PASS"}, {"margin": 0.5}]
)
def test_invalid_external_family_inputs_are_refused(override):
    base = Measurement(
        check_id="T02",
        suite_id="S2",
        endpoint_id="target",
        run_id="run",
        check_class="comparative",
        outcomes=[],
        seed=8,
    )
    with pytest.raises(ValueError):
        decide_family([base], load_policy(), external=[fingerprint(**override)])
