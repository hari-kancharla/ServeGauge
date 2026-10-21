import numpy as np
import pytest

from servegauge.core.models import CheckResult
from servegauge.runner.plan import Requirements
from servegauge.verdicts.config import load_policy
from servegauge.verdicts.engine import Measurement, decide_family, map_requirements


def measurement(**changes):
    return Measurement(
        check_id="T02",
        suite_id="S2",
        endpoint_id="e",
        run_id="r",
        check_class="comparative",
        outcomes=[1.0] * 500,
        reference=[1.0] * 500,
        reference_tier="BASELINE",
        noise_band=0,
        seed=1,
        **changes,
    )


def test_invariant_table_and_no_reference_paths() -> None:
    policy = load_policy()
    for outcomes, replay, expected in [
        ([0.0] * 400, False, "PASS"),
        ([0.0] * 10, False, "INCONCLUSIVE"),
        ([1.0, 1.0] + [0.0] * 398, True, "FAIL"),
        ([1.0, 1.0] + [0.0] * 398, False, "INCONCLUSIVE"),
        ([1.0] + [0.0] * 399, True, "INCONCLUSIVE"),
        ([], False, "INCONCLUSIVE"),
    ]:
        item = Measurement(
            check_id="P01",
            suite_id="S1",
            endpoint_id="e",
            run_id="r",
            check_class="invariant",
            outcomes=outcomes,
            seed=1,
            replay_confirmed=replay,
        )
        result = decide_family([item], policy)[0]
        assert result.verdict == expected
        assert result.n == len(outcomes)
        assert result.reference_estimate == 0
        if outcomes:
            assert result.ci_low is not None and result.ci_high is not None
    result = decide_family([measurement().model_copy(update={"reference": None})], policy)[0]
    assert result.verdict == "INCONCLUSIVE" and result.reason == "no baseline or reference"


@pytest.mark.parametrize(
    "updates,expected",
    [
        ({}, "PASS"),
        ({"reference": [1.0] * 500, "outcomes": [0.0] * 500}, "FAIL"),
        ({"outcomes": [1.0] * 5, "reference": [1.0] * 5}, "INCONCLUSIVE"),
        ({"noise_band": 0.1}, "INCONCLUSIVE"),
        ({"noise_band": None}, "INCONCLUSIVE"),
        ({"supported": False}, "NOT_SUPPORTED"),
        ({"applicable": False}, "NOT_APPLICABLE"),
        ({"error": "transport_failure"}, "ERROR"),
        ({"verified": False}, "INCONCLUSIVE"),
        ({"outcomes": [], "reference": []}, "INCONCLUSIVE"),
        ({"check_class": "report_only", "reference": None}, "INFO"),
        ({"two_sided": True}, "PASS"),
        ({"two_sided": True, "outcomes": [0.0] * 500}, "FAIL"),
        ({"reference_tier": None}, "INCONCLUSIVE"),
    ],
)
def test_every_verdict_path(updates, expected) -> None:
    result = decide_family([measurement().model_copy(update=updates)], load_policy())[0]
    assert result.verdict == expected
    if result.verdict == "FAIL":
        assert result.adjusted_p <= 0.05 and abs(result.delta) > result.margin


def test_continuous_clustered_paired_and_family_adjustment() -> None:
    rng = np.random.default_rng(123)
    base = rng.normal(size=400)
    candidate = base + rng.normal(scale=0.01, size=400)
    continuous = measurement().model_copy(
        update={
            "binary": False,
            "outcomes": candidate.tolist(),
            "reference": base.tolist(),
            "clusters": [str(i // 2) for i in range(400)],
            "two_sided": True,
        }
    )
    result = decide_family([continuous], load_policy())[0]
    assert result.verdict == "PASS" and result.n == 200
    assert result.details["requests"] == 400 and result.details["bootstrap_resamples"] >= 2000
    assert result.mde > 0 and result.details["worst_case_holm_alpha"] == 0.05
    worse = measurement().model_copy(update={"outcomes": [0.0] * 500})
    family = decide_family([measurement(), worse], load_policy())
    assert family[1].verdict == "FAIL" and family[1].adjusted_p >= family[1].p_value
    assert family[0].details["worst_case_holm_alpha"] == 0.025


def test_margins_requirements_and_invalid_paired_designs() -> None:
    policy = load_policy()
    assert policy.margin("T02") == 0.02 and policy.margin("T01") == 0.03
    assert policy.margin("T02", 0.01) == 0.01
    with pytest.raises(ValueError, match="tighten"):
        policy.margin("T02", 0.04)
    result = decide_family([measurement()], policy)[0]
    requirements = Requirements.model_validate(
        {"app": "a", "requires": [{"id": "tools", "tests": ["T02"], "margin": 0.01}]}
    )
    mapped = map_requirements(requirements, [measurement()], policy)
    assert mapped[0].requirement_ids == ["tools"]
    assert mapped[0].declared_margin == 0.01
    assert decide_family(mapped, policy)[0].margin == 0.01
    with pytest.raises(ValueError, match="paired"):
        decide_family([measurement().model_copy(update={"reference": [1.0]})], policy)
    with pytest.raises(ValueError, match="paired"):
        decide_family([measurement().model_copy(update={"clusters": ["a"]})], policy)
    assert CheckResult.model_validate(result.model_dump()).n == 500


def test_effort_ratio_rules_holm_and_overlapping_corner() -> None:
    import math

    policy = load_policy()
    template = measurement().model_copy(
        update={
            "check_id": "P07",
            "suite_id": "S1",
            "binary": False,
            "outcomes": [math.log(8)] * 30,
            "reference": [math.log(8)] * 30,
            "details": {"decision_rule": "effort"},
        }
    )
    clean = decide_family([template], policy)[0]
    assert clean.verdict == "PASS"
    assert clean.details["geometric_mean_ratio"] == pytest.approx(8)
    ignored = decide_family([template.model_copy(update={"outcomes": [0.0] * 30})], policy)[0]
    assert ignored.verdict == "FAIL" and ignored.adjusted_p < 0.05
    corner = decide_family(
        [template.model_copy(update={"outcomes": [0.0] * 30, "reference": [math.log(1.5)] * 30})],
        policy,
    )[0]
    assert corner.verdict == "INCONCLUSIVE"
    assert (
        decide_family([template.model_copy(update={"reference": None})], policy)[0].reason
        == "no baseline or reference"
    )
