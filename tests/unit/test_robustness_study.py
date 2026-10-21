import copy

import pytest

from servegauge.scoring.fingerprint_analysis import analyze_evidence, group_evidence
from servegauge.scoring.fingerprint_evidence import prepare_evidence
from servegauge.validation.robustness_study import (
    configuration,
    generate_evidence,
    guarded_evidence,
    primary_trial,
)
from servegauge.verdicts.config import load_policy


@pytest.mark.parametrize("profile", ["iid", "correlated", "cache_bursts", "linked_cache"])
def test_generator_records_actual_probabilities_and_known_effective_groups(profile):
    cfg = configuration()
    evidence = generate_evidence(919, profile, groups=4, noise=0.3, cfg=cfg)
    assert evidence == generate_evidence(919, profile, groups=4, noise=0.3, cfg=cfg)
    parsed = prepare_evidence(evidence, minimum_anchor_tokens=2, tolerance=0.001)
    assert parsed.logprob_reason is None
    groups = group_evidence(evidence, parsed)
    assert groups.reason is None and len(groups.pairs) == 4
    assert len(evidence.target) == 4 * cfg["observations_per_session"] * (
        2 if profile == "linked_cache" else 1
    )


def test_primary_statistics_match_actual_analysis_path_with_same_policy():
    cfg = configuration()
    evidence = generate_evidence(771, "correlated", groups=30, noise=0.3, cfg=cfg)
    trial = primary_trial(771, "correlated", 30, 0.3, cfg, legacy=False)
    result = analyze_evidence(
        evidence, policy=load_policy(), minimum_groups=30, minimum_anchor_tokens=2
    )
    for check in result["checks"]:
        assert check["p_value"] == trial["results"][check["check_id"]]["p_value"]
        assert check["adjusted_p"] == trial["results"][check["check_id"]]["adjusted_p"]


@pytest.mark.parametrize(
    "guard",
    [
        "missing_all",
        "missing_one",
        "missing_anchor",
        "unknown_dependence",
        "mismatched_sampling",
        "mismatched_cache",
        "one_cache_component",
        "cache_hit_missing",
        "invalid_probability",
    ],
)
def test_guard_scenarios_abstain_without_false_pass_fail_or_dropped_rows(guard):
    cfg = configuration()
    e = guarded_evidence(10, guard, cfg)
    before = copy.deepcopy(e)
    result = analyze_evidence(e, policy=load_policy(), minimum_groups=30, minimum_anchor_tokens=2)
    assert result["status"] == "INCONCLUSIVE"
    assert all(c["verdict"] not in {"PASS", "FAIL"} for c in result["checks"])
    assert result["quality"]["target_requests"] == len(e.target)
    assert e == before


def test_partial_topk_with_all_fixed_anchors_stays_measurable():
    cfg = configuration()
    e = guarded_evidence(10, "changing_topk", cfg)
    result = analyze_evidence(e, policy=load_policy(), minimum_groups=30, minimum_anchor_tokens=2)
    assert result["status"] == "INFO"
