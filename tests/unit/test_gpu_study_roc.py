"""Independent rank arithmetic, ties, missingness and held-out threshold controls."""

import math

import pytest

from servegauge.stats.gpu_roc import roc_analysis


def rows(partition, negative, positive):
    return [
        {
            "unit_id": f"{partition}-{label}-{i}",
            "partition": partition,
            "positive": label,
            "score": score,
        }
        for label, scores in [(False, negative), (True, positive)]
        for i, score in enumerate(scores)
    ]


def test_held_out_rank_auc_and_wilson_have_actual_denominators():
    data = rows("calibration", [0, 1], [2, 3]) + rows("evaluation", [0, 2, None], [1, 3, None])
    result = roc_analysis(data, alpha=0.05, independent=True)
    assert result["auc"]["estimate"] == 0.75
    assert result["auc"]["n_negative"] == result["auc"]["n_positive"] == 2
    assert result["auc"]["ci_low"] < 0.75 < result["auc"]["ci_high"]
    assert result["operating_point"]["threshold"] == 2
    point = result["operating_point"]
    assert point["tpr"]["successes"] == point["fpr"]["successes"] == 1
    assert point["tpr"]["n"] == point["fpr"]["n"] == 2
    assert result["missing"]["evaluation"] == {"negative": 1, "positive": 1}
    changed = rows("calibration", [0, 1], [2, 3]) + rows("evaluation", [99], [-99])
    assert roc_analysis(changed, alpha=0.05, independent=True)["operating_point"]["threshold"] == 2
    # Every ROC threshold is selected on calibration, before looking at evaluation.
    assert [p["threshold"] for p in result["curve"]] == [None, 3, 2, 1, 0]


def test_ties_are_half_a_pair_and_tiny_perfect_auc_has_honest_uncertainty():
    result = roc_analysis(
        rows("calibration", [0], [1]) + rows("evaluation", [1], [1]), alpha=0.05, independent=True
    )
    assert result["auc"]["estimate"] == 0.5
    assert result["auc"]["ci_low"] == 0 and result["auc"]["ci_high"] == 1
    result = roc_analysis(
        rows("calibration", [0], [1]) + rows("evaluation", [0], [1]), alpha=0.05, independent=True
    )
    assert result["auc"]["estimate"] == 1
    assert result["auc"]["ci_low"] == 0


def test_missing_class_and_unreviewed_independence_abstain():
    data = rows("calibration", [None], [1]) + rows("evaluation", [0], [None])
    result = roc_analysis(data, alpha=0.05, independent=False)
    assert result["auc"]["estimate"] is None and result["operating_point"] is None
    assert result["status"] == "INCONCLUSIVE"
    complete = rows("calibration", [0], [1]) + rows("evaluation", [0], [1])
    result = roc_analysis(complete, alpha=0.05, independent=False)
    assert result["auc"]["estimate"] == 1 and result["auc"]["ci_low"] is None
    assert result["curve"][1]["tpr"]["ci_low"] is None


@pytest.mark.parametrize(
    "mutation", ["duplicate", "nan", "infinity", "partition", "label", "alpha"]
)
def test_invalid_or_reused_units_refuse(mutation):
    data = rows("calibration", [0], [1]) + rows("evaluation", [0], [1])
    alpha = 0.05
    if mutation == "duplicate":
        data[-1]["unit_id"] = data[0]["unit_id"]
    elif mutation in {"nan", "infinity"}:
        data[0]["score"] = math.nan if mutation == "nan" else math.inf
    elif mutation == "partition":
        data[0]["partition"] = "test-tuning"
    elif mutation == "label":
        data[0]["positive"] = 1
    else:
        alpha = 0
    with pytest.raises(ValueError):
        roc_analysis(data, alpha=alpha, independent=True)
