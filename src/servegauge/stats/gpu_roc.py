"""Held-out rank ROC with conditional pointwise intervals and explicit missing units."""

from __future__ import annotations

import math
from typing import Any

from servegauge.stats.common import probability
from servegauge.stats.intervals import wilson


def _rate(
    scores: list[float], threshold: float | None, alpha: float, independent: bool
) -> dict[str, Any]:
    hits = sum(threshold is not None and score >= threshold for score in scores)
    low, high = wilson(hits, len(scores), alpha=alpha) if scores and independent else (None, None)
    return {
        "successes": hits,
        "n": len(scores),
        "estimate": hits / len(scores) if scores else None,
        "ci_low": low,
        "ci_high": high,
    }


def roc_analysis(rows: list[dict[str, Any]], *, alpha: float, independent: bool) -> dict[str, Any]:
    """Scores must already be label-blind raw-derived unit contrasts, larger = shift.

    One unit is four exclusive captures; requests within it are not ROC replicates.
    Thresholds (including the Youden operating point) use calibration only. Wilson
    intervals are approximate pointwise conditional on calibration, not simultaneous.
    The AUC interval uses bounded differences for independent positive/negative units;
    it is conservative and remains wide for tiny or apparently perfect experiments.
    Missingness can be intervention dependent: all estimates are complete-case only.
    """
    probability(alpha)
    if type(independent) is not bool:
        raise ValueError("explicit_study_independence_declaration_required")
    seen: set[str] = set()
    values: dict[str, dict[bool, list[float]]] = {
        part: {False: [], True: []} for part in ("calibration", "evaluation")
    }
    missing = {part: {"negative": 0, "positive": 0} for part in values}
    for row in rows:
        if (
            set(row) != {"unit_id", "partition", "positive", "score"}
            or not isinstance(row["unit_id"], str)
            or not row["unit_id"]
            or row["unit_id"] in seen
            or row["partition"] not in values
            or type(row["positive"]) is not bool
        ):
            raise ValueError("unique_declared_study_units_required")
        seen.add(row["unit_id"])
        score = row["score"]
        if score is None:
            missing[row["partition"]]["positive" if row["positive"] else "negative"] += 1
        elif type(score) not in {int, float} or not math.isfinite(score):
            raise ValueError("finite_study_scores_or_explicit_missing_required")
        else:
            values[row["partition"]][row["positive"]].append(float(score))
    calibration, evaluation = values["calibration"], values["evaluation"]
    negative, positive = evaluation[False], evaluation[True]
    thresholds: list[float | None] = [
        None,
        *sorted(set(calibration[False] + calibration[True]), reverse=True),
    ]
    curve = [
        {
            "threshold": threshold,
            "fpr": _rate(negative, threshold, alpha, independent),
            "tpr": _rate(positive, threshold, alpha, independent),
        }
        for threshold in thresholds
    ]
    operating = None
    if calibration[False] and calibration[True]:
        differences = [
            sum(threshold is not None and v >= threshold for v in calibration[True])
            / len(calibration[True])
            - sum(threshold is not None and v >= threshold for v in calibration[False])
            / len(calibration[False])
            for threshold in thresholds
        ]
        # First tied maximum is the highest threshold (including never-flag).
        operating = curve[max(range(len(thresholds)), key=lambda i: differences[i])]
    auc = None
    low = high = None
    if negative and positive:
        auc = sum(float(p > n) + 0.5 * (p == n) for p in positive for n in negative) / (
            len(negative) * len(positive)
        )
        if independent:
            radius = math.sqrt((1 / len(negative) + 1 / len(positive)) * math.log(2 / alpha) / 2)
            low, high = max(0.0, auc - radius), min(1.0, auc + radius)
    return {
        "version": "gpu-study-roc-v1",
        "reference_tier": "BASELINE",
        "status": "INFO"
        if independent and auc is not None and operating is not None
        else "INCONCLUSIVE",
        "alpha": alpha,
        "independence": "owner_declared" if independent else "unreviewed",
        "unit": "exclusive_target_and_three_reference_capture_quartet",
        "score_direction": "larger_means_greater_raw_fingerprint_contrast",
        "threshold_source": "calibration_only_Youden_highest_tie",
        "interval_scope": (
            "conditional_pointwise_Wilson_and_McDiarmid_AUC; no_simultaneous_guarantee"
        ),
        "missingness_scope": "complete_case_only; no_missing_at_random_assumption_or_correction",
        "auc": {
            "estimate": auc,
            "ci_low": low,
            "ci_high": high,
            "n_negative": len(negative),
            "n_positive": len(positive),
            "interval_method": "independent_unit_bounded_differences",
        },
        "missing": missing,
        "curve": curve,
        "operating_point": operating,
    }
