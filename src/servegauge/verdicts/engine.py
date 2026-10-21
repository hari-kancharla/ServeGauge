"""SPEC 12 decisions. Statistical evidence, replay and provenance are explicit inputs."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Literal

import numpy as np
from pydantic import Field

from servegauge.core.models import CheckClass, CheckResult, Tier, ValueModel
from servegauge.runner.plan import Requirements
from servegauge.scoring.toolcall_scoring import paired_f1, trigger_counts
from servegauge.stats.bootstrap import clustered_bootstrap, paired_bootstrap
from servegauge.stats.intervals import paired_difference_interval, wilson
from servegauge.stats.multiple import holm
from servegauge.stats.power import paired_mde
from servegauge.stats.tests import midp_mcnemar, paired_permutation, wilcoxon_test
from servegauge.verdicts.config import DecisionPolicy


class Measurement(ValueModel):
    check_id: str
    suite_id: str
    endpoint_id: str
    run_id: str
    check_class: CheckClass
    outcomes: list[float]
    reference: list[float] | None = None
    reference_tier: Tier | None = None
    clusters: list[str] | None = None
    metric: Literal["mean", "f1"] = "mean"
    labels: list[float] | None = None
    binary: bool = True
    two_sided: bool = False
    higher_is_better: bool = True
    declared_margin: float | None = Field(default=None, gt=0)
    noise_band: float | None = Field(default=None, ge=0)
    replay_confirmed: bool = False
    supported: bool = True
    applicable: bool = True
    verified: bool = True
    error: str | None = None
    seed: int
    requirement_ids: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


def _paired_items(item: Measurement) -> tuple[list[float], list[float] | None]:
    if item.reference is not None and len(item.reference) != len(item.outcomes):
        raise ValueError("paired_outcome_lengths_must_match")
    if item.clusters is None:
        return item.outcomes, item.reference
    if len(item.clusters) != len(item.outcomes):
        raise ValueError("paired_cluster_lengths_must_match")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(item.clusters):
        groups[label].append(index)
    first = [float(np.mean([item.outcomes[i] for i in indices])) for indices in groups.values()]
    second = (
        None
        if item.reference is None
        else [float(np.mean([item.reference[i] for i in indices])) for indices in groups.values()]
    )
    return first, second


def _interval(
    values: list[float], binary: bool, item: Measurement, policy: DecisionPolicy
) -> tuple[float, float]:
    repeats = item.details.get("binary_repeats")
    if repeats:
        return _average_binary_interval(repeats, policy.alpha)
    if binary and all(value in {0.0, 1.0} for value in values):
        return wilson(int(sum(values)), len(values), alpha=policy.alpha)
    result = clustered_bootstrap(
        values, seed=item.seed, resamples=policy.bootstrap_resamples, alpha=policy.alpha
    )
    return result.low, result.high


def _average_binary_interval(rows: list[list[float]], alpha: float) -> tuple[float, float]:
    draws = len(rows[0])
    if not draws or any(
        len(row) != draws or any(value not in {0, 1} for value in row) for row in rows
    ):
        raise ValueError("matched_binary_repeats_required")
    intervals = [
        wilson(int(sum(row[index] for row in rows)), len(rows), alpha=alpha / draws)
        for index in range(draws)
    ]
    return tuple(float(np.mean([bound[index] for bound in intervals])) for index in (0, 1))  # type: ignore[return-value]  # two fixed endpoints


def _measure(item: Measurement, policy: DecisionPolicy, family_size: int) -> CheckResult:
    first, second = _paired_items(item)
    if item.binary and any(value not in {0.0, 1.0} for value in item.outcomes):
        raise ValueError("binary_outcomes_required")
    n = len(first)
    result = CheckResult(
        check_id=item.check_id,
        check_class=item.check_class,
        suite_id=item.suite_id,
        endpoint_id=item.endpoint_id,
        run_id=item.run_id,
        requirement_ids=item.requirement_ids,
        n=n,
        reference_tier=item.reference_tier,
        verdict="INCONCLUSIVE",
        reason="no observations",
        evidence=item.evidence,
        details={
            **item.details,
            "requests": len(item.outcomes),
            "replay_confirmed": item.replay_confirmed,
            "verified": item.verified,
            "worst_case_holm_alpha": policy.alpha / max(1, family_size),
            "bootstrap_resamples": policy.bootstrap_resamples,
        },
    )
    if item.check_class == "invariant":
        result.reference_estimate = 0
        result.margin = policy.tolerance
    if item.error:
        result.verdict, result.reason = "ERROR", item.error
        return result
    if not item.applicable:
        result.verdict, result.reason = "NOT_APPLICABLE", "irrelevant for this profile"
        return result
    if not item.supported:
        result.verdict, result.reason = (
            "NOT_SUPPORTED",
            "feature absent or required length above cap",
        )
        return result
    if n and not (item.metric == "f1" and item.labels is None):
        result.estimate = float(np.mean(first))
        result.ci_low, result.ci_high = _interval(first, item.binary, item, policy)
    if item.check_class == "report_only":
        result.verdict, result.reason = "INFO", "informational observation"
        return result
    if not item.verified:
        result.reason = "unverified fact cannot drive a verdict"
        return result
    if not n:
        return result
    if item.check_class == "invariant":
        if not item.binary or item.clusters is not None:
            raise ValueError("invariant_requires_independent_binary_violation_trials")
        result.violations = int(sum(first))
        if result.violations >= policy.k_fail and item.replay_confirmed:
            result.verdict, result.reason = "FAIL", "repeated violations; replay reproduced"
        elif (
            result.violations == 0
            and not item.details.get("incomplete_trials")
            and result.ci_high is not None
            and result.ci_high <= policy.tolerance
        ):
            result.verdict, result.reason = (
                "PASS",
                "zero violations; 95% upper bound within tolerance",
            )
        else:
            result.reason = "needs more clean trials or a reproduced violation"
        return result
    declared = policy.margin(item.check_id, item.declared_margin)
    result.margin = max(declared, item.noise_band) if item.noise_band is not None else declared
    result.details.update(declared_margin=declared, noise_band=item.noise_band)
    if second is None or item.reference_tier is None:
        result.reason = "no baseline or reference"
        return result
    if not second:
        return result
    if item.metric == "f1":
        if item.clusters is not None or item.labels is None or len(item.labels) != n:
            raise ValueError("F1_requires_one_anchor_and_prediction_per_independent_item")
        contrast = paired_f1(
            item.labels,
            first,
            second,
            seed=item.seed,
            alpha=policy.alpha,
            resamples=policy.bootstrap_resamples,
            permutations=policy.permutation_resamples,
        )
        result.estimate, result.reference_estimate = contrast.estimate, contrast.reference_estimate
        result.ci_low, result.ci_high = contrast.interval
        result.delta = contrast.estimate - contrast.reference_estimate
        result.delta_ci_low, result.delta_ci_high = contrast.delta_interval
        result.p_value = contrast.p_value
        result.mde = paired_mde(
            n,
            variance=contrast.paired_variance,
            alpha=policy.alpha,
            power=policy.target_power,
            family_size=max(1, family_size),
        )
        result.details.update(
            metric="tool_call_f1",
            trigger_counts=trigger_counts(item.labels, first),
            reference_trigger_counts=trigger_counts(item.labels, second),
            reference_interval=list(contrast.reference_interval),
            one_sided_bounds=list(contrast.one_sided_bounds),
        )
        result.reason = (
            "matched-n reference calibration unavailable"
            if item.noise_band is None
            else "reference noise exceeds declared margin; increase n"
            if item.noise_band > declared
            else "awaiting family decision"
        )
        return result
    result.reference_estimate = float(np.mean(second))
    result.details["reference_interval"] = list(_interval(second, item.binary, item, policy))
    reference_rows = item.details.get("reference_binary_repeats")
    if reference_rows:
        result.details["reference_interval"] = list(
            _average_binary_interval(reference_rows, policy.alpha)
        )
    differences = np.asarray(first) - np.asarray(second)
    orientation = 1 if item.higher_is_better else -1
    differences *= orientation
    result.delta = float(np.mean(differences))
    binary_pairs = item.binary and all(v in {0.0, 1.0} for v in [*first, *second])
    if binary_pairs:
        a, b = (first, second) if orientation == 1 else (second, first)
        result.delta_ci_low, result.delta_ci_high = paired_difference_interval(
            a, b, alpha=policy.alpha
        )
        bounds = paired_difference_interval(a, b, alpha=policy.alpha, one_sided=True)
        positive = sum(x > y for x, y in zip(a, b, strict=True))
        negative = sum(x < y for x, y in zip(a, b, strict=True))
        result.p_value = midp_mcnemar(
            positive, negative, alternative="two-sided" if item.two_sided else "less"
        )
    else:
        if n < 2:
            result.reason = "insufficient independent items for paired bootstrap"
            return result
        report = paired_bootstrap(
            differences,
            np.zeros(n),
            seed=item.seed,
            alpha=policy.alpha,
            resamples=policy.bootstrap_resamples,
        )
        decision = paired_bootstrap(
            differences,
            np.zeros(n),
            seed=item.seed,
            alpha=2 * policy.alpha,
            resamples=policy.bootstrap_resamples,
        )
        result.delta_ci_low, result.delta_ci_high = report.low, report.high
        bounds = decision.low, decision.high
        result.p_value = paired_permutation(
            differences,
            seed=item.seed,
            resamples=policy.permutation_resamples,
            alternative="two-sided" if item.two_sided else "less",
        ).p_value
        rows = item.details.get("binary_repeats")
        if rows and reference_rows:
            draws = len(rows[0])
            if len(rows) != len(reference_rows) or any(len(row) != draws for row in reference_rows):
                raise ValueError("matched_binary_reference_repeats_required")
            reports, decisions = [], []
            for index in range(draws):
                a = [row[index] for row in rows]
                b = [row[index] for row in reference_rows]
                if orientation == -1:
                    a, b = b, a
                reports.append(paired_difference_interval(a, b, alpha=policy.alpha / draws))
                decisions.append(
                    paired_difference_interval(a, b, alpha=policy.alpha / draws, one_sided=True)
                )
            result.delta_ci_low = min(
                result.delta_ci_low, float(np.mean([bound[0] for bound in reports]))
            )
            result.delta_ci_high = max(
                result.delta_ci_high, float(np.mean([bound[1] for bound in reports]))
            )
            bounds = (
                min(bounds[0], float(np.mean([bound[0] for bound in decisions]))),
                max(bounds[1], float(np.mean([bound[1] for bound in decisions]))),
            )
            result.details["repeat_interval_method"] = (
                "whole-problem bootstrap hull with simultaneous per-draw paired-score bounds"
            )
    variance = float(np.var(differences, ddof=1)) if n > 1 else 1.0
    result.mde = paired_mde(
        n,
        variance=variance,
        alpha=policy.alpha,
        power=policy.target_power,
        family_size=max(1, family_size),
        two_sided=item.two_sided,
    )
    result.details["one_sided_bounds"] = list(bounds)
    if item.details.get("decision_rule") == "effort":
        assert (
            result.estimate is not None and result.ci_low is not None and result.ci_high is not None
        )
        result.details.update(
            geometric_mean_ratio=math.exp(result.estimate),
            ratio_interval=[math.exp(result.ci_low), math.exp(result.ci_high)],
            reference_geometric_mean_ratio=math.exp(result.reference_estimate),
            reference_ratio_interval=[math.exp(v) for v in result.details["reference_interval"]],
            high_greater_low_p=wilcoxon_test(first, alternative="greater").p_value,
            units="log(high+offset)-log(low+offset)",
        )
    if item.noise_band is None:
        result.reason = "matched-n reference calibration unavailable"
    elif item.noise_band > declared:
        result.reason = "reference noise exceeds declared margin; increase n"
    else:
        result.reason = "awaiting family decision"
    return result


def decide_family(
    items: list[Measurement], policy: DecisionPolicy, *, external: list[CheckResult] | None = None
) -> list[CheckResult]:
    """Call once per predeclared endpoint/run verdict family, including unsupported checks."""
    extra = [r.model_copy(deep=True) for r in external or []]
    identities = {(i.endpoint_id, i.run_id) for i in items} | {
        (r.endpoint_id, r.run_id) for r in extra
    }
    if len(identities) > 1:
        raise ValueError("one_endpoint_and_run_per_verdict_family_required")
    if any(r.verdict in {"PASS", "FAIL"} for r in extra):
        raise ValueError("external_evidence_must_precede_family_decision")
    checks = [i.check_id for i in items] + [r.check_id for r in extra]
    if extra and len(set(checks)) != len(checks):
        raise ValueError("unique_check_per_family_required")
    extra_indices = [
        len(items) + i
        for i, r in enumerate(extra)
        if r.verdict != "NOT_APPLICABLE"
        and (r.check_class == "comparative" or r.details.get("family_test"))
    ]
    family_size = sum(
        item.check_class == "comparative" and item.applicable for item in items
    ) + len(extra_indices)
    results = [_measure(item, policy, family_size) for item in items]
    for result in extra:
        if result.check_class == "comparative" and result.reason == "awaiting family decision":
            declared = policy.margin(
                result.check_id, result.details.get("declared_margin", result.margin)
            )
            noise = result.details.get("noise_band")
            result.margin = max(declared, noise) if noise is not None else declared
            result.details["declared_margin"] = declared
            if noise is None or noise > declared:
                result.reason = "matched-n reference noise unavailable or exceeds declared margin"
            elif result.n < 2 or any(
                v is None
                for v in (result.delta, result.delta_ci_low, result.delta_ci_high, result.p_value)
            ):
                result.reason = "complete independent comparative evidence required"
    results.extend(extra)
    indices = [
        i for i, item in enumerate(items) if item.check_class == "comparative" and item.applicable
    ] + extra_indices
    p_values: list[float] = []
    for index in indices:
        p_value = results[index].p_value
        p_values.append(1.0 if p_value is None else p_value)
    adjusted = holm(p_values)
    for index, p in zip(indices, adjusted, strict=True):
        result = results[index]
        item = items[index] if index < len(items) else None
        if result.p_value is not None:
            result.adjusted_p = p
        if item is None and result.check_class == "report_only":
            result.details["signal_detected"] = result.p_value is not None and p <= policy.alpha
            continue
        if result.reason != "awaiting family decision":
            continue
        assert result.delta is not None and result.margin is not None
        low, high = (
            (result.delta_ci_low, result.delta_ci_high)
            if item is None
            else result.details["one_sided_bounds"]
        )
        assert low is not None and high is not None
        two_sided = item.two_sided if item is not None else True
        if item is not None and item.details.get("decision_rule") == "effort":
            threshold = float(policy.entries["P07"]["effort_ignored_ratio"])
            fraction = float(policy.entries["P07"]["pass_reference_fraction"])
            ratio_low, ratio_high = result.details["ratio_interval"]
            reference_low = result.details["reference_ratio_interval"][0]
            ignored = ratio_high < threshold and reference_low > threshold
            if ignored and p <= policy.alpha and result.delta < -result.margin:
                result.verdict, result.reason = (
                    "FAIL",
                    "effort ratio below ignored boundary; Holm-significant versus reference",
                )
            elif (
                not ignored
                and low > -result.margin
                and ratio_low > fraction * result.details["reference_geometric_mean_ratio"]
                and reference_low > threshold
            ):
                result.verdict, result.reason = (
                    "PASS",
                    "effort ratio clears half of qualified reference",
                )
            else:
                result.reason = "effort ratio does not establish the specified effect; see MDE"
            continue
        if low > -result.margin and (not two_sided or high < result.margin):
            if item is None and result.details.get("equivalence_qualified") is not True:
                result.reason = "equivalence interval qualification unavailable"
                continue
            qualification = result.details.get("qualification")
            if item is None and qualification is not None:
                members = [results[i] for i in indices]
                if (
                    sorted(r.check_id for r in members) != qualification["family"]
                    or any(r.p_value is None or r.reference_tier != "MOCK" for r in members)
                    or result.details.get("declared_margin")
                    != qualification["margins"].get(result.check_id)
                ):
                    result.reason = "outside qualified complete verdict family or practical margin"
                    continue
            result.verdict, result.reason = (
                "PASS",
                "equivalent within margin" if two_sided else "non-inferior within margin",
            )
        elif p <= policy.alpha and (
            abs(result.delta) > result.margin if two_sided else result.delta < -result.margin
        ):
            result.verdict, result.reason = "FAIL", "deviation exceeds margin; Holm-significant"
        else:
            result.reason = "interval/test does not establish required effect; see MDE"
    return results


def map_requirements(
    requirements: Requirements,
    items: list[Measurement],
    policy: DecisionPolicy,
    *,
    available_checks: set[str] | None = None,
) -> list[Measurement]:
    available = {item.check_id for item in items} | (available_checks or set())
    if any(check not in available for req in requirements.requires for check in req.tests):
        raise ValueError("requirement_check_not_measured")
    mapped = []
    for item in items:
        matching = [req for req in requirements.requires if item.check_id in req.tests]
        margins = [
            policy.margin(item.check_id, req.margin)
            for req in matching
            if req.margin is not None and item.check_class == "comparative"
        ]
        mapped.append(
            item.model_copy(
                update={
                    "requirement_ids": [req.id for req in matching],
                    "declared_margin": min(margins) if margins else item.declared_margin,
                    "details": {
                        **item.details,
                        "requirement_params": {r.id: r.params for r in matching},
                    },
                }
            )
        )
    return mapped
