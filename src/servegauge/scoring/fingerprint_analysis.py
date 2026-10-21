"""Honest offline fingerprint diagnostics; no quality or model identity verdicts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from servegauge.core.hashing import content_hash
from servegauge.scoring.fingerprint_calibration import SignalCalibration, match_signal_calibration
from servegauge.scoring.fingerprint_evidence import (
    FingerprintEvidence,
    PreparedEvidence,
    prepare_evidence,
    unavailable_check,
)
from servegauge.stats.bootstrap import clustered_bootstrap
from servegauge.stats.fingerprint_blocks import Pairs, block_comparisons
from servegauge.stats.multiple import holm
from servegauge.verdicts.config import DecisionPolicy

METHODS = ("FP1", "FP3-mean", "FP3-dispersion")


@dataclass
class Grouping:
    nominal_sessions: int
    pairs: Pairs = field(default_factory=list)
    reason: str | None = None
    response_lineages: dict[str, int] = field(default_factory=dict)


def group_evidence(evidence: FingerprintEvidence, prepared: PreparedEvidence) -> Grouping:
    sessions = {r.session_id for r in evidence.target + evidence.reference}
    result = Grouping(len(sessions))
    maps = [
        {(r.session_id, r.draw): i for i, r in enumerate(rows)}
        for rows in (evidence.target, evidence.reference)
    ]
    if any(
        len(mapping) != len(rows)
        for mapping, rows in zip(maps, (evidence.target, evidence.reference), strict=True)
    ):
        result.reason = "duplicate_sample_identity"
        return result
    if set(maps[0]) != set(maps[1]) or not maps[0]:
        result.reason = "unmatched_sample_identities"
        return result
    parents = {s: s for s in sessions}

    def root(s: str) -> str:
        while parents[s] != s:
            parents[s] = parents[parents[s]]
            s = parents[s]
        return s

    partitions = []
    for name, rows, parsed in zip(
        ("target", "reference"),
        (evidence.target, evidence.reference),
        (prepared.first, prepared.second),
        strict=True,
    ):
        groups: dict[str, list[int]] = defaultdict(list)
        partition = []
        for i, row in enumerate(rows):
            if row.cache_group is None:
                partition.append(frozenset([(row.session_id, row.draw)]))
            else:
                groups[row.cache_group].append(i)
        for members in groups.values():
            texts = {parsed[i].text for i in members if parsed[i].text is not None}
            observed: dict[str, set[float]] = defaultdict(set)
            for i in members:
                for token, value in (parsed[i].logprobs or {}).items():
                    observed[token].add(value)
            # Cache hits may omit metadata. Compare only observed content; never
            # impute omitted probabilities or suppress otherwise usable text.
            if len(texts) > 1 or any(len(values) > 1 for values in observed.values()):
                result.reason = "inconsistent_cache_lineage"
                return result
            keys = frozenset((rows[i].session_id, rows[i].draw) for i in members)
            partition.append(keys)
            first = rows[members[0]].session_id
            for i in members[1:]:
                parents[root(rows[i].session_id)] = root(first)
        partitions.append(set(partition))
        result.response_lineages[name] = len(partition)
    if partitions[0] != partitions[1]:
        result.reason = "unmatched_cache_design"
        return result
    components: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key in sorted(maps[0]):
        components[root(key[0])].append(key)
    result.pairs = [
        ([maps[0][key] for key in keys], [maps[1][key] for key in keys])
        for keys in components.values()
    ]
    return result


def analyze_evidence(
    evidence: FingerprintEvidence,
    *,
    policy: DecisionPolicy,
    minimum_groups: int,
    minimum_anchor_tokens: int,
    signal_calibration: SignalCalibration | None = None,
    current_source_hash: str | None = None,
) -> dict[str, Any]:
    if minimum_groups < 2:
        raise ValueError("explicit_independent_group_policy_required")
    prepared = prepare_evidence(
        evidence,
        minimum_anchor_tokens=minimum_anchor_tokens,
        tolerance=float(policy.entries["P06"]["prob_sum_tolerance"]),
    )
    grouping = group_evidence(evidence, prepared)
    reason = (
        "independent_sessions_required"
        if not evidence.independent_sessions
        else "matched_sampling_design_required"
        if not evidence.matched_sampling_design
        else grouping.reason
        or ("insufficient_independent_groups" if len(grouping.pairs) < minimum_groups else None)
    )
    n = (
        len(grouping.pairs)
        if evidence.independent_sessions and evidence.matched_sampling_design
        else 0
    )
    with np.errstate(over="ignore", invalid="ignore"):
        comparisons = (
            {}
            if reason
            else block_comparisons(
                prepared,
                grouping.pairs,
                anchors=evidence.anchor_tokens,
                seed=evidence.seed,
                resamples=policy.permutation_resamples,
            )
        )
    checks = []
    for method in METHODS:
        unavailable = reason or (
            prepared.logprob_reason
            if method != "FP1"
            else "incomplete_text_evidence"
            if any(r.text is None for r in prepared.first + prepared.second)
            else None
        )
        if unavailable or method not in comparisons:
            checks.append(
                unavailable_check(
                    evidence,
                    check_id=method,
                    reason=unavailable or "statistic_unavailable",
                    groups=n,
                )
            )
            continue
        comparison = comparisons[method]
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                interval = clustered_bootstrap(
                    comparison.values,
                    seed=evidence.seed,
                    resamples=policy.bootstrap_resamples,
                    alpha=policy.alpha,
                    statistic=comparison.statistic,
                )
        except (ValueError, FloatingPointError, OverflowError):
            checks.append(
                unavailable_check(
                    evidence,
                    check_id=method,
                    reason="statistic_interval_unavailable",
                    groups=n,
                )
            )
            continue
        check = unavailable_check(
            evidence,
            check_id=method,
            reason="report only; no equivalence or quality determination",
            groups=n,
        )
        check.verdict = "INFO"
        check.estimate, check.ci_low, check.ci_high = interval.estimate, interval.low, interval.high
        check.reference_estimate = 0
        check.p_value = comparison.test.p_value
        check.details.update(
            metric=comparison.metric,
            permutation_statistic=comparison.test.statistic,
            permutations=comparison.test.permutations,
            exact=comparison.test.exact,
            bootstrap_resamples=policy.bootstrap_resamples,
        )
        checks.append(check)
    adjusted = holm([c.p_value if c.p_value is not None else 1.0 for c in checks])
    calibration_reason = None
    cutoff: float | None = policy.alpha
    if signal_calibration is not None:
        calibration_reason, cutoff = match_signal_calibration(
            signal_calibration,
            evidence,
            groups=n,
            alpha=policy.alpha,
            permutations=policy.permutation_resamples,
            current_source_hash=current_source_hash,
        )
    for check, p in zip(checks, adjusted, strict=True):
        if check.p_value is not None:
            check.adjusted_p = p
            check.details["distribution_signal"] = p <= cutoff if cutoff is not None else None
    return {
        "version": evidence.version,
        "status": "INCONCLUSIVE"
        if calibration_reason or any(c.verdict == "INCONCLUSIVE" for c in checks)
        else "INFO",
        "reference_tier": evidence.reference_tier,
        "signal_policy": {
            "cutoff": cutoff,
            "reason": calibration_reason,
            "qualification": "matched MOCK calibration"
            if signal_calibration is not None and calibration_reason is None
            else "nominal randomization only"
            if signal_calibration is None
            else "INCONCLUSIVE",
        },
        "checks": [c.model_dump(mode="json") for c in checks],
        "quality": {
            "target_requests": len(evidence.target),
            "reference_requests": len(evidence.reference),
            "nominal_sessions": grouping.nominal_sessions,
            "effective_groups": n,
            "response_lineages": grouping.response_lineages,
            "unavailable_reasons": prepared.unavailable_reasons,
            "anchor_count": len(evidence.anchor_tokens),
            "probe_configuration_hash": content_hash(
                [evidence.probe_id, evidence.anchor_tokens, evidence.anchor_source]
            ),
        },
        "assumptions": [
            "Independent, label-exchangeable session/cache components are declared by "
            "collection design, not inferred from text",
            "Matched response-cache lineage partitions; all within-group dependence retained",
            "Fixed probe anchors; no missing-value imputation, row deletion or "
            "top-k renormalization",
            "Report-only statistics; non-rejection is not equivalence; no monitoring "
            "calibration or provider certification",
        ],
        "model_api_calls": 0,
        "spent_usd": 0,
    }
