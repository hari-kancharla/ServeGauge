"""Prepare captured fingerprint contrasts for the endpoint's shared verdict family."""

from __future__ import annotations

from servegauge.core.hashing import content_hash
from servegauge.core.models import CheckResult, EndpointSpec, ReferencePack
from servegauge.refs.fingerprint import compare_fingerprint
from servegauge.scoring.common import Samples
from servegauge.scoring.fingerprint_collection import pending_checks
from servegauge.verdicts.config import DecisionPolicy


def fingerprint_results(
    samples: dict[str, Samples],
    *,
    endpoint_id: str,
    run_id: str,
    pack: ReferencePack | None,
    policy: DecisionPolicy,
    seed: int,
    endpoint: EndpointSpec | None = None,
) -> list[CheckResult]:
    results = pending_checks(samples, endpoint_id, run_id)
    if pack is None:
        return results
    for result in results:
        if result.verdict in {"ERROR", "NOT_SUPPORTED"}:
            continue
        sample = samples[result.check_id]
        methods = ["FP3-mean", "FP3-dispersion"] if result.check_id == "FP3" else [result.check_id]
        try:
            contrasts = {
                m: compare_fingerprint(
                    pack,
                    sample,
                    target=endpoint_id,
                    method=m,
                    policy=policy,
                    seed=int(content_hash([seed, endpoint_id, m])[:16], 16),
                    endpoint=endpoint,
                )
                for m in methods
            }
        except (LookupError, ValueError, TypeError):
            result.reason = (
                "fixed-anchor logprobs and matched qualified fingerprint reference required"
                if result.check_id == "FP3"
                else "independent evidence and matched qualified fingerprint reference required"
            )
            if result.check_id == "FP4":
                result.reason = (
                    "complete text, pinned local CPU model/runtime and matched "
                    "fingerprint reference required"
                )
            result.details["reference_pack_id"] = pack.pack_id
            continue
        primary = contrasts[methods[0]]
        for name in (
            "n",
            "estimate",
            "ci_low",
            "ci_high",
            "reference_estimate",
            "delta",
            "delta_ci_low",
            "delta_ci_high",
        ):
            setattr(result, name, primary[name])
        result.reference_tier = pack.tier
        result.p_value = min(1.0, len(methods) * min(c["p_value"] for c in contrasts.values()))
        result.details.update(
            reference_pack_id=pack.pack_id,
            fingerprint_metrics=contrasts,
            fingerprint_position=primary["position"],
            reference_interval=primary["reference_interval"],
            noise_band=primary["noise_band"],
            equivalence_qualified=all(c["equivalence_qualified"] for c in contrasts.values()),
            family_test=True,
        )
        if primary.get("qualification"):
            result.details["qualification"] = primary["qualification"]
        if result.check_id in {"FP3", "FP4"}:
            result.check_class = "report_only"
            result.verdict, result.reason = (
                "INFO",
                "logprob diagnostics; no declared log-unit practical margin"
                if result.check_id == "FP3"
                else "embedding energy diagnostics; no declared practical margin "
                "or equivalence qualification",
            )
            if result.check_id == "FP3":
                result.details["subtest_correction"] = (
                    "Bonferroni over mean and dispersion, then endpoint Holm"
                )
        else:
            result.margin = policy.margin(result.check_id)
            result.reason = "awaiting family decision"
    return results
