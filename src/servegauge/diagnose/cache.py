"""Complete new cache witness; legacy lab hypotheses retain their original rules."""

from __future__ import annotations

import math
from typing import Any

from servegauge.core.models import CheckResult
from servegauge.suites.s1_cache import CACHE_CHECKS, VERSION


def _finite(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(value)


def cache_witness(checks: list[CheckResult]) -> list[CheckResult]:
    support = [c for c in checks if c.check_id in CACHE_CHECKS]
    if (
        len(support) != 2
        or {c.check_id for c in support} != set(CACHE_CHECKS)
        or len({(c.run_id, c.endpoint_id) for c in support}) != 1
    ):
        return []
    by_id = {c.check_id: c for c in support}
    repeated, fresh = (by_id[c] for c in CACHE_CHECKS)
    contract = repeated.details.get("cache_contract")
    if (
        not isinstance(contract, dict)
        or contract.get("version") != VERSION
        or contract != fresh.details.get("cache_contract")
        or repeated.verdict != "FAIL"
        or fresh.verdict != "PASS"
        or repeated.n != fresh.n
        or repeated.n != contract.get("groups")
        or repeated.reference_tier != fresh.reference_tier
        or not set(repeated.evidence).isdisjoint(fresh.evidence)
    ):
        return []
    groups = []
    for check in support:
        inputs = check.details.get("input_hashes")
        signature = check.details.get("request_signature")
        if (
            check.check_class != "comparative"
            or check.n < 1
            or not check.evidence
            or check.reference_tier is None
            or not isinstance(signature, str)
            or not signature
            or not isinstance(inputs, dict)
            or len(inputs) != check.n
            or not all(
                _finite(v)
                for v in [
                    check.estimate,
                    check.ci_low,
                    check.ci_high,
                    check.reference_estimate,
                    check.delta,
                    check.delta_ci_low,
                    check.delta_ci_high,
                    check.margin,
                    check.p_value,
                    check.adjusted_p,
                ]
            )
        ):
            return []
        groups.append(set(inputs))
    if groups[0] != groups[1]:
        return []
    return [repeated, fresh]
