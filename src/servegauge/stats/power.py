"""Normal-approximation paired planning at the worst-case Holm alpha; no continuity correction."""

from __future__ import annotations

import argparse
import math

from scipy.stats import norm

from servegauge.stats.common import probability


def _z(alpha: float, power: float, family_size: int, two_sided: bool) -> tuple[float, float]:
    probability(alpha)
    probability(power)
    if family_size < 1:
        raise ValueError("positive_verdict_family_size_required")
    adjusted = alpha / family_size
    return float(norm.isf(adjusted / (2 if two_sided else 1))), float(norm.ppf(power))


def unpaired_sample_size(
    p_endpoint: float,
    p_reference: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    family_size: int = 1,
    two_sided: bool = True,
) -> int:
    probability(p_endpoint, closed=True)
    probability(p_reference, closed=True)
    if p_endpoint == p_reference:
        raise ValueError("nonzero_alternative_difference_required")
    za, zp = _z(alpha, power, family_size, two_sided)
    pooled = (p_endpoint + p_reference) / 2
    variance_null = 2 * pooled * (1 - pooled)
    variance_alt = p_endpoint * (1 - p_endpoint) + p_reference * (1 - p_reference)
    return math.ceil(
        (za * math.sqrt(variance_null) + zp * math.sqrt(variance_alt)) ** 2
        / (p_endpoint - p_reference) ** 2
    )


def paired_sample_size(
    p_endpoint: float,
    p_reference: float,
    *,
    correlation: float,
    alpha: float = 0.05,
    power: float = 0.8,
    family_size: int = 1,
    two_sided: bool = True,
) -> int:
    probability(p_endpoint, closed=True)
    probability(p_reference, closed=True)
    if p_endpoint == p_reference or not -1 <= correlation <= 1:
        raise ValueError("nonzero_difference_and_valid_correlation_required")
    covariance = correlation * math.sqrt(
        p_endpoint * (1 - p_endpoint) * p_reference * (1 - p_reference)
    )
    p11 = p_endpoint * p_reference + covariance
    if not max(0, p_endpoint + p_reference - 1) <= p11 <= min(p_endpoint, p_reference):
        raise ValueError("correlation_not_feasible_for_binary_marginals")
    za, zp = _z(alpha, power, family_size, two_sided)
    pooled = (p_endpoint + p_reference) / 2
    variance_null = 2 * pooled * (1 - pooled) * (1 - correlation)
    variance_alt = p_endpoint * (1 - p_endpoint) + p_reference * (1 - p_reference) - 2 * covariance
    return math.ceil(
        (za * math.sqrt(max(0, variance_null)) + zp * math.sqrt(max(0, variance_alt))) ** 2
        / (p_endpoint - p_reference) ** 2
    )


def paired_mde(
    n: int,
    *,
    variance: float,
    alpha: float = 0.05,
    power: float = 0.8,
    family_size: int = 1,
    two_sided: bool = False,
) -> float:
    if n < 1 or not math.isfinite(variance) or variance < 0:
        raise ValueError("positive_n_and_nonnegative_finite_paired_variance_required")
    za, zp = _z(alpha, power, family_size, two_sided)
    return (za + zp) * math.sqrt(variance / n)


def paired_power(
    n: int,
    *,
    effect: float,
    variance: float,
    alpha: float = 0.05,
    family_size: int = 1,
    two_sided: bool = False,
) -> float:
    if n < 1 or not math.isfinite(variance) or variance <= 0 or not math.isfinite(effect):
        raise ValueError("valid_n_effect_and_positive_variance_required")
    za, _ = _z(alpha, 0.8, family_size, two_sided)
    shift = abs(effect) * math.sqrt(n / variance)
    value = float(norm.sf(za - shift))
    return value + float(norm.cdf(-za - shift)) if two_sided else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unpaired", nargs=2, type=float, required=True, metavar=("ENDPOINT", "REFERENCE")
    )
    args = parser.parse_args()
    print(unpaired_sample_size(*args.unpaired))


if __name__ == "__main__":
    main()
