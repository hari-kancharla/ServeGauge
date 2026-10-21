"""Wilson intervals and independently derived Tango paired-score inversion."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.stats import norm

from servegauge.stats.common import paired_arrays, probability


def wilson(
    successes: int, n: int, *, alpha: float = 0.05, one_sided: bool = False
) -> tuple[float, float]:
    probability(alpha)
    if n < 1 or not 0 <= successes <= n:
        raise ValueError("valid_success_count_and_positive_n_required")
    z = float(norm.isf(alpha if one_sided else alpha / 2))
    p, z2 = successes / n, z * z
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    radius = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (
        0.0 if successes == 0 else max(0.0, center - radius),
        1.0 if successes == n else min(1.0, center + radius),
    )


def paired_score_z(positive: int, negative: int, n: int, delta: float) -> float:
    """Constrained multinomial MLE of discordance q under E[X-Y]=delta."""
    if n < 1 or min(positive, negative) < 0 or positive + negative > n or not -1 <= delta <= 1:
        raise ValueError("invalid_paired_score_inputs")
    observed = (positive - negative) / n
    if delta in {-1, 1}:
        return 0.0 if observed == delta else math.copysign(math.inf, observed - delta)
    m, s, concordant = positive + negative, positive - negative, n - positive - negative
    coefficient = m + s * delta
    discriminant = coefficient * coefficient - 4 * n * (s * delta - concordant * delta * delta)
    q = (coefficient + math.sqrt(max(0.0, discriminant))) / (2 * n)
    q = max(abs(delta), min(1.0, q))
    variance = max(0.0, q - delta * delta) / n
    if variance == 0:
        return 0.0 if observed == delta else math.copysign(math.inf, observed - delta)
    return (observed - delta) / math.sqrt(variance)


def paired_difference_interval(
    x: ArrayLike,
    y: ArrayLike,
    *,
    alpha: float = 0.05,
    one_sided: bool = False,
) -> tuple[float, float]:
    probability(alpha)
    a, b = paired_arrays(x, y)
    if a.ndim != 1 or not np.all(np.isin(a, [0, 1])) or not np.all(np.isin(b, [0, 1])):
        raise ValueError("paired_binary_observations_required")
    positive, negative = int(np.sum((a == 1) & (b == 0))), int(np.sum((a == 0) & (b == 1)))
    n = len(a)
    observed = (positive - negative) / n
    z = float(norm.isf(alpha if one_sided else alpha / 2))

    def score(delta: float) -> float:
        return paired_score_z(positive, negative, n, delta)

    lower = -1.0 if observed == -1 else float(brentq(lambda d: score(d) - z, -1, observed))
    upper = 1.0 if observed == 1 else float(brentq(lambda d: score(d) + z, observed, 1))
    return lower, upper
