"""Conservative rate bounds for independent bounded units with dependent repeats."""

import math

import numpy as np
from numpy.typing import ArrayLike

from servegauge.stats.common import finite_array, probability


def bounded_rate_interval(
    numerator: ArrayLike,
    denominator: ArrayLike,
    bounds: ArrayLike,
    *,
    alpha: float,
) -> tuple[float, float]:
    """Ratio of expected sums, using two Hoeffding bounds and a union bound.

    Unit pairs may be dependent internally and non-identically distributed. Units
    must be independent and their nonrandom bounds specified before observation.
    """
    probability(alpha)
    a, b, m = finite_array(numerator), finite_array(denominator), finite_array(bounds)
    if (
        a.ndim != 1
        or a.shape != b.shape
        or a.shape != m.shape
        or np.any(a < 0)
        or np.any(a > b)
        or np.any(b > m)
        or np.any(m < 0)
        or np.sum(b) <= 0
    ):
        raise ValueError("bounded_nonnegative_rate_unit_pairs_required")
    radius = math.sqrt(float(np.sum(m * m)) * math.log(4 / alpha) / 2)
    n, d = float(np.sum(a)), float(np.sum(b))
    low = max(0.0, n - radius) / (d + radius)
    high = 1.0 if d <= radius else min(1.0, (n + radius) / (d - radius))
    return low, high
