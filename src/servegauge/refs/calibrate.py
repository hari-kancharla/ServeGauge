"""Two independent item bootstrap draws at exactly the comparison n and configuration."""

from collections.abc import Callable

import numpy as np
from scipy.stats import binom

from servegauge.stats.common import FloatArray, finite_array


def calibrate(
    values: list[float] | list[list[float]],
    *,
    seed: int,
    resamples: int = 2000,
    statistic: Callable[[FloatArray], float] | None = None,
) -> dict[str, object]:
    if resamples < 2000:
        raise ValueError("at_least_2000_calibration_resamples_required")
    data = finite_array(values)
    rng = np.random.default_rng(seed)
    first: list[float] = []
    second: list[float] = []
    for start in range(0, resamples, 128):
        size = min(128, resamples - start)
        for output in (first, second):
            indices = rng.integers(0, len(data), size=(size, len(data)))
            if statistic is None:
                output.extend(np.mean(data[indices], axis=1).tolist())
            else:
                output.extend(statistic(data[row]) for row in indices)
    noise = np.sort(np.abs(np.asarray(first) - np.asarray(second)))
    ranks = np.clip(
        np.asarray(binom.ppf([0.025, 0.975], resamples, 0.95), dtype=np.int64), 0, resamples - 1
    )
    return {
        "n": len(data),
        "noise_band": float(np.quantile(noise, 0.95)),
        "noise_band_interval": [float(noise[ranks[0]]), float(noise[ranks[1]])],
        "reference_estimate": float(np.mean(data)) if statistic is None else statistic(data),
        "reference_interval": np.quantile(first, [0.025, 0.975]).tolist(),
        "resamples": resamples,
        "seed": seed,
        "method": "two_independent_whole_item_bootstrap_draws",
        "limitations": "empirical item bootstrap cannot observe unseen failure modes",
    }
