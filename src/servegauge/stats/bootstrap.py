"""Resample whole items; never treat an item's repeated requests as independent clusters."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from servegauge.stats.common import FloatArray, finite_array, paired_arrays, probability


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    low: float
    high: float
    n_clusters: int
    resamples: int


def _mean(values: FloatArray) -> float:
    return float(np.mean(values))


def clustered_bootstrap(
    values: ArrayLike,
    clusters: Sequence[Hashable] | None = None,
    *,
    seed: int,
    resamples: int = 2000,
    alpha: float = 0.05,
    statistic: Callable[[FloatArray], float] = _mean,
) -> BootstrapResult:
    probability(alpha)
    if resamples < 2000:
        raise ValueError("at_least_2000_bootstrap_resamples_required")
    data = finite_array(values)
    labels = list(range(len(data))) if clusters is None else list(clusters)
    if len(labels) != len(data):
        raise ValueError("cluster_label_count_must_match_rows")
    groups: dict[Hashable, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    indices = [np.asarray(group, dtype=np.int64) for group in groups.values()]
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    if all(len(index) == 1 for index in indices):
        # The lab repeats this hot path thousands of times. Batched singleton draws
        # consume the identical RNG sequence and retain whole-item resampling.
        for start in range(0, resamples, 128):
            stop = min(start + 128, resamples)
            draw = rng.integers(0, len(indices), size=(stop - start, len(indices)))
            selected = data[draw]
            if statistic is _mean:
                estimates[start:stop] = np.mean(selected, axis=tuple(range(1, selected.ndim)))
            else:
                estimates[start:stop] = [statistic(row) for row in selected]
    else:
        for replicate in range(resamples):
            draw = rng.integers(0, len(indices), size=len(indices))
            selected = np.concatenate([indices[int(i)] for i in draw])
            estimates[replicate] = statistic(data[selected])
    if not np.all(np.isfinite(estimates)):
        raise ValueError("bootstrap_statistic_must_be_finite")
    low, high = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    estimate = statistic(data)
    if not np.isfinite(estimate):
        raise ValueError("bootstrap_statistic_must_be_finite")
    return BootstrapResult(estimate, float(low), float(high), len(indices), resamples)


def paired_bootstrap(
    x: ArrayLike,
    y: ArrayLike,
    clusters: Sequence[Hashable] | None = None,
    *,
    seed: int,
    resamples: int = 2000,
    alpha: float = 0.05,
) -> BootstrapResult:
    a, b = paired_arrays(x, y)
    return clustered_bootstrap(a - b, clusters, seed=seed, resamples=resamples, alpha=alpha)
