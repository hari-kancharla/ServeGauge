"""Euclidean energy statistic: 2 E|X-Y| - E|X-X'| - E|Y-Y'| (V-statistic)."""

from __future__ import annotations

from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist

from servegauge.stats.common import FloatArray, finite_array
from servegauge.stats.tests import TestResult, two_sample_permutation


def _matrix(values: ArrayLike) -> FloatArray:
    data = finite_array(values)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.ndim != 2 or data.shape[1] == 0:
        raise ValueError("nonempty_feature_vectors_required")
    return data


def energy_distance(x: ArrayLike, y: ArrayLike) -> float:
    a, b = _matrix(x), _matrix(y)
    if a.shape[1] != b.shape[1]:
        raise ValueError("matching_feature_dimensions_required")
    value = 2 * cdist(a, b).mean() - cdist(a, a).mean() - cdist(b, b).mean()
    return max(0.0, float(value))


def energy_test(
    x: ArrayLike, y: ArrayLike, *, seed: int, resamples: int = 2000, paired: bool = True
) -> TestResult:
    return two_sample_permutation(
        _matrix(x), _matrix(y), energy_distance, seed=seed, resamples=resamples, paired=paired
    )
