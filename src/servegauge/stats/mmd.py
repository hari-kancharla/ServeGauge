"""Unbiased MMD with the paper's summed position-wise Unicode Hamming kernel."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial.distance import cdist

from servegauge.stats.common import FloatArray
from servegauge.stats.tests import TestResult, two_sample_permutation


def _encode(strings: Sequence[str], length: int) -> FloatArray:
    if length < 1 or len(strings) < 2:
        raise ValueError("positive_fixed_length_and_two_samples_per_group_required")
    result = np.full((len(strings), length), -1, dtype=np.float64)
    for row, text in enumerate(strings):
        values = [ord(char) for char in text[:length]]
        result[row, : len(values)] = values
    return result


def _hamming_statistic(a: FloatArray, b: FloatArray) -> float:
    n, m, length = len(a), len(b), a.shape[1]
    if min(n, m) < 2:
        raise ValueError("two_samples_per_group_required")
    aa = length * (1 - cdist(a, a, metric="hamming"))
    bb = length * (1 - cdist(b, b, metric="hamming"))
    ab = length * (1 - cdist(a, b, metric="hamming"))
    return float(
        (aa.sum() - np.trace(aa)) / (n * (n - 1))
        + (bb.sum() - np.trace(bb)) / (m * (m - 1))
        - 2 * ab.mean()
    )


def hamming_mmd(x: Sequence[str], y: Sequence[str], *, length: int) -> float:
    return _hamming_statistic(_encode(x, length), _encode(y, length))


def hamming_test(
    x: Sequence[str],
    y: Sequence[str],
    *,
    length: int,
    seed: int,
    resamples: int = 2000,
    paired: bool = True,
) -> TestResult:
    return two_sample_permutation(
        _encode(x, length),
        _encode(y, length),
        _hamming_statistic,
        seed=seed,
        resamples=resamples,
        paired=paired,
    )
