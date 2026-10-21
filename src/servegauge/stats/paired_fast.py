"""Batched paired-label randomization, algebraically identical to the scalar tests.

Kernel matrices are computed once. No practical margin, alpha or seed is chosen here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist

from servegauge.stats.common import FloatArray, finite_array, paired_arrays
from servegauge.stats.logprob_track import logprob_matrix
from servegauge.stats.mmd import _encode
from servegauge.stats.tests import TestResult, _extreme


def _signs(n: int, resamples: int, seed: int) -> Iterator[tuple[int, FloatArray]]:
    if n < 1 or resamples < 1:
        raise ValueError("positive_sample_and_permutation_counts_required")
    exact = n <= resamples.bit_length() and 2**n <= resamples
    count = 2**n if exact else resamples
    rng = np.random.default_rng(seed)
    for start in range(0, count, 128):
        size = min(128, count - start)
        if exact:
            rows = np.arange(start, start + size, dtype=np.uint64)[:, None]
            swaps = ((rows >> np.arange(n, dtype=np.uint64)) & 1).astype(np.float64)
        else:
            swaps = rng.integers(0, 2, size=(size, n)).astype(np.float64)
        yield start, 1 - 2 * swaps


def _test(
    n: int, statistic: Callable[[FloatArray], FloatArray], *, seed: int, resamples: int
) -> TestResult:
    if n < 1 or resamples < 1:
        raise ValueError("positive_sample_and_permutation_counts_required")
    exact = n <= resamples.bit_length() and 2**n <= resamples
    count = 2**n if exact else resamples
    observed = float(statistic(np.ones((1, n)))[0])
    null = np.empty(count, dtype=np.float64)
    for start, signs in _signs(n, resamples, seed):
        null[start : start + len(signs)] = statistic(signs)
    if not np.isfinite(observed) or not np.all(np.isfinite(null)):
        raise ValueError("finite_statistics_required")
    extreme = _extreme(null, observed, "greater")
    return TestResult(
        observed, extreme / count if exact else (extreme + 1) / (count + 1), n, n, count, exact
    )


def kernel_test(
    first: ArrayLike, second: ArrayLike, *, seed: int, resamples: int, kind: str, length: int = 1
) -> TestResult:
    a, b = paired_arrays(first, second)
    if a.ndim != 2 or len(a) < 2 or kind not in {"hamming", "energy"}:
        raise ValueError("matched_matrices_and_known_kernel_required")
    n = len(a)
    pooled = np.concatenate((a, b))
    gram = (
        length * (1 - cdist(pooled, pooled, metric="hamming"))
        if kind == "hamming"
        else cdist(pooled, pooled)
    )
    difference = gram[:n, :n] + gram[n:, n:] - gram[:n, n:] - gram[n:, :n]
    constant = gram.sum() / (2 * n * n * (n - 1)) - np.trace(gram) / (n * (n - 1))
    coefficient = (2 * n - 1) / (2 * n * n * (n - 1))

    def statistic(signs: FloatArray) -> FloatArray:
        quadratic = np.sum((signs @ difference) * signs, axis=1)
        return np.asarray(
            (
                constant + coefficient * quadratic
                if kind == "hamming"
                else np.maximum(0, -quadratic / (n * n))
            ),
            dtype=np.float64,
        )

    return _test(n, statistic, seed=seed, resamples=resamples)


def hamming_test_fast(
    first: Sequence[str], second: Sequence[str], *, length: int, seed: int, resamples: int
) -> TestResult:
    return kernel_test(
        _encode(first, length),
        _encode(second, length),
        length=length,
        kind="hamming",
        seed=seed,
        resamples=resamples,
    )


def energy_test_fast(
    first: ArrayLike, second: ArrayLike, *, seed: int, resamples: int
) -> TestResult:
    a, b = finite_array(first), finite_array(second)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    return kernel_test(a, b, kind="energy", seed=seed, resamples=resamples)


def logprob_test_fast(
    first: Sequence[dict[str, float]],
    second: Sequence[dict[str, float]],
    *,
    missing_floor: float,
    seed: int,
    resamples: int,
) -> TestResult:
    a, b, _ = logprob_matrix(first, second, missing_floor=missing_floor)
    a, b = paired_arrays(a, b)
    difference = a - b
    return _test(
        len(a),
        lambda signs: np.mean(np.abs(signs @ difference / len(a)), axis=1),
        seed=seed,
        resamples=resamples,
    )


def jsd_test_fast(
    first: Sequence[str], second: Sequence[str], *, seed: int, resamples: int
) -> TestResult:
    if not first or len(first) != len(second):
        raise ValueError("matched_nonempty_samples_required")
    vocabulary = {s: i for i, s in enumerate(sorted(set(first) | set(second)))}
    a = np.eye(len(vocabulary))[[vocabulary[s] for s in first]]
    b = np.eye(len(vocabulary))[[vocabulary[s] for s in second]]
    total, difference = np.sum(a + b, axis=0), a - b

    def statistic(signs: FloatArray) -> FloatArray:
        displacement = signs @ difference
        p, q = (total + displacement) / (2 * len(a)), (total - displacement) / (2 * len(a))
        mixture = (p + q) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            left = np.where(p > 0, p * np.log2(p / mixture), 0)
            right = np.where(q > 0, q * np.log2(q / mixture), 0)
        return np.asarray(np.sum(left + right, axis=1) / 2, dtype=np.float64)

    return _test(len(a), statistic, seed=seed, resamples=resamples)
