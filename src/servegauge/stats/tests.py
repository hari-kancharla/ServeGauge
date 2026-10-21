"""Paired/exchangeable tests with seeded randomization and conservative Monte Carlo p-values."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import binom, wilcoxon

from servegauge.stats.common import FloatArray, finite_array, paired_arrays

type Alternative = Literal["two-sided", "less", "greater"]


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float
    n: int
    n_reference: int | None = None
    permutations: int = 0
    exact: bool = False


def midp_mcnemar(positive: int, negative: int, *, alternative: Alternative = "two-sided") -> float:
    if min(positive, negative) < 0:
        raise ValueError("nonnegative_discordance_counts_required")
    total = positive + negative
    if total == 0:
        return 1.0
    mass = float(binom.pmf(positive, total, 0.5))
    left = float(binom.cdf(positive - 1, total, 0.5)) + mass / 2
    right = float(binom.sf(positive, total, 0.5)) + mass / 2
    return (
        min(1.0, 2 * min(left, right))
        if alternative == "two-sided"
        else left
        if alternative == "less"
        else right
    )


def wilcoxon_test(
    differences: ArrayLike,
    *,
    alternative: Alternative = "two-sided",
    method: Literal["auto", "exact", "asymptotic"] = "auto",
) -> TestResult:
    data = finite_array(differences)
    if data.ndim != 1:
        raise ValueError("one_dimensional_paired_differences_required")
    if not np.any(data):
        return TestResult(0, 1, len(data), exact=True)
    result = wilcoxon(
        data, alternative=alternative, method=method, correction=False, nan_policy="raise"
    )
    return TestResult(
        float(result.statistic), float(result.pvalue), len(data), exact=method == "exact"
    )


def _extreme(null: FloatArray, observed: float, alternative: Alternative) -> int:
    # Numeric tie guard follows floating arithmetic precision, not a practical effect margin.
    tolerance = 100 * np.finfo(np.float64).eps * max(1.0, abs(observed))
    if alternative == "greater":
        return int(np.sum(null >= observed - tolerance))
    if alternative == "less":
        return int(np.sum(null <= observed + tolerance))
    return int(np.sum(np.abs(null) >= abs(observed) - tolerance))


def paired_permutation(
    differences: ArrayLike,
    *,
    seed: int,
    resamples: int = 2000,
    alternative: Alternative = "two-sided",
) -> TestResult:
    data = finite_array(differences)
    if data.ndim != 1 or resamples < 1:
        raise ValueError("one_dimensional_data_and_positive_resamples_required")
    n = len(data)
    exact = n <= resamples.bit_length() and 2**n <= resamples
    count = 2**n if exact else resamples
    rng = np.random.default_rng(seed)
    null = np.empty(count, dtype=np.float64)
    for start in range(0, count, 256):
        size = min(256, count - start)
        if exact:
            rows = np.arange(start, start + size, dtype=np.uint64)[:, None]
            signs = ((rows >> np.arange(n, dtype=np.uint64)) & 1).astype(np.int8) * 2 - 1
        else:
            signs = rng.integers(0, 2, size=(size, n), dtype=np.int8) * 2 - 1
        null[start : start + size] = np.mean(signs * data, axis=1)
    observed = float(np.mean(data))
    extreme = _extreme(null, observed, alternative)
    p = extreme / count if exact else (extreme + 1) / (count + 1)
    return TestResult(observed, p, n, n, count, exact)


def two_sample_permutation(
    x: ArrayLike,
    y: ArrayLike,
    statistic: Callable[[FloatArray, FloatArray], float],
    *,
    seed: int,
    resamples: int = 2000,
    paired: bool = False,
    alternative: Alternative = "greater",
) -> TestResult:
    a, b = paired_arrays(x, y) if paired else (finite_array(x), finite_array(y))
    if a.shape[1:] != b.shape[1:] or resamples < 1:
        raise ValueError("matching_feature_shapes_and_positive_resamples_required")
    n, m = len(a), len(b)
    observed = float(statistic(a, b))
    if not math.isfinite(observed):
        raise ValueError("finite_statistic_required")
    possible = 2**n if paired else math.comb(n + m, n)
    exact = possible <= resamples
    count = possible if exact else resamples
    rng = np.random.default_rng(seed)
    pooled = np.concatenate((a, b))
    null = np.empty(count, dtype=np.float64)
    assignments = itertools.combinations(range(n + m), n) if exact and not paired else None
    for index in range(count):
        if paired:
            swaps = (
                np.asarray([(index >> bit) & 1 for bit in range(n)], dtype=bool)
                if exact
                else rng.integers(0, 2, size=n).astype(bool)
            )
            mask = swaps.reshape((n,) + (1,) * (a.ndim - 1))
            first, second = np.where(mask, b, a), np.where(mask, a, b)
        else:
            selected = (
                np.asarray(next(assignments), dtype=np.int64)
                if assignments is not None
                else rng.permutation(n + m)[:n]
            )
            mask = np.zeros(n + m, dtype=bool)
            mask[selected] = True
            first, second = pooled[mask], pooled[~mask]
        null[index] = statistic(first, second)
    if not np.all(np.isfinite(null)):
        raise ValueError("finite_permuted_statistics_required")
    extreme = _extreme(null, observed, alternative)
    p = extreme / count if exact else (extreme + 1) / (count + 1)
    return TestResult(observed, p, n, m, count, exact)
