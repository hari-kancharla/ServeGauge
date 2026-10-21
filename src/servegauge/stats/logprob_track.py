"""Mean absolute per-token logprob displacement; configured-floor variant of LT."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from servegauge.stats.common import FloatArray
from servegauge.stats.tests import TestResult, two_sample_permutation


def logprob_matrix(
    x: Sequence[dict[str, float]],
    y: Sequence[dict[str, float]],
    *,
    missing_floor: float,
) -> tuple[FloatArray, FloatArray, list[str]]:
    vocabulary = sorted({token for row in [*x, *y] for token in row})
    if not x or not y or not vocabulary or not math.isfinite(missing_floor) or missing_floor >= 0:
        raise ValueError("nonempty_logprob_samples_and_negative_finite_floor_required")
    values = [value for row in [*x, *y] for value in row.values()]
    if any(not math.isfinite(value) or value > 0 or value < missing_floor for value in values):
        raise ValueError("finite_nonpositive_logprobs_above_configured_floor_required")

    def matrix(rows: Sequence[dict[str, float]]) -> FloatArray:
        return np.asarray(
            [[row.get(token, missing_floor) for token in vocabulary] for row in rows],
            dtype=np.float64,
        )

    return matrix(x), matrix(y), vocabulary


def logprob_statistic(x: FloatArray, y: FloatArray) -> float:
    return float(np.mean(np.abs(np.mean(x, axis=0) - np.mean(y, axis=0))))


def logprob_test(
    x: Sequence[dict[str, float]],
    y: Sequence[dict[str, float]],
    *,
    missing_floor: float,
    seed: int,
    resamples: int = 2000,
    paired: bool = True,
) -> TestResult:
    a, b, _ = logprob_matrix(x, y, missing_floor=missing_floor)
    return two_sample_permutation(
        a, b, logprob_statistic, seed=seed, resamples=resamples, paired=paired
    )
