"""Paired dispersion companion to mean logprob tracking, with complete shared support.

The label-invariant pooled center is fixed during randomization. This tests squared
radius differences, not all distribution alternatives and not model identity.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from servegauge.stats.paired_fast import _test
from servegauge.stats.tests import TestResult


def logprob_dispersion_test(
    first: Sequence[dict[str, float]],
    second: Sequence[dict[str, float]],
    *,
    seed: int,
    resamples: int,
) -> TestResult:
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("matched_repeated_samples_required")
    vocabulary = sorted(first[0])
    if len(vocabulary) < 2 or any(set(row) != set(vocabulary) for row in [*first, *second]):
        raise ValueError("complete_shared_top_token_support_required")
    if any(not math.isfinite(v) or v > 0 for row in [*first, *second] for v in row.values()):
        raise ValueError("finite_nonpositive_logprobs_required")
    a, b = (
        np.asarray([[row[token] for token in vocabulary] for row in rows], dtype=np.float64)
        for rows in (first, second)
    )
    center = np.mean(np.concatenate((a, b)), axis=0)
    difference = np.sum((a - center) ** 2, axis=1) - np.sum((b - center) ** 2, axis=1)
    return _test(
        len(a),
        lambda signs: np.abs(signs @ difference / len(a)),
        seed=seed,
        resamples=resamples,
    )
