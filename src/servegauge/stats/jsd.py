"""Jensen-Shannon divergence in bits, with matched-cell paired label randomization."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from servegauge.stats.common import FloatArray, finite_array
from servegauge.stats.tests import TestResult, two_sample_permutation


def jsd_bits(x: ArrayLike, y: ArrayLike) -> float:
    a, b = finite_array(x), finite_array(y)
    if (
        a.shape != b.shape
        or a.ndim != 1
        or np.any(a < 0)
        or np.any(b < 0)
        or min(a.sum(), b.sum()) <= 0
    ):
        raise ValueError("matching_nonnegative_positive_mass_vectors_required")
    p, q = a / a.sum(), b / b.sum()
    mixture = (p + q) / 2
    left, right = p > 0, q > 0
    return float(
        (
            np.sum(p[left] * np.log2(p[left] / mixture[left]))
            + np.sum(q[right] * np.log2(q[right] / mixture[right]))
        )
        / 2
    )


def jsd_test(
    x: Sequence[Sequence[str]],
    y: Sequence[Sequence[str]],
    *,
    seed: int,
    resamples: int = 2000,
) -> TestResult:
    if (
        not x
        or len(x) != len(y)
        or any(not a or len(a) != len(b) for a, b in zip(x, y, strict=True))
    ):
        raise ValueError("matched_nonempty_cells_and_sample_counts_required")
    cells: list[tuple[int, int]] = []
    first: list[int] = []
    second: list[int] = []
    offset = 0
    for a, b in zip(x, y, strict=True):
        vocabulary = {value: index + offset for index, value in enumerate(sorted(set(a) | set(b)))}
        cells.append((offset, offset + len(vocabulary)))
        offset += len(vocabulary)
        first.extend(vocabulary[value] for value in a)
        second.extend(vocabulary[value] for value in b)

    def statistic(a: FloatArray, b: FloatArray) -> float:
        counts_a = np.bincount(a.astype(np.int64), minlength=offset)
        counts_b = np.bincount(b.astype(np.int64), minlength=offset)
        return float(
            np.mean([jsd_bits(counts_a[start:end], counts_b[start:end]) for start, end in cells])
        )

    return two_sample_permutation(
        first, second, statistic, seed=seed, resamples=resamples, paired=True
    )
