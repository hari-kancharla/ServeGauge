"""Whole-session paired randomization; arbitrary within-session dependence is retained."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass

import numpy as np

from servegauge.scoring.fingerprint_evidence import PreparedEvidence
from servegauge.stats.common import FloatArray
from servegauge.stats.jsd import jsd_bits
from servegauge.stats.paired_fast import _test
from servegauge.stats.tests import TestResult

type Pairs = list[tuple[list[int], list[int]]]


@dataclass
class BlockComparison:
    test: TestResult
    values: FloatArray
    statistic: Callable[[FloatArray], float]
    metric: str


def _mean(data: FloatArray) -> float:
    return float(np.mean(data))


def block_comparisons(
    prepared: PreparedEvidence, pairs: Pairs, *, anchors: list[str], seed: int, resamples: int
) -> dict[str, BlockComparison]:
    if len(pairs) < 2:
        raise ValueError("independent_paired_groups_required")
    groups = len(pairs)
    result = {}
    if all(r.text is not None for r in prepared.first + prepared.second):
        vocabulary = sorted({str(r.text) for r in prepared.first + prepared.second})

        def probabilities(rows: list[int], side: bool) -> list[float]:
            source = prepared.first if side else prepared.second
            return [sum(source[i].text == token for i in rows) / len(rows) for token in vocabulary]

        a = np.asarray([probabilities(x, True) for x, _ in pairs], dtype=np.float64)
        b = np.asarray([probabilities(y, False) for _, y in pairs], dtype=np.float64)
        total, difference = np.sum(a + b, axis=0), a - b

        def jsd(signs: FloatArray) -> FloatArray:
            displacement = signs @ difference
            p, q = (total + displacement) / (2 * groups), (total - displacement) / (2 * groups)
            mixture = (p + q) / 2
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.asarray(
                    np.sum(
                        np.where(p > 0, p * np.log2(p / mixture), 0)
                        + np.where(q > 0, q * np.log2(q / mixture), 0),
                        axis=1,
                    )
                    / 2,
                    dtype=np.float64,
                )

        def text_statistic(data: FloatArray) -> float:
            return jsd_bits(
                np.mean(data[:, : len(vocabulary)], axis=0),
                np.mean(data[:, len(vocabulary) :], axis=0),
            )

        result["FP1"] = BlockComparison(
            _test(groups, jsd, seed=seed, resamples=resamples),
            np.concatenate((a, b), axis=1),
            text_statistic,
            "JSD bits, equally weighted independent groups",
        )
    if prepared.logprob_reason is not None:
        return result
    a, b = (
        np.asarray([[row[token] for token in anchors] for row in rows], dtype=np.float64)
        for rows in (prepared.first_logs, prepared.second_logs)
    )
    first = np.asarray([np.mean(a[x], axis=0) for x, _ in pairs])
    second = np.asarray([np.mean(b[y], axis=0) for _, y in pairs])
    difference = first - second

    def mean_statistic(data: FloatArray) -> float:
        return float(np.mean(np.abs(np.mean(data, axis=0))))

    with suppress(ValueError):
        result["FP3-mean"] = BlockComparison(
            _test(
                groups,
                lambda s: np.mean(np.abs(s @ difference / groups), axis=1),
                seed=seed,
                resamples=resamples,
            ),
            difference,
            mean_statistic,
            "mean absolute logprob displacement on fixed anchors",
        )
    center = np.mean(first + second, axis=0) / 2
    with np.errstate(over="ignore", invalid="ignore"):
        radius_difference = np.asarray(
            [
                np.mean(np.sum((a[x] - center) ** 2, axis=1))
                - np.mean(np.sum((b[y] - center) ** 2, axis=1))
                for x, y in pairs
            ]
        )
        with suppress(ValueError):
            result["FP3-dispersion"] = BlockComparison(
                _test(
                    groups,
                    lambda s: np.abs(s @ radius_difference / groups),
                    seed=seed,
                    resamples=resamples,
                ),
                radius_difference,
                _mean,
                "signed squared-logprob-radius difference on fixed anchors",
            )
    return result
