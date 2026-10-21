"""Matched prompt strata and whole-component reference contrasts.

A target (a) and three independent reference captures (b,c,d) provide
distance(a,b)-distance(c,d). Randomization swaps a/c components within each prompt
stratum, retaining the two other captures and every within-component dependence.
This needs exchangeable target/reference sampling, never inferred by this module.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

from servegauge.stats.common import FloatArray, paired_arrays, probability
from servegauge.stats.paired_fast import _test
from servegauge.stats.tests import TestResult


def _jsd(a: FloatArray, b: FloatArray) -> FloatArray:
    middle = (a + b) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(
            np.sum(
                np.where(a > 0, a * np.log2(a / middle), 0)
                + np.where(b > 0, b * np.log2(b / middle), 0),
                axis=-1,
            )
            / 2,
            dtype=np.float64,
        )


def _within(data: FloatArray) -> FloatArray:
    n = data.shape[-2]
    total = np.sum(data, axis=-2)
    return np.asarray(
        (np.sum(total**2, axis=-1) - np.sum(data**2, axis=(-2, -1))) / (n * (n - 1)),
        dtype=np.float64,
    )


@dataclass
class Cell:
    method: str
    a: FloatArray
    b: FloatArray
    c: FloatArray
    d: FloatArray

    def __post_init__(self) -> None:
        if self.method not in {"FP1", "FP2", "FP3-mean", "FP3-dispersion"}:
            raise ValueError("known_fingerprint_metric_required")
        self.a, self.b = paired_arrays(self.a, self.b)
        self.c, self.d = paired_arrays(self.c, self.d)
        self.a, self.c = paired_arrays(self.a, self.c)
        if self.a.ndim != 2 or self.a.shape[0] < 2 or self.a.shape[1] < 1:
            raise ValueError("matched_independent_component_matrices_required")
        if self.method == "FP3-dispersion" and self.a.shape[1] % 2:
            raise ValueError("mean_and_second_moment_features_required")
        if self.method == "FP1" and any(
            np.any(array < 0)
            or not np.allclose(
                array.sum(axis=1),
                1,
                rtol=0,
                atol=float(100 * np.finfo(float).eps * array.shape[1]),
            )
            for array in (self.a, self.b, self.c, self.d)
        ):
            raise ValueError("observed_text_probability_vectors_required")

    @property
    def center(self) -> FloatArray:
        width = self.a.shape[-1] // 2
        return np.asarray(
            np.mean(np.concatenate((self.a, self.b, self.c, self.d))[:, :width], axis=0),
            dtype=np.float64,
        )

    def metric(self, a: FloatArray, b: FloatArray) -> FloatArray:
        am, bm = np.mean(a, axis=-2), np.mean(b, axis=-2)
        if self.method == "FP1":
            return _jsd(am, bm)
        if self.method == "FP2":
            return np.asarray(
                _within(a) + _within(b) - 2 * np.sum(am * bm, axis=-1), dtype=np.float64
            )
        if self.method == "FP3-mean":
            return np.asarray(np.mean(np.abs(am - bm), axis=-1), dtype=np.float64)
        width = am.shape[-1] // 2
        difference = am - bm
        return np.asarray(
            np.abs(
                np.sum(difference[..., width:] - 2 * self.center * difference[..., :width], axis=-1)
            ),
            dtype=np.float64,
        )

    def distance(self, a: FloatArray, b: FloatArray) -> float:
        return float(self.metric(a, b))

    def permuted(self, signs: FloatArray) -> FloatArray:
        n = len(self.a)
        total = np.sum(self.a + self.c, axis=0)
        difference = self.a - self.c
        shift = signs @ difference
        am, cm = (total + shift) / (2 * n), (total - shift) / (2 * n)
        bm, dm = np.mean(self.b, axis=0), np.mean(self.d, axis=0)
        if self.method == "FP1":
            return _jsd(am, bm) - _jsd(cm, dm)
        if self.method == "FP3-mean":
            return np.asarray(
                np.mean(np.abs(am - bm), axis=1) - np.mean(np.abs(cm - dm), axis=1),
                dtype=np.float64,
            )
        if self.method == "FP3-dispersion":
            width = am.shape[-1] // 2

            def radius(x: FloatArray) -> FloatArray:
                return np.asarray(
                    np.sum(x[..., width:] - 2 * self.center * x[..., :width], axis=-1),
                    dtype=np.float64,
                )

            return np.asarray(
                np.abs(radius(am) - radius(bm)) - np.abs(radius(cm) - radius(dm)), dtype=np.float64
            )
        norm_difference = np.sum(self.a**2 - self.c**2, axis=1)
        coefficient = (difference @ total - norm_difference) / (n * (n - 1))
        coefficient -= difference @ np.sum(self.b + self.d, axis=0) / n**2
        constant = float(_within(self.b) - _within(self.d))
        constant += float(total @ np.sum(self.d - self.b, axis=0)) / n**2
        return np.asarray(constant + signs @ coefficient, dtype=np.float64)


def text_cell(groups: list[list[list[str]]], *, method: str, length: int) -> Cell:
    if (
        method not in {"FP1", "FP2"}
        or length < 1
        or len(groups) != 4
        or len({len(side) for side in groups}) != 1
        or len(groups[0]) < 2
        or any(
            not row or any(not isinstance(s, str) or not s for s in row)
            for side in groups
            for row in side
        )
    ):
        raise ValueError("four_complete_matched_grouped_text_samples_required")
    texts = {s for side in groups for row in side for s in row}
    if method == "FP1":
        vocabulary = {s: i for i, s in enumerate(sorted(texts))}
        vectors = {s: np.eye(len(vocabulary))[vocabulary[s]] for s in texts}
    else:
        characters = {-1, *(ord(ch) for s in texts for ch in s[:length])}
        alphabet = {ch: i for i, ch in enumerate(sorted(characters))}
        vectors = {}
        for text in texts:
            vector = np.zeros((length, len(alphabet)))
            for position in range(length):
                char = ord(text[position]) if position < len(text) else -1
                vector[position, alphabet[char]] = 1 / np.sqrt(length)
            vectors[text] = vector.ravel()
    arrays = [
        np.asarray(
            [
                np.sum(
                    [vectors[s] * (count / len(row)) for s, count in sorted(Counter(row).items())],
                    axis=0,
                )
                for row in side
            ]
        )
        for side in groups
    ]
    return Cell(method, *arrays)


@dataclass(frozen=True)
class Comparison:
    """Empirical component interval; equivalence qualification is a separate gate."""

    estimate: float
    reference_estimate: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int
    permutations: int
    exact: bool
    estimate_interval: tuple[float, float]
    reference_interval: tuple[float, float]


def bootstrap_cells(
    cells: list[Cell],
    *,
    seed: int,
    resamples: int,
    independent: bool = False,
    batch_size: int = 32,
) -> FloatArray:
    if not cells or resamples < 2000:
        raise ValueError("cells_and_at_least_2000_bootstrap_resamples_required")
    if type(batch_size) is not int or not 1 <= batch_size <= resamples:
        raise ValueError("bounded_positive_bootstrap_batch_size_required")
    rng = np.random.default_rng(seed)
    output = np.zeros((resamples, 3), dtype=np.float64)
    for cell in cells:
        n = len(cell.a)
        for start in range(0, resamples, batch_size):
            stop = min(start + batch_size, resamples)
            draws = [
                rng.integers(0, n, size=(stop - start, n)) for _ in range(4 if independent else 1)
            ]
            arrays = [
                array[draws[i if independent else 0]]
                for i, array in enumerate((cell.a, cell.b, cell.c, cell.d))
            ]
            first, second = cell.metric(arrays[0], arrays[1]), cell.metric(arrays[2], arrays[3])
            output[start:stop, 0] += first / len(cells)
            output[start:stop, 1] += second / len(cells)
            output[start:stop, 2] += (first - second) / len(cells)
    if not np.all(np.isfinite(output)):
        raise ValueError("finite_fingerprint_bootstrap_required")
    return output


def randomization_test(cells: list[Cell], *, seed: int, permutations: int) -> TestResult:
    if not cells or len({c.method for c in cells}) != 1:
        raise ValueError("nonempty_same_method_prompt_strata_required")
    n = sum(len(c.a) for c in cells)

    def statistic(signs: FloatArray) -> FloatArray:
        offset = 0
        values = np.zeros(len(signs))
        for cell in cells:
            count = len(cell.a)
            values += cell.permuted(signs[:, offset : offset + count]) / len(cells)
            offset += count
        return np.abs(values)

    return _test(n, statistic, seed=seed, resamples=permutations)


def compare_cells(
    cells: list[Cell], *, seed: int, permutations: int, resamples: int, alpha: float
) -> Comparison:
    probability(alpha)
    result = randomization_test(cells, seed=seed, permutations=permutations)
    n = result.n
    draws = bootstrap_cells(cells, seed=seed, resamples=resamples)
    first = float(np.mean([c.distance(c.a, c.b) for c in cells]))
    second = float(np.mean([c.distance(c.c, c.d) for c in cells]))
    low, high = np.quantile(draws[:, 2], [alpha / 2, 1 - alpha / 2])
    first_low, first_high = np.quantile(draws[:, 0], [alpha / 2, 1 - alpha / 2])
    second_low, second_high = np.quantile(draws[:, 1], [alpha / 2, 1 - alpha / 2])
    return Comparison(
        first,
        second,
        first - second,
        float(low),
        float(high),
        result.p_value,
        n,
        result.permutations,
        result.exact,
        (float(first_low), float(first_high)),
        (float(second_low), float(second_high)),
    )


def calibrate_cells(
    cells: list[Cell], *, seed: int, resamples: int, alpha: float
) -> dict[str, object]:
    probability(alpha)
    # Two independent bootstrap draws of reference b, then references c/d.
    # Target a never participates in its own reference calibration.
    reference = [Cell(c.method, c.b, c.b, c.c, c.d) for c in cells]
    draws = bootstrap_cells(reference, seed=seed, resamples=resamples, independent=True)
    noise = np.sort(np.abs(draws[:, 2]))
    ranks = np.clip(
        np.asarray(binom.ppf([alpha / 2, 1 - alpha / 2], resamples, 1 - alpha), dtype=np.int64),
        0,
        resamples - 1,
    )
    return {
        "noise_band": float(np.quantile(noise, 1 - alpha)),
        "noise_band_interval": [float(noise[ranks[0]]), float(noise[ranks[1]])],
        "n": sum(len(c.a) for c in cells),
        "groups_per_cell": [len(c.a) for c in cells],
        "seed": seed,
        "resamples": resamples,
        "method": "independent_reference_only_stratified_component_bootstrap",
        "limitations": "Empirical conditional bootstrap; no unseen dependence or universal "
        "near-null interval-coverage guarantee",
    }
