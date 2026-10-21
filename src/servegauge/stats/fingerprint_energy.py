"""FP4 energy of response distributions, with whole independent component swaps.

Euclidean distance induces k(x,y)=(|x|+|y|-|x-y|)/2. Its finite Gram features
preserve energy = 2 |mean(phi(x))-mean(phi(y))|². Average phi within sessions,
never the original embeddings: averaging embeddings would erase distribution
differences. Components receive equal weight, responses equal weight within each
component. Prompt energies are summed. Exchangeability is a caller precondition.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import binom

from servegauge.stats.common import FloatArray, finite_array, paired_arrays, probability
from servegauge.stats.fingerprint_strata import (
    Cell,
    Comparison,
    bootstrap_cells,
    compare_cells,
    randomization_test,
)
from servegauge.stats.tests import TestResult


@dataclass
class EnergyCell(Cell):
    def __post_init__(self) -> None:
        if self.method != "FP4":
            raise ValueError("embedding_energy_method_required")
        self.a, self.b = paired_arrays(self.a, self.b)
        self.c, self.d = paired_arrays(self.c, self.d)
        self.a, self.c = paired_arrays(self.a, self.c)
        if self.a.ndim != 2 or self.a.shape[0] < 2 or self.a.shape[1] < 1:
            raise ValueError("matched_independent_energy_component_matrices_required")

    def metric(self, a: FloatArray, b: FloatArray) -> FloatArray:
        difference = np.mean(a, axis=-2) - np.mean(b, axis=-2)
        return np.asarray(2 * np.sum(difference**2, axis=-1), dtype=np.float64)

    def permuted(self, signs: FloatArray) -> FloatArray:
        n = len(self.a)
        middle = np.mean(self.a + self.c, axis=0) / 2
        bm, dm = np.mean(self.b, axis=0), np.mean(self.d, axis=0)
        constant = 2 * (np.sum((middle - bm) ** 2) - np.sum((middle - dm) ** 2))
        coefficient = 2 * (self.a - self.c) @ (2 * middle - bm - dm) / n
        return np.asarray(constant + signs @ coefficient, dtype=np.float64)


def energy_cell(groups: list[list[FloatArray]]) -> EnergyCell:
    if len(groups) != 4 or len({len(s) for s in groups}) != 1 or len(groups[0]) < 2:
        raise ValueError("four_matched_independent_grouped_embedding_samples_required")
    rows = [finite_array(row) for side in groups for row in side]
    if (
        any(row.ndim != 2 or row.shape[1] < 1 for row in rows)
        or len({row.shape[1] for row in rows}) != 1
    ):
        raise ValueError("complete_matching_embedding_feature_vectors_required")
    # Exact duplicate vectors share one Gram coordinate, retaining their weights.
    pooled, inverse = np.unique(np.concatenate(rows), axis=0, return_inverse=True)
    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.linalg.norm(pooled, axis=1)
        distances = cdist(pooled, pooled)
        gram = (norms[:, None] + norms[None, :] - distances) / 2
    if not np.all(np.isfinite(gram)):
        raise ValueError("finite_energy_geometry_required")
    try:
        values, vectors = np.linalg.eigh(gram)
    except np.linalg.LinAlgError:
        raise ValueError("energy_geometry_cannot_be_decomposed") from None
    # Floating precision guard, unrelated to a practical equivalence margin.
    tolerance = 100 * np.finfo(float).eps * len(pooled) * max(1, float(np.max(np.abs(gram))))
    if float(np.min(values)) < -tolerance:
        raise ValueError("positive_semidefinite_energy_geometry_required")
    features = vectors * np.sqrt(np.maximum(values, 0))
    output = []
    offset = 0
    for side in groups:
        grouped = []
        for row in side:
            grouped.append(np.mean(features[inverse[offset : offset + len(row)]], axis=0))
            offset += len(row)
        output.append(np.asarray(grouped))
    return EnergyCell("FP4", *output)


def _checked(cells: list[EnergyCell]) -> list[Cell]:
    if not cells or any(not isinstance(c, EnergyCell) or c.method != "FP4" for c in cells):
        raise ValueError("nonempty_fp4_prompt_strata_required")
    return list(cells)


def energy_randomization_test(
    cells: list[EnergyCell], *, seed: int, permutations: int
) -> TestResult:
    result = randomization_test(_checked(cells), seed=seed, permutations=permutations)
    return replace(result, statistic=result.statistic * len(cells))


def compare_energy_cells(
    cells: list[EnergyCell], *, seed: int, permutations: int, resamples: int, alpha: float
) -> Comparison:
    mean = compare_cells(
        _checked(cells), seed=seed, permutations=permutations, resamples=resamples, alpha=alpha
    )
    scale = len(cells)
    return replace(
        mean,
        estimate=mean.estimate * scale,
        reference_estimate=mean.reference_estimate * scale,
        delta=mean.delta * scale,
        ci_low=mean.ci_low * scale,
        ci_high=mean.ci_high * scale,
        estimate_interval=(mean.estimate_interval[0] * scale, mean.estimate_interval[1] * scale),
        reference_interval=(mean.reference_interval[0] * scale, mean.reference_interval[1] * scale),
    )


def calibrate_energy_cells(
    cells: list[EnergyCell], *, seed: int, resamples: int, alpha: float
) -> dict[str, object]:
    _checked(cells)
    probability(alpha)
    reference: list[Cell] = [EnergyCell("FP4", c.b, c.b, c.c, c.d) for c in cells]
    draws = bootstrap_cells(reference, seed=seed, resamples=resamples, independent=True) * len(
        cells
    )
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
        "prompt_aggregation": "sum",
        "seed": seed,
        "resamples": resamples,
        "method": "independent_reference_only_stratified_energy_component_bootstrap",
        "limitations": "Empirical conditional bootstrap with fixed embeddings; no unseen "
        "dependence or universal near-null interval-coverage guarantee",
    }
