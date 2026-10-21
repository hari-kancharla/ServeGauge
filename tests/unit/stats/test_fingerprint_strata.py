import itertools

import numpy as np
import pytest

from servegauge.stats.fingerprint_strata import Cell, calibrate_cells, compare_cells, text_cell


@pytest.mark.parametrize("method", ["FP1", "FP2"])
def test_quartet_randomization_matches_every_whole_component_label_swap(method):
    groups = [
        [["red", "red"], ["blue"], ["red", "blue"]],
        [["blue"], ["red", "blue"], ["red"]],
        [["blue", "blue"], ["red"], ["blue", "red"]],
        [["red"], ["red", "red"], ["blue"]],
    ]
    cell = text_cell(groups, method=method, length=8)
    result = compare_cells([cell], seed=7, permutations=64, resamples=2000, alpha=0.05)
    values = []
    for swap in itertools.product([False, True], repeat=3):
        a, c = cell.a.copy(), cell.c.copy()
        for i, yes in enumerate(swap):
            if yes:
                a[i], c[i] = cell.c[i], cell.a[i]
        oracle = Cell(method, a, cell.b, c, cell.d)
        values.append(abs(oracle.distance(a, cell.b) - oracle.distance(c, cell.d)))
    observed = abs(cell.distance(cell.a, cell.b) - cell.distance(cell.c, cell.d))
    assert result.exact and result.permutations == 8 and result.n == 3
    assert result.delta == pytest.approx(
        cell.distance(cell.a, cell.b) - cell.distance(cell.c, cell.d)
    )
    assert result.p_value == sum(v >= observed - 1e-12 for v in values) / len(values)


def test_permutation_preserves_prompt_strata_and_component_counts():
    first = text_cell([[["red"], ["blue"]]] * 4, method="FP1", length=8)
    second = text_cell([[["red"], ["red"], ["blue"]]] * 4, method="FP1", length=8)
    result = compare_cells([first, second], seed=11, permutations=64, resamples=2000, alpha=0.05)
    assert result.n == 5 and result.exact and result.permutations == 32
    assert result.estimate == pytest.approx(result.reference_estimate)
    assert result.ci_low <= result.delta <= result.ci_high


@pytest.mark.parametrize("method", ["FP1", "FP2"])
def test_repeating_a_cached_reply_does_not_create_information(method):
    base = [
        [["red"], ["blue"], ["red"]],
        [["blue"], ["blue"], ["red"]],
        [["blue"], ["red"], ["red"]],
        [["red"], ["red"], ["blue"]],
    ]
    repeated = [[row * 20 for row in side] for side in base]
    args = dict(seed=123, permutations=64, resamples=2000, alpha=0.05)
    a = compare_cells([text_cell(base, method=method, length=8)], **args)
    b = compare_cells([text_cell(repeated, method=method, length=8)], **args)
    assert a == b


def test_invalid_or_unmatched_components_fail_without_filling_values():
    with pytest.raises(ValueError):
        text_cell([[["red"]]] * 4, method="FP1", length=8)
    with pytest.raises(ValueError):
        text_cell([[["red"], []]] * 4, method="FP1", length=8)
    with pytest.raises(ValueError):
        Cell("FP3", np.array([[np.nan]]), np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)))


@pytest.mark.parametrize("method", ["FP1", "FP2"])
def test_reference_calibration_is_independent_of_target_observations(method):
    groups = [[["red"], ["blue"], ["green"], ["gold"]]] * 4
    cell = text_cell(groups, method=method, length=8)
    original = calibrate_cells([cell], seed=19, resamples=2000, alpha=0.05)
    changed = Cell(method, np.tile(cell.a[0], (len(cell.a), 1)), cell.b, cell.c, cell.d)
    assert calibrate_cells([changed], seed=19, resamples=2000, alpha=0.05) == original


@pytest.mark.parametrize("method", ["FP1", "FP2"])
def test_stratified_test_detects_shift_instead_of_swapping_symmetric_cell_distances(method):
    cells = [
        text_cell(
            [[["blue"]] * 4, [["red"]] * 4, [["red"]] * 4, [["red"]] * 4], method=method, length=8
        )
        for _ in range(2)
    ]
    result = compare_cells(cells, seed=13, permutations=256, resamples=2000, alpha=0.05)
    assert result.exact and result.p_value <= 0.05 and result.delta > 0


@pytest.mark.parametrize("method", ["FP3-mean", "FP3-dispersion"])
def test_logprob_quartet_matches_exhaustive_component_swaps(method):
    rng = np.random.default_rng(83)
    features = []
    for _ in range(4):
        logits = rng.normal(size=(4, 3, 4))
        probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        logs = np.log(probabilities[..., :2])
        value = np.mean(logs, axis=1)
        if method == "FP3-dispersion":
            value = np.concatenate((value, np.mean(logs**2, axis=1)), axis=1)
        features.append(value)
    cell = Cell(method, *features)
    result = compare_cells([cell], seed=18, permutations=16, resamples=2000, alpha=0.05)
    null = []
    for swap in itertools.product([False, True], repeat=4):
        a, c = cell.a.copy(), cell.c.copy()
        for i, yes in enumerate(swap):
            if yes:
                a[i], c[i] = cell.c[i], cell.a[i]
        null.append(abs(cell.distance(a, cell.b) - cell.distance(c, cell.d)))
    observed = abs(cell.distance(cell.a, cell.b) - cell.distance(cell.c, cell.d))
    assert result.p_value == sum(v >= observed - 1e-12 for v in null) / 16
