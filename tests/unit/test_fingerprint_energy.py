import itertools

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from servegauge.stats.energy import energy_distance
from servegauge.stats.fingerprint_energy import (
    calibrate_energy_cells,
    compare_energy_cells,
    energy_cell,
    energy_randomization_test,
)


def groups(seed=17, count=3):
    rng = np.random.default_rng(seed)
    return [[rng.normal(size=(i % 2 + 1, 3)) for i in range(count)] for _ in range(4)]


def weighted_energy(first, second):
    a, b = np.concatenate(first), np.concatenate(second)
    wa = np.concatenate([np.full(len(row), 1 / (len(first) * len(row))) for row in first])
    wb = np.concatenate([np.full(len(row), 1 / (len(second) * len(row))) for row in second])
    return float(2 * wa @ cdist(a, b) @ wb - wa @ cdist(a, a) @ wa - wb @ cdist(b, b) @ wb)


def test_geometry_preserves_response_energy_with_unequal_component_sizes():
    raw = groups()
    cell = energy_cell(raw)
    assert cell.method == "FP4" and len(cell.a) == 3
    assert cell.distance(cell.a, cell.b) == pytest.approx(weighted_energy(*raw[:2]), abs=1e-12)
    assert cell.distance(cell.c, cell.d) == pytest.approx(weighted_energy(*raw[2:]), abs=1e-12)


def test_grouping_embedding_means_would_destroy_distribution_signal():
    raw = [
        [np.array([[-1.0], [1.0]]) for _ in range(2)],
        [np.array([[0.0], [0.0]]) for _ in range(2)],
    ]
    cell = energy_cell([*raw, *raw])
    assert cell.distance(cell.a, cell.b) == pytest.approx(1)
    assert (
        energy_distance(
            np.array([row.mean(0) for row in raw[0]]), np.array([row.mean(0) for row in raw[1]])
        )
        == 0
    )


def test_fast_whole_component_swaps_match_every_brute_force_allocation():
    raw = groups()
    cell = energy_cell(raw)
    signs = np.array(list(itertools.product([1.0, -1.0], repeat=3)))
    expected = []
    for row in signs:
        a = [raw[0 if sign > 0 else 2][i] for i, sign in enumerate(row)]
        c = [raw[2 if sign > 0 else 0][i] for i, sign in enumerate(row)]
        expected.append(weighted_energy(a, raw[1]) - weighted_energy(c, raw[3]))
    assert np.allclose(cell.permuted(signs), expected, rtol=0, atol=1e-12)
    result = energy_randomization_test([cell], seed=11, permutations=2000)
    observed = abs(expected[0])
    p = sum(abs(x) >= observed - 1e-12 for x in expected) / len(expected)
    assert result.exact and result.permutations == 8 and result.n == 3
    assert result.p_value == pytest.approx(p)


def test_prompts_sum_without_changing_randomization_or_component_bootstrap():
    from servegauge.stats.fingerprint_strata import compare_cells

    cells = [energy_cell(groups(21)), energy_cell(groups(22))]
    kwargs = {"seed": 12, "permutations": 2000, "resamples": 2000, "alpha": 0.05}
    mean = compare_cells(cells, **kwargs)
    total = compare_energy_cells(cells, **kwargs)
    assert total.estimate == pytest.approx(2 * mean.estimate)
    assert total.reference_estimate == pytest.approx(2 * mean.reference_estimate)
    assert total.delta == pytest.approx(2 * mean.delta)
    assert (total.ci_low, total.ci_high) == pytest.approx((2 * mean.ci_low, 2 * mean.ci_high))
    assert total.p_value == mean.p_value and total.n == 6
    assert total.estimate_interval == pytest.approx(np.array(mean.estimate_interval) * 2)
    assert total.reference_interval == pytest.approx(np.array(mean.reference_interval) * 2)


def test_identical_dependent_groups_have_zero_signal_and_calibration():
    raw = [[np.ones((7, 2)) for _ in range(3)] for _ in range(4)]
    cell = energy_cell(raw)
    kwargs = {"seed": 9, "resamples": 2000, "alpha": 0.05}
    result = compare_energy_cells([cell], permutations=2000, **kwargs)
    assert result.estimate == result.delta == result.ci_low == result.ci_high == 0
    assert result.n == 3 and result.p_value == 1
    calibration = calibrate_energy_cells([cell], **kwargs)
    assert calibration["noise_band"] == 0 and calibration["noise_band_interval"] == [0, 0]


def test_reference_calibration_does_not_observe_target_and_scales_with_prompt_count():
    raw = groups()
    first = energy_cell(raw)
    changed = energy_cell([[row + 100 for row in raw[0]], *raw[1:]])
    kwargs = {"seed": 2, "resamples": 2000, "alpha": 0.05}
    a = calibrate_energy_cells([first], **kwargs)
    b = calibrate_energy_cells([changed], **kwargs)
    assert a["noise_band"] == pytest.approx(b["noise_band"], abs=1e-11)
    assert a["noise_band_interval"] == pytest.approx(b["noise_band_interval"], abs=1e-11)


@pytest.mark.parametrize(
    "case", ["empty", "three_sides", "one_group", "bad_shape", "nan", "dimension", "unequal_n"]
)
def test_missing_or_invalid_grouped_vectors_are_refused(case):
    raw = groups()
    if case == "empty":
        raw[0][0] = np.empty((0, 3))
    elif case == "three_sides":
        raw.pop()
    elif case == "one_group":
        raw = [[side[0]] for side in raw]
    elif case == "bad_shape":
        raw[0][0] = np.ones(3)
    elif case == "nan":
        raw[0][0][0, 0] = np.nan
    elif case == "dimension":
        raw[0][0] = np.ones((2, 4))
    else:
        raw[0].pop()
    with pytest.raises(ValueError):
        energy_cell(raw)


def test_fresh_iid_null_uses_exact_whole_group_randomization():
    # Independent families, not a provider qualification. Every family is kept.
    ps = [
        energy_randomization_test(
            [energy_cell(groups(91000 + i))], seed=1, permutations=2000
        ).p_value
        for i in range(300)
    ]
    # With only three swappable groups the smallest exact p is 1/8; alpha=.05 abstains.
    assert min(ps) >= 1 / 8 and all(p > 0.05 for p in ps)


def test_exact_conditional_randomization_controls_every_null_allocation():
    raw = groups(seed=711, count=7)
    ps = []
    for signs in itertools.product([1, -1], repeat=7):
        a = [raw[0 if sign > 0 else 2][i] for i, sign in enumerate(signs)]
        c = [raw[2 if sign > 0 else 0][i] for i, sign in enumerate(signs)]
        cell = energy_cell([a, raw[1], c, raw[3]])
        ps.append(energy_randomization_test([cell], seed=1, permutations=2000).p_value)
    assert sum(p <= 0.05 for p in ps) <= 6


@pytest.mark.parametrize(
    "operation", [energy_randomization_test, compare_energy_cells, calibrate_energy_cells]
)
def test_empty_prompt_sets_are_refused(operation):
    kwargs = {"seed": 1}
    if operation != calibrate_energy_cells:
        kwargs["permutations"] = 2000
    if operation != energy_randomization_test:
        kwargs.update(resamples=2000, alpha=0.05)
    with pytest.raises(ValueError):
        operation([], **kwargs)
