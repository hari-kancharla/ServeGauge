import numpy as np
import pytest

from servegauge.stats.fingerprint_strata import bootstrap_cells, text_cell
from servegauge.validation.fingerprint_product import (
    _required_scale,
    _saved_rows,
    _widen,
    categorical_cells,
    configuration,
    finite_truth,
    interval_cases,
    kernel_features,
)


def test_finite_jsd_truth_includes_small_sample_bias_and_null_cancels():
    p = np.array([0.5, 0.5])
    assert finite_truth("FP1", p, p, 2) == pytest.approx(0, abs=1e-14)
    # Enumerate all two-draw categorical sequences independently of binomial marginals.
    import itertools

    from servegauge.stats.jsd import jsd_bits

    q = np.array([1.0, 0.0])
    samples = list(itertools.product(range(2), repeat=2))

    def expected(first, second):
        result = 0
        for a in samples:
            for b in samples:
                pa = np.bincount(a, minlength=2) / 2
                pb = np.bincount(b, minlength=2) / 2
                result += np.prod(first[list(a)]) * np.prod(second[list(b)]) * jsd_bits(pa, pb)
        return result

    assert finite_truth("FP1", p, q, 2) == pytest.approx(expected(q, p) - expected(p, p))


def test_compact_hamming_geometry_is_the_actual_unicode_padding_kernel():
    words = ["red", "blue", "green", "gold"]
    features = kernel_features(words, length=32)
    for a in range(4):
        for b in range(4):
            padded_a = list(words[a]) + [None] * (32 - len(words[a]))
            padded_b = list(words[b]) + [None] * (32 - len(words[b]))
            expected = sum(x == y for x, y in zip(padded_a, padded_b, strict=True)) / 32
            assert features[a] @ features[b] == pytest.approx(expected)
    groups = [[[w] for w in words]] * 4
    cell = text_cell(groups, method="FP2", length=32)
    assert features @ features.T == pytest.approx(cell.a @ cell.a.T)


def test_categorical_strata_are_seeded_independent_complete_components():
    p, q = np.array([0.55, 0.25, 0.13, 0.07]), np.array([0.07, 0.13, 0.25, 0.55])
    a = categorical_cells("FP1", p, q, cells=3, groups=5, seed=13)
    b = categorical_cells("FP1", p, q, cells=3, groups=5, seed=13)
    assert len(a) == 3 and all(c.a.shape == (5, 4) for c in a)
    assert all(np.array_equal(x.a, y.a) for x, y in zip(a, b, strict=True))
    assert all(np.array_equal(c.a.sum(axis=1), np.ones(5)) for c in a)
    assert finite_truth("FP2", p, p, 5) == pytest.approx(0)
    assert finite_truth("FP2", p, q, 5) > 0


@pytest.mark.parametrize("method", ["FP1", "FP2"])
def test_larger_validation_batches_preserve_every_paired_bootstrap_draw(method):
    p = np.array([0.55, 0.25, 0.13, 0.07])
    cells = categorical_cells(method, p, p[::-1], cells=2, groups=6, seed=17)
    original = bootstrap_cells(cells, seed=28, resamples=2000)
    batched = bootstrap_cells(cells, seed=28, resamples=2000, batch_size=2000)
    assert np.array_equal(original, batched)


def test_predeclared_known_truth_grid_reaches_both_sides_of_the_fixed_margin():
    cases = interval_cases(configuration())
    assert len(cases) == 24 and len({c["id"] for c in cases}) == 24
    assert sum(c["null"] for c in cases) == 6
    assert sum(c["outside_equivalence"] for c in cases) == 12
    assert all(c["margin"] == 0.03 for c in cases)
    assert all(
        c["truth"] == pytest.approx(0.03, abs=1e-12) for c in cases if "boundary-1.0" in c["id"]
    )


def test_calibration_only_widens_intervals_and_keeps_the_declared_margin():
    row = {"estimate": 0.1, "truth": 0.03, "ci_low": 0.08, "ci_high": 0.12, "margin": 0.03}
    scale = _required_scale(row)
    assert scale == pytest.approx(3.5)
    wider = _widen(row, scale)
    assert wider["ci_low"] == pytest.approx(0.03)
    assert wider["ci_high"] == pytest.approx(0.17)
    assert wider["margin"] == 0.03 and row["ci_low"] == 0.08


def test_saved_cases_require_complete_hashes_matching_seeds_and_input_layout(tmp_path):
    from servegauge.core.hashing import content_hash
    from servegauge.validation.common import named_seed, write_json

    case = {"id": "FP1-case", "groups": 6, "cells": 2, "truth": 0.0}
    cfg, header = {"seed": 101}, {"source_hash": "source", "configuration_hash": "config"}
    seed = named_seed(named_seed(101, "calibration", case["id"], 0), "bootstrap")
    rows = [
        {"seed": seed, "n": 12, "truth": 0.0, "estimate": 0.01, "ci_low": -0.02, "ci_high": 0.02}
    ]
    path = tmp_path / "case.json"
    assert _saved_rows(path, header, case, "calibration", 1, cfg) is None
    write_json(path, {**header, "case": case, "trials": rows, "trials_hash": content_hash(rows)})
    assert _saved_rows(path, header, case, "calibration", 1, cfg) == rows
    with pytest.raises(ValueError, match="seeded"):
        _saved_rows(path, header, case, "evaluation", 1, cfg)
    rows[0]["n"] = 13
    write_json(path, {**header, "case": case, "trials": rows, "trials_hash": content_hash(rows)})
    with pytest.raises(ValueError, match="seeded"):
        _saved_rows(path, header, case, "calibration", 1, cfg)


@pytest.mark.parametrize("batch", [0, 2001, True])
def test_invalid_bootstrap_memory_batch_is_refused(batch):
    p = np.array([0.25] * 4)
    cells = categorical_cells("FP1", p, p, cells=2, groups=2, seed=12)
    with pytest.raises(ValueError, match="batch_size"):
        bootstrap_cells(cells, seed=5, resamples=2000, batch_size=batch)


@pytest.mark.parametrize(
    "p,q", [([1, 0], [1]), ([-0.1, 1.1], [0.5, 0.5]), ([0.5, 0.6], [0.5, 0.5])]
)
def test_unknown_or_invalid_probability_processes_refuse_truth(p, q):
    with pytest.raises(ValueError):
        finite_truth("FP1", np.array(p), np.array(q), 5)
