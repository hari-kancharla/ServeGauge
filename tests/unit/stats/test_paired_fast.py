import numpy as np
import pytest

from servegauge.stats.energy import energy_test
from servegauge.stats.jsd import jsd_test
from servegauge.stats.logprob_track import logprob_test
from servegauge.stats.mmd import hamming_test
from servegauge.stats.paired_fast import (
    energy_test_fast,
    hamming_test_fast,
    jsd_test_fast,
    kernel_test,
    logprob_test_fast,
)


@pytest.mark.parametrize("n,resamples", [(4, 32), (12, 199)])
def test_batched_tests_match_scalar_independent_implementations(n, resamples):
    rng = np.random.default_rng(23)
    a = rng.choice(["red", "blue", "gold"], n).tolist()
    b = rng.choice(["red", "blue", "gold"], n).tolist()
    x, y = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
    logs_a = [{s: -0.2} for s in a]
    logs_b = [{s: -0.4} for s in b]
    pairs = [
        (
            hamming_test(a, b, length=16, seed=11, resamples=resamples),
            hamming_test_fast(a, b, length=16, seed=11, resamples=resamples),
        ),
        (
            energy_test(x, y, seed=11, resamples=resamples),
            energy_test_fast(x, y, seed=11, resamples=resamples),
        ),
        (
            jsd_test([a], [b], seed=11, resamples=resamples),
            jsd_test_fast(a, b, seed=11, resamples=resamples),
        ),
        (
            logprob_test(logs_a, logs_b, missing_floor=-30, seed=11, resamples=resamples),
            logprob_test_fast(logs_a, logs_b, missing_floor=-30, seed=11, resamples=resamples),
        ),
    ]
    for scalar, batched in pairs:
        assert batched.statistic == pytest.approx(scalar.statistic, abs=1e-12)
        assert batched.p_value == scalar.p_value
        assert batched.exact == scalar.exact


def test_batched_tests_reject_invalid_designs_and_keep_scalar_vectors():
    assert energy_test_fast([1, 2, 3], [2, 3, 4], seed=1, resamples=8).exact
    with pytest.raises(ValueError):
        jsd_test_fast([], [], seed=1, resamples=8)
    with pytest.raises(ValueError):
        kernel_test([[1]], [[1]], kind="unknown", seed=1, resamples=8)
    with pytest.raises(ValueError):
        jsd_test_fast(["a"], ["b"], seed=1, resamples=0)
