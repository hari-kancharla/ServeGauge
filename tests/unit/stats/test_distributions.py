import numpy as np
import pytest

from servegauge.stats.energy import energy_distance, energy_test
from servegauge.stats.jsd import jsd_bits, jsd_test
from servegauge.stats.logprob_track import logprob_matrix, logprob_statistic, logprob_test
from servegauge.stats.mmd import hamming_mmd, hamming_test
from servegauge.stats.tests import two_sample_permutation


def test_jsd_bits_known_answers_and_configuration_matching() -> None:
    assert jsd_bits([1, 0], [0, 1]) == 1
    assert jsd_bits([1, 1], [1, 1]) == 0
    result = jsd_test([["a"] * 20], [["b"] * 20], seed=1, resamples=199)
    assert result.statistic == 1 and result.p_value <= 0.01
    assert jsd_test([["a"] * 20], [["a"] * 20], seed=1, resamples=199).p_value == 1
    with pytest.raises(ValueError):
        jsd_bits([0], [0])
    with pytest.raises(ValueError):
        jsd_test([["a"]], [["a", "a"]], seed=1)


def test_hamming_kernel_uses_positions_and_preserves_fixed_length() -> None:
    assert hamming_mmd(["ab", "ab"], ["cd", "cd"], length=2) == 4
    assert hamming_mmd(["ab", "ab"], ["ab", "ab"], length=2) == 0
    assert hamming_mmd(["ab", "ab"], ["ac", "ac"], length=2) == 2
    result = hamming_test(["ab"] * 20, ["cd"] * 20, length=2, seed=2, resamples=199)
    assert result.p_value <= 0.01
    with pytest.raises(ValueError):
        hamming_mmd(["a"], ["b"], length=1)


def test_energy_and_logprob_mean_distances() -> None:
    assert energy_distance([[0]], [[1]]) == 2
    assert energy_distance([[1], [2]], [[1], [2]]) == 0
    result = energy_test(np.zeros((20, 2)), np.ones((20, 2)), seed=2, resamples=199)
    assert result.p_value <= 0.01
    x, y, vocabulary = logprob_matrix([{"a": -1}], [{"b": -2}], missing_floor=-30)
    assert vocabulary == ["a", "b"]
    assert x.tolist() == [[-1, -30]] and y.tolist() == [[-30, -2]]
    assert logprob_statistic(x, y) == 28.5
    assert (
        logprob_test(
            [{"a": -1}] * 20, [{"a": -2}] * 20, missing_floor=-30, seed=1, resamples=199
        ).p_value
        <= 0.01
    )
    with pytest.raises(ValueError):
        logprob_matrix([{}], [{}], missing_floor=-30)
    with pytest.raises(ValueError):
        logprob_matrix([{"a": -40}], [{"a": -1}], missing_floor=-30)


def test_paired_permutations_keep_rows_paired() -> None:
    x = np.arange(10, dtype=float)
    y = x + 10
    result = two_sample_permutation(
        x,
        y,
        lambda a, b: float(np.mean(a - b)),
        paired=True,
        alternative="less",
        seed=1,
        resamples=2048,
    )
    assert result.p_value == pytest.approx(1 / 1024)
    with pytest.raises(ValueError):
        two_sample_permutation([1], [1, 2], lambda a, b: 0, paired=True, seed=1)
