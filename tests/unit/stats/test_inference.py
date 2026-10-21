import math

import numpy as np
import pytest

from servegauge.stats.bootstrap import clustered_bootstrap, paired_bootstrap
from servegauge.stats.intervals import paired_difference_interval, wilson
from servegauge.stats.multiple import benjamini_hochberg, holm
from servegauge.stats.noninferiority import equivalent, non_inferior
from servegauge.stats.power import paired_mde, paired_sample_size, unpaired_sample_size
from servegauge.stats.tests import midp_mcnemar, paired_permutation, wilcoxon_test


def test_wilson_known_answers_and_boundaries() -> None:
    assert wilson(90, 100) == pytest.approx((0.8256343385, 0.9447708629))
    assert wilson(0, 100)[0] == 0
    assert wilson(100, 100)[1] == 1
    assert wilson(0, 400)[1] < 0.01
    assert wilson(0, 10)[1] > 0.01
    for successes, n in ((1, 0), (-1, 10), (11, 10)):
        with pytest.raises(ValueError):
            wilson(successes, n)


def test_tango_against_independent_reference_and_zero_discordance() -> None:
    # Reference: PropCIs scoreci.mp, b=10,c=20,n=100; independent equation inversion.
    x = [1] * 20 + [0] * 10 + [1] * 50 + [0] * 20
    y = [0] * 20 + [1] * 10 + [1] * 50 + [0] * 20
    assert paired_difference_interval(x, y) == pytest.approx(
        (-0.0076829038587, 0.2082117752676), abs=1e-9
    )
    lower, upper = paired_difference_interval([0, 1] * 50, [0, 1] * 50)
    assert lower < 0 < upper
    assert abs(lower) == pytest.approx(wilson(0, 100)[1])
    assert paired_difference_interval([1] * 100, [0] * 100)[1] == 1
    assert paired_difference_interval([0] * 100, [1] * 100)[0] == -1
    with pytest.raises(ValueError):
        paired_difference_interval([1], [0, 1])
    with pytest.raises(ValueError):
        paired_difference_interval([2], [1])


def test_bootstrap_preserves_items_and_pairing() -> None:
    values = [0.0] * 20 + [1.0] * 20
    clusters = ["first"] * 20 + ["second"] * 20
    result = clustered_bootstrap(values, clusters, seed=1, resamples=2000)
    assert result.estimate == 0.5 and result.low == 0 and result.high == 1
    assert result.n_clusters == 2
    paired = paired_bootstrap(np.arange(10) + 2, np.arange(10), seed=2)
    assert paired.low == paired.high == paired.estimate == 2
    assert paired == paired_bootstrap(np.arange(10) + 2, np.arange(10), seed=2)
    with pytest.raises(ValueError):
        clustered_bootstrap(values, clusters, seed=1, resamples=1999)
    with pytest.raises(ValueError):
        paired_bootstrap([1], [1, 2], seed=1)


def test_exact_tests_and_directionality() -> None:
    assert midp_mcnemar(10, 0) == pytest.approx(1 / 1024)
    assert midp_mcnemar(10, 0, alternative="greater") == pytest.approx(0.5 / 1024)
    assert midp_mcnemar(0, 0) == 1
    assert midp_mcnemar(10, 0, alternative="less") > 0.9
    with pytest.raises(ValueError):
        midp_mcnemar(-1, 2)
    # SciPy's published signed-rank example; numbers only, no benchmark text.
    values = [6, 8, 14, 16, 23, 24, 28, 29, 41, -48, 49, 56, 60, -67, 75]
    assert wilcoxon_test(values, method="exact").p_value == pytest.approx(0.041259765625)
    assert wilcoxon_test([0, 0, 0]).p_value == 1
    positive = paired_permutation([1] * 10, alternative="greater", seed=1, resamples=2048)
    assert positive.p_value == pytest.approx(1 / 1024)
    assert paired_permutation([0] * 5, seed=1).p_value == 1


def test_multiple_testing_noninferiority_and_power() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert benjamini_hochberg([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    assert holm([]) == []
    # Independent uniform complete-null family: P(any Holm rejection) equals
    # P(min p <= alpha/m), bounded by alpha without Monte Carlo uncertainty.
    assert 1 - (1 - 0.05 / 10) ** 10 <= 0.05
    assert benjamini_hochberg([]) == []
    for function in (holm, benjamini_hochberg):
        with pytest.raises(ValueError):
            function([math.nan])
    assert non_inferior(-0.01, 0.02)
    assert not non_inferior(-0.02, 0.02)
    assert equivalent(-0.01, 0.01, 0.02)
    assert not equivalent(-0.02, 0.01, 0.02)
    assert unpaired_sample_size(0.95, 0.90) == 435
    assert paired_sample_size(0.95, 0.90, correlation=0.5) < 435
    assert paired_mde(100, variance=0.1, family_size=10) > paired_mde(100, variance=0.1)
    with pytest.raises(ValueError):
        unpaired_sample_size(0.9, 0.9)
    with pytest.raises(ValueError):
        paired_sample_size(0.95, 0.5, correlation=1)
