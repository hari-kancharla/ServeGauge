import math

import pytest

from servegauge.stats.changepoint import ADWIN, CUSUM, EValueAccumulator


def test_cusum_requires_explicit_calibration_and_detects_shift() -> None:
    detector = CUSUM(baseline_mean=0, baseline_sd=1, slack=0.5, threshold=8)
    assert not any(detector.update(0) for _ in range(20))
    assert any(detector.update(3) for _ in range(10))
    detector.reset()
    assert not detector.update(0)
    with pytest.raises(ValueError):
        CUSUM(baseline_mean=0, baseline_sd=0, slack=0.5, threshold=8)


def test_evalue_calibrator_and_shared_baseline_warning() -> None:
    evidence = EValueAccumulator(kappa=0.5, log_threshold=math.log(100))
    assert not evidence.update(0.001)
    assert evidence.update(0.001)
    assert evidence.requires_dependence_calibration
    evidence.reset()
    assert evidence.log_evidence == 0
    assert math.isfinite(evidence.calibrated_log_e(0))
    with pytest.raises(ValueError):
        evidence.update(1.1)


def test_adwin_uses_union_bound_and_binary_stream() -> None:
    detector = ADWIN(delta=0.01)
    assert not any(detector.update(1) for _ in range(200))
    assert any(detector.update(0) for _ in range(200))
    assert len(detector.window) < 400
    with pytest.raises(ValueError):
        detector.update(2)
    with pytest.raises(ValueError):
        ADWIN(delta=0)
