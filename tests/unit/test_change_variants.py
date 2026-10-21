import math

import numpy as np
import pytest

from servegauge.stats.change_variants import BatchADWIN, RestartingEvidence
from servegauge.stats.changepoint import EValueAccumulator


def test_restart_removes_negative_history_without_changing_calibrator():
    old = EValueAccumulator(kappa=0.5, log_threshold=3)
    new = RestartingEvidence(kappa=0.5, log_threshold=3)
    for _ in range(100):
        assert not old.update(1)
        assert not new.update(1)
    assert new.log_evidence == 0 and old.log_evidence < -60
    assert new.calibrated_log_e(0.01) == old.calibrated_log_e(0.01)
    assert not new.update(0.01)
    assert new.update(0.01)
    assert not old.update(0.01)
    new.reset()
    assert new.log_evidence == 0
    assert new.requires_dependence_calibration


def test_batch_detector_retains_all_bits_and_shrinks_on_known_change():
    detector = BatchADWIN(delta=1e-5)
    assert not detector.update([1] * 64)
    for _ in range(40):
        assert not detector.update([1] * 32)
    assert detector.observations == 1344
    assert detector.update([0] * 256)
    assert detector.observations == 256
    assert not detector.update([0] * 32)
    before = detector.observations
    with pytest.raises(ValueError):
        detector.update([0, 2])
    assert detector.observations == before
    with pytest.raises(ValueError):
        detector.update([])


def test_batch_boundary_bound_agrees_with_independent_concatenated_cut_oracle():
    # The oracle works on raw observations. Its candidate cuts are fingerprint boundaries.
    rng = np.random.default_rng(18)
    batches = [rng.binomial(1, p, 32).tolist() for p in [0.95] * 15 + [0.3] * 25]
    detector = BatchADWIN(delta=0.001)
    window = []
    lengths = []
    events = []
    for batch in batches:
        window.extend(batch)
        lengths.append(len(batch))
        changed = False
        while len(lengths) > 1:
            n = len(window)
            cut = next(
                (
                    c
                    for c in np.cumsum(lengths)[:-1]
                    if abs(sum(window[:c]) / c - sum(window[c:]) / (n - c))
                    > math.sqrt(0.5 * (1 / c + 1 / (n - c)) * math.log(4 * n / 0.001))
                ),
                None,
            )
            if cut is None:
                break
            drop = list(np.cumsum(lengths)).index(cut) + 1
            window = window[cut:]
            lengths = lengths[drop:]
            changed = True
        events.append(changed)
        assert detector.update(batch) == changed
        assert detector.observations == len(window)
    assert any(events)
