import numpy as np
import pytest

from servegauge.stats.multiple import holm
from servegauge.validation.changepoint_sim import permutation_p
from servegauge.validation.common import settings
from servegauge.validation.detection_study import (
    _P,
    configuration,
    delay_values,
    noise_observations,
    noise_trial,
    variant_events,
)


def test_noise_trial_uses_full_topk_real_probabilities_and_correct_holm():
    words, logs = noise_observations(8, seed=2, jitter=0.1, noise=0.3)
    assert len(words) == len(logs) == 8
    assert all(
        len(row) == 4 and sum(np.exp(list(row.values()))) == pytest.approx(1) for row in logs
    )
    assert len({tuple(r.values()) for r in logs}) > 1
    trial = noise_trial(8, seed=2, jitter=0.1, noise=0.3, permutations=99)
    assert trial == noise_trial(8, seed=2, jitter=0.1, noise=0.3, permutations=99)
    assert [r["adjusted_p"] for r in trial["results"].values()] == holm(
        [r["p_value"] for r in trial["results"].values()]
    )


def test_binary_lookup_equals_exact_permutation_including_extreme_counts():
    for a in (0, 1, 16, 31, 32):
        for b in (0, 1, 32, 63, 64):
            assert _P[a, b] == permutation_p(a, b, 32, 64)


def test_v2_alarm_replaces_with_latest_window_and_counts_batches(monkeypatch):
    import servegauge.validation.detection_study as module

    observed = []

    class Evidence:
        def __init__(self, **kwargs):
            self.index = 0
            self.log_evidence = 0

        def update(self, p):
            observed.append(p)
            self.index += 1
            return self.index == 2

        def reset(self):
            self.log_evidence = 0

    monkeypatch.setattr(module, "RestartingEvidence", Evidence)
    cfg = {**settings()["changepoint"], "baseline_size": 4, "binary_batch": 2}
    data = {
        "baseline": np.array([-1, 1, -1, 1]),
        "values": np.zeros(3),
        "binary_baseline": np.ones(4, dtype=int),
        "batches": np.zeros((3, 2), dtype=int),
    }
    defaults = {"D-A": {"threshold": 100}, "D-B": {"threshold": 100}, "D-C": {"delta": 1e-5}}
    result = variant_events(data, cfg, defaults)
    assert result["D-B"] == [1]
    assert observed == [permutation_p(0, 4, 2, 4)] * 2 + [1.0]


def test_delays_include_misses_and_ignore_prechange_alarm_as_detection():
    cfg = settings()["changepoint"]
    conditional, restricted = delay_values(
        [{"events": {"D-B": [2, 44]}}, {"events": {"D-B": [2]}}], "D-B", cfg
    )
    assert conditional == [1]
    assert restricted == [1, 20]
    assert configuration()["noise"]["fault_noise"] == 0.3
