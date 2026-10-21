import numpy as np
import pytest

from servegauge.mock.faults import FaultConfig
from servegauge.stats.tests import two_sample_permutation
from servegauge.validation.changepoint_sim import (
    calibrated_threshold,
    evaluate_trajectory,
    permutation_p,
    trajectory,
)
from servegauge.validation.common import settings
from servegauge.validation.power_study import observations, plot_power, power_tests


def test_binary_permutation_probability_matches_exhaustive_label_assignments():
    scalar = two_sample_permutation(
        [0, 0],
        [1, 1],
        lambda a, b: float(np.mean(a) - np.mean(b)),
        seed=1,
        resamples=100,
        alternative="less",
    )
    assert permutation_p(0, 2, 2, 2) == pytest.approx(scalar.p_value)


def test_threshold_uses_calibration_budget_and_refuses_insufficient_monte_carlo():
    cfg = settings()["changepoint"]
    result = calibrated_threshold(np.linspace(0, 1, 5000).tolist(), cfg)
    assert result["calibration_tail"]["ci_high"] <= result["per_stream_method_probability_budget"]
    assert result["threshold"] > 0.99
    with pytest.raises(ValueError, match="more_null_calibration"):
        calibrated_threshold([1, 2, 3], cfg)


def test_shift_simulation_keeps_noise_and_baseline_estimation_explicit():
    cfg = settings()["changepoint"]
    null = trajectory(19, "null", cfg)
    abrupt = trajectory(19, "abrupt", cfg)
    start = cfg["shift_day"] * cfg["observations_per_day"]
    assert np.array_equal(null["baseline"], abrupt["baseline"])
    assert np.allclose(abrupt["values"][:start], null["values"][:start])
    assert np.allclose(abrupt["values"][start:] - null["values"][start:], cfg["standardized_shift"])
    assert not np.array_equal(null["baseline"], trajectory(20, "null", cfg)["baseline"])
    with pytest.raises(ValueError):
        trajectory(1, "unknown", cfg)


def test_evidence_alarm_replaces_baseline_with_latest_observed_window(monkeypatch):
    import servegauge.validation.changepoint_sim as module

    calls = []

    class Evidence:
        def __init__(self, **kwargs):
            self.index = 0

        def update(self, p):
            self.index += 1
            return self.index == 2

        def reset(self):
            pass

    monkeypatch.setattr(module, "EValueAccumulator", Evidence)

    def observed(current, baseline, n_current, n_baseline):
        calls.append(baseline)
        return 1.0

    monkeypatch.setattr(module, "permutation_p", observed)
    cfg = {**settings()["changepoint"], "baseline_size": 4, "binary_batch": 2}
    data = {
        "baseline": np.array([-1, 1, -1, 1]),
        "values": np.zeros(3),
        "binary_baseline": np.ones(4, dtype=int),
        "batches": np.zeros((3, 2), dtype=int),
    }
    defaults = {"D-A": {"threshold": 100}, "D-B": {"threshold": 100}, "D-C": {"delta": 0.001}}
    found = evaluate_trajectory(data, cfg, defaults)
    assert found["D-B"] == [1] and calls == [4, 4, 0]


def test_power_uses_original_faulted_scripted_outputs_and_all_four_methods():
    assert observations(5, seed=11, faults=FaultConfig()) == observations(
        5, seed=11, faults=FaultConfig()
    )
    result = power_tests(4, seed=11, faults=FaultConfig(route_to_other_model=1), permutations=32)
    assert set(result) == {"FP1", "FP2", "FP3", "FP4-statistic"}
    assert all(0 <= r["adjusted_p"] <= 1 for r in result.values())


def test_scientific_plots_are_standalone_and_use_only_owner_attribution(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "cache"))
    rows = [
        {
            "scenario": s,
            "method": m,
            "queries": q,
            "scenario_usd": q / 1000000,
            "power": {"estimate": 0.5, "ci_low": 0.2, "ci_high": 0.8},
        }
        for s in ("F-SWAP", "F-MIX", "F-QSIM", "clean")
        for m in ("FP1", "FP2", "FP3", "FP4-statistic")
        for q in (40, 100)
    ]
    plot_power({"configuration": {"replicates": 60}, "rows": rows}, tmp_path)
    svg = (tmp_path / "power_queries.svg").read_text(encoding="utf-8")
    assert "Hari Krishna Kancharla" in svg
    assert (tmp_path / "power_usd.png").stat().st_size > 1000
