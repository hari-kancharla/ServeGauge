"""Explicitly calibrated detectors. No observatory operating thresholds are selected here."""

from __future__ import annotations

import math

import numpy as np

from servegauge.stats.common import probability


class CUSUM:
    def __init__(
        self, *, baseline_mean: float, baseline_sd: float, slack: float, threshold: float
    ) -> None:
        if (
            not all(math.isfinite(x) for x in (baseline_mean, baseline_sd, slack, threshold))
            or baseline_sd <= 0
            or min(slack, threshold) < 0
            or threshold == 0
        ):
            raise ValueError("finite_baseline_positive_sd_and_explicit_calibration_required")
        self.mean, self.sd, self.slack, self.threshold = (
            baseline_mean,
            baseline_sd,
            slack,
            threshold,
        )
        self.positive = 0.0
        self.negative = 0.0

    def update(self, value: float) -> bool:
        if not math.isfinite(value):
            raise ValueError("finite_observation_required")
        standardized = (value - self.mean) / self.sd
        self.positive = max(0, self.positive + standardized - self.slack)
        self.negative = max(0, self.negative - standardized - self.slack)
        return max(self.positive, self.negative) > self.threshold

    def reset(self) -> None:
        self.positive = self.negative = 0


class EValueAccumulator:
    requires_dependence_calibration = True

    def __init__(self, *, kappa: float, log_threshold: float) -> None:
        probability(kappa)
        if not math.isfinite(log_threshold) or log_threshold <= 0:
            raise ValueError("explicit_positive_log_threshold_required")
        self.kappa, self.log_threshold, self.log_evidence = kappa, log_threshold, 0.0

    def calibrated_log_e(self, p_value: float) -> float:
        probability(p_value, closed=True)
        p = max(p_value, float(np.finfo(np.float64).tiny))
        return math.log(self.kappa) + (self.kappa - 1) * math.log(p)

    def update(self, p_value: float) -> bool:
        self.log_evidence += self.calibrated_log_e(p_value)
        return self.log_evidence > self.log_threshold

    def reset(self) -> None:
        self.log_evidence = 0


class ADWIN:
    """Original full-cut Hoeffding ADWIN; per-update union bound delta/n.

    This deliberately uses the original rigorous bound rather than the faster bucketed
    approximation. Observatory-wide repeated-testing behavior is calibrated in M12.
    """

    def __init__(self, *, delta: float) -> None:
        probability(delta)
        self.delta = delta
        self.window: list[int] = []

    def update(self, value: int) -> bool:
        if value not in {0, 1}:
            raise ValueError("binary_observations_required")
        self.window.append(value)
        changed = False
        while len(self.window) >= 2:
            data = np.asarray(self.window, dtype=np.float64)
            n = len(data)
            cuts = np.arange(1, n)
            prefix = np.cumsum(data)
            differences = np.abs(prefix[:-1] / cuts - (prefix[-1] - prefix[:-1]) / (n - cuts))
            bound = np.sqrt(0.5 * (1 / cuts + 1 / (n - cuts)) * math.log(4 * n / self.delta))
            violations = np.flatnonzero(differences > bound)
            if len(violations) == 0:
                break
            self.window = self.window[int(cuts[int(violations[0])]) :]
            changed = True
        return changed
