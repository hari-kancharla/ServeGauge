"""Opt-in change detectors; new sampling rules require new operating calibration."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from servegauge.stats.changepoint import EValueAccumulator
from servegauge.stats.common import probability


class RestartingEvidence(EValueAccumulator):
    """Reflected p-to-e score. Not an e-process or an anytime-valid guarantee.

    Discarding negative accumulated evidence changes the repeated-testing rule.
    Thresholds must be recalibrated, including the shared baseline and restarts.
    """

    def update(self, p_value: float) -> bool:
        self.log_evidence = max(0.0, self.log_evidence + self.calibrated_log_e(p_value))
        return self.log_evidence > self.log_threshold


class BatchADWIN:
    """Hoeffding ADWIN with cuts restricted to observed batch boundaries.

    Uses all binary observations; checks once per fingerprint. The original bound
    uses the number of raw bits, which exceeds the number of candidate cuts here.
    This is a conservative boundary-cut variant, not the original every-bit update.
    """

    def __init__(self, *, delta: float) -> None:
        probability(delta)
        self.delta = delta
        self.batches: list[tuple[int, int]] = []

    @property
    def observations(self) -> int:
        return sum(size for _, size in self.batches)

    def update(self, values: Sequence[int]) -> bool:
        if not values or any(value not in {0, 1} for value in values):
            raise ValueError("nonempty_binary_batch_required")
        self.batches.append((sum(values), len(values)))
        changed = False
        while len(self.batches) > 1:
            sums, sizes = np.asarray(self.batches, dtype=np.float64).T
            prefix, cuts = np.cumsum(sums), np.cumsum(sizes)
            total, n = prefix[-1], cuts[-1]
            left, right = cuts[:-1], n - cuts[:-1]
            difference = np.abs(prefix[:-1] / left - (total - prefix[:-1]) / right)
            bound = np.sqrt(0.5 * (1 / left + 1 / right) * math.log(4 * n / self.delta))
            violations = np.flatnonzero(difference > bound)
            if len(violations) == 0:
                break
            self.batches = self.batches[int(violations[0]) + 1 :]
            changed = True
        return changed
