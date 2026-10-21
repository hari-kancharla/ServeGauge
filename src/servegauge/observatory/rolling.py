"""Continuous-history candidate with summable start weights; no alarm admission."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from servegauge.core.models import ValueModel
from servegauge.observatory.bounds import (
    cut_limit,
    effect_interval,
    floor_fraction,
    upper_cut_bound,
)
from servegauge.observatory.changes import BatchSize, CutEvidence
from servegauge.suites.config import suite_config


class RollingTransportPolicy(ValueModel):
    version: Literal["monitor-transport-v2"]
    horizon_days: Literal[30]
    max_streams: int = Field(strict=True, ge=1, le=100)
    max_updates: int = Field(strict=True, ge=1, le=120)
    max_history_batches: int = Field(strict=True, ge=2, le=121)
    family_alpha: float = Field(strict=True, gt=0, le=0.05)
    max_false_events_30_days: float = Field(strict=True, gt=0, le=1)
    batch_sizes: dict[Literal["core", "full"], BatchSize]

    def _budget(self) -> Fraction:
        return min(
            Fraction(str(self.family_alpha)), Fraction(str(self.max_false_events_30_days))
        ) / (self.max_streams * self.max_updates)

    @property
    def update_budget(self) -> float:
        return floor_fraction(self._budget())

    def delta_for(self, batches: int) -> float:
        if type(batches) is not int or not 2 <= batches <= self.max_history_batches:
            raise ValueError("bounded_integer_history_length_required")
        return floor_fraction(self._budget() * Fraction(2, batches * (batches + 1)))

    @model_validator(mode="after")
    def finite_budget(self) -> RollingTransportPolicy:
        if set(self.batch_sizes) != {"core", "full"}:
            raise ValueError("canonical_transport_levels_required")
        if not 0 < self.delta_for(self.max_history_batches) < 1:
            raise ValueError("finite_rolling_error_allocation_required")
        return self


def load_rolling_policy(
    path: Path = Path("configs/monitor_transport_v2.yaml"),
) -> RollingTransportPolicy:
    try:
        policy = RollingTransportPolicy.model_validate(yaml.safe_load(path.read_text("utf-8")))
        if policy.batch_sizes != {
            level: suite_config()["protocol"][level]["invariant_trials"]
            for level in ("core", "full")
        }:
            raise ValueError("canonical_transport_trial_counts_required")
        return policy
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError):
        raise ValueError("invalid_or_unavailable_rolling_policy") from None


def first_count_cut(
    successes: Sequence[int], batch_size: int, policy: RollingTransportPolicy
) -> CutEvidence | None:
    """Counts are sufficient only for these complete, uniform independent batches."""
    if type(batch_size) is not int or batch_size not in policy.batch_sizes.values():
        raise ValueError("canonical_transport_batch_size_required")
    if len(successes) > policy.max_history_batches:
        raise ValueError("transport_history_budget_exceeded")
    if any(type(s) is not int or not 0 <= s <= batch_size for s in successes):
        raise ValueError("complete_integer_transport_counts_required")
    if len(successes) < 2:
        return None
    delta = policy.delta_for(len(successes))
    total_n = len(successes) * batch_size
    total_successes = sum(successes)
    left_successes = 0
    for index, count in enumerate(successes[:-1]):
        left_n = (index + 1) * batch_size
        left_successes += count
        right_n = total_n - left_n
        right_successes = total_successes - left_successes
        if abs(right_successes * left_n - left_successes * right_n) > cut_limit(
            left_n, right_n, delta
        ):
            bound = upper_cut_bound(left_n, right_n, delta)
            difference, low, high = effect_interval(
                left_successes, left_n, right_successes, right_n, bound
            )
            return CutEvidence(
                left_batches=index + 1,
                left_n=left_n,
                left_successes=left_successes,
                right_n=right_n,
                right_successes=right_successes,
                difference=difference,
                bound=bound,
                ci_low=low,
                ci_high=high,
            )
    return None


def first_rolling_cut(
    batches: Sequence[Sequence[int]], policy: RollingTransportPolicy
) -> CutEvidence | None:
    if any(
        not row or any(type(bit) is not int or bit not in {0, 1} for bit in row) for row in batches
    ):
        raise ValueError("complete_binary_transport_batches_required")
    if not batches:
        return None
    size = len(batches[0])
    if any(len(row) != size for row in batches):
        raise ValueError("uniform_transport_batch_sizes_required")
    return first_count_cut([sum(row) for row in batches], size, policy)
