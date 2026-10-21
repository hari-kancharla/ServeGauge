"""Inactive transport-channel contract and numerical diagnostics, without alarms."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, model_validator

from servegauge.core.models import ValueModel
from servegauge.suites.config import suite_config

BatchSize = Annotated[int, Field(strict=True, ge=1, le=100000)]


class TransportPolicy(ValueModel):
    version: Literal["monitor-transport-v1"]
    epoch_days: Literal[30]
    max_streams: int = Field(strict=True, ge=1, le=100)
    max_updates: int = Field(strict=True, ge=1, le=120)
    family_alpha: float = Field(strict=True, gt=0, le=0.05)
    epoch_false_event_budget: float = Field(strict=True, gt=0, le=0.5)
    batch_sizes: dict[Literal["core", "full"], BatchSize]

    @property
    def delta(self) -> float:
        # Both expected-event and simultaneous interval budgets must hold. Two
        # fixed epochs can intersect any rolling 30-day interval.
        return min(self.epoch_false_event_budget, self.family_alpha / 2) / (
            self.max_streams * self.max_updates**2
        )

    @model_validator(mode="after")
    def finite_budget(self) -> TransportPolicy:
        if set(self.batch_sizes) != {"core", "full"}:
            raise ValueError("canonical_transport_levels_required")
        if not 0 < self.delta < 1 or not math.isfinite(
            4 * self.max_updates * max(self.batch_sizes.values()) / self.delta
        ):
            raise ValueError("finite_transport_error_allocation_required")
        return self


def load_transport_policy(
    path: Path = Path("configs/monitor_transport_v1.yaml"),
) -> TransportPolicy:
    try:
        policy = TransportPolicy.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        if policy.batch_sizes != {
            level: suite_config()["protocol"][level]["invariant_trials"]
            for level in ("core", "full")
        }:
            raise ValueError("canonical_transport_trial_counts_required")
        return policy
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError):
        raise ValueError("invalid_or_unavailable_transport_policy") from None


def transport_epoch(now: datetime, policy: TransportPolicy) -> tuple[datetime, datetime]:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("aware_transport_epoch_time_required")
    origin = datetime(1970, 1, 1, tzinfo=UTC)
    duration = timedelta(days=policy.epoch_days)
    try:
        start = origin + ((now.astimezone(UTC) - origin) // duration) * duration
        return start, start + duration
    except OverflowError:
        raise ValueError("unsupported_transport_epoch_range") from None


@dataclass(frozen=True)
class CutEvidence:
    left_batches: int
    left_n: int
    left_successes: int
    right_n: int
    right_successes: int
    difference: float
    bound: float
    ci_low: float
    ci_high: float
    reference_tier: Literal["MOCK"] = field(default="MOCK", init=False)
    qualified: Literal[False] = field(default=False, init=False)


def first_cut(batches: Sequence[Sequence[int]], policy: TransportPolicy) -> CutEvidence | None:
    """Locate the frozen batch ADWIN's first candidate cut; never admit an alarm."""
    if len(batches) > policy.max_updates:
        raise ValueError("transport_epoch_update_budget_exceeded")
    if any(
        not row or any(type(bit) is not int or bit not in {0, 1} for bit in row) for row in batches
    ):
        raise ValueError("complete_binary_transport_batches_required")
    if any(len(row) > max(policy.batch_sizes.values()) for row in batches):
        raise ValueError("transport_batch_size_budget_exceeded")
    total_n = sum(len(row) for row in batches)
    total_successes = sum(sum(row) for row in batches)
    left_n = left_successes = 0
    for index, row in enumerate(batches[:-1]):
        left_n += len(row)
        left_successes += sum(row)
        right_n = total_n - left_n
        right_successes = total_successes - left_successes
        difference = right_successes / right_n - left_successes / left_n
        bound = math.sqrt(0.5 * (1 / left_n + 1 / right_n) * math.log(4 * total_n / policy.delta))
        if abs(difference) > bound:
            return CutEvidence(
                left_batches=index + 1,
                left_n=left_n,
                left_successes=left_successes,
                right_n=right_n,
                right_successes=right_successes,
                difference=difference,
                bound=bound,
                ci_low=max(-1.0, difference - bound),
                ci_high=min(1.0, difference + bound),
            )
    return None
