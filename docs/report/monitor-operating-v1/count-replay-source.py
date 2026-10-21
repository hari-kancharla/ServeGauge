"""Count-based offline replay for the declared transport model; no admission."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt

from servegauge.observatory.bounds import cut_limit, effect_interval, upper_cut_bound
from servegauge.observatory.rolling import RollingTransportPolicy


@dataclass(frozen=True)
class CountCut:
    start: int
    end: int
    left_batches: int
    left_n: int
    right_n: int
    left_successes: int
    right_successes: int
    difference: float
    bound: float
    ci_low: float
    ci_high: float


@lru_cache(maxsize=16)
def _tables(size: int, policy_json: str) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    policy = RollingTransportPolicy.model_validate_json(policy_json)
    horizon = policy.max_history_batches
    limits = np.zeros((horizon + 1, horizon), dtype=np.int64)
    bounds = np.zeros((horizon + 1, horizon), dtype=np.float64)
    for length in range(2, horizon + 1):
        delta = policy.delta_for(length)
        for cut in range(1, length):
            left, right = cut * size, (length - cut) * size
            limits[length, cut - 1] = cut_limit(left, right, delta)
            bounds[length, cut - 1] = upper_cut_bound(left, right, delta)
    limits.setflags(write=False)
    bounds.setflags(write=False)
    return limits, bounds


def replay_counts(
    counts: npt.NDArray[np.int64],
    available: npt.NDArray[np.bool_],
    size: int,
    policy: RollingTransportPolicy,
) -> list[list[CountCut]]:
    """Replay complete fixed-size potential batches, with explicit availability barriers.

    Rows are independent constructed streams. This kernel does not admit source,
    runtime, cadence or family scope; the public controller supplies those guards.
    Truth, fault labels, shift times and expected probabilities cannot enter it.
    """
    if (
        type(size) is not int
        or size not in policy.batch_sizes.values()
        or counts.ndim != 2
        or min(counts.shape) < 1
        or not np.issubdtype(counts.dtype, np.integer)
        or available.shape != counts.shape
        or available.dtype != np.bool_
        or np.any(counts < 0)
        or np.any(counts > size)
    ):
        raise ValueError("complete_bounded_transport_count_matrix_required")
    limits, bounds = _tables(size, policy.model_dump_json())
    rows, updates = counts.shape
    prefix = np.pad(np.cumsum(counts, axis=1, dtype=np.int64), ((0, 0), (1, 0)))
    starts = np.zeros(rows, dtype=np.int64)
    events: list[list[CountCut]] = [[] for _ in range(rows)]
    for end in range(updates):
        starts = np.maximum(starts, end - policy.max_history_batches + 1)
        starts[~available[:, end]] = end + 1
        lengths = end - starts + 1
        for raw_length in np.unique(lengths):
            length = int(raw_length)
            if length < 2:
                continue
            ids = np.flatnonzero(lengths == length)
            begin = starts[ids]
            cuts = np.arange(1, length)
            lefts = prefix[ids[:, None], begin[:, None] + cuts] - prefix[ids, begin, None]
            totals = prefix[ids, end + 1] - prefix[ids, begin]
            rights = totals[:, None] - lefts
            crosses = np.abs(rights * (cuts * size) - lefts * ((length - cuts) * size))
            violations = crosses > limits[length, : length - 1]
            for selected in np.flatnonzero(np.any(violations, axis=1)):
                row = int(ids[selected])
                cut = int(np.argmax(violations[selected])) + 1
                left, right = cut * size, (length - cut) * size
                left_successes = int(lefts[selected, cut - 1])
                right_successes = int(rights[selected, cut - 1])
                bound = float(bounds[length, cut - 1])
                difference, low, high = effect_interval(
                    left_successes, left, right_successes, right, bound
                )
                events[row].append(
                    CountCut(
                        start=int(begin[selected]),
                        end=end,
                        left_batches=cut,
                        left_n=left,
                        right_n=right,
                        left_successes=left_successes,
                        right_successes=right_successes,
                        difference=difference,
                        bound=bound,
                        ci_low=low,
                        ci_high=high,
                    )
                )
                starts[row] = end
    return events
