"""Fast study counts must reproduce the complete public raw-bit controller."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from servegauge.core.hashing import content_hash
from servegauge.observatory.rolling import load_rolling_policy
from servegauge.observatory.temporal import TemporalFrame, replay_candidates
from servegauge.observatory.transport import TransportBatch
from servegauge.validation.transport_operating import replay_counts


def test_known_cuts_and_missing_window_barriers():
    policy = load_rolling_policy()
    counts = np.array([[400, 0, 0, 400], [400, 0, 0, 400]], dtype=np.int64)
    available = np.array([[True, True, True, True], [True, False, True, True]])
    events = replay_counts(counts, available, 400, policy)
    # The first eligible boundary on [0, 0, 400] compares 0 against mean 200/400.
    assert [(e.end, e.difference) for e in events[0]] == [(1, -1), (3, 0.5)]
    assert [(e.end, e.difference) for e in events[1]] == [(3, 1)]
    altered = counts.copy()
    altered[1, 1] = 127
    assert replay_counts(altered, available, 400, policy) == events


def test_disjoint_held_out_paths_agree_with_public_controller_and_version_bridge():
    rng = np.random.default_rng(91943)
    policy = load_rolling_policy()
    for size in [400, 800]:
        means = np.r_[np.full(10, 0.95), np.linspace(0.95, 0.35, 25)]
        counts = rng.binomial(size, means, size=(6, len(means)))
        available = rng.random(counts.shape) > 0.08
        expected = replay_counts(counts, available, size, policy)
        for trial in range(6):
            frames = []
            for day, count in enumerate(counts[trial]):
                start = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=day)
                batch = TransportBatch(
                    ready=bool(available[trial, day]),
                    reason="constructed",
                    run_id=f"r-{trial}-{day}",
                    endpoint_id="mock:study",
                    observed_units=size if available[trial, day] else 0,
                    expected_units=size,
                    values=(1,) * int(count) + (0,) * (size - int(count))
                    if available[trial, day]
                    else (),
                    role_seed=trial * 1000 + day,
                    request_design_hash=content_hash(["fixed", size]),
                    capture_hash=content_hash([trial, day, int(count)]),
                    reference_tier="MOCK",
                )
                frames.append(
                    TemporalFrame(
                        "study",
                        f"w-{day}",
                        "daily",
                        start,
                        start + timedelta(days=1),
                        "a" * 64 if day < 18 else "b" * 64,
                        batch,
                        measurement_source_hash="f" * 64,
                        producer_runtime_hash="d" * 64,
                    )
                )
            rows = replay_candidates(frames, policy, expected_source_hash="f" * 64)
            actual = [(i, r.event) for i, r in enumerate(rows) if r.event]
            assert len(actual) == len(expected[trial])
            for (day, event), want in zip(actual, expected[trial], strict=True):
                assert day == want.end
                assert event.input_window_ids == tuple(
                    f"w-{j}" for j in range(want.start, want.end + 1)
                )
                assert (
                    event.left_n,
                    event.right_n,
                    event.left_successes,
                    event.right_successes,
                    event.difference,
                    event.ci_low,
                    event.ci_high,
                ) == (
                    want.left_n,
                    want.right_n,
                    want.left_successes,
                    want.right_successes,
                    want.difference,
                    want.ci_low,
                    want.ci_high,
                )


@pytest.mark.parametrize("case", ["size", "shape", "dtype", "outside", "available"])
def test_invalid_or_noncanonical_count_inputs_refuse(case):
    counts = np.array([[400, 0]], dtype=np.int64)
    available = np.array([[True, True]])
    size = 400
    if case == "size":
        size = 399
    elif case == "shape":
        counts = counts[0]
    elif case == "dtype":
        counts = counts.astype(float)
    elif case == "outside":
        counts[0, 0] = 401
    else:
        available = np.array([[1, 1]])
    with pytest.raises(ValueError):
        replay_counts(counts, available, size, load_rolling_policy())
