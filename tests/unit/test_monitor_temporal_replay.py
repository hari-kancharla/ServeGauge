"""Candidate rolling/reset mechanics; no accepted operating alarm is emitted."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from servegauge.core.hashing import content_hash
from servegauge.observatory.rolling import load_rolling_policy
from servegauge.observatory.schedule import cadence_window
from servegauge.observatory.temporal import TemporalFrame, replay_candidates
from servegauge.observatory.transport import TransportBatch

SOURCE = "a" * 64


def frame(index, successes=400, *, stream="stream-one", start=None, cadence="daily", n=400):
    moment = start or datetime(2026, 10, 20, tzinfo=UTC) + timedelta(days=index)
    window = cadence_window(cadence, moment)
    batch = TransportBatch(
        ready=True,
        reason="candidate_only",
        run_id=f"run-{stream}-{index}",
        endpoint_id="mock:one",
        observed_units=n,
        expected_units=n,
        values=(1,) * successes + (0,) * (n - successes),
        role_seed=10000 + index,
        request_design_hash=content_hash(["canonical", n]),
        capture_hash=content_hash([stream, index, n, successes]),
        reference_tier="MOCK",
    )
    return TemporalFrame(
        stream_id=stream,
        window_id=f"window-{index}",
        cadence=cadence,
        start=window.start,
        end=window.end,
        producer_source_hash=SOURCE,
        batch=batch,
    )


def replay(frames, policy=None):
    return replay_candidates(frames, policy or load_rolling_policy(), expected_source_hash=SOURCE)


def test_unchanged_batches_are_observations_not_quality_passes():
    rows = replay([frame(0), frame(1), frame(2)])
    assert [r.status for r in rows] == ["BASELINE", "NO_CANDIDATE", "NO_CANDIDATE"]
    assert all(not r.qualified and r.verdict == "INCONCLUSIVE" for r in rows)
    assert not any(r.event for r in rows)


def test_change_resets_to_latest_complete_window_and_keeps_direction():
    rows = replay([frame(0), frame(1), frame(2, 0), frame(3, 0), frame(4)])
    down = rows[2].event
    assert down and down.difference < 0 and down.ci_high < 0
    assert down.baseline_after_window_id == "window-2"
    assert rows[3].status == "NO_CANDIDATE"
    up = rows[4].event
    assert up and up.difference > 0 and up.ci_low > 0
    assert "window-0" not in up.input_window_ids and "window-1" not in up.input_window_ids
    assert all(not r.qualified for r in rows)


def test_exact_duplicates_are_idempotent_and_conflicts_refuse_before_replay():
    frames = [frame(0), frame(1, 0)]
    assert replay([*frames, frames[0], frames[1]]) == replay(frames)
    with pytest.raises(ValueError, match="conflicting"):
        replay([frame(0), replace(frame(0), batch=frame(0, 0).batch)])


@pytest.mark.parametrize("case", ["missing", "reused_seed", "changed_design", "batch_size_changed"])
def test_incomplete_or_dependent_rows_are_barriers_and_next_valid_row_restarts(case):
    frames = [frame(0), frame(1), frame(2, 0), frame(3, 0)]
    if case == "missing":
        frames[1] = replace(frames[1], batch=replace(frames[1].batch, ready=False, values=()))
    elif case == "reused_seed":
        frames[1] = replace(
            frames[1], batch=replace(frames[1].batch, role_seed=frames[0].batch.role_seed)
        )
    elif case == "changed_design":
        frames[1] = replace(frames[1], batch=replace(frames[1].batch, request_design_hash="b" * 64))
    else:
        frames[1] = replace(
            frames[1],
            batch=replace(
                frames[1].batch, observed_units=800, expected_units=800, values=(1,) * 800
            ),
        )
    rows = replay(frames)
    assert rows[1].status == ("INCONCLUSIVE" if case in {"missing", "reused_seed"} else "BASELINE")
    assert rows[2].status == "BASELINE" and rows[3].status == "NO_CANDIDATE"
    assert not any(r.event for r in rows)


def test_missing_calendar_window_cannot_be_skipped_into_an_alarm():
    rows = replay([frame(0), frame(2, 0), frame(3, 0)])
    assert rows[1].status == "BASELINE" and rows[1].reason == "missing_calendar_window"
    assert rows[2].status == "NO_CANDIDATE" and not any(r.event for r in rows)


def test_continuous_monthly_history_detects_preserved_slow_change_counterexample():
    frames = []
    for i in range(36):
        p = 0.95 if i < 12 else 0.95 - 0.2 * min((i - 11) / 12, 1)
        frames.append(
            frame(
                i,
                round(800 * p),
                start=datetime(2026 + i // 12, i % 12 + 1, 1, tzinfo=UTC),
                cadence="monthly",
                n=800,
            )
        )
    rows = replay(frames)
    events = [r.event for r in rows if r.event]
    assert events and events[0].difference < 0 and events[0].ci_high < 0
    assert len(events[0].input_window_ids) > 2
    assert all(not r.qualified for r in rows)


def test_history_cap_retains_only_recent_derived_batches_without_epoch_reset():
    policy = load_rolling_policy().model_copy(update={"max_history_batches": 4})
    rows = replay([frame(i, 400 if i < 5 else 0) for i in range(6)], policy)
    assert rows[-1].event.input_window_ids == ("window-2", "window-3", "window-4", "window-5")


def test_rolling_boundary_cannot_reset_update_budget():
    policy = load_rolling_policy().model_copy(update={"max_updates": 1})
    start = datetime(2026, 1, 30, tzinfo=UTC)
    with pytest.raises(ValueError, match="update_budget"):
        replay([frame(0, start=start), frame(1, start=start + timedelta(days=1))], policy)
    rows = replay([frame(0, start=start), frame(30, start=start + timedelta(days=30))], policy)
    assert [r.update_index for r in rows] == [1, 1]
    assert rows[-1].status == "BASELINE"


def test_measurement_switches_consume_separate_rolling_stream_budgets():
    policy = load_rolling_policy().model_copy(update={"max_streams": 1})
    first = frame(0)
    for second in [
        replace(frame(1), producer_source_hash="b" * 64),
        replace(frame(1), batch=replace(frame(1).batch, request_design_hash="b" * 64)),
        frame(1, n=800),
    ]:
        with pytest.raises(ValueError, match="stream_budget"):
            replay([first, second], policy)


def test_stream_and_update_budgets_refuse_without_partial_candidates():
    p = load_rolling_policy().model_copy(update={"max_streams": 1})
    with pytest.raises(ValueError, match="stream_budget"):
        replay([frame(0, stream="one"), frame(0, stream="two")], p)
    p = load_rolling_policy().model_copy(update={"max_updates": 1})
    with pytest.raises(ValueError, match="update_budget"):
        replay([frame(0), frame(1)], p)


def test_aware_canonical_windows_required_and_source_versions_not_pooled():
    with pytest.raises(ValueError, match="aware"):
        replay([replace(frame(0), start=frame(0).start.replace(tzinfo=None))])
    old = replace(frame(0), producer_source_hash="b" * 64)
    rows = replay([old, frame(1, 0)])
    assert rows[0].status == "INCONCLUSIVE" and rows[1].status == "BASELINE"
    assert not any(r.event for r in rows)


def test_distinct_window_ids_at_same_time_and_changed_cadence_refuse():
    with pytest.raises(ValueError, match="colliding"):
        replay([frame(0), replace(frame(0), window_id="other")])
    with pytest.raises(ValueError, match="identity_changed"):
        replay([frame(0), frame(1, cadence="weekly")])


def test_malformed_ready_batch_cannot_be_an_alarm():
    for batch in [
        replace(frame(0).batch, values=(True,) * 400),
        replace(frame(0).batch, observed_units=399),
        replace(frame(0).batch, capture_hash=None),
        replace(frame(0).batch, request_design_hash="fake"),
        replace(frame(0).batch, role_seed=True),
    ]:
        rows = replay([replace(frame(0), batch=batch), frame(1, 0)])
        assert rows[0].status == "INCONCLUSIVE" and rows[1].status == "BASELINE"
        assert not any(r.event for r in rows)


def test_same_capture_cannot_supply_a_second_independent_window():
    a = frame(0)
    b = replace(frame(1, 0), batch=replace(frame(1, 0).batch, capture_hash=a.batch.capture_hash))
    rows = replay([a, b, frame(2, 0)])
    assert rows[1].status == "INCONCLUSIVE" and rows[1].reason == "reused_transport_capture"
    assert rows[2].status == "BASELINE"


def test_observation_range_and_event_identity_bind_exact_inputs_and_policy():
    a, b = frame(0), frame(1, 0)
    event = replay([a, b])[1].event
    assert event.observation_start == a.start.isoformat()
    assert event.observation_end == b.end.isoformat()
    other = replay([a, replace(b, batch=replace(b.batch, capture_hash="e" * 64))])[1].event
    assert event.event_id != other.event_id
    other = replay([a, b], load_rolling_policy().model_copy(update={"max_streams": 1}))[1].event
    assert event.event_id != other.event_id


def test_reviewed_measurement_identity_can_span_actual_producer_versions_without_hiding_them():
    generator = "f" * 64
    first = replace(frame(0), measurement_source_hash=generator, producer_runtime_hash="d" * 64)
    second = replace(
        frame(1, 0),
        producer_source_hash="b" * 64,
        measurement_source_hash=generator,
        producer_runtime_hash="d" * 64,
    )
    rows = replay_candidates([first, second], load_rolling_policy(), expected_source_hash=generator)
    assert rows[1].event and rows[1].event.ci_high < 0
    assert not rows[1].qualified and rows[1].verdict == "INCONCLUSIVE"
    altered = replace(second, producer_source_hash="c" * 64)
    changed = replay_candidates(
        [first, altered], load_rolling_policy(), expected_source_hash=generator
    )
    assert changed[1].event.input_hashes != rows[1].event.input_hashes


def test_unknown_or_malformed_actual_producer_cannot_borrow_a_measurement_identity():
    generator = "f" * 64
    for source in [None, "unknown"]:
        row = replace(frame(0), producer_source_hash=source, measurement_source_hash=generator)
        steps = replay_candidates([row], load_rolling_policy(), expected_source_hash=generator)
        assert steps[0].status == "INCONCLUSIVE"


def test_missing_known_size_consumes_an_update_without_manufacturing_a_second_variant():
    policy = load_rolling_policy().model_copy(update={"max_streams": 1})
    rows = [
        frame(0),
        replace(frame(1), batch=replace(frame(1).batch, ready=False, values=(), observed_units=0)),
        frame(2),
    ]
    result = replay(rows, policy)
    assert [r.status for r in result] == ["BASELINE", "INCONCLUSIVE", "BASELINE"]
    assert [r.update_index for r in result] == [1, 2, 3]


def test_declared_random_seed_from_an_unavailable_window_cannot_be_reused():
    first = replace(
        frame(0), batch=replace(frame(0).batch, ready=False, values=(), observed_units=0)
    )
    second = replace(frame(1), batch=replace(frame(1).batch, role_seed=first.batch.role_seed))
    rows = replay([first, second])
    assert rows[1].status == "INCONCLUSIVE" and rows[1].reason == "reused_transport_role_seed"
