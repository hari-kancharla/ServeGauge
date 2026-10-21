"""Operating events require a bound admission and preserve immutable input history."""

from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta

import pytest
from test_monitor_admission_loader import qualification

from servegauge.core.hashing import content_hash
from servegauge.observatory.admission import load_admission
from servegauge.observatory.evaluation import OperatingEvaluation, evaluate_frames, evaluate_history
from servegauge.observatory.rolling import load_rolling_policy
from servegauge.observatory.temporal import TemporalFrame
from servegauge.observatory.transport import TransportBatch
from servegauge.storage.transport_events import TransportEventJournal


def setup_admission(root):
    producer, _ = qualification(root)
    admission = load_admission(root).admission
    assert admission
    return admission, producer


def frames(admission):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    result = []
    for i, successes in enumerate([400, 400, 0]):
        batch = TransportBatch(
            True,
            "candidate",
            f"run-{i}",
            "mock:one",
            400,
            400,
            tuple([1] * successes + [0] * (400 - successes)),
            i,
            content_hash("design"),
            content_hash(["capture", i]),
            reference_tier="MOCK",
        )
        result.append(
            TemporalFrame(
                "stream-one",
                f"window-{i}",
                "daily",
                start + timedelta(days=i),
                start + timedelta(days=i + 1),
                admission.current_source_hash,
                batch,
                admission.measurement_source_hash,
                admission.runtime_hash,
            )
        )
    return result


def inputs(rows):
    return {
        f.window_id: {
            "spec_hash": content_hash(f.window_id),
            "state_hash": content_hash([f.window_id]),
            "event_hashes": [content_hash(f.window_id)],
            "complete": True,
        }
        for f in rows
    }


def evaluation(admission, rows):
    return evaluate_frames("local", rows, inputs(rows), {}, admission, load_rolling_policy())


def test_accepted_event_keeps_counts_interval_actual_producers_and_endpoint_inconclusive(tmp_path):
    admission, producer = setup_admission(tmp_path)
    rows = frames(admission)
    rows[0] = replace(rows[0], producer_source_hash="b" * 64)
    result = evaluation(admission, rows)
    assert isinstance(result, OperatingEvaluation)
    value = result.value()
    assert len(value["events"]) == 1
    event = value["events"][0]
    assert event["qualified"] is True and event["reference_tier"] == "MOCK"
    assert event["quality_verdict"] == "INCONCLUSIVE"
    assert event["contrast"]["left_n"] == 400 and event["contrast"]["right_n"] == 800
    assert event["contrast"]["ci_high"] < 0
    assert event["producer_sources"]["window-0"] == "b" * 64
    assert event["qualification"]["sha256"] == admission.receipt_hash
    assert event["interval_scope"] == "fixed_declared_cadence_family"
    assert all(s["qualified"] is False for s in value["steps"])
    assert producer["source_hash"] == admission.current_source_hash


@pytest.mark.parametrize("case", ["actual_source", "runtime", "measurement", "missing", "seed"])
def test_unsupported_frame_breaks_history_and_cannot_create_event(tmp_path, case):
    admission, _ = setup_admission(tmp_path)
    rows = frames(admission)
    if case == "actual_source":
        rows[-1] = replace(rows[-1], producer_source_hash="c" * 64)
    elif case == "runtime":
        rows[-1] = replace(rows[-1], producer_runtime_hash="c" * 64)
    elif case == "measurement":
        rows[-1] = replace(rows[-1], measurement_source_hash=None)
    elif case == "missing":
        rows[-1] = replace(rows[-1], batch=replace(rows[-1].batch, ready=False, values=()))
    else:
        rows[-1] = replace(rows[-1], batch=replace(rows[-1].batch, role_seed=0))
    value = evaluation(admission, rows).value()
    assert not value["events"] and value["steps"][-1]["status"] == "INCONCLUSIVE"


def test_journal_idempotence_receipt_refresh_and_input_conflict_are_atomic(tmp_path):
    admission, _ = setup_admission(tmp_path)
    rows = frames(admission)
    first = evaluation(admission, rows)
    journal = TransportEventJournal(tmp_path / "data/raw", "local")
    assert journal.record(first) and not journal.record(first)
    assert len(journal.accepted_events()) == 1 and len(journal.events()) == 1
    refresh_root = tmp_path / "refresh"
    refresh_root.mkdir()
    qualification(refresh_root, alter=lambda r: r.update({"review_note": "additional review"}))
    other = load_admission(refresh_root).admission
    assert other is not None and other.receipt_hash != admission.receipt_hash
    refreshed = evaluation(other, rows)
    assert journal.record(refreshed) and len(journal.accepted_events()) == 1
    assert journal.accepted_events()[0]["qualification"]["sha256"] == admission.receipt_hash
    assert not journal.record(refreshed)
    before = [asdict(row) for row in rows]
    with pytest.raises(ValueError, match="history_evidence"):
        journal.record(evaluation(other, rows[1:]))
    assert len(journal.events()) == 2 and [asdict(row) for row in rows] == before
    with pytest.raises((ValueError, TypeError)):
        journal.record(first.value())


def test_absent_qualification_evaluation_is_read_only_and_never_dispatches(tmp_path):
    result = evaluate_history(tmp_path, "empty", proof_root=tmp_path)
    assert result["status"] == "unqualified" and result["events"] == []
    assert not list(tmp_path.iterdir())


def test_changed_current_source_cannot_borrow_loaded_capability(tmp_path, monkeypatch):
    admission, _ = setup_admission(tmp_path)
    monkeypatch.setattr("servegauge.observatory.evaluation.implementation_hash", lambda: "e" * 64)
    with pytest.raises(ValueError, match="consumer_source"):
        evaluation(admission, frames(admission))


def test_missing_input_or_duplicate_frame_conflict_cannot_be_committed(tmp_path):
    admission, _ = setup_admission(tmp_path)
    rows = frames(admission)
    with pytest.raises(ValueError, match="operating_input"):
        evaluate_frames("local", rows, {}, {}, admission, load_rolling_policy())
    with pytest.raises(ValueError, match="conflicting_temporal"):
        evaluation(admission, [*rows, replace(rows[-1], producer_source_hash="b" * 64)])
    assert not (tmp_path / "data").exists()


def test_evaluate_cli_without_proof_or_read_only_never_creates_evidence(tmp_path):
    from typer.testing import CliRunner

    from servegauge.cli import app

    response = CliRunner().invoke(
        app,
        [
            "monitor",
            "evaluate",
            "--observatory-id",
            "none",
            "--root",
            str(tmp_path),
            "--proof-root",
            str(tmp_path),
            "--read-only",
        ],
    )
    assert response.exit_code == 0 and '"unqualified"' in response.stdout
    assert not list(tmp_path.iterdir())


def test_backfill_cannot_create_a_fresh_historical_decision_budget(tmp_path):
    admission, _ = setup_admission(tmp_path)
    rows = frames(admission)
    journal = TransportEventJournal(tmp_path / "data/raw", "local")
    journal.record(evaluation(admission, rows[:2]))
    prior = replace(
        rows[0],
        window_id="earlier",
        start=rows[0].start - timedelta(days=1),
        end=rows[0].end - timedelta(days=1),
        batch=replace(rows[0].batch, role_seed=99, capture_hash=content_hash("earlier")),
    )
    with pytest.raises(ValueError, match="backfill"):
        journal.record(evaluation(admission, [prior, *rows[:2]]))
    assert len(journal.events()) == 1


def test_private_value_refuses_before_journal_creation(tmp_path, monkeypatch):
    admission, _ = setup_admission(tmp_path)
    monkeypatch.setenv("SERVEGAUGE_PRIVATE_TEST_SECRET", "CANARY_SECRET_EVENT_123456789")
    result = evaluate_frames(
        "local",
        frames(admission),
        inputs(frames(admission)),
        {"private": "CANARY_SECRET_EVENT_123456789"},
        admission,
        load_rolling_policy(),
    )
    with pytest.raises(ValueError, match="private_credentials"):
        TransportEventJournal(tmp_path / "data/raw", "local").record(result)
    assert not (tmp_path / "data").exists()


def test_tick_evaluation_also_runs_for_a_completed_repeat(tmp_path, monkeypatch):
    import asyncio
    import json
    from types import SimpleNamespace

    from typer.testing import CliRunner

    from servegauge.cli import app
    from servegauge.validation.async_offline import SocketlessLoop

    def socketless_run(coroutine):
        with asyncio.Runner(loop_factory=SocketlessLoop) as runner:
            return runner.run(coroutine)

    monkeypatch.setattr("servegauge.cli.asyncio.run", socketless_run)
    calls = []

    async def completed(*args, **kwargs):
        return {"status": "already_completed", "measurement_exit_code": 0}

    def evaluated(root, identifier, **kwargs):
        calls.append((root, identifier, kwargs))
        return {"status": "unchanged", "events": []}

    configuration_reads = []

    def configuration(_):
        configuration_reads.append(True)
        return SimpleNamespace(id="local" if len(configuration_reads) == 1 else "changed")

    monkeypatch.setattr("servegauge.observatory.schedule.load_observatory", configuration)
    monkeypatch.setattr("servegauge.observatory.tick.tick", completed)
    monkeypatch.setattr("servegauge.observatory.evaluation.evaluate_history", evaluated)
    result = CliRunner().invoke(
        app,
        [
            "monitor",
            "tick",
            "--cadence",
            "daily",
            "--config",
            "unused.yaml",
            "--evaluate",
            "--root",
            str(tmp_path),
        ],
    )
    assert (
        result.exit_code == 0 and json.loads(result.stdout)["monitoring"]["status"] == "unchanged"
    )
    assert len(calls) == 1 and calls[0][1] == "local"


def test_generic_append_cannot_bypass_qualified_event_storage(tmp_path):
    admission, _ = setup_admission(tmp_path)
    journal = TransportEventJournal(tmp_path / "data/raw", "local")
    journal.record(evaluation(admission, frames(admission)))
    before = journal.events()
    with pytest.raises(ValueError, match="typed_evaluation"):
        journal.append({"version": "unverified", "new_events": []})
    assert journal.events() == before and len(journal.accepted_events()) == 1
