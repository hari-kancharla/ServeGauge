"""Read immutable event/notice journals into tables without granting new qualification."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.ids import safe_filename
from servegauge.core.models import Exchange
from servegauge.observatory.history import _job_plan
from servegauge.observatory.rolling import first_count_cut, load_rolling_policy
from servegauge.observatory.schedule import ObservatoryConfig
from servegauge.storage.monitor import MonitorJournal
from servegauge.storage.notices import guard_path, notice_snapshots
from servegauge.storage.raw_store import RawStore
from servegauge.storage.transport_events import TransportEventJournal
from servegauge.suites.registry import prepare_item
from servegauge.suites.s1_protocol import protocol_items


def _verify_event(
    raw: Path,
    event: dict[str, Any],
    snapshot: dict[str, Any],
    groups: dict[tuple[str, str], list[Exchange]],
) -> None:
    policy = load_rolling_policy()
    if (
        event.get("version") != "transport-operating-event-v1"
        or event.get("qualified") is not True
        or event.get("reference_tier") != "MOCK"
        or event.get("quality_verdict") != "INCONCLUSIVE"
        or event.get("interval_scope") != "fixed_declared_cadence_family"
        or event.get("policy_hash") != content_hash(policy.model_dump(mode="json"))
        or event.get("event_id")
        != "transport-event-"
        + content_hash(
            [
                "transport-operating-event-v1",
                event["observatory_id"],
                event["stream_id"],
                event["ending_window_id"],
                event["policy_hash"],
            ]
        )
    ):
        raise ValueError("operating_recorded_event_schema_invalid")
    contrast = event["contrast"]
    windows = contrast["input_window_ids"]
    if not windows or len(windows) != len(set(windows)) or windows[-1] != event["ending_window_id"]:
        raise ValueError("operating_recorded_event_windows_invalid")
    counts = []
    size = None
    for window_id in windows:
        journal = MonitorJournal(raw, window_id)
        spec, states = journal.spec(), journal.events()
        expected = snapshot["inputs"][window_id]
        if (
            content_hash(spec) != expected["spec_hash"]
            or not expected["complete"]
            or content_hash(states) != expected["state_hash"]
            or [content_hash(e) for e in states] != expected["event_hashes"]
            or not states
            or states[-1].get("type") != "window_completed"
            or spec["observatory_id"] != event["observatory_id"]
            or spec["binding"]["source_hash"] != event["producer_sources"][window_id]
        ):
            raise ValueError("operating_recorded_event_inputs_missing_or_changed")
        config = ObservatoryConfig.model_validate(spec["config"])
        matches = []
        for job in spec["jobs"]:
            plan = _job_plan(job, config, window_id)
            for endpoint in plan.endpoints:
                stream = "transport-" + content_hash(
                    [event["observatory_id"], job["model_id"], endpoint.id, spec["cadence"]]
                )
                if stream == event["stream_id"]:
                    matches.append((job, plan, endpoint))
        if len(matches) != 1:
            raise ValueError("operating_recorded_event_stream_changed")
        job, plan, endpoint = matches[0]
        if plan.level not in {"core", "full"}:
            raise ValueError("operating_recorded_event_size_invalid")
        n = policy.batch_sizes[plan.level]
        size = n if size is None else size
        if size != n:
            raise ValueError("operating_recorded_event_size_changed")
        canonical = [
            i for i in protocol_items(plan.level, plan.seed) if i.expected.get("check_id") == "P01"
        ]
        selected = groups.get((job["run_id"], endpoint.id), [])
        subset = {r.item_id: r for r in selected}
        if (
            len(selected) != n
            or len(subset) != n
            or len(canonical) != n
            or len({r.exchange_id for r in selected}) != n
        ):
            raise ValueError("operating_recorded_event_raw_units_missing_or_repeated")
        successes = 0
        for item in canonical:
            record = subset[item.item_id]
            if (
                record.request != prepare_item(item, endpoint).body
                or record.suite_id != "S1"
                or record.request_url != endpoint.base_url.rstrip("/") + "/chat/completions"
                or record.sample_idx != 0
                or not record.attempts
                or record.attempts[0].n != 1
                or record.attempts[0].status not in {200, 503}
            ):
                raise ValueError("operating_recorded_event_first_attempt_changed")
            successes += record.attempts[0].status == 200
        counts.append(successes)
    assert size is not None
    cut = first_count_cut(counts, size, policy)
    if cut is None or any(
        contrast.get(k) != v
        for k, v in asdict(cut).items()
        if k not in {"left_batches", "reference_tier", "qualified"}
    ):
        raise ValueError("operating_recorded_event_contrast_changed")


def event_inputs(raw: Path, event: dict[str, Any]) -> list[tuple[str, str]]:
    result = []
    for identifier in event["contrast"]["input_window_ids"]:
        spec = MonitorJournal(raw, identifier).spec()
        found = [
            (job["run_id"], endpoint["id"])
            for job in spec["jobs"]
            for endpoint in job["plan"]["endpoints"]
            if "transport-"
            + content_hash(
                [event["observatory_id"], job["model_id"], endpoint["id"], spec["cadence"]]
            )
            == event["stream_id"]
        ]
        if len(found) != 1:
            raise ValueError("operating_recorded_event_stream_changed")
        result.extend(found)
    return result


def observatory_rows(
    raw: Path, records: Sequence[Exchange] | None = None, *, selected_runs: set[str] | None = None
) -> dict[str, list[dict[str, str]]]:
    guard_path(raw)
    groups: dict[tuple[str, str], list[Exchange]] = defaultdict(list)
    for record in records if records is not None else RawStore(raw).exchanges():
        if record.item_id.startswith("P01-"):
            groups[(record.run_id, record.endpoint_id)].append(record)
    changes: list[dict[str, str]] = []
    parent = raw / "transport-events"
    guard_path(parent)
    for directory in sorted(parent.iterdir()) if parent.exists() else []:
        guard_path(directory)
        if not directory.is_dir():
            raise ValueError("operating_foreign_namespace_entry")
        journal = TransportEventJournal(raw, "unbound")
        journal.directory = directory
        spec = journal.spec()
        identifier = spec.get("observatory_id")
        if (
            spec.get("version") != "transport-event-journal-v1"
            or not isinstance(identifier, str)
            or directory.name != safe_filename(identifier)
        ):
            raise ValueError("operating_journal_identity_changed")
        journal.identifier = identifier
        snapshots = journal.events()
        selected_snapshots = snapshots if selected_runs is None else []
        for event in journal.accepted_events():
            if selected_runs is not None and not {r for r, _ in event_inputs(raw, event)}.issubset(
                selected_runs
            ):
                continue
            origin = next(
                r
                for r in snapshots
                if any(e["event_id"] == event["event_id"] for e in r["new_events"])
            )
            if selected_runs is not None and origin not in selected_snapshots:
                selected_snapshots.append(origin)
            _verify_event(raw, event, origin, groups)
            changes.append(
                {"record_id": event["event_id"], "record_json": canonical_json(event).decode()}
            )
        for snapshot in selected_snapshots:
            if snapshot.get("version") != "transport-event-record-v1":
                raise ValueError("operating_recorded_snapshot_schema_invalid")
            for window_id, expected in snapshot["inputs"].items():
                window = MonitorJournal(raw, window_id)
                spec, states = window.spec(), window.events()
                prefix = states[: len(expected["event_hashes"])]
                if (
                    content_hash(spec) != expected["spec_hash"]
                    or content_hash(prefix) != expected["state_hash"]
                    or [content_hash(e) for e in prefix] != expected["event_hashes"]
                    or (expected["complete"] and prefix != states)
                ):
                    raise ValueError("operating_recorded_history_missing_or_changed")
                if expected["complete"]:
                    for job in states[-1]["result"]["jobs"]:
                        seal = raw / safe_filename(job["run_id"]) / "analysis.json"
                        guard_path(seal)
                        if hashlib.sha256(seal.read_bytes()).hexdigest() != job["analysis_hash"]:
                            raise ValueError("operating_recorded_raw_seal_missing_or_changed")
    identifiers = [r["record_id"] for r in changes]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("operating_duplicate_event_across_journals")
    notices = []
    for snapshot in notice_snapshots(raw):
        spec = snapshot["spec"]
        value: dict[str, Any] = {
            "notice_id": spec["notice_id"],
            "subject": spec["subject"],
            "prepared_at": spec["prepared_at"],
            "proof_bundle_sha256": spec["bundle_sha256"],
            "assertions": snapshot["events"],
            "delivery_verified": False,
            "state": snapshot["events"][-1]["type"] if snapshot["events"] else "prepared",
        }
        notices.append(
            {"record_id": spec["notice_id"], "record_json": canonical_json(value).decode()}
        )
    return {
        "change_events": sorted(changes, key=lambda r: r["record_id"]),
        "notifications": sorted(notices, key=lambda r: r["record_id"]),
    }
