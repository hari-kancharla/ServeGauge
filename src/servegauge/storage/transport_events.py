"""Immutable operating event journal, separate from unqualified candidate storage."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock

from servegauge.core.hashing import content_hash
from servegauge.core.ids import safe_filename
from servegauge.core.redaction import Redactor
from servegauge.observatory.evaluation import OperatingEvaluation
from servegauge.storage.monitor import MonitorJournal


def _stable(event: dict[str, Any]) -> dict[str, Any]:
    # An extended receipt can approve additional versions without rewriting or
    # duplicating an already committed physical decision. Its original proof stays.
    return {k: v for k, v in event.items() if k != "qualification"}


def _extends(old: dict[str, Any], new: dict[str, Any]) -> None:
    for key, before in old.items():
        after = new.get(key)
        if (
            not isinstance(after, dict)
            or after.get("spec_hash") != before.get("spec_hash")
            or after.get("event_hashes", [])[: len(before["event_hashes"])]
            != before["event_hashes"]
            or (
                after.get("event_hashes") == before["event_hashes"]
                and after.get("state_hash") != before["state_hash"]
            )
            or (before["complete"] and after != before)
        ):
            raise ValueError("operating_history_evidence_missing_or_changed")


class TransportEventJournal(MonitorJournal):
    def __init__(self, raw: Path, identifier: str) -> None:
        if any(p.lower() == "starter files" for p in raw.parts) or raw.parent.is_symlink():
            raise ValueError("operating_journal_path_refused")
        super().__init__(raw, identifier)
        self.identifier = identifier
        self.directory = raw / "transport-events" / safe_filename(identifier)

    def append(self, data: dict[str, Any]) -> None:
        raise ValueError("operating_typed_evaluation_required")

    def accepted_events(self) -> list[dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for record in self.events():
            for event in record["new_events"]:
                if event["event_id"] in result:
                    raise ValueError("operating_duplicate_committed_event")
                result[event["event_id"]] = event
        return list(result.values())

    def record(self, evaluation: OperatingEvaluation) -> bool:
        if not isinstance(evaluation, OperatingEvaluation):
            raise TypeError("operating_evaluation_capability_required")
        value = evaluation.value()
        if value.get("observatory_id") != self.identifier:
            raise ValueError("operating_evaluation_namespace_mismatch")
        if Redactor.from_environment().redact(value) != value:
            raise ValueError("operating_evaluation_contains_private_credentials")
        self.create({"version": "transport-event-journal-v1", "observatory_id": self.identifier})
        with FileLock(str(self.directory / ".record.lock"), timeout=0):
            records = self._records()
            previous = [r["value"]["data"] for r in records]
            if previous:
                last = previous[-1]
                _extends(last["inputs"], value["inputs"])
                old_times, new_times = last["window_starts"], value["window_starts"]
                if any(new_times.get(w) != t for w, t in old_times.items()):
                    raise ValueError("operating_history_window_time_changed")
                latest = max(old_times.values()) if old_times else None
                if latest is not None and any(
                    t < latest
                    and (
                        w not in old_times
                        or (not last["inputs"][w]["complete"] and value["inputs"][w]["complete"])
                    )
                    for w, t in new_times.items()
                ):
                    raise ValueError("operating_backfill_or_late_completion_refused")
                if previous[-1]["evaluation_id"] == value["evaluation_id"]:
                    return False
            existing = {e["event_id"]: e for r in previous for e in r["new_events"]}
            current = {e["event_id"]: e for e in value["events"]}
            if len(current) != len(value["events"]):
                raise ValueError("operating_duplicate_evaluated_event")
            for identifier, old in existing.items():
                if identifier not in current or _stable(current[identifier]) != _stable(old):
                    raise ValueError("operating_accepted_event_missing_or_changed")
            data = {
                "version": "transport-event-record-v1",
                "evaluation_id": value["evaluation_id"],
                "inputs": value["inputs"],
                "window_starts": value["window_starts"],
                "source_hash": value["source_hash"],
                "policy_hash": value["policy_hash"],
                "qualification": value["qualification"],
                "new_events": [e for e in value["events"] if e["event_id"] not in existing],
                "evaluation_hash": content_hash(value),
            }
            self._commit(
                self.directory / f"event-{len(records) + 1:08}.json",
                {
                    "version": "monitor-event-v1",
                    "sequence": len(records) + 1,
                    "spec_hash": self._read(self.directory / "spec.json")["hash"],
                    "previous_hash": records[-1]["hash"] if records else None,
                    "created_at": datetime.now(UTC).isoformat(),
                    "data": data,
                },
            )
            return True
