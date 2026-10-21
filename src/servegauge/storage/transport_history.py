"""Append-only candidate projections, never operating alarm records."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.ids import safe_filename
from servegauge.core.redaction import Redactor
from servegauge.storage.monitor import MonitorJournal


class TransportHistoryJournal(MonitorJournal):
    def __init__(self, raw: Path, identifier: str) -> None:
        super().__init__(raw, identifier)
        self.identifier = identifier
        self.directory = raw / "transport-history" / safe_filename(identifier)

    def projections(self) -> list[dict[str, Any]]:
        return self.events()

    def record(self, value: dict[str, Any]) -> bool:
        # The storage boundary accepts a JSON value. Tuple/list normalization is
        # distinct from credential redaction, and must precede equality checks.
        value = json.loads(canonical_json(value))
        if Redactor.from_environment().redact(value) != value:
            raise ValueError("history_projection_contains_private_credentials")
        if (
            value.get("version") != "transport-history-v1"
            or value.get("observatory_id") != self.identifier
            or value.get("qualified") is not False
            or value.get("verdict") != "INCONCLUSIVE"
            or not isinstance(value.get("inputs"), dict)
            or not isinstance(value.get("steps"), list)
            or any(
                not isinstance(s, dict)
                or s.get("qualified") is not False
                or s.get("verdict") != "INCONCLUSIVE"
                or (
                    s.get("event") is not None
                    and (
                        not isinstance(s["event"], dict) or s["event"].get("qualified") is not False
                    )
                )
                for s in value["steps"]
            )
            or value.get("projection_id")
            != "transport-history-"
            + content_hash({k: v for k, v in value.items() if k != "projection_id"})
        ):
            raise ValueError("invalid_or_operating_history_projection")
        for item in value["inputs"].values():
            if (
                not isinstance(item, dict)
                or set(item) != {"spec_hash", "state_hash", "event_hashes", "complete"}
                or type(item["complete"]) is not bool
                or not isinstance(item["event_hashes"], list)
                or any(
                    not isinstance(d, str)
                    or len(d) != 64
                    or any(c not in "0123456789abcdef" for c in d)
                    for d in [item["spec_hash"], item["state_hash"], *item["event_hashes"]]
                )
            ):
                raise ValueError("invalid_history_input_binding")
        self.create({"version": "transport-history-journal-v1", "observatory_id": self.identifier})
        with FileLock(str(self.directory / ".record.lock"), timeout=0):
            records = self._records()
            if records:
                last = records[-1]["value"]["data"]
                for name, old in last["inputs"].items():
                    new = value["inputs"].get(name)
                    if (
                        new is None
                        or new["spec_hash"] != old["spec_hash"]
                        or new["event_hashes"][: len(old["event_hashes"])] != old["event_hashes"]
                        or (
                            new["event_hashes"] == old["event_hashes"]
                            and new["state_hash"] != old["state_hash"]
                        )
                        or (old["complete"] and new != old)
                    ):
                        raise ValueError("history_evidence_missing_or_changed")
                if last == value:
                    return False
            sequence = len(records) + 1
            # The record is committed under the same lock as rollback/dedup checks.
            self._commit(
                self.directory / f"event-{sequence:08}.json",
                {
                    "version": "monitor-event-v1",
                    "sequence": sequence,
                    "spec_hash": self._read(self.directory / "spec.json")["hash"],
                    "previous_hash": records[-1]["hash"] if records else None,
                    "created_at": datetime.now(UTC).isoformat(),
                    "data": value,
                },
            )
            return True
