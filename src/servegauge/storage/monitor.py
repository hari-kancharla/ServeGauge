"""Immutable monitor specifications and atomically committed, hash-chained records."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.ids import safe_filename
from servegauge.core.redaction import Redactor


class MonitorJournal:
    def __init__(self, raw: Path, identifier: str) -> None:
        self.raw = raw
        self.directory = raw / "observatory" / safe_filename(identifier)

    @classmethod
    def from_directory(cls, raw: Path, directory: Path) -> MonitorJournal:
        """Open an already encoded path without applying safe_filename twice."""
        if directory.parent != raw / "observatory":
            raise ValueError("monitor_journal_directory_outside_namespace")
        journal = cls(raw, "unbound")
        journal.directory = directory
        spec = journal.spec()
        identifier = spec.get("window_id")
        if not isinstance(identifier, str) or directory.name != safe_filename(identifier):
            raise ValueError("monitor_journal_directory_identity_mismatch")
        return journal

    def _guard(self) -> None:
        if any(
            path.is_symlink()
            for path in [
                self.raw,
                self.directory.parent,
                self.directory,
                self.directory / ".record.lock",
            ]
        ):
            raise ValueError("monitor_journal_symlink")

    def _read(self, path: Path) -> dict[str, Any]:
        self._guard()
        if path.is_symlink() or not path.is_file():
            raise ValueError("monitor_journal_record_missing_or_symlinked")
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("value"), dict)
                or row.get("hash") != content_hash(row["value"])
            ):
                raise ValueError("monitor_journal_hash_mismatch")
            return dict(row)
        except (KeyError, TypeError, ValueError):
            raise ValueError("monitor_journal_invalid_record") from None

    def _commit(self, path: Path, value: dict[str, Any]) -> None:
        payload = canonical_json({"value": value, "hash": content_hash(value)}) + b"\n"
        if path.exists() or path.is_symlink():
            raise ValueError("monitor_journal_immutable_record_conflict")
        temporary = self.directory / (".pending-" + uuid.uuid4().hex)
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        if os.name == "posix":
            descriptor = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def create(self, specification: dict[str, Any]) -> None:
        self._guard()
        self.directory.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.directory / ".record.lock"), timeout=0):
            path = self.directory / "spec.json"
            clean = dict(Redactor.from_environment().redact(specification))
            if path.exists() or path.is_symlink():
                if self._read(path)["value"] != clean:
                    raise ValueError("monitor_journal_specification_conflict")
                return
            if any(self.directory.glob("event-*.json")):
                raise ValueError("monitor_journal_specification_missing")
            if any(
                path.name != ".record.lock" and not path.name.startswith(".pending-")
                for path in self.directory.iterdir()
            ):
                raise ValueError("monitor_journal_foreign_window_directory")
            self._commit(path, clean)

    def spec(self) -> dict[str, Any]:
        return dict(self._read(self.directory / "spec.json")["value"])

    def _records(self) -> list[dict[str, Any]]:
        spec_hash = self._read(self.directory / "spec.json")["hash"]
        result: list[dict[str, Any]] = []
        for path in sorted(self.directory.iterdir()):
            if path.name in {"spec.json", ".record.lock"} or path.name.startswith(".pending-"):
                continue
            if not re.fullmatch(r"event-\d{8}\.json", path.name):
                raise ValueError("monitor_journal_unexpected_record")
            row = self._read(path)
            value = row["value"]
            expected_sequence = len(result) + 1
            previous = result[-1]["hash"] if result else None
            if (
                value.get("version") != "monitor-event-v1"
                or value.get("sequence") != expected_sequence
                or path.name != f"event-{expected_sequence:08}.json"
                or value.get("spec_hash") != spec_hash
                or value.get("previous_hash") != previous
                or not isinstance(value.get("data"), dict)
            ):
                raise ValueError("monitor_journal_chain_mismatch")
            result.append(row)
        return result

    def events(self) -> list[dict[str, Any]]:
        return [dict(row["value"]["data"]) for row in self._records()]

    def append(self, data: dict[str, Any]) -> None:
        self._guard()
        with FileLock(str(self.directory / ".record.lock"), timeout=0):
            previous = self._records()
            sequence = len(previous) + 1
            value = {
                "version": "monitor-event-v1",
                "sequence": sequence,
                "spec_hash": self._read(self.directory / "spec.json")["hash"],
                "previous_hash": previous[-1]["hash"] if previous else None,
                "created_at": datetime.now(UTC).isoformat(),
                "data": Redactor.from_environment().redact(data),
            }
            self._commit(self.directory / f"event-{sequence:08}.json", value)
