"""Local immutable notice assertions; no transport, email or support integration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock

from servegauge.core.hashing import content_hash
from servegauge.core.ids import safe_filename
from servegauge.core.redaction import Redactor
from servegauge.observatory.notices import (
    FailureSubject,
    NoticeAssertion,
    aware,
    channel_origin,
    validate_bundle_reference,
    validate_history,
)
from servegauge.storage.monitor import MonitorJournal


def guard_path(path: Path) -> None:
    if any(p.lower() == "starter files" for p in path.parts):
        raise ValueError("notice_path_refused")
    if any(p.is_symlink() for p in [*reversed(path.absolute().parents), path]):
        raise ValueError("notice_path_symlink_refused")


class NoticeJournal(MonitorJournal):
    def __init__(self, raw: Path, identifier: str) -> None:
        guard_path(raw)
        super().__init__(raw, identifier)
        self.identifier = identifier
        self.directory = raw / "notices" / safe_filename(identifier)

    def _guard(self) -> None:
        guard_path(self.raw)
        super()._guard()

    @classmethod
    def from_directory(cls, raw: Path, directory: Path) -> NoticeJournal:
        guard_path(directory)
        if directory.parent != raw / "notices":
            raise ValueError("notice_directory_outside_namespace")
        journal = cls(raw, "unbound")
        journal.directory = directory
        spec = journal.spec()
        identifier = spec["notice_id"]
        if directory.name != safe_filename(identifier):
            raise ValueError("notice_directory_identity_changed")
        journal.identifier = identifier
        return journal

    @classmethod
    def prepare(
        cls,
        root: Path,
        subject: FailureSubject,
        bundle_reference: str,
        bundle_sha256: str,
        *,
        now: datetime,
    ) -> NoticeJournal:
        aware(now)
        validate_bundle_reference(bundle_reference)
        identifier = "notice-" + content_hash([subject.subject_id, bundle_sha256])
        journal = cls(root / "data/raw", identifier)
        proposed = {
            "version": "notice-spec-v1",
            "notice_id": identifier,
            "subject_id": subject.subject_id,
            "subject": subject.model_dump(mode="json"),
            "bundle_reference": bundle_reference,
            "bundle_sha256": bundle_sha256,
            "prepared_at": aware(now).isoformat(),
        }
        validate_history(proposed, [])
        if Redactor.from_environment().redact(proposed) != proposed:
            raise ValueError("notice_private_credentials_refused")
        # Preparation and its immutable initial timestamp share one process lock.
        journal._guard()
        journal.directory.mkdir(parents=True, exist_ok=True)
        with FileLock(str(journal.directory / ".record.lock"), timeout=0):
            spec_file = journal.directory / "spec.json"
            if spec_file.exists():
                original = journal.spec()
                if {k: v for k, v in original.items() if k != "prepared_at"} != {
                    k: v for k, v in proposed.items() if k != "prepared_at"
                }:
                    raise ValueError("notice_preparation_conflict")
                validate_history(original, journal.events())
            else:
                if any(
                    p.name != ".record.lock" and not p.name.startswith(".pending-")
                    for p in journal.directory.iterdir()
                ):
                    raise ValueError("notice_foreign_directory")
                journal._commit(spec_file, proposed)
        return journal

    def append(self, data: dict[str, Any]) -> None:
        raise ValueError("notice_typed_assertion_required")

    def snapshot(self) -> dict[str, Any]:
        spec, events = self.spec(), self.events()
        validate_history(spec, events)
        return {"spec": spec, "events": events}

    def record(
        self,
        kind: str,
        *,
        at: datetime,
        reference_sha256: str,
        owner_assertion: bool,
        channel: str | None = None,
        now: datetime | None = None,
    ) -> bool:
        now = aware(now or datetime.now(UTC))
        at = aware(at)
        if owner_assertion is not True or at > now:
            raise ValueError("notice_explicit_owner_assertion_and_past_time_required")
        row = NoticeAssertion.model_validate(
            {
                "type": kind,
                "at": at.isoformat(),
                "reference_sha256": reference_sha256,
                "owner_asserted": True,
                "channel_origin": channel_origin(channel),
            }
        )
        value = row.model_dump(mode="json")
        if Redactor.from_environment().redact(value) != value:
            raise ValueError("notice_private_credentials_refused")
        with FileLock(str(self.directory / ".record.lock"), timeout=0):
            records = self._records()
            events = [r["value"]["data"] for r in records]
            validate_history(self.spec(), events)
            if value in events:
                return False
            validate_history(self.spec(), [*events, value])
            self._commit(
                self.directory / f"event-{len(records) + 1:08}.json",
                {
                    "version": "monitor-event-v1",
                    "sequence": len(records) + 1,
                    "spec_hash": self._read(self.directory / "spec.json")["hash"],
                    "previous_hash": records[-1]["hash"] if records else None,
                    "created_at": now.isoformat(),
                    "data": value,
                },
            )
            return True


def notice_snapshots(raw: Path) -> list[dict[str, Any]]:
    guard_path(raw)
    parent = raw / "notices"
    guard_path(parent)
    result = []
    for path in sorted(parent.iterdir()) if parent.exists() else []:
        guard_path(path)
        if not path.is_dir():
            raise ValueError("notice_foreign_namespace_entry")
        result.append(NoticeJournal.from_directory(raw, path).snapshot())
    return result
