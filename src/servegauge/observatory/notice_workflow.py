"""Bind local notice preparation/publication policy to actual sealed failure evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from servegauge.core.models import Diagnosis
from servegauge.observatory.notices import (
    FailureSubject,
    failure_subject,
    notice_state,
    validate_bundle_reference,
)
from servegauge.storage.analysis import reconstruct
from servegauge.storage.notices import NoticeJournal, guard_path, notice_snapshots


def bundle_file(root: Path, relative: str) -> Path:
    guard_path(root)
    validate_bundle_reference(relative)
    parts = PurePosixPath(relative)
    if (
        parts.is_absolute()
        or ".." in parts.parts
        or not parts.parts
        or parts.parts[0] != "bundles"
        or parts.name != "bundle.json"
    ):
        raise ValueError("notice_bundle_reference_refused")
    path = root.joinpath(*parts.parts)
    guard_path(path)
    if not path.is_file():
        raise ValueError("notice_bundle_missing")
    return path


def _bound_failure(
    root: Path,
    run_id: str,
    endpoint_id: str,
    check_id: str,
    bundle_reference: str,
) -> tuple[FailureSubject, str]:
    path = bundle_file(root, bundle_reference)
    blob = path.read_bytes()
    data = json.loads(blob)
    if (
        not isinstance(data, dict)
        or data.get("version") != "bundle-v2"
        or data.get("redact_content") is not False
    ):
        raise ValueError("notice_original_sanitized_bundle_required")
    analysis = reconstruct(root / "data/raw", run_id)
    subject = failure_subject(analysis, endpoint_id, check_id)
    check = next(
        c for c in analysis.checks if c.check_id == check_id and c.endpoint_id == endpoint_id
    )
    diagnosis = Diagnosis.model_validate(data["diagnosis"])
    if (
        diagnosis.run_id != run_id
        or diagnosis.endpoint_id != endpoint_id
        or check_id not in diagnosis.check_ids
        or not diagnosis.evidence
        or not set(diagnosis.evidence).issubset(
            {r.exchange_id for r in analysis.records if r.endpoint_id == endpoint_id}
        )
        or check.model_dump(mode="json") not in data["expected_vs_observed"]
        or data["configuration"] != analysis.plan.model_dump(mode="json")
        or data["manifest"] != analysis.manifest
    ):
        raise ValueError("notice_bundle_does_not_bind_current_failure")
    return subject, hashlib.sha256(blob).hexdigest()


def prepare_notice(
    root: Path,
    run_id: str,
    endpoint_id: str,
    check_id: str,
    bundle_reference: str,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    subject, digest = _bound_failure(root, run_id, endpoint_id, check_id, bundle_reference)
    journal = NoticeJournal.prepare(
        root, subject, bundle_reference, digest, now=now or datetime.now(UTC)
    )
    return {
        "notice_id": journal.identifier,
        "state": "prepared",
        "subject_id": subject.subject_id,
        "delivery_verified": False,
        "network_writes": 0,
    }


def publication_status(
    root: Path, subject: FailureSubject, *, now: datetime | None = None
) -> dict[str, Any]:
    """An owner assertion is local evidence, never independent delivery verification."""
    if subject.synthetic_mock:
        return {
            "eligible": True,
            "basis": "scripted_mock",
            "owner_asserted": False,
            "delivery_verified": False,
        }
    snapshots = notice_snapshots(root / "data/raw")
    valid = []
    for snapshot in snapshots:
        spec = snapshot["spec"]
        if spec["subject"] != subject.model_dump(mode="json"):
            continue
        try:
            bound, digest = _bound_failure(
                root,
                subject.run_id,
                subject.endpoint_id,
                subject.check_id,
                spec["bundle_reference"],
            )
        except (OSError, ValueError, KeyError, TypeError, StopIteration):
            continue
        if bound == subject and digest == spec["bundle_sha256"]:
            valid.append(snapshot)
    state = notice_state(subject, valid, now=now or datetime.now(UTC))
    return {**state, "delivery_verified": False}
