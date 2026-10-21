"""Read-only qualification inputs from guarded, reconstructed captures and sessions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.ids import safe_filename
from servegauge.core.models import ModelProfile, ReferencePack
from servegauge.refs.owner import (
    CapturedReference,
    Qualification,
    SelfhostOrigin,
    origin_review,
    qualify_capture,
)
from servegauge.refs.pack import ReferenceRun
from servegauge.runner.ledger import Ledger
from servegauge.selfhost.runtime import validate_runtime
from servegauge.storage.analysis import export_outcomes, reconstruct
from servegauge.storage.reference_export import _guard, _guard_capture
from servegauge.storage.selfhost import SessionJournal, _clean


def read_review(root: Path, path: Path) -> dict[str, Any]:
    path = path if path.is_absolute() else root / path
    _guard(root)
    _guard(path)
    if (
        not path.absolute().is_relative_to(root.absolute() / "configs")
        or any(p.casefold().startswith(".env") for p in path.parts)
        or path.suffix != ".json"
        or not path.is_file()
    ):
        raise ValueError("reference_review_owned_json_configuration_required")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("reference_review_object_required")
    _clean(value)
    return value


def _session(root: Path, origin: SelfhostOrigin) -> dict[str, Any]:
    journal = SessionJournal(root, origin.session_id)
    status = journal.status()
    plan = status["plan"]
    history = [row["value"] for row in journal._records()]
    running_rows = [
        row
        for row in history
        if row["data"]["state"] == "running"
        and row["data"]["facts"].get("runtime_receipt_sha256") == origin.runtime_sha256
    ]
    running = [row["data"]["facts"] for row in running_rows]
    if (
        not running
        or any(e != running[0] for e in running)
        or "reservation_id" not in status["facts"]
    ):
        raise ValueError("reference_observed_running_session_required")
    path = (
        root
        / "data/raw/selfhost-runtime"
        / journal.directory.name
        / (origin.runtime_sha256 + ".json")
    )
    _guard(path)
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != origin.runtime_sha256:
        raise ValueError("reference_runtime_receipt_bytes_changed")
    runtime = validate_runtime(origin.session_id, plan, json.loads(blob))
    if canonical_json(runtime) + b"\n" != blob:
        raise ValueError("reference_canonical_runtime_receipt_required")
    _clean(runtime)
    ledger = Ledger(root)
    _guard(ledger.path)
    _guard(ledger.summary)
    ledger.refresh()
    events = [json.loads(line) for line in ledger.path.read_bytes().splitlines()]
    rid = status["facts"]["reservation_id"]
    rows = [e for e in events if e.get("type") == "reserve" and e.get("reservation_id") == rid]
    dispatches = [
        e for e in events if e.get("type") == "dispatch" and e.get("reservation_id") == rid
    ]
    if len(rows) != 1 or len(dispatches) != 1:
        raise ValueError("reference_original_session_reservation_and_dispatch_required")
    return {
        "session_id": origin.session_id,
        "plan": plan,
        "runtime": runtime,
        "running": running[0],
        "running_at": running_rows[0]["created_at"],
        "terminated_at": next(
            (row["created_at"] for row in history if row["data"]["state"] == "terminated"), None
        ),
        "reservation_id": rid,
        "reservation": rows[0] | {"dispatched": True},
    }


def captured_reference(
    root: Path, run: str, profile: ModelProfile, request: Qualification
) -> CapturedReference:
    _guard_capture(root / "data/raw", run)
    # RawStore scans recursively. Guard every declared supporting namespace before
    # that scan, and require its actual two-level endpoint/file layout. Reject a
    # protected name or symlink before statting or descending into it.
    descriptor = root / "data/raw" / safe_filename(run) / "analysis.json"
    sealed = json.loads(descriptor.read_bytes())["value"]
    supporting = set(sealed["run_ids"]) | {run}
    capability = sealed["metadata"].get("capability_run_id")
    if capability is not None:
        supporting.add(capability)
    for identifier in supporting:
        namespace = root / "data/raw" / safe_filename(identifier)
        for entry in namespace.iterdir():
            _guard(entry)
            if entry.is_dir():
                for child in entry.iterdir():
                    _guard(child)
                    if not child.is_file():
                        raise ValueError("reference_flat_endpoint_raw_layout_required")
    analysis = reconstruct(root / "data/raw", run)
    if content_hash(analysis.metadata["profile"]) != request.profile_hash or analysis.metadata[
        "profile"
    ] != profile.model_dump(mode="json"):
        raise ValueError("reference_original_captured_profile_required")
    source = ReferenceRun.model_validate(export_outcomes(analysis))
    if source.run_id != run:
        raise ValueError("reference_original_capture_identity_changed")
    session = _session(root, request.origin) if isinstance(request.origin, SelfhostOrigin) else None
    return CapturedReference(source, analysis.records, session)


def validate_pack_origin(root: Path, pack: ReferencePack) -> None:
    """Publication rechecks local owner-origin proof independently of rights/notices."""
    saved = pack.manifest["owner_qualification"]
    request = Qualification.model_validate(saved["request"])
    if pack.tier != request.tier or not saved["captures"]:
        raise ValueError("reference_saved_owner_origin_changed")
    runs = {row["source_run"] for row in saved["captures"]}
    expected_runs = {
        pack.manifest["source_run"],
        *[row["run_id"] for row in pack.fingerprints.get("reference_sources", {}).values()],
    }
    if (
        runs != expected_runs
        or len(runs) != len(saved["captures"])
        or pack.manifest.get("origin_review") != origin_review(request, saved["captures"])
    ):
        raise ValueError("reference_saved_origin_inventory_changed")
    for expected in saved["captures"]:
        run = expected["source_run"]
        _guard_capture(root / "data/raw", run)
        descriptor = root / "data/raw" / safe_filename(run) / "analysis.json"
        profile = ModelProfile.model_validate(
            json.loads(descriptor.read_bytes())["value"]["metadata"]["profile"]
        )
        capture = captured_reference(root, run, profile, request)
        if (
            qualify_capture(
                capture.source, profile, request, capture.records, session=capture.session
            )
            != expected
        ):
            raise ValueError("reference_saved_capture_origin_changed")
