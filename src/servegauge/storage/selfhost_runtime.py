"""Immutable private runtime observations, separate from session transition records."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from servegauge.core.hashing import canonical_json
from servegauge.selfhost.runtime import validate_runtime
from servegauge.storage.notices import guard_path
from servegauge.storage.selfhost import SessionJournal, _clean


def save_runtime(root: Path, identifier: str, plan: dict[str, Any], value: dict[str, Any]) -> str:
    journal = SessionJournal(root, identifier)
    if journal.status()["plan"] != plan:
        raise ValueError("selfhost_runtime_reviewed_plan_changed")
    value = validate_runtime(identifier, plan, value)
    _clean(value)
    blob = canonical_json(value) + b"\n"
    digest = hashlib.sha256(blob).hexdigest()
    directory = root / "data/raw/selfhost-runtime" / journal.directory.name
    guard_path(directory)
    path = directory / (digest + ".json")
    guard_path(path)
    if path.exists():
        if not path.is_file() or path.read_bytes() != blob:
            raise ValueError("selfhost_runtime_immutable_observation_changed")
        return digest
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    return digest
