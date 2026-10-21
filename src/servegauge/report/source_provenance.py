"""Safe recorded code provenance; missing attempt facts are never inferred from context."""

from __future__ import annotations

import re
from typing import Any

from servegauge.core.hashing import content_hash
from servegauge.core.models import Exchange


def _digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _source(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not _digest(value.get("source_tree_hash")):
        return None
    digest = value["source_tree_hash"]
    identity = value.get("source_identity")
    if "source_identity" in value:
        if (
            not isinstance(identity, dict)
            or identity.get("version") != "capture-source-v2"
            or identity.get("scope") != "loaded-package-python-v1"
            or identity.get("package") != "servegauge"
            or identity.get("source_hash") != digest
            or identity.get("import_source_hash") != digest
            or identity.get("status") != "consistent_with_import_snapshot"
            or type(identity.get("python_files")) is not int
            or identity["python_files"] < 1
        ):
            return None
        scope, status = "loaded-package-python-v1", "consistent_with_import_snapshot"
    else:
        scope = "legacy-cwd-git-inventory"
        status = (
            "legacy_empty_inventory" if digest == content_hash({}) else "legacy_recorded_inventory"
        )
    git_sha = value.get("git_sha")
    if not isinstance(git_sha, str) or (
        git_sha != "uncommitted" and re.fullmatch(r"[a-f0-9]{7,64}", git_sha) is None
    ):
        git_sha = None
    git_state = value.get("git_state")
    return {
        "source_tree_hash": digest,
        "scope": scope,
        "status": status,
        "git_sha": git_sha,
        "git_dirty": value.get("git_dirty") if type(value.get("git_dirty")) is bool else None,
        "git_state": git_state
        if isinstance(git_state, str)
        and git_state in {"bound_checkout", "not_checkout", "unavailable"}
        else "legacy_or_unknown",
    }


def source_provenance(manifest: dict[str, Any], records: list[Exchange]) -> dict[str, Any]:
    """Describe original context and recorded attempts, retaining unknown legacy data."""
    sources: dict[str, dict[str, Any]] = {}
    total = unknown = loaded = legacy = 0
    for record in records:
        for attempt in record.attempts:
            total += 1
            source = _source(attempt.metadata.get("execution_source"))
            if source is None:
                unknown += 1
                continue
            if source["scope"] == "loaded-package-python-v1":
                loaded += 1
            else:
                legacy += 1
            key = content_hash(source)
            sources.setdefault(key, {**source, "attempts": 0})["attempts"] += 1
    return {
        "context": _source(manifest)
        or {
            "source_tree_hash": None,
            "scope": "unknown",
            "status": "not_recorded_or_unverified",
            "git_sha": None,
            "git_dirty": None,
            "git_state": "legacy_or_unknown",
        },
        "sources": [sources[key] for key in sorted(sources)],
        "recorded_attempts": total,
        "unknown_attempts": unknown,
        "loaded_package_attempts": loaded,
        "legacy_attempts": legacy,
    }
