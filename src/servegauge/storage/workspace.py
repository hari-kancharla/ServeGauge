"""Explicit, offline installation of versioned workspace defaults and evidence context."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any

from servegauge.refs import fingerprint_qualification as qualification
from servegauge.storage.notices import guard_path

MARKER = ".servegauge-workspace.json"


def _guard(path: Path) -> None:
    if ".." in path.parts or any(p.casefold() == ".git" for p in path.parts):
        raise ValueError("workspace_path_refused")
    guard_path(path)


def template_payloads() -> dict[str, bytes]:
    """Read only the packaged manifest's canonical runtime files; no root discovery."""
    base = files("servegauge").joinpath("resources/workspace-v1")
    manifest = json.loads(base.joinpath("manifest.json").read_bytes())
    if manifest["version"] != "workspace-template-v1":
        raise ValueError("workspace_template_version_invalid")
    result = {}
    for name, digest in manifest["files"].items():
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or not relative.parts
            or relative.parts[0] not in {"configs", "profiles", "docs"}
            or any(p.casefold() in {"starter files", ".git"} for p in relative.parts)
            or str(relative) != name
        ):
            raise ValueError("workspace_template_path_invalid")
        raw = base.joinpath(name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("workspace_template_checksum_invalid")
        result[name] = raw
    return result


def _evidence(root: Path) -> dict[str, str]:
    """Use the unchanged qualification verifier after guarding every declared read."""
    _guard(root)
    root = root.absolute()
    registry_path = root / "configs/fingerprint_qualification.json"
    _guard(registry_path)
    raw = registry_path.read_bytes()
    registry = json.loads(raw)
    previous = qualification.ROOT, qualification.REGISTRY
    try:
        qualification.ROOT, qualification.REGISTRY = root, registry_path
        for name in registry["implementation"]:
            _guard(qualification._path(name))
        assets = registry["assets"]
        for asset in assets.values():
            _guard(qualification._path(asset["path"]))
        interval = json.loads(
            qualification._path(assets["interval_acceptance"]["path"]).read_bytes()
        )
        wire = json.loads(qualification._path(assets["wire"]["path"]).read_bytes())
        for name in [
            *interval["study_code_sha256"],
            *wire["value"]["configuration"]["wire"]["null_source_files"],
        ]:
            _guard(qualification._path(name))
        proof = qualification.load_qualification(registry["id"])
        return {
            "root": str(root),
            "id": proof["id"],
            "binding": proof["binding"],
            "registry_sha256": hashlib.sha256(raw).hexdigest(),
        }
    finally:
        qualification.ROOT, qualification.REGISTRY = previous


def _preflight(path: Path, raw: bytes) -> bool:
    _guard(path)
    for parent in path.parents:
        if parent.exists() and not parent.is_dir():
            raise ValueError("workspace_parent_conflict")
    if path.exists():
        if not path.is_file() or path.read_bytes() != raw:
            raise ValueError("workspace_existing_file_conflict")
        return False
    return True


def initialize(root: Path, *, evidence_root: Path | None = None) -> dict[str, Any]:
    """Preflight all named files, then create missing files exclusively. Never overwrite."""
    _guard(root)
    root = root.absolute()
    evidence = _evidence(evidence_root) if evidence_root is not None else None
    payloads = template_payloads()
    marker = {"version": "servegauge-workspace-v1", "evidence": evidence}
    payloads[MARKER] = (json.dumps(marker, indent=2, sort_keys=True) + "\n").encode()
    needed = {name: _preflight(root / name, raw) for name, raw in payloads.items()}
    created = 0
    for name, raw in payloads.items():
        path = root / name
        if needed[name] and _preflight(path, raw):
            path.parent.mkdir(parents=True, exist_ok=True)
            _guard(path)
            with path.open("xb") as stream:
                stream.write(raw)
            created += 1
    return {
        "status": "created" if created else "unchanged",
        "files_created": created,
        "live_budget_usd": 0,
        "network_writes": 0,
        "qualification_evidence": "verified_scripted_scope" if evidence else "unqualified",
    }


def configure_workspace(root: Path) -> str | None:
    """An initialized CLI cannot silently inherit the source tree's qualification."""
    _guard(root)
    marker = root / MARKER
    _guard(marker)
    if not marker.exists():
        return None
    value = json.loads(marker.read_bytes())
    if value["version"] != "servegauge-workspace-v1":
        raise ValueError("workspace_marker_invalid")
    qualification.ROOT = root.absolute()
    qualification.REGISTRY = qualification.ROOT / "configs/fingerprint_qualification.json"
    evidence = value.get("evidence")
    if evidence is not None:
        try:
            verified = _evidence(Path(evidence["root"]))
            if verified == evidence:
                qualification.ROOT = Path(verified["root"])
                qualification.REGISTRY = (
                    qualification.ROOT / "configs/fingerprint_qualification.json"
                )
                return "verified_scripted_scope"
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return "unqualified"
