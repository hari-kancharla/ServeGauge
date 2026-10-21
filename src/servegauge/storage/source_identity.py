"""Local package-source snapshots; optional Git context never comes from caller CWD."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

from servegauge.core.errors import SafeError
from servegauge.core.hashing import content_hash


def _guard(path: Path) -> None:
    if any(part.casefold() == "starter files" for part in path.parts) or ".." in path.parts:
        raise ValueError("source_path_refused")
    if any(p.is_symlink() for p in [*reversed(path.absolute().parents), path]):
        raise ValueError("source_symlink_refused")


def _inventory(package: Path) -> dict[str, str]:
    _guard(package)
    for name in ["__init__.py", "core/manifest.py", "storage/source_identity.py"]:
        path = package / name
        _guard(path)
        if not path.is_file():
            raise ValueError("source_required_file_unavailable")
    result = {}
    pending = [package]
    while pending:
        directory = pending.pop()
        for path in sorted(directory.iterdir()):
            # Reject protected names and links before any descent or file read.
            _guard(path)
            if path.is_dir():
                pending.append(path)
            elif path.suffix == ".py":
                if not path.is_file():
                    raise ValueError("source_regular_python_file_required")
                result["src/servegauge/" + path.relative_to(package).as_posix()] = content_hash(
                    path.read_text(encoding="utf-8")
                )
    return dict(sorted(result.items()))


def _git(root: Path, args: list[str]) -> subprocess.CompletedProcess[str] | None:
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    environment.update(
        GIT_OPTIONAL_LOCKS="0", GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull
    )
    try:
        return subprocess.run(
            ["git", "-c", "core.fsmonitor=false", *args],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None


def _checkout(package: Path) -> dict[str, Any]:
    unknown = {"git_sha": "uncommitted", "git_dirty": False, "git_state": "not_checkout"}
    if package.parent.name != "src":
        return unknown
    root = package.parent.parent
    marker = root / ".git"
    _guard(marker)
    if not (marker.is_dir() or marker.is_file()):
        return unknown
    version = _git(root, ["--version"])
    parsed = re.match(r"git version (\d+)\.(\d+)", version.stdout) if version is not None else None
    # Older Git treats fsmonitor Booleans as hook names; Git context is optional.
    if parsed is None or (int(parsed[1]), int(parsed[2])) < (2, 36):
        return {**unknown, "git_state": "unavailable"}
    top = _git(root, ["rev-parse", "--show-toplevel"])
    if top is None or top.returncode != 0:
        return {**unknown, "git_state": "unavailable"}
    if Path(top.stdout.strip()).absolute() != root.absolute():
        return unknown
    revision = _git(root, ["rev-parse", "HEAD"])
    dirty = _git(
        root,
        [
            "status",
            "--porcelain",
            "--untracked-files=normal",
            "--",
            "src",
            "scripts",
            "configs",
            "profiles",
            "pyproject.toml",
            "uv.lock",
        ],
    )
    if (
        revision is None
        or revision.returncode != 0
        or dirty is None
        or dirty.returncode != 0
        or re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", revision.stdout.strip()) is None
    ):
        return {**unknown, "git_state": "unavailable"}
    return {
        "git_sha": revision.stdout.strip(),
        "git_dirty": bool(dirty.stdout),
        "git_state": "bound_checkout",
    }


def _snapshot(package: Path, import_hash: str | None) -> dict[str, Any]:
    try:
        inventory = _inventory(package)
        current = content_hash(inventory)
        if import_hash is None:
            raise SafeError("capture_source_inventory_unavailable")
        if current != import_hash:
            raise SafeError("capture_source_changed_since_import")
        git = _checkout(package)
        if _inventory(package) != inventory:
            raise SafeError("capture_source_changed_during_snapshot")
        return {
            **git,
            "source_tree_hash": current,
            "source_identity": {
                "version": "capture-source-v2",
                "scope": "loaded-package-python-v1",
                "package": "servegauge",
                "source_hash": current,
                "import_source_hash": import_hash,
                "python_files": len(inventory),
                "status": "consistent_with_import_snapshot",
            },
        }
    except (OSError, ValueError, UnicodeError):
        raise SafeError("capture_source_inventory_unavailable") from None


_PACKAGE = Path(__file__).absolute().parents[1]
try:
    _IMPORT_HASH: str | None = content_hash(_inventory(_PACKAGE))
except (OSError, ValueError, UnicodeError):
    _IMPORT_HASH = None


def capture_source() -> dict[str, Any]:
    """Hash package Python files at import/capture; not bytecode or hardware attestation."""
    return _snapshot(_PACKAGE, _IMPORT_HASH)
