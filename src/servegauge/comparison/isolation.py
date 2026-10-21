"""Restricted external-tool processes with no inherited account or network access."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from servegauge.storage.comparison import _guard, _private_json

PROMPTFOO_VERSION = "0.124.1"


def _lexical(paths: list[Path]) -> None:
    # Check every supplied path before any filesystem operation on any of them.
    for path in paths:
        if (
            ".." in path.parts
            or any(part.casefold() == "starter files" for part in path.parts)
            or any(ord(char) < 32 for char in str(path))
        ):
            raise ValueError("comparison_path_refused")


def _executable(path: Path) -> Path:
    """Resolve file aliases without following an unchecked protected target."""
    seen: set[Path] = set()
    current = path.absolute()
    while True:
        _lexical([current])
        _guard(current.parent)
        if current in seen:
            raise ValueError("comparison_executable_alias_cycle")
        seen.add(current)
        if not current.is_symlink():
            if not current.is_file():
                raise ValueError("comparison_executable_missing")
            return current
        target = current.readlink()
        # abspath normalizes dots lexically, without touching the target.
        current = Path(os.path.abspath(current.parent / target))


@dataclass(frozen=True)
class ToolRuntime:
    node: Path
    tool: Path
    python: Path
    package_root: Path


@dataclass(frozen=True)
class IsolatedProcess:
    runtime: ToolRuntime
    directory: Path
    environment: dict[str, str]
    profile: Path

    def wrap(self, executable: Path, arguments: list[str]) -> list[str]:
        return ["/usr/bin/sandbox-exec", "-f", str(self.profile), str(executable), *arguments]

    def tool_command(self, arguments: list[str]) -> list[str]:
        entry = self.runtime.tool / "node_modules/promptfoo/dist/src/entrypoint.js"
        return self.wrap(self.runtime.node, [str(entry), *arguments])

    def eval_command(self) -> list[str]:
        return self.tool_command(
            [
                "eval",
                "--config",
                "promptfoo.json",
                "--no-cache",
                "--no-write",
                "--no-share",
                "--no-table",
                "--no-progress-bar",
                "--max-concurrency",
                "1",
                "--repeat",
                "1",
                "--output",
                "promptfoo-results.json",
            ]
        )


def _profile(read_roots: list[Path], directory: Path) -> str:
    readers = " ".join(f"(subpath {json.dumps(str(path))})" for path in read_roots)
    return (
        '(version 1)\n(deny default)\n(import "system.sb")\n'
        "(allow process-fork process-exec)\n(allow signal)\n"
        "(allow sysctl-read)\n(allow mach-lookup)\n(allow file-read-metadata)\n"
        f"(allow file-read* {readers})\n"
        f"(allow file-write* (subpath {json.dumps(str(directory))}))\n"
        "(deny network*)\n"
        '(deny mach-lookup (global-name "com.apple.securityd") '
        '(global-name "com.apple.securityd.xpc") (global-name "com.apple.mDNSResponder"))\n'
    )


def prepare_process(
    runtime: ToolRuntime, directory: Path, *, platform: str | None = None
) -> IsolatedProcess:
    """Prepare a fresh restricted child, never launch it or inherit environment.

    macOS's native boundary is implemented here. Other platforms refuse until an
    equivalent separately verified OS/container backend is available; opt-out
    flags or a monkeypatch must never silently replace that requirement.
    """
    _lexical([runtime.node, runtime.tool, runtime.python, runtime.package_root, directory])
    if (platform or sys.platform) != "darwin":
        raise ValueError("comparison_os_boundary_unavailable")
    for root in [runtime.tool, runtime.package_root, directory]:
        if len(root.absolute().parts) < 3:
            raise ValueError("comparison_read_scope_too_broad")
        _guard(root)
    node = _executable(runtime.node)
    python = _executable(runtime.python)
    for path in [runtime.tool, runtime.package_root, directory]:
        if not path.is_dir():
            raise ValueError("comparison_runtime_directory_missing")
    package = runtime.tool / "node_modules/promptfoo"
    for path in [
        package,
        package / "package.json",
        runtime.tool / "package-lock.json",
        package / "dist/src/entrypoint.js",
    ]:
        _guard(path)
    metadata = _private_json((package / "package.json").read_bytes())
    lock = _private_json((runtime.tool / "package-lock.json").read_bytes())
    packages = lock.get("packages") if isinstance(lock, dict) else None
    locked = packages.get("node_modules/promptfoo") if isinstance(packages, dict) else None
    if (
        not isinstance(metadata, dict)
        or metadata.get("name") != "promptfoo"
        or metadata.get("version") != PROMPTFOO_VERSION
        or not isinstance(locked, dict)
        or locked.get("version") != PROMPTFOO_VERSION
        or not (package / "dist/src/entrypoint.js").is_file()
    ):
        raise ValueError("comparison_pinned_tool_required")
    directory = directory.absolute()
    new_paths = [
        directory / name for name in ["tool-state", "tool-tmp", "empty.env", "boundary.sb"]
    ]
    for path in new_paths:
        _guard(path)
    if any(path.exists() for path in new_paths):
        raise ValueError("comparison_new_isolated_state_required")
    readers = [
        Path(name)
        for name in [
            "/System",
            "/usr",
            "/bin",
            "/dev",
            "/Library/Apple",
            "/private/var/db/dyld",
            "/opt/homebrew",
        ]
    ]
    readers.extend(
        [
            runtime.tool.absolute(),
            runtime.package_root.absolute(),
            directory,
            runtime.python.absolute().parent.parent,
            python.parent.parent,
            node.parent.parent,
        ]
    )
    owner_home = Path.home().absolute()
    if any(owner_home.is_relative_to(root) for root in readers):
        raise ValueError("comparison_read_scope_too_broad")
    environment = {
        "PATH": os.pathsep.join(
            [str(node.parent), str(runtime.python.absolute().parent), "/usr/bin", "/bin"]
        ),
        "LANG": "C.UTF-8",
        "TZ": "UTC",
        "NO_COLOR": "1",
        "TMPDIR": str(directory / "tool-tmp"),
        "PROMPTFOO_CONFIG_DIR": str(directory / "tool-state"),
        "DOTENV_PATH": str(directory / "empty.env"),
        "PROMPTFOO_DISABLE_TELEMETRY": "1",
        "PROMPTFOO_DISABLE_UPDATE": "1",
        "PROMPTFOO_DISABLE_REMOTE_GENERATION": "true",
        "PROMPTFOO_DISABLE_SHARING": "1",
        "PROMPTFOO_SELF_HOSTED": "1",
        "PROMPTFOO_PYTHON_WORKERS": "1",
        "PROMPTFOO_PYTHON": str(runtime.python.absolute()),
        "PYTHONPATH": str(runtime.package_root.absolute()),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONSAFEPATH": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    (directory / "tool-state").mkdir(mode=0o700)
    (directory / "tool-tmp").mkdir(mode=0o700)
    with (directory / "empty.env").open("xb"):
        pass
    profile = directory / "boundary.sb"
    with profile.open("x") as stream:
        stream.write(_profile(readers, directory))
    for path in [directory / "empty.env", profile]:
        path.chmod(0o600)
    return IsolatedProcess(runtime, directory, environment, profile)
