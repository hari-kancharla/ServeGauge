"""Secret scan that reports locations, never matching text or environment values."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

PATTERNS = (
    re.compile(rb"\bsk-(?:or-v1-)?[A-Za-z0-9_-]{20,}"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{25,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"(?i)(?:api[_-]?key|authorization)\s*[:=]\s*['\"]?(?:Bearer )?[A-Za-z0-9_-]{24,}"),
)


def configured_secrets() -> list[str]:
    """Only nonempty credential-like env values; never disclose their names or contents."""
    return [
        value
        for name, value in os.environ.items()
        if re.search(r"(?:^|_)(?:KEY|TOKEN|SECRET|PASSWORD)(?:$|_)", name.upper())
        and len(value) >= 8
    ]


def scan_paths(paths: list[Path]) -> list[str]:
    values = [value.encode() for value in configured_secrets()]
    findings: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        blob = path.read_bytes()
        for number, line in enumerate(blob.splitlines(), 1):
            if any(p.search(line) for p in PATTERNS) or any(v in line for v in values):
                findings.append(f"{path.as_posix()}:{number}: possible secret (value suppressed)")
    return findings


def scan_repository(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    paths = {root / os.fsdecode(p) for p in result.stdout.split(b"\0") if p}
    cassettes = root / "tests" / "cassettes"
    if cassettes.exists():
        paths.update(p for p in cassettes.rglob("*") if p.is_file())
    return scan_paths(sorted(paths))
