"""Cross-platform developer task runner. Invoke with uv run python scripts/dev.py TARGET."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    subprocess.run(list(args), cwd=ROOT, check=True)


def verify_coverage(packages: list[str]) -> None:
    report = json.loads((ROOT / ".cache/coverage.json").read_text(encoding="utf-8"))
    for package in packages:
        summaries = [
            data["summary"]
            for filename, data in report["files"].items()
            if Path(filename).parent == Path("src/servegauge") / package
        ]
        total = sum(s["num_statements"] + s["num_branches"] for s in summaries)
        missing = sum(s["missing_lines"] + s["missing_branches"] for s in summaries)
        percent = 100 * (total - missing) / total if total else 0
        print(f"Coverage {package}: {percent:.2f}% (required >=85%)", flush=True)
        if percent < 85:
            raise SystemExit(f"Coverage below requirement in {package}")


def main() -> None:
    target = sys.argv[1] if len(sys.argv) == 2 else "help"
    os.environ.setdefault("UV_CACHE_DIR", str(ROOT / ".cache/uv"))
    os.environ.setdefault("UV_PYTHON_INSTALL_DIR", str(ROOT / ".cache/python"))
    python = sys.executable
    if target == "setup":
        run("uv", "sync", "--locked")
        run(python, "-m", "servegauge.cli", "setup-cache")
    elif target == "check":
        run(python, "-m", "ruff", "format", "--check", "src", "tests", "scripts")
        run(python, "-m", "ruff", "check", "src", "tests", "scripts")
        run(python, "-m", "mypy", "--strict", "src")
        run(python, "-m", "servegauge.cli", "doctor", "--scan-secrets")
        packages = [
            p
            for p in ("core", "stats", "verdicts", "diagnose", "runner", "scoring")
            if (ROOT / "src/servegauge" / p).is_dir()
        ]
        run(
            python,
            "-m",
            "pytest",
            *(f"--cov=servegauge.{p}" for p in packages),
            "--cov-report=term",
            "--cov-report=json:.cache/coverage.json",
        )
        verify_coverage(packages)
    elif target == "test":
        run(python, "-m", "pytest")
    elif target == "fmt":
        run(python, "-m", "ruff", "format", "src", "tests", "scripts")
        run(python, "-m", "ruff", "check", "--fix", "src", "tests", "scripts")
        run(python, "-m", "ruff", "format", "src", "tests", "scripts")
    elif target == "site":
        run(python, "-m", "servegauge.cli", "site", "build")
    elif target == "validate":
        for kind in ("faults", "power", "changepoint"):
            run(python, "-m", "servegauge.cli", "validate", kind)
    else:
        raise SystemExit("Targets: setup check test fmt site validate")


if __name__ == "__main__":
    main()
