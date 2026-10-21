"""Bind to actual package files in checkouts and installed wheels alike."""

from pathlib import Path

from servegauge.core.hashing import content_hash


def implementation_hash(package: Path | None = None) -> str:
    directory = package if package is not None else Path(__file__).resolve().parents[1]
    required = [directory / "__init__.py", directory / "observatory/tick.py"]
    if (
        any(part.lower() == "starter files" for part in directory.parts)
        or directory.is_symlink()
        or any(p.is_symlink() or not p.is_file() for p in required)
    ):
        raise ValueError("monitor_package_source_inventory_required")
    files = sorted(directory.rglob("*.py"))
    if any(p.is_symlink() for p in files):
        raise ValueError("monitor_package_source_symlink")
    return content_hash(
        {
            "src/servegauge/" + p.relative_to(directory).as_posix(): content_hash(
                p.read_text(encoding="utf-8")
            )
            for p in files
        }
    )
