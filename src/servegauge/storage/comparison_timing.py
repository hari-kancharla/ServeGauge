"""Private observed timing marker around the separately pure external assertion."""

from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from servegauge.comparison.independent import get_assert as independent_assert


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("comparison_timing_duplicate_keys")
    return value


def record_first_valid(
    directory: Path, *, procedure: Literal["ordinary", "external"]
) -> dict[str, Any]:
    """Write once; this measures an observed result, never complete setup effort."""
    path = directory / "first-valid-result.json"
    if ".." in path.parts or any(part.casefold() == "starter files" for part in path.parts):
        raise ValueError("comparison_timing_path_refused")
    if any(part.is_symlink() for part in [*reversed(path.absolute().parents), path]):
        raise ValueError("comparison_timing_symlink_refused")
    if procedure not in {"ordinary", "external"}:
        raise ValueError("comparison_timing_procedure_invalid")
    value = {
        "procedure": procedure,
        "utc": datetime.now(UTC).isoformat(),
        "scope": "first_valid_independent_assertion"
        if procedure == "external"
        else "first_valid_ordinary_check",
    }
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            previous = json.loads(path.read_bytes(), object_pairs_hook=_unique)
            if (
                not isinstance(previous, dict)
                or set(previous) != set(value)
                or previous.get("procedure") != procedure
                or previous.get("scope") != value["scope"]
                or datetime.fromisoformat(previous["utc"]).utcoffset() is None
            ):
                raise ValueError
        except (ValueError, TypeError, KeyError):
            raise ValueError("comparison_timing_marker_changed") from None
        return dict(previous)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write((json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())
    return value


def get_assert(output: Any, context: dict[str, Any]) -> dict[str, Any]:
    """Record actual first-valid time without supplying or changing a verdict."""
    result = independent_assert(output, context)
    summary = json.loads(result["reason"])
    if (
        summary["verdict"] in {"PASS", "FAIL", "INCONCLUSIVE"}
        and type(summary.get("n")) is int
        and summary["n"] > 0
        and all(
            isinstance(summary.get(name), (int, float)) and math.isfinite(summary[name])
            for name in ["ci_low", "ci_high"]
        )
    ):
        record_first_valid(Path.cwd(), procedure="external")
    return result
