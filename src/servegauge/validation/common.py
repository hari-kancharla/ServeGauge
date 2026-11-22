"""Shared validation provenance, predeclared configuration and offline guards."""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.stats.intervals import wilson

ROOT = Path(__file__).resolve().parents[3]


def settings() -> dict[str, Any]:
    value = yaml.safe_load((ROOT / "configs/validation.yaml").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("validation_configuration_required")
    return value


def named_seed(*parts: object) -> int:
    return int(content_hash(list(parts))[:15], 16)


def proportion(successes: int, n: int) -> dict[str, Any]:
    low, high = wilson(successes, n)
    return {
        "successes": successes,
        "n": n,
        "estimate": successes / n,
        "ci_low": low,
        "ci_high": high,
        "reference_tier": "MOCK",
        "reference_estimate": None,
    }


def source_hash() -> str:
    return content_hash(
        {
            p.relative_to(ROOT).as_posix(): content_hash(p.read_text(encoding="utf-8"))
            for p in sorted((ROOT / "src/servegauge").rglob("*.py"))
        }
    )


def receipt(kind: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "study": kind,
        "synthetic": True,
        "reference_tier": "MOCK",
        "configuration": config,
        "configuration_hash": content_hash(config),
        "source_hash": source_hash(),
        "model_api_calls": 0,
        "spent_usd": 0,
        "limitations": [
            "Scripted responses and simulation distributions; not real-provider reliability",
            "No model weights, hardware or quantization attestation",
        ],
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


@contextmanager
def offline_only() -> Iterator[None]:
    """Fail closed even if a future study accidentally introduces a network client."""

    def blocked(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("validation_is_offline_network_forbidden")

    with (
        patch.object(socket, "socket", blocked),
        patch.object(socket, "create_connection", blocked),
    ):
        yield


def cached(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
