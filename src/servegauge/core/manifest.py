"""Run manifests record reproducibility inputs, never credentials."""

from __future__ import annotations

import importlib.metadata
from datetime import UTC, datetime
from typing import Any

from servegauge.core.hashing import content_hash
from servegauge.core.redaction import Redactor
from servegauge.storage.source_identity import capture_source


def build_manifest(
    run_id: str,
    config: dict[str, Any],
    item_hashes: dict[str, str],
    endpoints: list[dict[str, Any]],
    *,
    redactor: Redactor | None = None,
    reference_pack_id: str | None = None,
    suite_versions: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    scrubber = redactor or Redactor.from_environment()
    clean = scrubber.redact(config)
    source = capture_source()
    packages = {
        name: importlib.metadata.version(name)
        for name in (
            "servegauge",
            "httpx",
            "pydantic",
            "numpy",
            "scipy",
            "openai-harmony",
            "tiktoken",
        )
    }
    return {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        **source,
        "config": clean,
        "config_hash": content_hash(clean),
        "packages": packages,
        "suite_versions": suite_versions or {key: ["unverified"] for key in item_hashes},
        "item_set_hashes": item_hashes,
        "endpoint_metadata": scrubber.redact(endpoints),
        "price_snapshot": [
            scrubber.redact(e.get("metadata", {}).get("pricing", {})) for e in endpoints
        ],
        "reference_pack_id": reference_pack_id,
        "router_flags": [scrubber.redact(e.get("openrouter_pin")) for e in endpoints],
    }
