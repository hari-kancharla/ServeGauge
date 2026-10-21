"""Actual local GLM counting dependencies for matched accounting references."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from servegauge.core.errors import SafeError
from servegauge.observatory.identity import generator_manifest
from servegauge.tokencount.glm_assets import FILES, MODEL, REVISION

COUNTER_VERSION = "glm-hf-api-adapter-v1"
COUNTER_ROOTS = (
    "servegauge.scoring.accounting_scoring",
    "servegauge.scoring.completion_accounting",
    "servegauge.scoring.tokenizer_hints",
    "servegauge.storage.accounting_identity",
    "servegauge.scoring.json_scoring",
    "servegauge.suites.s1_protocol",
    "servegauge.suites.s3_structured",
    "servegauge.suites.s4_longctx",
    "servegauge.suites.s5_longout",
)


def accounting_counter_identity(profile: str) -> dict[str, Any] | None:
    """No identity lookup or new fields for the frozen gpt-oss configurations."""
    if profile != "glm-5.3":
        return None
    try:
        closure = generator_manifest(roots=COUNTER_ROOTS)
        if not closure["complete"]:
            raise SafeError("accounting_counter_identity_unavailable")
        runtime = {name: version(name) for name in ("tokenizers", "Jinja2")}
    except (OSError, ValueError, PackageNotFoundError):
        raise SafeError("accounting_counter_identity_unavailable") from None
    return {
        "counter_version": COUNTER_VERSION,
        "hf_model": MODEL,
        "hf_revision": REVISION,
        "files_sha256": dict(FILES),
        "implementation_roots": list(COUNTER_ROOTS),
        "implementation_sha256": closure["pins"],
        "runtime_versions": runtime,
        "projection": "pinned local template; complete visible and typed raw reasoning text",
        "provider_rendering_verified": False,
        "billing_correctness_proven": False,
    }
