"""Explicit experimental controls around the unchanged, owned fingerprint corpus."""

from __future__ import annotations

from typing import Any

from servegauge.core.hashing import content_hash
from servegauge.core.models import Item
from servegauge.selfhost.remote_recipe import STUDY_COLLECTION, STUDY_PROFILE
from servegauge.suites.s6_fingerprint import fingerprint_items


def study_fingerprint_items(
    level: str,
    seed: int,
    overrides: dict[str, Any] | None = None,
    *,
    profile_id: str | None = None,
) -> list[Item]:
    """Create a distinct study corpus without changing any qualified legacy item."""
    selected = dict(overrides or {})
    if profile_id != STUDY_PROFILE or selected.pop("collection_recipe", None) != STUDY_COLLECTION:
        raise ValueError("known_profile_bound_fingerprint_recipe_required")
    original = fingerprint_items(level, seed, selected)
    result = []
    for probe in original:
        item = probe.model_copy(deep=True)
        params = item.payload["params"]
        params.pop("reasoning_effort", None)
        params["top_p"] = 1.0
        design = item.expected["design"]
        design["version"] += "-qwen-study-v1"
        design.update(
            collection_recipe=STUDY_COLLECTION,
            template_defaults={"enable_thinking": False},
            top_p=1.0,
        )
        item.expected["design_hash"] = content_hash(design)
        item.version += "-qwen-study-v1"
        result.append(item)
    return result
