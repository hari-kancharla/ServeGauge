"""Complete experiment declarations, separate from the preserved matrix-only schema."""

from __future__ import annotations

import re
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from servegauge.core.hashing import content_hash
from servegauge.core.models import ValueModel
from servegauge.selfhost.study_design import StudyDesign, compile_matrix
from servegauge.selfhost.study_recipe import study_template


def _unknown_engine_builds() -> dict[str, str | None]:
    return {"0.31.0": None, "0.32.0": None}


class CollectionSpec(ValueModel):
    version: Literal["gpu-study-collection-v1"]
    design: StudyDesign
    scope: Literal["owner_gpu", "mock_software_qa"]
    backend: Literal["ssh", "modal"]
    fingerprints: dict[str, Any] = Field(
        default_factory=lambda: {"methods": ["FP1", "FP2", "FP3", "FP4"]}
    )
    engine_images: dict[str, str | None] = Field(default_factory=_unknown_engine_builds)
    modal_image_ids: dict[str, str | None] = Field(default_factory=_unknown_engine_builds)
    session_size: int = Field(default=1, ge=1, le=10000, strict=True)
    requests_per_minute: float = Field(default=60, gt=0, le=60000, strict=True)
    concurrency_per_endpoint: int = Field(default=4, ge=1, le=16, strict=True)
    within_capture_components_reviewed: bool = Field(default=False, strict=True)
    iid_condition_quartets_reviewed: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def complete(self) -> Self:
        if self.fingerprints.get("methods") != ["FP1", "FP2", "FP3", "FP4"]:
            raise ValueError("study_all_four_ordered_methods_required")
        if self.iid_condition_quartets_reviewed and not self.design.independence_reviewed:
            raise ValueError("study_independent_quartets_required_for_IID_declaration")
        for mapping, pattern in (
            (self.engine_images, r"vllm/vllm-openai@sha256:[a-f0-9]{64}"),
            (self.modal_image_ids, r"im-[A-Za-z0-9_-]{1,128}"),
        ):
            if set(mapping) != {"0.31.0", "0.32.0"} or any(
                value is not None and re.fullmatch(pattern, value) is None
                for value in mapping.values()
            ):
                raise ValueError("study_two_pinned_image_builds_required")
            selected = [value for value in mapping.values() if value is not None]
            if len(selected) != len(set(selected)):
                raise ValueError("study_distinct_engine_version_builds_required")
        return self

    def sampling_declaration(self) -> dict[str, Any]:
        """Owner assumptions about whole components/cache strata, never observations."""
        return {
            "independent_sessions": self.within_capture_components_reviewed,
            "independent_prompt_strata": self.within_capture_components_reviewed,
            "independent_reference_captures": self.design.independence_reviewed,
            "matched_sampling_design": True,
            "session_size": self.session_size,
            "cache_groups": {},
        }


def collection_matrix(spec: CollectionSpec) -> list[dict[str, Any]]:
    rows = compile_matrix(spec.design)
    for row in rows:
        digest = content_hash([spec.model_dump(mode="json"), row["slot_id"]])
        row.update(
            session_id="selfhost-" + digest[:32],
            run_id="gpu-study-" + digest,
            capture_seed=int(digest[:16], 16),
        )
    if any(
        len({r[key] for r in rows}) != len(rows) for key in ("run_id", "session_id", "capture_seed")
    ):
        raise ValueError("study_collection_identity_collision")
    return rows


def collection_template(backend: str = "ssh", scope: str = "owner_gpu") -> dict[str, Any]:
    return CollectionSpec.model_validate(
        {
            "version": "gpu-study-collection-v1",
            "design": StudyDesign(
                version="gpu-study-v1", study_id="pilot", seed=101, replicates=2
            ).model_dump(mode="json"),
            "scope": scope,
            "backend": backend,
        }
    ).model_dump(mode="json")


def collection_owner_template(spec: CollectionSpec, variant: str) -> dict[str, Any]:
    value = study_template(spec.backend, variant)
    value["image"] = spec.engine_images[value["engine_version"]]
    if spec.backend == "modal":
        value["modal_image_id"] = spec.modal_image_ids[value["engine_version"]]
    return value
