"""Finite independent experiment units; no sessions, funding or network effects."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from servegauge.core.hashing import content_hash
from servegauge.core.models import ValueModel
from servegauge.selfhost.study_recipe import VARIANTS


class StudyDesign(ValueModel):
    version: Literal["gpu-study-v1"]
    study_id: str = Field(pattern=r"^[a-z][a-z0-9-]{0,39}$")
    seed: int = Field(ge=0, le=2**32 - 1, strict=True)
    replicates: int = Field(ge=2, le=1000, strict=True)
    level: Literal["smoke", "core", "full"] = "smoke"
    independence_reviewed: bool = Field(default=False, strict=True)


def compile_matrix(design: StudyDesign) -> list[dict[str, Any]]:
    """Every score gets four exclusive engine sessions, including three BF16 controls.

    Capture seeds identify inventories; they do not set or attest server RNG state.
    Independence remains an explicit owner sampling assumption, not a measured fact.
    """
    rows = []
    for partition in ("calibration", "evaluation"):
        for condition in VARIANTS:
            for replicate in range(design.replicates):
                unit = f"{design.study_id}-{partition}-{condition}-{replicate:04d}"
                for role in ("target", "reference-b", "reference-c", "reference-d"):
                    digest = content_hash([design.model_dump(mode="json"), unit, role])
                    rows.append(
                        {
                            "slot_id": unit + "-" + role,
                            "unit_id": unit,
                            "partition": partition,
                            "condition": condition,
                            "replicate": replicate,
                            "role": role,
                            "recipe": condition if role == "target" else "bf16",
                            "session_id": "selfhost-" + digest[:32],
                            "run_id": "gpu-study-" + digest,
                            "capture_seed": int(digest[:16], 16),
                        }
                    )
    if any(
        len({r[k] for r in rows}) != len(rows) for k in ("run_id", "session_id", "capture_seed")
    ):
        raise ValueError("study_independent_slot_identity_collision")
    return rows
