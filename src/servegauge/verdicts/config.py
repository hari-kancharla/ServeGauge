"""Validated decision controls, all operating thresholds come from the configuration."""

import math
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from servegauge.core.models import ValueModel


class DecisionPolicy(ValueModel):
    alpha: float = Field(gt=0, lt=1)
    target_power: float = Field(gt=0, lt=1)
    bootstrap_resamples: int = Field(ge=2000)
    permutation_resamples: int = Field(ge=1)
    k_fail: int = Field(ge=1)
    tolerance: float = Field(gt=0, lt=1)
    entries: dict[str, dict[str, Any]]

    def margin(self, check_id: str, requested: float | None = None) -> float:
        if check_id == "P03.cache.fresh":
            check_id = "P03.cache"
        declared = (
            -math.log(float(self.entries["P07"]["pass_reference_fraction"]))
            if check_id == "P07"
            else float(
                self.entries.get(check_id, {}).get(
                    "margin", self.entries["comparative_defaults"]["margin"]
                )
            )
        )
        if requested is not None and not 0 < requested <= declared:
            raise ValueError("requirements_may_only_tighten_declared_margins")
        return declared if requested is None else requested


def load_policy(path: Path = Path("configs/margins.yaml")) -> DecisionPolicy:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or any(
        not isinstance(entry, dict) or not entry.get("rationale") for entry in raw.values()
    ):
        raise ValueError("every_margin_entry_requires_a_rationale")
    return DecisionPolicy(
        **{k: v for k, v in raw["statistics"].items() if k != "rationale"},
        **{k: v for k, v in raw["invariant_defaults"].items() if k != "rationale"},
        entries=raw,
    )
