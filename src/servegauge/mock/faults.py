"""One binding YAML key per injected fault; labels never appear in scored responses."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from servegauge.core.models import ValueModel

Probability = Annotated[float, Field(ge=0, le=1)]


class FaultConfig(ValueModel):
    fault_id: str | None = None
    seed: int | None = None
    ignore_temperature: bool = False
    ignore_top_p: bool = False
    ignore_max_tokens: bool = False
    ignore_stop: bool = False
    ignore_reasoning_effort: bool = False
    drop_tool_calls: Probability = 0
    leak_markers: Probability = 0
    stream_delta_without_index: bool = False
    corrupt_arguments: Probability = 0
    plain_text_after_tool_result: Probability = 0
    truncate_prompt_head_over: int | None = Field(default=None, gt=0)
    truncate_output_after: int | None = Field(default=None, gt=0)
    misreport_usage_pct: float = 0
    stale_cache: Probability = 0
    error_rate: Probability = 0
    route_to_other_model: Probability = 0
    logit_noise: float = Field(default=0, ge=0)
