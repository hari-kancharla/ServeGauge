"""Binding measurement interfaces from SPEC 7.2; validation never echoes input values."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CheckClass = Literal["invariant", "comparative", "report_only"]
Verdict = Literal[
    "PASS", "FAIL", "INCONCLUSIVE", "NOT_SUPPORTED", "NOT_APPLICABLE", "INFO", "ERROR"
]
FailureClass = Literal["API_INCOMPATIBILITY", "MODEL_BEHAVIOR", "ROUTING_OR_INFRA", "ACCOUNTING"]
Level = Literal["smoke", "core", "full"]
Tier = Literal["BASELINE", "R1", "R2", "R3", "MOCK"]


class ValueModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, allow_inf_nan=False)


class OpenRouterPin(ValueModel):
    tag: str
    require_parameters: bool = True
    allow_fallbacks: bool = False


class EndpointSpec(ValueModel):
    id: str
    kind: Literal["openai_compat", "openrouter", "selfhosted", "mock"]
    base_url: str
    model: str
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    claimed_profile: str
    openrouter_pin: OpenRouterPin | None = None
    defaults: dict[str, Any] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(ValueModel):
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    stream: bool = False
    params: dict[str, Any] = Field(default_factory=dict)
    minimum_output_tokens: int | None = Field(default=None, ge=1)
    unsupported_probe: Literal["P13"] | None = None


class SseEvent(ValueModel):
    t_ms: float
    event: str | None = None
    data: str
    is_comment: bool = False
    event_id: str | None = None
    retry_ms: int | None = None
    raw_lines: list[str] = Field(default_factory=list)


class Timings(ValueModel):
    start_unix: float
    ttft_ms: float | None = None
    total_ms: float


class Attempt(ValueModel):
    n: int = Field(ge=1)
    status: int | None = None
    error: str | None = None
    timings: Timings
    response: dict[str, Any] | None = None
    response_text: str | None = None
    sse_events: list[SseEvent] | None = None
    response_headers: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Exchange(ValueModel):
    exchange_id: str
    run_id: str
    endpoint_id: str
    suite_id: str
    item_id: str
    sample_idx: int = Field(ge=0)
    request: dict[str, Any]
    attempts: list[Attempt]
    status: int | None = None
    response: dict[str, Any] | None = None
    sse_events: list[SseEvent] | None = None
    provider_meta: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    cost_usd: float | None = Field(default=None, ge=0)
    cost_source: Literal["provider", "generation_record", "estimate"] | None = None
    error: str | None = None
    request_url: str | None = None
    request_headers: dict[str, str] = Field(default_factory=dict)
    response_headers: dict[str, str] = Field(default_factory=dict)
    response_text: str | None = None
    timings: Timings | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Item(ValueModel):
    item_id: str
    suite_id: str
    version: str
    payload: dict[str, Any]
    expected: dict[str, Any]
    tags: list[str] = Field(default_factory=list)
    source: str = "generated"
    license: str = "CC-BY-4.0"


class ItemScore(ValueModel):
    item_id: str
    endpoint_id: str
    outcomes: list[float | bool]
    details: dict[str, Any] = Field(default_factory=dict)


class CheckResult(ValueModel):
    check_id: str
    check_class: CheckClass
    suite_id: str
    endpoint_id: str
    run_id: str
    requirement_ids: list[str] = Field(default_factory=list)
    n: int = Field(ge=0)
    estimate: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    reference_tier: Tier | None = None
    reference_estimate: float | None = None
    delta: float | None = None
    delta_ci_low: float | None = None
    delta_ci_high: float | None = None
    margin: float | None = None
    p_value: float | None = None
    adjusted_p: float | None = None
    violations: int | None = None
    mde: float | None = None
    verdict: Verdict
    reason: str
    evidence: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class Diagnosis(ValueModel):
    diagnosis_id: str
    failure_class: FailureClass
    code: str
    endpoint_id: str
    run_id: str
    confidence: Literal["low", "medium", "high"]
    summary: str
    evidence: list[str]
    check_ids: list[str]
    alternatives: list[str]
    falsifiers: list[str]
    bundle_path: str | None = None


class ModelProfile(ValueModel):
    id: str
    weights: str
    openrouter_slug: str | None = None
    family: str | None = None
    chat_format: str | None = None
    tokenizer: dict[str, Any] = Field(default_factory=dict)
    reasoning: dict[str, Any] = Field(default_factory=dict)
    recommended_sampling: dict[str, Any] = Field(default_factory=dict)
    format_markers: list[str] = Field(default_factory=list)
    immutable_params: dict[str, Any] = Field(default_factory=dict)
    tool_id_format: str | None = None
    reference_tiers: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    field_sources: dict[str, str] = Field(default_factory=dict)
    unverified: list[str] = Field(default_factory=list)


class ReferencePack(ValueModel):
    pack_id: str
    profile: str
    tier: Tier
    manifest: dict[str, Any] = Field(default_factory=dict)
    outcomes: dict[str, Any] = Field(default_factory=dict)
    calibration: dict[str, Any] = Field(default_factory=dict)
    fingerprints: dict[str, Any] = Field(default_factory=dict)


class SuiteConfig(ValueModel):
    level: Level = "smoke"
    seeds: list[int] = Field(default_factory=lambda: [20261018])
    size_overrides: dict[str, int] = Field(default_factory=dict)
    enabled_checks: list[str] = Field(default_factory=list)
