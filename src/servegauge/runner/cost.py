"""Prices are per token. Missing live prices fail closed; MDE assumptions are explicit."""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from scipy.stats import norm

from servegauge.core.hashing import canonical_json
from servegauge.core.models import EndpointSpec
from servegauge.core.pricing import reservation_rates
from servegauge.tokencount.counters import mock_prompt_tokens


def prices(endpoint: EndpointSpec) -> tuple[Decimal, Decimal]:
    if endpoint.kind == "mock":
        return Decimal(0), Decimal(0)
    snapshot = endpoint.metadata.get("pricing")
    if not isinstance(snapshot, dict) or "prompt" not in snapshot or "completion" not in snapshot:
        raise ValueError("live_endpoint_requires_price_snapshot")
    rates = reservation_rates(snapshot)
    return (
        max(
            rates.get(key, Decimal(0))
            for key in ("prompt", "input_cache_read", "input_cache_write")
        ),
        max(rates.get(key, Decimal(0)) for key in ("completion", "internal_reasoning")),
    )


def request_fee(endpoint: EndpointSpec) -> Decimal:
    return (
        Decimal(0)
        if endpoint.kind == "mock"
        else reservation_rates(endpoint.metadata["pricing"]).get("request", Decimal(0))
    )


def prompt_reservation_tokens(body: dict[str, Any], endpoint: EndpointSpec) -> int:
    if endpoint.kind == "mock":
        return mock_prompt_tokens(body)
    # Native rendering/overhead is unqualified; the official local template supplies
    # a scenario proxy, while reservations retain the explicit advertised bound.
    bound = endpoint.metadata.get("max_prompt_tokens") or endpoint.metadata.get("context_length")
    if not isinstance(bound, int) or bound <= 0:
        raise ValueError("live_endpoint_requires_prompt_token_bound")
    wire_bytes = len(canonical_json(body))
    if wire_bytes > bound * 16:
        raise ValueError("request_may_exceed_prompt_token_bound")
    # Deliberate over-context probes may be accepted by a faulty endpoint. Reserve
    # their byte-level input upper envelope as well as the advertised template bound.
    return bound + wire_bytes if wire_bytes > bound else bound


def worst_case_cost(body: dict[str, Any], endpoint: EndpointSpec) -> Decimal:
    prompt, completion = prices(endpoint)
    cap = body.get("max_completion_tokens", body.get("max_tokens"))
    if not isinstance(cap, int) or cap < 1:
        raise ValueError("explicit_positive_output_cap_required")
    advertised = endpoint.metadata.get("max_completion_tokens")
    if isinstance(advertised, int) and cap > advertised:
        raise ValueError("output_cap_exceeds_endpoint_limit")
    return (
        prompt * prompt_reservation_tokens(body, endpoint)
        + completion * cap
        + request_fee(endpoint)
    )


def actual_cost(
    body: dict[str, Any] | None,
    endpoint: EndpointSpec,
    reserved: Decimal,
    *,
    generation: dict[str, Any] | None = None,
) -> tuple[Decimal, Literal["provider", "estimate", "generation_record"]]:
    if endpoint.kind == "mock":
        return Decimal(0), "estimate"
    if generation and generation.get("total_cost") is not None:
        try:
            value = Decimal(str(generation["total_cost"]))
            if value.is_finite() and value >= 0:
                return value, "generation_record"
        except InvalidOperation:
            pass
    usage = body.get("usage") if body else None
    if isinstance(usage, dict):
        if usage.get("cost") is not None:
            try:
                value = Decimal(str(usage["cost"]))
            except InvalidOperation:
                return reserved, "estimate"
            if value.is_finite() and value >= 0:
                return value, "provider"
        prompt, completion = prices(endpoint)
        if (
            type(usage.get("prompt_tokens")) is int
            and type(usage.get("completion_tokens")) is int
            and usage["prompt_tokens"] >= 0
            and usage["completion_tokens"] >= 0
        ):
            return prompt * usage["prompt_tokens"] + completion * usage[
                "completion_tokens"
            ] + request_fee(endpoint), "estimate"
    return reserved, "estimate"


def conservative_paired_mde(
    n: int, family_size: int = 1, alpha: float = 0.05, power: float = 0.8
) -> float:
    if n <= 0 or family_size < 1:
        raise ValueError("positive_n_and_family_size_required")
    z = float(norm.ppf(1 - alpha / family_size)) + float(norm.ppf(power))
    return z / math.sqrt(n)
