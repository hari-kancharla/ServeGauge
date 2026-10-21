"""Pure USD token pricing, cache subsets and documented conditional overrides."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

RATE_KEYS = {
    "prompt",
    "completion",
    "input_cache_read",
    "input_cache_write",
    "internal_reasoning",
    "request",
}
CONDITIONS = {"min_prompt_tokens", "utc_start", "utc_end", "utc_days"}
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


class PricingUnavailable(ValueError):
    """Fixed public reason code; never include captured values in exceptions."""


def money(value: Any) -> Decimal:
    if not isinstance(value, (int, float, str, Decimal)) or isinstance(value, bool):
        raise PricingUnavailable("invalid_monetary_value")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise PricingUnavailable("invalid_monetary_value") from None
    if not result.is_finite() or result < 0:
        raise PricingUnavailable("nonnegative_finite_money_required")
    return result


def token_count(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PricingUnavailable("nonnegative_integer_token_count_required")
    return value


def _rates(value: dict[str, Any], *, base: bool = False) -> dict[str, Decimal]:
    if base and not {"prompt", "completion"} <= value.keys():
        raise PricingUnavailable("prompt_and_completion_prices_required")
    result = {}
    for key, raw in value.items():
        if key == "overrides" or (not base and key in CONDITIONS):
            continue
        rate = money(raw)
        if key in RATE_KEYS:
            result[key] = rate
        elif rate != 0:
            raise PricingUnavailable("unsupported_non_token_fee_or_discount")
    return result


def _overrides(pricing: dict[str, Any]) -> list[dict[str, Any]]:
    entries = pricing.get("overrides", [])
    if not isinstance(entries, list) or any(not isinstance(e, dict) for e in entries):
        raise PricingUnavailable("invalid_pricing_overrides")
    for entry in entries:
        if set(entry) - RATE_KEYS - CONDITIONS:
            raise PricingUnavailable("unknown_pricing_override_condition")
        if "min_prompt_tokens" in entry:
            token_count(entry["min_prompt_tokens"])
        if ("utc_start" in entry) != ("utc_end" in entry):
            raise PricingUnavailable("complete_price_time_window_required")
        for key in ("utc_start", "utc_end"):
            if key in entry:
                clock = token_count(entry[key])
                if clock > 2359 or clock % 100 > 59:
                    raise PricingUnavailable("invalid_price_time_window")
        if "utc_days" in entry and (
            not isinstance(entry["utc_days"], list)
            or not entry["utc_days"]
            or any(day not in WEEKDAYS for day in entry["utc_days"])
            or len(set(entry["utc_days"])) != len(entry["utc_days"])
        ):
            raise PricingUnavailable("invalid_price_weekdays")
        _rates(entry)
    return entries


def effective_rates(pricing: dict[str, Any], prompt: int, started: float) -> dict[str, Decimal]:
    rates = _rates(pricing, base=True)
    token_count(prompt)
    try:
        instant = datetime.fromtimestamp(started, UTC)
    except (ValueError, OverflowError, OSError):
        raise PricingUnavailable("price_request_time_unavailable") from None
    clock = instant.hour * 100 + instant.minute
    for entry in _overrides(pricing):
        if "min_prompt_tokens" in entry and prompt <= entry["min_prompt_tokens"]:
            continue
        if "utc_days" in entry and WEEKDAYS[instant.weekday()] not in entry["utc_days"]:
            continue
        if "utc_start" in entry:
            start, end = entry["utc_start"], entry["utc_end"]
            applies = start <= clock < end if end > start else clock >= start or clock < end
            if not applies:
                continue
        rates.update(_rates(entry))
    return rates


def reservation_rates(pricing: dict[str, Any]) -> dict[str, Decimal]:
    """Upper envelope across every declared tariff, independent of current time."""
    rates = _rates(pricing, base=True)
    for entry in _overrides(pricing):
        for key, rate in _rates(entry).items():
            rates[key] = max(rates.get(key, Decimal(0)), rate)
    return rates


def expected_token_cost(
    usage: dict[str, Any],
    pricing: dict[str, Any],
    started: float,
) -> tuple[Decimal, dict[str, Any]]:
    prompt, completion = (
        token_count(usage.get("prompt_tokens")),
        token_count(usage.get("completion_tokens")),
    )
    rates = effective_rates(pricing, prompt, started)
    prompt_details, output_details = (
        usage.get("prompt_tokens_details", {}),
        usage.get("completion_tokens_details", {}),
    )
    if not isinstance(prompt_details, dict) or not isinstance(output_details, dict):
        raise PricingUnavailable("invalid_token_details")
    for key, value in prompt_details.items():
        if key not in {"cached_tokens", "cache_write_tokens", "text_tokens"} and token_count(value):
            raise PricingUnavailable("unsupported_prompt_channel")
    for key, value in output_details.items():
        if key not in {
            "reasoning_tokens",
            "text_tokens",
            "accepted_prediction_tokens",
        } and token_count(value):
            raise PricingUnavailable("unsupported_output_channel")
    cache_counts: dict[str, int] = {}
    for name, price in (
        ("cached_tokens", "input_cache_read"),
        ("cache_write_tokens", "input_cache_write"),
    ):
        if price in rates and name not in prompt_details:
            raise PricingUnavailable("priced_cache_count_unavailable")
        count = token_count(prompt_details.get(name, 0))
        if count and price not in rates:
            raise PricingUnavailable("cache_rate_unavailable")
        cache_counts[name] = count
    read, write = cache_counts["cached_tokens"], cache_counts["cache_write_tokens"]
    if read + write > prompt:
        raise PricingUnavailable("cache_subsets_exceed_prompt_total")
    reason = token_count(output_details.get("reasoning_tokens", 0))
    if "internal_reasoning" in rates and "reasoning_tokens" not in output_details:
        raise PricingUnavailable("priced_reasoning_count_unavailable")
    if reason > completion:
        raise PricingUnavailable("reasoning_exceeds_completion_total")
    input_cost = (
        (prompt - read - write) * rates["prompt"]
        + read * rates.get("input_cache_read", Decimal(0))
        + write * rates.get("input_cache_write", Decimal(0))
    )
    output_cost = (completion - reason) * rates["completion"] + reason * rates.get(
        "internal_reasoning", rates["completion"]
    )
    expected = input_cost + output_cost + rates.get("request", Decimal(0))
    return expected, {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "uncached_prompt_tokens": prompt - read - write,
        "cache_read_tokens": read,
        "cache_write_tokens": write,
        "reasoning_tokens": reason,
        "cache_savings_usd": str(prompt * rates["prompt"] - input_cost),
        "effective_rates_usd": {key: str(rate) for key, rate in sorted(rates.items())},
    }
