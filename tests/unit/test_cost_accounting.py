import copy
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from tests_support import exchange

from servegauge.core.models import Attempt, Timings
from servegauge.core.pricing import effective_rates, expected_token_cost, reservation_rates
from servegauge.scoring.cost_accounting import cost_observation


def test_cache_reads_writes_and_reasoning_use_native_subsets_once():
    pricing = {
        "prompt": ".01",
        "completion": ".02",
        "input_cache_read": ".002",
        "input_cache_write": ".0125",
        "internal_reasoning": ".03",
        "request": ".1",
    }
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 30, "cache_write_tokens": 10},
        "completion_tokens_details": {"reasoning_tokens": 5},
    }
    expected, details = expected_token_cost(usage, pricing, 1)
    assert expected == Decimal("1.335")
    assert details["uncached_prompt_tokens"] == 60
    assert Decimal(details["cache_savings_usd"]) == Decimal(".215")


def test_price_overrides_apply_strict_token_boundary_time_weekday_and_order():
    pricing = {
        "prompt": "1",
        "completion": "2",
        "overrides": [
            {"min_prompt_tokens": 100, "prompt": "3"},
            {"utc_start": 1630, "utc_end": 30, "utc_days": ["monday"], "completion": "4"},
            {"min_prompt_tokens": 100, "prompt": "5"},
        ],
    }
    monday = datetime(2026, 10, 19, 16, 30, tzinfo=UTC).timestamp()
    assert effective_rates(pricing, 100, monday)["prompt"] == 1
    assert effective_rates(pricing, 101, monday)["prompt"] == 5
    assert effective_rates(pricing, 101, monday)["completion"] == 4
    assert effective_rates(pricing, 101, monday - 1)["completion"] == 2
    assert effective_rates(pricing, 101, monday + 86400)["completion"] == 2
    assert reservation_rates(pricing)["prompt"] == 5
    assert reservation_rates(pricing)["completion"] == 4


def test_reservations_cover_all_tariffs_cache_write_reasoning_and_request_fee():
    from servegauge.core.models import EndpointSpec
    from servegauge.runner.cost import actual_cost, worst_case_cost

    pricing = {
        "prompt": "1",
        "completion": "2",
        "request": "3",
        "overrides": [
            {"utc_start": 0, "utc_end": 1200, "input_cache_write": "6", "request": "7"},
            {"min_prompt_tokens": 1000, "internal_reasoning": "4"},
        ],
    }
    endpoint = EndpointSpec(
        id="fixture",
        kind="openai_compat",
        base_url="http://127.0.0.1:8011/v1",
        model="mock",
        claimed_profile="gpt-oss-120b",
        metadata={"pricing": pricing, "context_length": 100},
    )
    assert worst_case_cost({"messages": [], "max_tokens": 20}, endpoint) == Decimal(687)
    amount, source = actual_cost(
        {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}, endpoint, Decimal(687)
    )
    assert amount == Decimal(87) and source == "estimate"


@pytest.mark.parametrize(
    "change",
    [
        "missing-cache",
        "cache-overlap",
        "write-rate",
        "bool-count",
        "bad-price",
        "negative",
        "unknown-fee",
        "unknown-rule",
        "unknown-discount",
        "by-channel",
        "bad-clock",
        "bad-days",
        "missing-reasoning",
    ],
)
def test_unqualified_price_and_count_shapes_cannot_supply_cost(change):
    pricing = {"prompt": ".01", "completion": ".02", "input_cache_read": ".002"}
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 30},
    }
    if change == "missing-cache":
        usage.pop("prompt_tokens_details")
    elif change == "cache-overlap":
        usage["prompt_tokens_details"]["cached_tokens"] = 101
    elif change == "write-rate":
        usage["prompt_tokens_details"]["cache_write_tokens"] = 5
    elif change == "bool-count":
        usage["completion_tokens"] = True
    elif change == "bad-price":
        pricing["prompt"] = "NaN"
    elif change == "negative":
        pricing["completion"] = "-1"
    elif change == "unknown-fee":
        pricing["image"] = "1"
    elif change == "unknown-rule":
        pricing["overrides"] = [{"unrecognized_condition": True, "prompt": "5"}]
    elif change == "unknown-discount":
        pricing["discount"] = 0.5
    elif change == "by-channel":
        usage["prompt_tokens_details"]["audio_tokens"] = 1
    elif change == "bad-clock":
        pricing["overrides"] = [{"utc_start": 1261, "utc_end": 0, "prompt": "5"}]
    elif change == "bad-days":
        pricing["overrides"] = [{"utc_days": ["notaday"], "prompt": "5"}]
    elif change == "missing-reasoning":
        pricing["internal_reasoning"] = "1"
    with pytest.raises(ValueError):
        expected_token_cost(usage, pricing, 1)


def billed_record():
    from servegauge.adapters.openrouter import price_provenance

    pricing = {"prompt": ".01", "completion": ".02"}
    snapshot = {
        "pricing": pricing,
        "provenance": price_provenance(
            "http://127.0.0.1:8011/api/v1/models/mock/endpoints",
            "mock",
            "fixture",
            pricing,
            captured_at="1970-01-01T00:00:00+00:00",
        ),
    }
    response = {
        "id": "gen-one",
        "model": "mock",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.2, "is_byok": False},
    }
    attempts = [
        Attempt(
            n=1,
            status=503,
            timings=Timings(start_unix=1, total_ms=2),
            response=copy.deepcopy(response),
            metadata={
                "cost_source": "provider",
                "actual_cost_usd": ".2",
                "accounting_price_snapshot": snapshot,
            },
        )
    ]
    attempts.append(
        Attempt(
            n=2,
            status=200,
            timings=Timings(start_unix=2, total_ms=2),
            response=copy.deepcopy(response),
            metadata={
                "cost_source": "generation_record",
                "actual_cost_usd": ".3",
                "accounting_price_snapshot": snapshot,
                "generation_record": {
                    "id": "gen-one",
                    "model": "mock",
                    "cancelled": False,
                    "is_byok": False,
                    "native_tokens_prompt": 20,
                    "native_tokens_completion": 5,
                    "total_cost": 0.3,
                },
            },
        )
    )
    attempts[-1].response["usage"]["prompt_tokens"] = 20
    attempts[-1].response["usage"]["cost"] = 0.3
    return exchange(
        attempts=attempts,
        cost_usd=0.5,
        cost_source="estimate",
        request={
            "model": "mock",
            "messages": [{"role": "user", "content": "test"}],
            "provider": {"only": ["fixture"], "allow_fallbacks": False},
        },
    )


def test_all_known_retry_costs_are_compared_with_each_attempts_own_tokens():
    record = billed_record()
    value, details, layout = cost_observation(record)
    assert value == 0 and details["expected_usd"] == "0.50"
    assert len(details["attempts"]) == len(layout["attempts"]) == 2
    # The mixed aggregate source is not a reason to discard known per-attempt charges.
    assert record.cost_source == "estimate"


@pytest.mark.parametrize(
    "change",
    [
        "unknown-hold",
        "price-hash",
        "currency",
        "generation-id",
        "generation-count",
        "byok",
        "aggregate",
        "missing-attempt",
        "transport",
        "redacted",
        "price-time",
        "cost-conflict",
        "metadata-only-charge",
        "unknown-charge-basis",
    ],
)
def test_incomplete_retry_or_price_evidence_abstains(change):
    record = billed_record()
    first, last = record.attempts
    if change == "unknown-hold":
        first.metadata["cost_source"] = "estimate"
    elif change == "price-hash":
        first.metadata["accounting_price_snapshot"]["pricing"]["prompt"] = "2"
    elif change == "currency":
        first.metadata["accounting_price_snapshot"]["provenance"]["currency"] = "unknown"
    elif change == "generation-id":
        last.metadata["generation_record"]["id"] = "gen-other"
    elif change == "generation-count":
        last.metadata["generation_record"]["native_tokens_prompt"] = 21
    elif change == "byok":
        last.metadata["generation_record"]["is_byok"] = True
    elif change == "aggregate":
        record.cost_usd = 0.3
    elif change == "missing-attempt":
        first.n = 2
    elif change == "transport":
        first.error = "transport_timeout"
    elif change == "redacted":
        from servegauge.core.redaction import REDACTED

        record.request["messages"][0]["content"] = REDACTED
    elif change == "price-time":
        first.metadata["accounting_price_snapshot"]["provenance"]["captured_at"] = (
            "2100-01-01T00:00:00+00:00"
        )
    elif change == "cost-conflict":
        last.response["usage"]["cost"] = 0.4
    elif change == "metadata-only-charge":
        first.response["usage"].pop("cost")
    elif change == "unknown-charge-basis":
        first.response["usage"].pop("is_byok")
    with pytest.raises(ValueError):
        cost_observation(record)
