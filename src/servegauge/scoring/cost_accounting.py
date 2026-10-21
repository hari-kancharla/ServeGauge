"""B04 per-attempt charges versus captured list prices, never reservation estimates."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Any
from urllib.parse import unquote, urlsplit

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.models import Attempt, Exchange, Item
from servegauge.core.pricing import PricingUnavailable, expected_token_cost, money, token_count
from servegauge.core.redaction import REDACTED
from servegauge.scoring.common import Samples
from servegauge.suites.s7_accounting import cost_contract, cost_group


def _snapshot(attempt: Attempt, request: dict[str, Any]) -> dict[str, Any]:
    snapshot = attempt.metadata.get("accounting_price_snapshot")
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("pricing"), dict):
        raise PricingUnavailable("per_attempt_price_snapshot_required")
    provenance = snapshot.get("provenance")
    if not isinstance(provenance, dict) or (
        not isinstance(request.get("model"), str)
        or not request["model"]
        or not isinstance(provenance.get("endpoint_tag"), str)
        or not provenance["endpoint_tag"]
        or provenance.get("version") != "openrouter-token-prices-v1"
        or provenance.get("verified") is not True
        or provenance.get("currency") != "USD"
        or provenance.get("pricing_hash") != content_hash(snapshot["pricing"])
        or provenance.get("model") != request.get("model")
    ):
        raise PricingUnavailable("qualified_bound_price_provenance_required")
    source = provenance.get("source")
    if not isinstance(source, str):
        raise PricingUnavailable("price_source_required")
    parts = urlsplit(source)
    if (
        parts.username
        or parts.query
        or parts.fragment
        or not (
            (parts.scheme == "https" and parts.hostname == "openrouter.ai")
            or (parts.scheme == "http" and parts.hostname in {"127.0.0.1", "localhost", "::1"})
        )
        or not unquote(parts.path).endswith("/models/" + str(request.get("model")) + "/endpoints")
    ):
        raise PricingUnavailable("supported_public_price_source_required")
    provider = request.get("provider")
    if (
        not isinstance(provider, dict)
        or provider.get("only") != [provenance.get("endpoint_tag")]
        or provider.get("allow_fallbacks") is not False
    ):
        raise PricingUnavailable("price_snapshot_pin_mismatch")
    try:
        instant = datetime.fromisoformat(provenance["captured_at"])
        if instant.tzinfo is None or instant.timestamp() > attempt.timings.start_unix:
            raise ValueError
    except (ValueError, TypeError, KeyError, OverflowError):
        raise PricingUnavailable("price_snapshot_time_unqualified") from None
    return snapshot


def _charged(attempt: Attempt, request: dict[str, Any]) -> tuple[Decimal, dict[str, Any]]:
    if (
        attempt.error
        or attempt.status is None
        or not attempt.response
        or attempt.metadata.get("exclude_from_scoring")
        or attempt.metadata.get("incomplete")
        or attempt.metadata.get("invalid_json_events")
    ):
        raise PricingUnavailable("attempt_charge_capture_incomplete")
    response = attempt.response
    usage = response.get("usage")
    if not isinstance(usage, dict) or usage.get("is_byok") is True:
        raise PricingUnavailable("native_account_usage_required")
    if response.get("model") != request.get("model"):
        raise PricingUnavailable("charged_model_mismatch")
    source = attempt.metadata.get("cost_source")
    if source == "generation_record":
        generation = attempt.metadata.get("generation_record")
        if (
            not isinstance(generation, dict)
            or not isinstance(response.get("id"), str)
            or generation.get("id") != response["id"]
            or generation.get("model") != response.get("model")
            or generation.get("cancelled") is not False
            or generation.get("is_byok") is not False
        ):
            raise PricingUnavailable("matched_account_generation_charge_required")
        charge = money(generation.get("total_cost"))
        for channel in ("prompt", "completion"):
            if token_count(generation.get("native_tokens_" + channel)) != token_count(
                usage.get(channel + "_tokens")
            ):
                raise PricingUnavailable("conflicting_native_usage_counts")
        if "cost" in usage and money(usage["cost"]) != charge:
            raise PricingUnavailable("conflicting_reported_charges")
        cached = generation.get("native_tokens_cached")
        details = usage.get("prompt_tokens_details", {})
        if (
            cached is not None
            and token_count(cached) > 0
            and (
                not isinstance(details, dict)
                or not (details.get("cached_tokens") or details.get("cache_write_tokens"))
            )
        ):
            raise PricingUnavailable("generation_cache_usage_not_explained")
    elif source == "provider":
        if usage.get("is_byok") is not False:
            raise PricingUnavailable("account_charge_basis_unknown_or_BYOK")
        charge = money(usage.get("cost"))
    else:
        raise PricingUnavailable("estimated_or_unknown_charge_cannot_be_tested")
    if money(attempt.metadata.get("actual_cost_usd")) != charge:
        raise PricingUnavailable("ledger_charge_does_not_match_capture")
    return charge, usage


def cost_observation(record: Exchange) -> tuple[float, dict[str, Any], dict[str, Any]]:
    if record.metadata.get("request_not_sent") or REDACTED.encode() in canonical_json(
        [record.request, [attempt.response for attempt in record.attempts]]
    ):
        raise PricingUnavailable("unredacted_sent_accounting_capture_required")
    if not record.attempts or [attempt.n for attempt in record.attempts] != list(
        range(1, len(record.attempts) + 1)
    ):
        raise PricingUnavailable("complete_attempt_sequence_required")
    expected, charged = Decimal(0), Decimal(0)
    rows, layouts = [], []
    for attempt in record.attempts:
        snapshot = _snapshot(attempt, record.request)
        charge, usage = _charged(attempt, record.request)
        subtotal, counts = expected_token_cost(
            usage, snapshot["pricing"], attempt.timings.start_unix
        )
        expected += subtotal
        charged += charge
        generation = attempt.metadata.get("generation_record") or {}
        rows.append(
            {
                "attempt": attempt.n,
                "expected_usd": str(subtotal),
                "charged_usd": str(charge),
                "price_snapshot_hash": content_hash(snapshot),
                "counts": counts,
                "cost_source": attempt.metadata["cost_source"],
                "reported_cache_discount": generation.get("cache_discount"),
            }
        )
        layouts.append(
            {
                "attempt": attempt.n,
                "status": attempt.status,
                "cost_source": attempt.metadata["cost_source"],
                "currency": "USD",
            }
        )
    if (
        record.cost_usd is None
        or not math.isfinite(record.cost_usd)
        or float(charged) != record.cost_usd
    ):
        raise PricingUnavailable("aggregate_charge_does_not_match_all_attempts")
    if expected <= 0:
        raise PricingUnavailable("positive_list_cost_denominator_required")
    value = float((charged - expected) / expected)
    if not math.isfinite(value):
        raise PricingUnavailable("finite_cost_ratio_required")
    return (
        value,
        {"expected_usd": str(expected), "charged_usd": str(charged), "attempts": rows},
        {
            "request": {k: v for k, v in record.request.items() if k not in {"model", "provider"}},
            "attempts": layouts,
            "projection": "native-total-cache-subsets-v1",
        },
    )


def cost_samples(items: list[Item], records: list[Exchange], endpoint: str) -> dict[str, Samples]:
    contract = cost_contract(items)
    if contract is None:
        return {}
    by_item: dict[str, list[Exchange]] = defaultdict(list)
    for record in records:
        if record.endpoint_id == endpoint:
            by_item[record.item_id].append(record)
    result: dict[str, Samples] = {}
    for item in items:
        group, unit = cost_group(item)
        sample = result.setdefault(
            group, Samples(check_id=group, check_class="comparative", binary=False, two_sided=True)
        )
        sample.details.update(
            accounting_contract=contract,
            units="(sum reported attempt charges - sum captured list-price costs) / sum list costs",
            scope="captured native usage and endpoint price snapshots; not an invoice attestation",
            calibration="matched actual request/attempt layouts; each endpoint uses its own prices",
        )
        captures = sorted(by_item[item.item_id], key=lambda r: r.sample_idx)
        if [r.sample_idx for r in captures] != list(range(int(item.expected.get("samples", 1)))):
            sample.mark("ERROR", "cost accounting capture missing or duplicated")
            continue
        for record in captures:
            sample.evidence.append(record.exchange_id)
            try:
                value, details, layout = cost_observation(record)
                sample.repeats.setdefault(unit, []).append(value)
                sample.values[unit] = math.fsum(sample.repeats[unit]) / len(sample.repeats[unit])
                hashes = sample.details.setdefault("input_hashes", {}).setdefault(unit, [])
                hashed = content_hash([layout, contract, record.sample_idx])
                if hashed not in hashes:
                    hashes.append(hashed)
                    hashes.sort()
                sample.details.setdefault("counts", {}).setdefault(unit, []).append(
                    {"exchange_id": record.exchange_id, **details}
                )
            except PricingUnavailable as error:
                sample.mark("INCONCLUSIVE", "cost evidence unavailable: " + str(error))
            except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
                sample.mark("INCONCLUSIVE", "cost evidence shape or arithmetic invalid")
        if not sample.values and sample.status is None:
            sample.mark("INCONCLUSIVE", "no qualified per-attempt costs")
    return result
