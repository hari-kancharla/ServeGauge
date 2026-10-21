"""Provider-level pin contradictions from linked generation metadata, not tag claims."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from servegauge.core.models import EndpointSpec, Exchange, Item
from servegauge.core.redaction import REDACTED
from servegauge.scoring.common import Samples

ROUTING_CHECK = "S8.routing.provider"
ROUTING_CONTRACT = {
    "version": "provider-routing-v1",
    "population": "capability-and-primary-initial-procedures",
    "unit": "suite/check/trial; variants and repeated draws clustered",
    "identity": "request-pin-and-generation-id-linked-provider-name",
    "inference": "configured-invariant-tolerance-and-reproduction",
    "tag_verified": False,
    "fallback_qualified": False,
}


def routing_unit(item: Item) -> str:
    return ":".join(
        (
            item.suite_id,
            str(item.expected.get("check_id", item.item_id)),
            str(item.expected.get("trial", item.item_id)),
        )
    )


def _identity(record: Exchange, endpoint: EndpointSpec) -> bool | None:
    """True is a provider contradiction; None is an unobservable identity."""
    pin = endpoint.openrouter_pin
    expected = endpoint.labels.get("provider_name", "").strip()
    provider = record.request.get("provider")
    if (
        pin is None
        or pin.allow_fallbacks
        or not pin.tag.strip()
        or REDACTED in pin.tag
        or not expected
        or REDACTED in expected
        or not isinstance(provider, dict)
        or provider.get("only") != [pin.tag]
        or provider.get("allow_fallbacks") is not False
        or record.request.get("model") != endpoint.model
        or record.status != 200
        or record.error not in {None, "ROUTING_MISMATCH"}
        or not record.attempts
        or [a.n for a in record.attempts] != list(range(1, len(record.attempts) + 1))
        or any(
            record.metadata.get(k)
            for k in (
                "request_not_sent",
                "incomplete",
                "invalid_utf8",
                "invalid_json_events",
                "stream_identity_conflict",
                "sensitive_response_redacted",
            )
        )
    ):
        return None
    final = record.attempts[-1]
    generation = record.metadata.get("generation_record")
    identifier = (record.response or {}).get("id")
    if (
        final.status != 200
        or final.error is not None
        or not isinstance(generation, dict)
        or not isinstance(identifier, str)
        or not re.fullmatch(r"gen-[0-9A-Za-z-]{1,124}", identifier)
        or generation.get("id") != identifier
        or (final.response or {}).get("id") != identifier
        or final.metadata.get("generation_record") != generation
        or record.metadata.get("generation_record_status") != "available"
        or final.metadata.get("generation_record_status") != "available"
    ):
        return None
    actual = generation.get("provider_name")
    if not isinstance(actual, str) or not actual.strip() or REDACTED in actual:
        return None
    return actual.strip().casefold() != expected.casefold()


def routing_samples(
    items: list[Item],
    records: list[Exchange],
    endpoint: EndpointSpec,
    *,
    run_id: str,
    capability_run_id: str | None = None,
    contract: dict[str, Any] | None = None,
) -> Samples | None:
    """A conditional complete-root cohort; missing roots cannot establish a PASS.

    Positive, linked contradictions remain evidence when unrelated roots are
    missing. Their interval describes observed complete roots, not all traffic.
    """
    if contract is None:
        return None
    if contract != ROUTING_CONTRACT:
        raise ValueError("unsupported_routing_contract")
    result = Samples(
        check_id=ROUTING_CHECK,
        check_class="invariant",
        details={
            "routing_contract": dict(contract),
            "independent_unit": contract["unit"],
            "population": "observed complete initial procedures; not all attempts or task turns",
            "independence_assumption": (
                "independent captured roots; temporal dependence unqualified"
            ),
            "tag_verified": False,
            "fallback_qualified": False,
            "operational_alarm": False,
        },
    )
    if endpoint.kind != "openrouter":
        result.mark("NOT_APPLICABLE", "provider identity contract requires OpenRouter metadata")
        return result
    if len({i.item_id for i in items}) != len(items):
        raise ValueError("routing_requires_unique_inventory")
    inventory = {i.item_id: i for i in items}
    by_item: dict[str, list[Exchange]] = defaultdict(list)
    exchange_ids: Counter[str] = Counter()
    generation_ids: Counter[str] = Counter()
    for record in records:
        item = inventory.get(record.item_id)
        if item is None or record.endpoint_id != endpoint.id:
            continue
        expected_run = capability_run_id if item.suite_id == "S0" else run_id
        if record.run_id == (expected_run or run_id):
            by_item[item.item_id].append(record)
            exchange_ids[record.exchange_id] += 1
            identifier = (record.response or {}).get("id")
            if isinstance(identifier, str):
                generation_ids[identifier] += 1
    groups: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        groups[routing_unit(item)].append(item)
    excluded: Counter[str] = Counter()
    opportunities = 0
    unsent = 0
    for unit, roots in sorted(groups.items()):
        adverse = []
        evidence = []
        reason = None
        complete = []
        for item in roots:
            found = sorted(by_item[item.item_id], key=lambda r: r.sample_idx)
            draws = item.expected.get("samples", 1)
            if (
                type(draws) is not int
                or draws < 1
                or [r.sample_idx for r in found] != list(range(draws))
            ):
                reason = "missing_or_duplicate_capture"
                break
            complete.extend(found)
            for record in found:
                if record.metadata.get("request_not_sent"):
                    continue
                identifier = (record.response or {}).get("id")
                if (
                    record.suite_id != item.suite_id
                    or exchange_ids[record.exchange_id] != 1
                    or not isinstance(identifier, str)
                    or generation_ids[identifier] != 1
                ):
                    reason = "ambiguous_or_reused_capture_identity"
                    break
                identity = _identity(record, endpoint)
                if identity is None:
                    reason = "unobservable_or_unbound_provider_identity"
                    break
                adverse.append(identity)
                evidence.append(record.exchange_id)
            if reason:
                break
        if reason:
            excluded[reason] += 1
            continue
        if complete and all(r.metadata.get("request_not_sent") for r in complete):
            unsent += 1
            continue
        if any(r.metadata.get("request_not_sent") for r in complete):
            excluded["partially_unsent_procedure"] += 1
            continue
        result.values[unit] = float(any(adverse))
        result.evidence.extend(evidence)
        opportunities += len(evidence)
        if any(adverse):
            result.replay_candidates.append([i.item_id for i in roots])
    result.evidence = sorted(set(result.evidence))
    result.details.update(
        planned_units=len(groups),
        observable_units=len(result.values),
        excluded_units=sum(excluded.values()),
        unsent_units=unsent,
        excluded_unit_reasons=dict(sorted(excluded.items())),
        observed_requests=opportunities,
        incomplete_trials=bool(excluded),
    )
    if not result.values:
        result.mark("INCONCLUSIVE", "no complete generation-linked provider identity procedures")
    return result
