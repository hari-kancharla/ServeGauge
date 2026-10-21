"""Observable, complete-group cache comparisons; never read backend fault flags."""

from __future__ import annotations

from collections import Counter, defaultdict

from servegauge.core.hashing import content_hash
from servegauge.core.models import EndpointSpec, Exchange, Item
from servegauge.core.redaction import REDACTED
from servegauge.scoring.common import Samples, choice, content, message
from servegauge.suites.s1_cache import (
    CACHE_CHECKS,
    VERSION,
    cache_check,
    cache_contract,
    cache_layout,
    request_signature,
)


def cache_samples(
    items: list[Item], records: list[Exchange], endpoint: EndpointSpec
) -> dict[str, Samples]:
    contract = cache_contract(items)
    if contract is None:
        return {}
    present = {cache_check(i) for i in items if i.version == VERSION}
    output = {
        check: Samples(
            check_id=check,
            check_class="comparative",
            binary=False,
            details={
                "cache_contract": contract,
                "metric": "adjacent visible completion-change fraction",
                "independent_unit": "complete prompt group; adjacent draws clustered",
                "independence_assumption": (
                    "independent prompt groups; cross-group dependence unqualified"
                ),
                "planned_units": contract["groups"],
                "draws_per_group": contract["draws"],
                "input_hashes": {},
            },
        )
        for check in CACHE_CHECKS
        if check in present
    }
    try:
        layout = cache_layout(items, endpoint)
    except LookupError:
        for sample in output.values():
            sample.mark("INCONCLUSIVE", "cache request capability unavailable")
        return output
    indexed: dict[tuple[str, int], list[Exchange]] = defaultdict(list)
    by_item: dict[str, list[Exchange]] = defaultdict(list)
    identities: Counter[str] = Counter()
    cache_ids = {i.item_id for i in items if i.version == VERSION}
    for row in records:
        if row.endpoint_id == endpoint.id:
            indexed[(row.item_id, row.sample_idx)].append(row)
            by_item[row.item_id].append(row)
            if row.item_id in cache_ids:
                identities[row.exchange_id] += 1
    for check, groups in layout.items():
        sample = output[check]
        used = set()
        hashes = {}
        for group, expected in groups.items():
            texts = []
            group_hashes = []
            for item_id, index, digest in expected:
                rows = indexed.get((item_id, index), [])
                expected_rows = sum(r[0] == item_id for r in expected)
                if len(rows) != 1 or len(by_item[item_id]) != expected_rows:
                    sample.mark("INCONCLUSIVE", "complete unique cache captures required")
                    continue
                row = rows[0]
                sample.evidence.append(row.exchange_id)
                try:
                    text = content(row)
                    msg = message(row)
                    attempt = row.attempts[-1] if row.attempts else None
                    final_choice = (attempt.response or {}).get("choices", [])[0] if attempt else {}
                    if not isinstance(final_choice, dict):
                        raise LookupError
                    final_message = final_choice.get("message", {})
                    if not isinstance(final_message, dict):
                        raise LookupError
                    if (
                        row.exchange_id in used
                        or identities[row.exchange_id] != 1
                        or row.status != 200
                        or row.error
                        or row.metadata.get("request_not_sent")
                        or not row.attempts
                        or row.attempts[-1].status != 200
                        or row.attempts[-1].error
                        or final_message.get("content") != text
                        or final_choice.get("finish_reason") != "stop"
                        or request_signature(row.request) != digest
                        or not text.strip()
                        or REDACTED in text
                        or REDACTED in str(row.request)
                        or (msg.get("refusal") is not None and msg.get("refusal") != "")
                        or msg.get("tool_calls")
                        or choice(row).get("finish_reason") != "stop"
                    ):
                        raise LookupError
                    used.add(row.exchange_id)
                    texts.append(text)
                    group_hashes.append(digest)
                except (LookupError, ValueError, TypeError, KeyError, IndexError):
                    sample.mark(
                        "INCONCLUSIVE", "complete observable request-bound cache cohort required"
                    )
            if len(texts) == contract["draws"]:
                sample.values[group] = sum(
                    a != b for a, b in zip(texts[:-1], texts[1:], strict=True)
                ) / (len(texts) - 1)
                hashes[group] = group_hashes
        if sample.status is not None or len(sample.values) != contract["groups"]:
            sample.mark("INCONCLUSIVE", "complete observable request-bound cache cohort required")
            sample.values.clear()
        else:
            sample.details["input_hashes"] = hashes
            sample.details["request_signature"] = content_hash(hashes)
        sample.evidence = sorted(set(sample.evidence))
    return output
