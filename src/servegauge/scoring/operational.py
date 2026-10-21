"""Operational and cache observations from captures, without inspecting fault labels."""

from __future__ import annotations

from collections import defaultdict

from servegauge.core.completion import completion_error
from servegauge.core.models import Exchange, Item
from servegauge.scoring.common import Samples, content


def cache_samples(items: list[Item], records: list[Exchange], endpoint: str) -> Samples:
    sample = Samples(check_id="P03.cache", check_class="comparative", binary=False)
    by_item: dict[str, list[Exchange]] = defaultdict(list)
    for record in records:
        if record.endpoint_id == endpoint:
            by_item[record.item_id].append(record)
    for item in items:
        if item.expected.get("check_id") != "P03.cache":
            continue
        rows = sorted(by_item[item.item_id], key=lambda r: r.sample_idx)
        sample.evidence.extend(r.exchange_id for r in rows)
        if len(rows) != item.expected["samples"] or any(r.error or r.status != 200 for r in rows):
            sample.mark("ERROR", "incomplete cache observations")
            continue
        values = [content(r) for r in rows]
        sample.values[item.item_id] = sum(
            a != b for a, b in zip(values[:-1], values[1:], strict=True)
        ) / (len(values) - 1)
    sample.details["independent_unit"] = "prompt; adjacent draws remain clustered"
    return sample


def operational_samples(records: list[Exchange], endpoint: str) -> Samples:
    sample = Samples(check_id="S8.success", check_class="comparative")
    for record in records:
        if record.endpoint_id != endpoint or record.metadata.get("request_not_sent"):
            continue
        sample.evidence.append(record.exchange_id)
        sample.values[record.item_id + ":" + str(record.sample_idx)] = float(
            not record.error and record.status == 200 and not completion_error(record.response)
        )
    sample.details["metric"] = "completed request fraction; operational signal, not model quality"
    return sample
