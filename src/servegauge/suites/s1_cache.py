"""Versioned repeated/fresh cache probes, separate from frozen protocol settings."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from servegauge.core.hashing import content_hash
from servegauge.core.models import EndpointSpec, Item

VERSION = "cache-observations-v1"
CACHE_CHECKS = ("P03.cache", "P03.cache.fresh")


def cache_items(level: str, seed: int, overrides: dict[str, Any] | None = None) -> list[Item]:
    cfg = json.loads(Path("configs/cache_probes_v1.json").read_text(encoding="utf-8"))
    if level not in {"smoke", "core", "full"} or cfg.get("version") != VERSION:
        raise ValueError("known_cache_design_required")
    design = {**cfg[level], "output_cap": cfg["output_cap"]}
    if overrides and set(overrides) - set(design):
        raise ValueError("known_cache_design_required")
    design.update(overrides or {})
    if (
        any(type(v) is not int or v < 1 for v in design.values())
        or design["draws"] < 2
        or 2 * design["groups"] * design["draws"] > cfg["maximum_requests_per_endpoint"]
    ):
        raise ValueError("bounded_positive_cache_design_required")
    contract = {
        "version": VERSION,
        "groups": design["groups"],
        "draws": design["draws"],
        "metric": "adjacent-visible-completion-change-frequency",
        "unit": "complete-prompt-group",
        "fresh": "unique-prompt-nonce-per-draw",
        "params": {
            "max_tokens": design["output_cap"],
            "temperature": cfg["temperature"],
            "reasoning_effort": cfg["reasoning_effort"],
        },
        "choices": cfg["choices"],
    }
    items = []
    for group in range(design["groups"]):
        key = f"cache-{seed}-{group:06}"
        prompt = (
            f"Observation {key}. Name one ordinary color at random from "
            + ", ".join(cfg["choices"])
            + ". Return only its name."
        )
        for condition in ("repeated", "fresh"):
            for draw in range(1 if condition == "repeated" else design["draws"]):
                fresh = condition == "fresh"
                expected: dict[str, Any] = {
                    "check_id": CACHE_CHECKS[0],
                    "cache_condition": condition,
                    "trial": key,
                    "cache_group": key,
                    "samples": 1 if fresh else design["draws"],
                    "cache_contract": copy.deepcopy(contract),
                    # Stable embedded JSON order survives canonical inventory
                    # seals; these keys are already in sorted order.
                    "mock_task": {"choices": list(cfg["choices"]), "type": "sample"},
                    "requires": ["temperature"],
                }
                if fresh:
                    expected["cache_draw"] = draw
                items.append(
                    Item(
                        item_id=key + (f"-fresh-{draw:04}" if fresh else "-repeated"),
                        suite_id="S1",
                        version=VERSION,
                        payload={
                            "messages": [
                                {
                                    "role": "user",
                                    "content": prompt + (f" Fresh nonce {draw}." if fresh else ""),
                                }
                            ],
                            "params": copy.deepcopy(contract["params"]),
                            "stream": False,
                        },
                        expected=expected,
                        tags=["generated", "paired-cache-observation"],
                    )
                )
    return items


def cache_contract(items: list[Item]) -> dict[str, Any] | None:
    candidates = [i for i in items if i.version == VERSION or "cache_contract" in i.expected]
    if not candidates:
        return None
    first = candidates[0].expected.get("cache_contract")
    if (
        not isinstance(first, dict)
        or set(first)
        != {"version", "groups", "draws", "metric", "unit", "fresh", "params", "choices"}
        or first.get("version") != VERSION
        or type(first.get("groups")) is not int
        or first["groups"] < 1
        or type(first.get("draws")) is not int
        or first["draws"] < 2
        or first.get("metric") != "adjacent-visible-completion-change-frequency"
        or first.get("unit") != "complete-prompt-group"
        or first.get("fresh") != "unique-prompt-nonce-per-draw"
        or not isinstance(first.get("params"), dict)
        or set(first["params"]) != {"max_tokens", "temperature", "reasoning_effort"}
        or type(first["params"].get("max_tokens")) is not int
        or first["params"]["max_tokens"] < 1
        or type(first["params"].get("temperature")) not in {int, float}
        or not math.isfinite(first["params"]["temperature"])
        or first["params"]["temperature"] <= 0
        or not isinstance(first["params"].get("reasoning_effort"), str)
        or first["params"].get("reasoning_effort") not in {"low", "medium", "high"}
        or not isinstance(first.get("choices"), list)
        or len(first["choices"]) < 2
        or any(not isinstance(v, str) or not v for v in first["choices"])
        or len(set(first["choices"])) != len(first["choices"])
        or any(
            i.version != VERSION
            or i.suite_id != "S1"
            or i.expected.get("cache_contract") != first
            or i.expected.get("check_id") != CACHE_CHECKS[0]
            or not isinstance(i.expected.get("cache_condition"), str)
            or i.expected.get("cache_condition") not in {"repeated", "fresh"}
            for i in candidates
        )
    ):
        raise ValueError("unsupported_or_mixed_cache_contract")
    return copy.deepcopy(first)


def cache_check(item: Item) -> str:
    if (
        item.version != VERSION
        or not isinstance(item.expected.get("cache_condition"), str)
        or item.expected.get("cache_condition") not in {"repeated", "fresh"}
    ):
        raise ValueError("unsupported_cache_contract")
    return CACHE_CHECKS[int(item.expected["cache_condition"] == "fresh")]


def request_signature(body: dict[str, Any]) -> str:
    from servegauge.scoring.fingerprint_collection import sampling_request_hash

    return sampling_request_hash(body)


def cache_layout(
    items: list[Item], endpoint: EndpointSpec
) -> dict[str, dict[str, list[tuple[str, int, str]]]]:
    from servegauge.suites.registry import prepare_item

    contract = cache_contract(items)
    if contract is None:
        return {}
    output: dict[str, dict[str, list[tuple[int, str, int, str]]]] = {}
    identifiers = set()
    for item in items:
        if item.version != VERSION:
            continue
        check = cache_check(item)
        group = item.expected.get("cache_group")
        count = item.expected.get("samples")
        if (
            not isinstance(group, str)
            or not group
            or item.expected.get("trial") != group
            or item.item_id in identifiers
            or type(count) is not int
            or count != (contract["draws"] if check == CACHE_CHECKS[0] else 1)
            or content_hash(item.payload.get("params")) != content_hash(contract["params"])
            or item.payload.get("stream") is not False
            or item.expected.get("mock_task") != {"choices": contract["choices"], "type": "sample"}
        ):
            raise ValueError("complete_cache_inventory_required")
        identifiers.add(item.item_id)
        prepared = prepare_item(item, endpoint)
        if prepared.body is None:
            raise LookupError("cache_request_capability_unavailable")
        rows = output.setdefault(check, {}).setdefault(group, [])
        for sample in range(count):
            draw = sample if check == CACHE_CHECKS[0] else item.expected.get("cache_draw")
            if type(draw) is not int or not 0 <= draw < contract["draws"]:
                raise ValueError("complete_cache_inventory_required")
            rows.append((draw, item.item_id, sample, request_signature(prepared.body)))
    result: dict[str, dict[str, list[tuple[str, int, str]]]] = {}
    for check, groups in output.items():
        if len(groups) != contract["groups"]:
            raise ValueError("complete_cache_inventory_required")
        result[check] = {}
        for group, rows in sorted(groups.items()):
            if len(rows) != contract["draws"] or sorted(r[0] for r in rows) != list(
                range(contract["draws"])
            ):
                raise ValueError("complete_cache_inventory_required")
            if len({r[3] for r in rows}) != (1 if check == CACHE_CHECKS[0] else contract["draws"]):
                raise ValueError("complete_cache_inventory_required")
            result[check][group] = [(i, n, h) for _, i, n, h in sorted(rows)]
    return result


def cache_request_signatures(
    items: list[Item], endpoints: list[EndpointSpec]
) -> dict[str, str | None]:
    if cache_contract(items) is None:
        return {}
    present = {cache_check(i) for i in items if i.version == VERSION}
    observed: dict[str, list[str | None]] = {check: [] for check in present}
    for endpoint in endpoints:
        try:
            layout = cache_layout(items, endpoint)
        except LookupError:
            for check in present:
                observed[check].append(None)
            continue
        for check in present:
            observed[check].append(
                content_hash({group: [r[2] for r in rows] for group, rows in layout[check].items()})
            )
    return {
        check: values[0] if values and values[0] is not None and len(set(values)) == 1 else None
        for check, values in observed.items()
    }
