"""Tool schemas and exact call targets; no external benchmark text."""

from __future__ import annotations

import copy
import json
import random
from typing import Any

from servegauge.core.hashing import content_hash
from servegauge.core.models import Item

VERSION = "toolcalls-v1"
DOMAINS = ("weather", "calendar", "units", "finance", "files", "database", "search", "ticketing")


def _object(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties) if required is None else required,
        "additionalProperties": False,
    }


def domain_case(domain: str, seed: int) -> tuple[str, dict[str, Any], dict[str, Any]]:
    cities = ["Zürich", "北京", "Montréal", "São Paulo"]
    city = cities[seed % len(cities)]
    if domain == "weather":
        return (
            "weather_lookup",
            _object(
                {
                    "city": {"type": "string"},
                    "days": {"type": "integer", "minimum": 1, "maximum": 7},
                }
            ),
            {"city": city, "days": seed % 7 + 1},
        )
    if domain == "calendar":
        return (
            "schedule_event",
            _object(
                {
                    "title": {"type": "string"},
                    "date": {"type": "string", "pattern": r"^2030-03-\d{2}$"},
                    "attendees": {"type": "array", "items": {"type": "string"}},
                    "note": {"type": "string"},
                },
                ["title", "date", "attendees"],
            ),
            {
                "title": f"Planning {city}",
                "date": f"2030-03-{seed % 28 + 1:02}",
                "attendees": ["person-a", "person-b"],
            },
        )
    if domain == "units":
        return (
            "convert_units",
            _object(
                {
                    "value": {"type": "number", "minimum": -10000, "maximum": 10000},
                    "from_unit": {"type": "string", "enum": ["m", "km", "cm"]},
                    "to_unit": {"type": "string", "enum": ["m", "km", "cm"]},
                }
            ),
            {"value": (seed % 10000) / 100, "from_unit": "km", "to_unit": "m"},
        )
    if domain == "finance":
        return (
            "invoice_total",
            _object(
                {
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "items": _object(
                            {
                                "quantity": {"type": "integer", "minimum": 1, "maximum": 100},
                                "price": {"type": "number", "minimum": 0, "maximum": 10000},
                            }
                        ),
                    },
                    "currency": {"type": "string", "enum": ["USD", "EUR", "INR"]},
                }
            ),
            {
                "items": [{"quantity": seed % 9 + 1, "price": (seed % 9999 + 1) / 100}],
                "currency": "EUR",
            },
        )
    if domain == "files":
        return (
            "write_document",
            _object(
                {
                    "path": {"type": "string"},
                    "document": _object(
                        {
                            "label": {"type": "string"},
                            "count": {"type": "integer"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        }
                    ),
                    "overwrite": {"type": "boolean"},
                },
                ["path", "document"],
            ),
            {
                "path": f"notes/item-{seed % 1000}.json",
                "document": {"label": city, "count": seed % 100, "tags": ["synthetic", "café"]},
                **({"overwrite": False} if seed % 2 else {}),
            },
        )
    if domain == "database":
        return (
            "lookup_records",
            _object(
                {
                    "table": {"type": "string", "enum": ["inventory", "orders"]},
                    "filter": _object(
                        {
                            "field": {"type": "string", "enum": ["city", "name"]},
                            "values": {"type": "array", "items": {"type": "string"}},
                        }
                    ),
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                }
            ),
            {
                "table": "orders",
                "filter": {"field": "city", "values": [city]},
                "limit": seed % 20 + 1,
            },
        )
    if domain == "search":
        return (
            "search_catalog",
            _object(
                {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                    "locale": {"type": "string", "enum": ["en", "fr", "zh"]},
                }
            ),
            {"query": f"guide to {city}", "limit": seed % 10 + 1, "locale": "en"},
        )
    if domain == "ticketing":
        return (
            "create_ticket",
            _object(
                {
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                    "summary": {"type": "string"},
                    "labels": {"type": "array", "items": {"type": "string"}},
                }
            ),
            {
                "priority": ["low", "medium", "high"][seed % 3],
                "summary": f"Synthetic request {city}",
                "labels": ["generated", str(seed % 10)],
            },
        )
    raise ValueError("unknown_synthetic_tool_domain")


def toolcall_item(seed: int, *, max_tools: int, samples: int, numeric_tolerance: float) -> Item:
    if not 1 <= max_tools <= 30 or samples < 1 or numeric_tolerance < 0:
        raise ValueError("valid_corpus_bounds_required")
    rng = random.Random(seed)
    domain = DOMAINS[seed % len(DOMAINS)]
    base, schema, arguments = domain_case(domain, seed)
    count = rng.randint(1, max_tools)
    name = f"{base}_{content_hash(seed)[:8]}"
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": name if index == 0 else f"{name}_alt_{index}",
                "description": (
                    f"Synthetic {domain} account {index}. "
                    "Use the named account requested by the user."
                ),
                "parameters": copy.deepcopy(schema),
            },
        }
        for index in range(count)
    ]
    should_call = seed % 5 != 0
    expected = [{"name": name, "arguments": arguments}] if should_call else []
    if should_call and seed % 13 == 0 and count > 1:
        second = copy.deepcopy(arguments)
        expected.append({"name": tools[1]["function"]["name"], "arguments": second})
    history = should_call and seed % 3 == 0
    messages: list[dict[str, Any]] = []
    if history:
        messages.extend(
            [
                {"role": "user", "content": "The earlier synthetic request has already completed."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_prior",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_prior", "content": '{"ok":true}'},
            ]
        )
    prompt = (
        f"Case {seed}. Use these exact named synthetic accounts, once each, with these arguments: "
        + json.dumps(expected, ensure_ascii=False)
        + ". Preserve argument types and values. Return the calls together."
        if should_call
        else (
            f"Case {seed}. Define the word hello briefly in ordinary text. "
            "None of the offered tools applies."
        )
    )
    messages.append({"role": "user", "content": prompt})
    return Item(
        item_id=f"tool-{seed}-{content_hash([domain, expected])[:8]}",
        suite_id="S2",
        version=VERSION,
        payload={
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "stream": seed % 2 == 0,
            "params": {},
        },
        expected={
            "check_id": "tools-corpus",
            "trial": str(seed),
            "samples": samples,
            "should_call": should_call,
            "calls": expected,
            "matching_rule": "unordered_calls",
            "numeric_tolerance": numeric_tolerance,
            "history": history,
            "mock_task": {"type": "tool" if should_call else "no_call", "calls": expected},
            "requires": ["tools"],
        },
        tags=[domain, "history" if history else "single_turn"],
    )
