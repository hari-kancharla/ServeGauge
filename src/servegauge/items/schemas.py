"""Own seeded schema ladder and valid gold objects; no benchmark text."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

LADDER = (
    "flat",
    "nested",
    "arrays",
    "enums",
    "required_optional",
    "additional_properties",
    "numeric_bounds",
    "patterns",
    "any_of",
    "depth6",
    "enum200",
)


@dataclass(frozen=True)
class SchemaCase:
    seed: int
    category: str
    schema: dict[str, Any]
    value: dict[str, Any]
    prompt: str


def object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties) if required is None else required,
        "additionalProperties": False,
    }


def schema_case(seed: int, category: str) -> SchemaCase:
    if category not in LADDER:
        raise ValueError("unknown_schema_ladder_category")
    number = seed % 97
    schema = object_schema({"id": {"type": "integer"}, "label": {"type": "string"}})
    value: dict[str, Any] = {"id": number, "label": f"café 東京 {seed}"}
    if category == "nested":
        schema, value = object_schema({"record": schema}), {"record": value}
    elif category == "arrays":
        schema = object_schema(
            {
                "values": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 3,
                    "maxItems": 3,
                }
            }
        )
        value = {"values": [number, number + 1, number + 2]}
    elif category == "enums":
        schema = object_schema({"priority": {"type": "string", "enum": ["low", "medium", "high"]}})
        value = {"priority": ["low", "medium", "high"][seed % 3]}
    elif category == "required_optional":
        schema = object_schema({"id": {"type": "integer"}, "note": {"type": "string"}}, ["id"])
        value = {"id": number, **({"note": f"note {seed}"} if seed % 2 else {})}
    elif category == "additional_properties":
        schema, value = object_schema({"ok": {"type": "boolean"}}), {"ok": bool(seed % 2)}
    elif category == "numeric_bounds":
        schema = object_schema(
            {
                "number": {"type": "integer", "minimum": 0, "maximum": 100},
                "fraction": {"type": "number", "minimum": -1, "maximum": 1},
            }
        )
        value = {"number": number, "fraction": (number - 48) / 100}
    elif category == "patterns":
        schema = object_schema({"code": {"type": "string", "pattern": "^TW-[0-9]{4}$"}})
        value = {"code": f"TW-{seed % 10000:04}"}
    elif category == "any_of":
        schema = object_schema({"value": {"anyOf": [{"type": "integer"}, {"type": "string"}]}})
        value = {"value": number if seed % 2 else str(number)}
    elif category == "depth6":
        schema, value = object_schema({"value": {"type": "integer"}}), {"value": number}
        for depth in reversed(range(5)):
            schema = object_schema({f"level_{depth}": schema})
            value = {f"level_{depth}": value}
    elif category == "enum200":
        choices = [f"category_{index:03}" for index in range(200)]
        schema = object_schema({"category": {"type": "string", "enum": choices}})
        value = {"category": choices[seed % len(choices)]}
    prompt = (
        f"Schema conversion case {seed}. Return exactly this information as a JSON object: "
        + json.dumps(value, ensure_ascii=False)
        + ". No markdown or commentary."
    )
    return SchemaCase(seed, category, schema, value, prompt)
