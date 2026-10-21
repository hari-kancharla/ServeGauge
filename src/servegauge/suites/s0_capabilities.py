"""Declared and observed capabilities before behavioral protocol measurements."""

from __future__ import annotations

from typing import Any

from servegauge.core.models import Item
from servegauge.suites.config import suite_config
from servegauge.suites.s1_protocol import probe, tool


def capability_items() -> list[Item]:
    cap = suite_config()["protocol"]["ordinary_output_cap"]
    common = {"max_tokens": cap}
    items = [
        probe(
            "S0.answers",
            0,
            "plain",
            "Reply with exactly OK.",
            {"type": "echo", "text": "OK"},
            params=common,
        )
    ]
    specifications: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
        ("streaming", {"type": "echo", "text": "OK"}, {"stream": True}),
        (
            "tools",
            {"type": "tool", "name": "convert", "arguments": {"value": 3, "unit": "m"}},
            {"tools": [tool()]},
        ),
        (
            "parallel_tool_calls",
            {
                "type": "tool",
                "calls": [
                    {"name": "convert", "arguments": {"value": i, "unit": "m"}} for i in (1, 2)
                ],
            },
            {"tools": [tool()], "params": {**common, "parallel_tool_calls": True}},
        ),
        (
            "json_object",
            {"type": "json", "value": {"ok": True}},
            {"response_format": {"type": "json_object"}},
        ),
        (
            "json_schema",
            {"type": "json", "value": {"ok": True}},
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ok",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"ok": {"type": "boolean"}},
                            "required": ["ok"],
                            "additionalProperties": False,
                        },
                    },
                }
            },
        ),
        (
            "logprobs",
            {"type": "echo", "text": "OK"},
            {"params": {**common, "logprobs": True, "top_logprobs": 1}},
        ),
        ("seed", {"type": "sample"}, {"params": {**common, "seed": 42}}),
        ("temperature", {"type": "sample"}, {"params": {**common, "temperature": 1.0}}),
        ("top_p", {"type": "sample"}, {"params": {**common, "top_p": 0.5}}),
        ("top_k", {"type": "sample"}, {"params": {**common, "top_k": 2}}),
        (
            "stop",
            {"type": "echo", "text": "alpha TW_STOP omega"},
            {"params": {**common, "stop": "TW_STOP"}},
        ),
        (
            "reasoning",
            {"type": "math", "a": 17, "b": 5},
            {"params": {"max_tokens": 2048, "reasoning_effort": "low"}},
        ),
    ]
    for mode in ("none", "required", "auto", "named"):
        choice: str | dict[str, Any] = (
            {"type": "function", "function": {"name": "convert"}} if mode == "named" else mode
        )
        specifications.append(
            (
                f"tool_choice.{mode}",
                {"type": "tool", "arguments": {"value": 3, "unit": "m"}},
                {"tools": [tool()], "tool_choice": choice},
            )
        )
    for name, task, kwargs in specifications:
        actual = {"params": common, **kwargs}
        items.append(
            probe(
                "S0." + name,
                0,
                "capability",
                "Use a conversion tool if offered. Return a JSON object with ok true "
                "when requested; otherwise reply OK.",
                task,
                extra={"allowed_statuses": [200, 400, 404, 422, 501], "capability": name},
                **actual,
            )
        )
    return items
