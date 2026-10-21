"""S2 generated corpus, paired structural probes and simulated multi-turn goals."""

from __future__ import annotations

import copy
from typing import Any

from servegauge.core.models import Item
from servegauge.items.agent_tasks import agent_item
from servegauge.items.toolcalls import toolcall_item
from servegauge.suites.config import suite_config


def tool_items(level: str, seed: int, overrides: dict[str, Any] | None = None) -> list[Item]:
    cfg = suite_config()["tool_calling"]
    counts = {**cfg[level], **(overrides or {})}
    if set(counts) != set(cfg[level]) or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in counts.values()
    ):
        raise ValueError("positive_known_S2_design_counts_required")
    corpus = []
    for index in range(counts["items"]):
        item = toolcall_item(
            seed + index,
            max_tools=cfg["max_tools"],
            samples=cfg["samples_per_item"],
            numeric_tolerance=float(cfg["numeric_tolerance"]),
        )
        item.payload["params"]["max_tokens"] = cfg["output_cap"]
        corpus.append(item)
    probes = []
    for probe in (
        "single",
        "parallel",
        "irrelevant",
        "nested",
        "unicode",
        "empty",
        "forced",
        "after_tool",
        "reasoning",
    ):
        for trial in range(counts["structural_repeats"]):
            args: dict[str, Any] = {"value": f"probe-{probe}-{seed}-{trial}"}
            schema: dict[str, Any] = {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            }
            if probe == "nested":
                args = {"value": {"numbers": [1, 2], "flag": True}}
                schema["properties"]["value"] = {
                    "type": "object",
                    "properties": {
                        "numbers": {"type": "array", "items": {"type": "integer"}},
                        "flag": {"type": "boolean"},
                    },
                    "required": ["numbers", "flag"],
                    "additionalProperties": False,
                }
            if probe == "unicode":
                args["value"] = f"東京 café e\u0301 {seed}-{trial}"
            if probe == "empty":
                args, schema = (
                    {},
                    {"type": "object", "properties": {}, "additionalProperties": False},
                )
            tool = {
                "type": "function",
                "function": {
                    "name": "record_value",
                    "description": "Record the provided value in this synthetic example.",
                    "parameters": schema,
                },
            }
            expected = [{"name": "record_value", "arguments": args}]
            if probe == "parallel":
                expected.append({"name": "record_value", "arguments": {"value": "second"}})
            if probe == "irrelevant":
                expected = []
            for stream in (False, True):
                variant = "stream" if stream else "json"
                messages: list[dict[str, Any]] = []
                if probe == "after_tool":
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "prior",
                                        "type": "function",
                                        "function": {
                                            "name": "record_value",
                                            "arguments": '{"value":"prior"}',
                                        },
                                    }
                                ],
                            },
                            {"role": "tool", "tool_call_id": "prior", "content": '{"ok":true}'},
                        ]
                    )
                import json

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Probe {probe}-{seed}-{trial}. Return these calls: "
                            + json.dumps(expected, ensure_ascii=False)
                            if expected
                            else f"Probe {seed}-{trial}. Say hello. No offered tool applies."
                        ),
                    }
                )
                params: dict[str, Any] = {"max_tokens": cfg["output_cap"]}
                if probe == "reasoning":
                    params["reasoning_effort"] = "low"
                probes.append(
                    Item(
                        item_id=f"structure-{seed}-{probe}-{trial}-{variant}",
                        suite_id="S2",
                        version="tool-structure-v1",
                        payload={
                            "messages": messages,
                            "tools": [copy.deepcopy(tool)],
                            "tool_choice": {
                                "type": "function",
                                "function": {"name": "record_value"},
                            }
                            if probe == "forced"
                            else "auto",
                            "stream": stream,
                            "params": params,
                        },
                        expected={
                            "check_id": f"T04.{probe}.{variant}",
                            "probe": probe,
                            "trial": f"{probe}-{trial}",
                            "variant": variant,
                            "calls": expected,
                            "samples": 1,
                            "requires": ["tools", *(["streaming"] if stream else [])],
                            "numeric_tolerance": float(cfg["numeric_tolerance"]),
                            "mock_task": {
                                "type": "tool" if expected else "no_call",
                                "calls": expected,
                            },
                        },
                        tags=["generated", "CC-BY-4.0"],
                    )
                )
    agents = [
        agent_item(
            seed + index,
            pass_reasoning=passback,
            minimum=cfg["agent_min_steps"],
            maximum=cfg["agent_max_steps"],
            cap=cfg["output_cap"],
            turn_headroom=cfg["agent_turn_headroom"],
        )
        for index in range(counts["agent_tasks"])
        for passback in (False, True)
    ]
    return [*corpus, *probes, *agents]
