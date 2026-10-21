"""Own seeded multi-step tasks and pure, bounded simulated tool implementations."""

from __future__ import annotations

import copy
import json
import random
from typing import Any

from jsonschema import Draft202012Validator

from servegauge.core.models import Item


def _schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Compute and store a named arithmetic result in the simulated ledger.",
            "parameters": _schema(
                {
                    "operation": {"enum": ["add", "multiply"]},
                    "a": {"type": "integer", "minimum": -1000, "maximum": 1000},
                    "b": {"type": "integer", "minimum": -1000, "maximum": 1000},
                    "store": {"type": "string", "maxLength": 80},
                }
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "key_value",
            "description": "Set one key in the simulated key-value store.",
            "parameters": _schema(
                {
                    "key": {"type": "string", "maxLength": 80},
                    "value": {"type": "string", "maxLength": 80},
                }
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar",
            "description": "Create or replace one event in a simulated calendar.",
            "parameters": _schema(
                {
                    "id": {"type": "string", "maxLength": 80},
                    "title": {"type": "string", "maxLength": 80},
                    "date": {"type": "string", "pattern": "^2030-03-[0-9]{2}$"},
                }
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inventory",
            "description": "Adjust one simulated SKU by an integer delta.",
            "parameters": _schema(
                {
                    "sku": {"type": "string", "maxLength": 80},
                    "delta": {"type": "integer", "minimum": -1000, "maximum": 1000},
                }
            ),
        },
    },
]


def execute_tool(state: dict[str, Any], name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Return new simulated state; no eval, I/O, clock or external account exists."""
    offered = {tool["function"]["name"]: tool["function"]["parameters"] for tool in TOOLS}
    if name not in offered:
        raise ValueError("unknown_simulated_tool")
    Draft202012Validator(offered[name]).validate(arguments)
    output = copy.deepcopy(state)
    if name == "calculator":
        a, b = arguments["a"], arguments["b"]
        output[name][arguments["store"]] = a + b if arguments["operation"] == "add" else a * b
    elif name == "key_value":
        output[name][arguments["key"]] = arguments["value"]
    elif name == "calendar":
        output[name][arguments["id"]] = {k: arguments[k] for k in ("title", "date")}
    else:
        sku = arguments["sku"]
        output[name][sku] = output[name].get(sku, 0) + arguments["delta"]
    return output


def agent_item(
    seed: int, *, pass_reasoning: bool, minimum: int, maximum: int, cap: int, turn_headroom: int
) -> Item:
    rng = random.Random(seed)
    state: dict[str, Any] = {"calculator": {}, "key_value": {}, "calendar": {}, "inventory": {}}
    steps: list[dict[str, Any]] = []
    for index in range(rng.randint(minimum, maximum)):
        kind = (seed + index) % 4
        key = f"task_{seed}_{index}"
        if kind == 0:
            name, args = (
                "calculator",
                {
                    "operation": rng.choice(["add", "multiply"]),
                    "a": rng.randint(-20, 20),
                    "b": rng.randint(1, 20),
                    "store": key,
                },
            )
        elif kind == 1:
            name, args = "key_value", {"key": key, "value": f"value_{rng.randrange(10000)}"}
        elif kind == 2:
            name, args = (
                "calendar",
                {"id": key, "title": f"Event {index}", "date": f"2030-03-{rng.randint(1, 28):02}"},
            )
        else:
            name, args = "inventory", {"sku": key, "delta": rng.randint(1, 9)}
        steps.append({"name": name, "arguments": args})
    goal = state
    for step in steps:
        goal = execute_tool(goal, step["name"], step["arguments"])
    variant = "passback" if pass_reasoning else "no_passback"
    instructions = "\n".join(
        f"{index + 1}. Use {step['name']} with {json.dumps(step['arguments'], ensure_ascii=False)}"
        for index, step in enumerate(steps)
    )
    return Item(
        item_id=f"agent-{seed}-{variant}",
        suite_id="S2",
        version="agent-v1",
        payload={
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Task {seed}. All simulated stores start empty. "
                        "Carry out these updates in order. Use tools and inspect their results. "
                        "Do not claim an update without calling its tool.\n" + instructions
                    ),
                }
            ],
            "tools": copy.deepcopy(TOOLS),
            "tool_choice": "auto",
            "stream": False,
            "params": {"max_tokens": cap},
        },
        expected={
            "check_id": f"T07.{variant}",
            "trial": str(seed),
            "variant": variant,
            "samples": 1,
            "requires": ["tools"],
            "agent_initial": state,
            "agent_goal": goal,
            "max_turns": len(steps) + turn_headroom,
            "pass_reasoning": pass_reasoning,
            "mock_task": {"type": "agent", "steps": steps},
        },
        tags=["generated", "simulated-tools", "CC-BY-4.0"],
    )
