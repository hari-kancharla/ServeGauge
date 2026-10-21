"""Versioned client-observed operational probes and simulated actions; no alarms."""

import copy
import json
from pathlib import Path
from typing import Any

from servegauge.core.models import Item
from servegauge.items.agent_tasks import agent_item

VERSION = "client-operations-v1"
HEALTH_CHECKS = ("S8.health.requests", "S8.health.attempts", "S8.health.nonempty")
HEALTH_CONTRACT = {
    "version": "operational-health-v1",
    "units": "ordinary-suite-trial-procedure",
    "margin_source": "comparative_defaults",
    "pairing": "exact-initial-requests-and-protocol",
    "outcome": "all-observed-opportunities-successful",
}
OPERATIONAL_CHECKS = (
    "S8.ttft_ms",
    "S8.first_body_ms",
    "S8.output_tokens_per_second",
    "S8.delivery_tokens_per_second",
    "S8.request_error_rate",
    "S8.attempt_error_rate",
    "S8.request_timeout_rate",
    "S8.attempt_timeout_rate",
    "S8.empty_response_rate",
    "S8.completion_tokens_per_action",
    "S8.agent_task_success_rate",
)
AGENT_CHECKS = tuple(c for c in OPERATIONAL_CHECKS if c.endswith(("_rate", "_action")))
STATIC_CHECKS = tuple(
    c
    for c in OPERATIONAL_CHECKS
    if c not in {"S8.completion_tokens_per_action", "S8.agent_task_success_rate"}
)


def operational_unit(item: Item) -> str:
    return item.suite_id + ":" + str(item.expected.get("trial", item.item_id))


def operational_suffix(item: Item) -> str:
    negative = item.expected.get("check_id") == "P12" or (
        item.expected.get("check_id") == "P13" and item.expected.get("variant") == "required"
    )
    return ".negative_probe" if negative else ""


def operational_items(level: str, seed: int, overrides: dict[str, Any] | None = None) -> list[Item]:
    cfg = json.loads(Path("configs/operations_v1.json").read_text(encoding="utf-8"))
    health = json.loads(Path("configs/operations_health_v1.json").read_text(encoding="utf-8"))
    if health != HEALTH_CONTRACT:
        raise ValueError("versioned_operational_health_design_required")
    if (
        not isinstance(cfg, dict)
        or cfg.get("version") != VERSION
        or set(cfg)
        != {
            "version",
            "smoke",
            "core",
            "full",
            "output_cap",
            "agent_output_cap",
            "agent_steps",
            "agent_turn_headroom",
        }
    ):
        raise ValueError("versioned_operational_design_required")
    for key in ("output_cap", "agent_output_cap", "agent_steps", "agent_turn_headroom"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError("positive_operational_design_required")
    counts = {**cfg[level], **(overrides or {})}
    if (
        set(counts) != {"items", "agent_tasks"}
        or any(type(v) is not int for v in counts.values())
        or counts["items"] < 1
        or counts["agent_tasks"] < 0
    ):
        raise ValueError("bounded_operational_item_and_agent_counts_required")
    items = []
    for index in range(counts["items"]):
        for stream in (False, True):
            items.append(
                Item(
                    item_id=f"ops-{seed + index}-{'stream' if stream else 'json'}",
                    suite_id="S8",
                    version=VERSION,
                    payload={
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Synthetic client timing case {seed + index}. "
                                "Reply only: one two three four",
                            }
                        ],
                        "stream": stream,
                        "params": {"max_tokens": cfg["output_cap"]},
                    },
                    expected={
                        "check_id": "S8.transport",
                        "trial": str(seed + index),
                        "samples": 1,
                        "operations_contract": copy.deepcopy(cfg),
                        "operational_health_contract": copy.deepcopy(health),
                        "mock_task": {"type": "echo", "text": "one two three four"},
                    },
                    tags=["generated", "client-observed-operations"],
                )
            )
    for index in range(counts["agent_tasks"]):
        item = agent_item(
            seed + 100000 + index,
            pass_reasoning=True,
            minimum=cfg["agent_steps"],
            maximum=cfg["agent_steps"],
            cap=cfg["agent_output_cap"],
            turn_headroom=cfg["agent_turn_headroom"],
        )
        item.item_id = "ops-" + item.item_id
        item.suite_id = "S8"
        item.version = VERSION
        item.expected.update(
            check_id="S8.agent_task",
            operations_contract=copy.deepcopy(cfg),
            operational_health_contract=copy.deepcopy(health),
        )
        items.append(item)
    return items


def operations_contract(items: list[Item]) -> dict[str, Any] | None:
    found = [i.expected.get("operations_contract") for i in items if i.version == VERSION]
    if not found:
        return None
    first = found[0]
    if (
        not isinstance(first, dict)
        or first.get("version") != VERSION
        or any(c != first for c in found)
    ):
        raise ValueError("unsupported_or_mixed_operations_contract")
    return dict(first)


def operational_health_contract(items: list[Item]) -> dict[str, Any] | None:
    found = [i.expected.get("operational_health_contract") for i in items if i.suite_id == "S8"]
    if not found or all(c is None for c in found):
        return None
    if any(c != HEALTH_CONTRACT for c in found):
        raise ValueError("unsupported_or_mixed_operational_health_contract")
    return dict(HEALTH_CONTRACT)
