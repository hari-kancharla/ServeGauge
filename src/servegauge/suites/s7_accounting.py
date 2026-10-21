"""Frozen B01 inventory and separately versioned completion, cost and tokenizer probes."""

import copy
import json
from pathlib import Path
from typing import Any

from servegauge.core.models import Item
from servegauge.suites.config import suite_config


def accounting_items(level: str, seed: int, overrides: dict[str, Any] | None = None) -> list[Item]:
    cfg = suite_config()["accounting"]
    design = {**cfg[level], **(overrides or {})}
    count = design.get("items")
    if (
        set(design) != {"items"}
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count < 1
    ):
        raise ValueError("positive_accounting_item_count_required")
    texts = (
        "café 北京 東京",
        "العربية हिन्दी",
        "tabs\tand spaces",
        "🙂 雨",
        "literal <|start|> marker",
        "1234567890 arithmetic",
    )
    prompts = [
        Item(
            item_id=f"accounting-{seed + index}",
            suite_id="S7",
            version="prompt-count-v1",
            payload={
                "messages": [
                    {
                        "role": "user",
                        "content": f"Synthetic tokenizer case {seed + index}. "
                        + (texts[index % len(texts)] + " ") * (index % 19 + 1)
                        + "Reply only OK.",
                    }
                ],
                "params": {"max_tokens": cfg["output_cap"]},
                "stream": False,
            },
            expected={
                "check_id": "B01",
                "trial": str(seed + index),
                "samples": 1,
                "mock_task": {"type": "echo", "text": "OK"},
            },
            tags=["generated", "tokenizer-sensitive"],
        )
        for index in range(count)
    ]
    contract = json.loads(Path("configs/accounting_completion_v1.json").read_text(encoding="utf-8"))
    if (
        contract.get("version") != "completion-count-v1"
        or contract.get("counter_version") != "native-text-channels-v1"
        or contract.get("generation_version") != "generation-field-pairs-v1"
        or not isinstance(contract.get("output_cap"), int)
        or isinstance(contract["output_cap"], bool)
        or contract["output_cap"] < 1
        or contract.get("reasoning_effort") not in {"low", "medium", "high"}
        or set(contract)
        != {"version", "counter_version", "generation_version", "output_cap", "reasoning_effort"}
    ):
        raise ValueError("supported_completion_accounting_contract_required")
    cost = json.loads(Path("configs/accounting_cost_v1.json").read_text(encoding="utf-8"))
    if cost != {"version": "cost-accounting-v1", "counter_version": "per-attempt-native-cost-v1"}:
        raise ValueError("supported_cost_accounting_contract_required")
    hint = json.loads(Path("configs/accounting_tokenizer_v1.json").read_text(encoding="utf-8"))
    if hint != TOKENIZER_CONTRACT:
        raise ValueError("supported_tokenizer_hint_contract_required")
    return [
        *prompts,
        *[
            Item(
                item_id=f"completion-accounting-{seed + index}",
                suite_id="S7",
                version=contract["version"],
                payload={
                    "messages": [
                        {
                            "role": "user",
                            "content": "Copy this synthetic text exactly, "
                            "without quotation marks or explanation: "
                            f"case {seed + index}: "
                            + (texts[index % len(texts)] + " ")
                            * (index % 19 + 1),
                        }
                    ],
                    "params": {
                        "max_tokens": contract["output_cap"],
                        "reasoning_effort": contract["reasoning_effort"],
                    },
                    "stream": False,
                },
                expected={
                    "check_id": "B02",
                    "trial": str(seed + index),
                    "samples": 1,
                    "accounting_contract": contract,
                    "mock_task": {
                        "type": "echo",
                        "text": f"case {seed + index}: "
                        + (texts[index % len(texts)] + " ") * (index % 19 + 1),
                    },
                },
                tags=["generated", "tokenizer-sensitive", "completion-accounting"],
            )
            for index in range(count)
        ],
        *[
            item.model_copy(
                deep=True,
                update={
                    "item_id": f"cost-accounting-{seed + index}",
                    "version": cost["version"],
                    "payload": {
                        **item.payload,
                        "messages": [
                            {
                                **item.payload["messages"][0],
                                "content": "Cost accounting. "
                                + item.payload["messages"][0]["content"],
                            }
                        ],
                    },
                    "expected": {**item.expected, "check_id": "B04", "cost_contract": cost},
                },
            )
            for index, item in enumerate(prompts)
        ],
        *[
            Item(
                item_id=f"tokenizer-hint-{seed + index}",
                suite_id="S7",
                version=hint["version"],
                payload={
                    **copy.deepcopy(item.payload),
                    "messages": [
                        {
                            "role": "user",
                            "content": "Tokenizer-sensitive count probe. "
                            + item.payload["messages"][0]["content"],
                        }
                    ],
                },
                expected={
                    **copy.deepcopy(item.expected),
                    "check_id": "B05",
                    "tokenizer_contract": dict(hint),
                },
                tags=["generated", "tokenizer-sensitive", "claimed-counter-hint"],
            )
            for index, item in enumerate(prompts)
        ],
    ]


def billing_group(item: Item) -> tuple[str, str]:
    check = item.expected.get("check_id", item.suite_id)
    if check == "context":
        return f"B01.context.{item.expected['length']}", item.item_id
    if check == "L06":
        return "B01.prefix", "prefix-" + item.expected["trial"]
    if check == "B01":
        return "B01", item.item_id
    if check == "P09":
        check += "." + item.expected["variant"]
    return "B01." + check, item.item_id


def extended_groups(item: Item) -> list[tuple[str, str]]:
    """Static item units, preserving matched fresh/warm prefix pairs."""
    prompt, unit = billing_group(item)
    suffix = "" if item.expected.get("check_id") == "B02" else prompt.removeprefix("B01") or ".B01"
    return [(root + suffix, unit) for root in ("B02", "B03.prompt", "B03.completion")]


def extended_contract(items: list[Item]) -> dict[str, Any] | None:
    """Read the sealed item contract, never today's config when reconstructing."""
    contracts = [
        item.expected["accounting_contract"]
        for item in items
        if item.version == "completion-count-v1" and "accounting_contract" in item.expected
    ]
    if not contracts:
        return None
    first = contracts[0]
    if (
        not isinstance(first, dict)
        or first.get("version") != "completion-count-v1"
        or first.get("counter_version") != "native-text-channels-v1"
        or first.get("generation_version") != "generation-field-pairs-v1"
        or any(contract != first for contract in contracts)
    ):
        raise ValueError("unsupported_or_mixed_accounting_contract")
    return dict(first)


def cost_group(item: Item) -> tuple[str, str]:
    prompt, unit = billing_group(item)
    return (
        "B04"
        if item.expected.get("check_id") == "B04"
        else "B04" + (prompt.removeprefix("B01") or ".B01")
    ), unit


def cost_contract(items: list[Item]) -> dict[str, Any] | None:
    contracts = [
        item.expected.get("cost_contract") for item in items if item.version == "cost-accounting-v1"
    ]
    if not contracts:
        return None
    required = {"version": "cost-accounting-v1", "counter_version": "per-attempt-native-cost-v1"}
    if any(contract != required for contract in contracts):
        raise ValueError("unsupported_or_mixed_cost_accounting_contract")
    return required


TOKENIZER_CONTRACT = {
    "version": "claimed-tokenizer-v1",
    "projection": "native-prompt-over-claimed-template-v1",
    "counter_version": "hf-api-adapter-v1",
    "counter_date": "2026-10-18",
}


def tokenizer_contract(items: list[Item]) -> dict[str, Any] | None:
    hints = [item for item in items if item.version == TOKENIZER_CONTRACT["version"]]
    if not hints:
        return None
    if any(item.expected.get("tokenizer_contract") != TOKENIZER_CONTRACT for item in hints):
        raise ValueError("unsupported_or_mixed_tokenizer_hint_contract")
    return dict(TOKENIZER_CONTRACT)
