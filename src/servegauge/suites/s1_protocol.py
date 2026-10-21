"""Natural-language protocol probes; the mock interpreter sees only explicit mock tasks."""

from __future__ import annotations

import random
from typing import Any

from servegauge.core.models import EndpointSpec, Item, ModelProfile
from servegauge.items.mathgen import math_problem
from servegauge.suites.config import suite_config
from servegauge.tokencount.counters import text_tokens
from servegauge.tokencount.model_text import profile_codec

VERSION = "protocol-v2"


def tool(name: str = "convert") -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Return the value converted to the requested unit.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "number"}, "unit": {"type": "string"}},
                "required": ["value", "unit"],
                "additionalProperties": False,
            },
        },
    }


def probe(
    check: str,
    index: int,
    variant: str,
    prompt: str,
    task: dict[str, Any],
    *,
    params: dict[str, Any] | None = None,
    samples: int = 1,
    stream: bool = False,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    response_format: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Item:
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "params": params or {},
        "stream": stream,
    }
    for name, value in (
        ("tools", tools),
        ("tool_choice", tool_choice),
        ("response_format", response_format),
    ):
        if value is not None:
            payload[name] = value
    return Item(
        item_id=f"{check}-{index:04}-{variant}",
        suite_id="S0" if check.startswith("S0.") else "S1",
        version=VERSION,
        payload=payload,
        expected={
            "check_id": check,
            "trial": index,
            "variant": variant,
            "samples": samples,
            "mock_task": task,
            **(extra or {}),
        },
    )


def protocol_items(
    level: str,
    seed: int,
    *,
    endpoint: EndpointSpec | None = None,
    profile: ModelProfile | None = None,
    overrides: dict[str, Any] | None = None,
) -> list[Item]:
    if profile is None and endpoint is not None and endpoint.claimed_profile == "glm-5.3":
        raise ValueError("glm_protocol_requires_verified_profile")
    codec = profile_codec(profile)
    cfg = suite_config()["protocol"]
    sizes = {**cfg[level], **(overrides or {})}
    rng = random.Random(seed)
    items: list[Item] = []
    cap = cfg["cap_tokens"]
    text = " token" * cfg["cap_text_repetitions"]
    for index in range(sizes["invariant_trials"]):
        items.append(
            probe(
                "P01",
                index,
                "cap",
                f"Trial {index}. Copy the following text exactly, without explanation:{text}",
                {"type": "echo", "text": text},
                params={"max_tokens": cap},
                extra={"full_text": text},
            )
        )
        stop = cfg["stop_string"]
        items.append(
            probe(
                "P02",
                index,
                "stop",
                f"Copy exactly: alpha {stop} omega. Case {index}.",
                {"type": "echo", "text": f"alpha {stop} omega"},
                params={"max_tokens": cfg["ordinary_output_cap"], "stop": stop},
                extra={"stop": stop, "requires": ["stop"]},
            )
        )
        for mode in ("none", "required", "named"):
            choice = (
                {"type": "function", "function": {"name": "convert"}} if mode == "named" else mode
            )
            items.append(
                probe(
                    "P09",
                    index,
                    mode,
                    f"Convert {index + 1} to unit m using the offered tool if allowed.",
                    {
                        "type": "tool",
                        "name": "convert",
                        "arguments": {"value": index + 1, "unit": "m"},
                    },
                    tools=[tool(), tool("distractor")],
                    tool_choice=choice,
                    params={"max_tokens": cfg["ordinary_output_cap"]},
                    extra={"mode": mode, "requires": ["tools", "tool_choice." + mode]},
                )
            )
        calls = [
            {"name": "convert", "arguments": {"value": value, "unit": "m"}}
            for value in (index, index + 1)
        ]
        items.append(
            probe(
                "P10",
                index,
                "serial",
                f"Convert {index} and {index + 1} using the tool.",
                {"type": "tool", "calls": calls},
                tools=[tool()],
                params={"max_tokens": cfg["ordinary_output_cap"], "parallel_tool_calls": False},
                extra={"requires": ["parallel_tool_calls", "tools"]},
            )
        )
        for stream in (False, True):
            items.append(
                probe(
                    "P11",
                    index,
                    "stream" if stream else "json",
                    f"Convert {index} to m with convert.",
                    {"type": "tool", "name": "convert", "arguments": {"value": index, "unit": "m"}},
                    tools=[tool()],
                    params={"max_tokens": cfg["ordinary_output_cap"]},
                    stream=stream,
                    extra={"requires": ["streaming", "tools"]},
                )
            )
        items.append(
            probe(
                "P06.structure",
                index,
                "logprobs",
                f"Trial {index}. Reply exactly OK.",
                {"type": "echo", "text": "OK"},
                params={
                    "max_tokens": cfg["ordinary_output_cap"],
                    "logprobs": True,
                    "top_logprobs": 1,
                },
                extra={"requires": ["logprobs"]},
            )
        )
    for check, parameter, low in (
        ("P03", "temperature", 0.0),
        ("P04.top_p", "top_p", 0.01),
        ("P04.top_k", "top_k", 1),
    ):
        for index in range(sizes["diversity_prompts"]):
            nonce = rng.getrandbits(64)
            for variant, value in (("low", low), ("high", 16 if parameter == "top_k" else 1.0)):
                items.append(
                    probe(
                        check,
                        index,
                        variant,
                        f"Name one ordinary color at random. Return only its name. Trial {nonce}.",
                        {
                            "type": "sample",
                            "choices": ["red", "blue", "green", "gold", "orange", "violet"],
                        },
                        params={"max_tokens": cfg["ordinary_output_cap"], parameter: value},
                        samples=sizes["diversity_repeats"],
                        extra={"requires": [parameter]},
                    )
                )
    for index in range(sizes["effort_items"]):
        problem = math_problem(seed + index, tier=3)
        for effort in ("low", "high"):
            item = probe(
                "P07",
                index,
                effort,
                problem.prompt,
                {"type": "math", "operation": problem.kind, **problem.parameters},
                params={"max_tokens": cfg["effort_output_cap"], "reasoning_effort": effort},
                extra={
                    "requires": ["reasoning"],
                    "math_tier": problem.tier,
                    "math_seed": problem.seed,
                },
            )
            item.payload["minimum_output_tokens"] = cfg["effort_minimum_output"]
            item.version = "protocol-effort-math-v1"
            items.append(item)
    for index in range(sizes["seed_trials"]):
        for variant in ("first", "repeat"):
            items.append(
                probe(
                    "P05",
                    index,
                    variant,
                    f"Name a random color, one word. Trial {index}.",
                    {"type": "sample"},
                    params={"max_tokens": cfg["ordinary_output_cap"], "seed": seed + index},
                    extra={"requires": ["seed"]},
                )
            )
    # Context and router probes depend on verified endpoint metadata, so the caller
    # supplies a shared target length and builds endpoint-specific requests below.
    if (
        endpoint is not None
        and isinstance(endpoint.metadata.get("context_length"), int)
        and endpoint.metadata["context_length"] <= cfg["maximum_probe_context"]
    ):
        context = endpoint.metadata["context_length"]
        over = " pad" * (context + cfg["output_headroom"])
        count_text = codec.count if codec is not None else text_tokens
        assert count_text(over) > context
        for index in range(sizes["invariant_trials"]):
            items.append(
                probe(
                    "P12",
                    index,
                    "over_context",
                    f"HEAD_{index}\n{over}\nTAIL_{index}\nReply OK.",
                    {"type": "echo", "text": "OK"},
                    params={"max_tokens": cfg["ordinary_output_cap"]},
                    extra={"allowed_statuses": [200, 400, 413, 422], "context_length": context},
                )
            )
            if codec is not None:
                items[-1].version = "protocol-context-glm-v1"
                items[-1].expected.update(
                    text_counter_profile=codec.profile,
                    text_counter_revision=codec.revision,
                    visible_prompt_tokens=count_text(items[-1].payload["messages"][0]["content"]),
                )
    if profile is not None and profile.immutable_params:
        for index in range(sizes["invariant_trials"]):
            for key, value in profile.immutable_params.items():
                items.append(
                    probe(
                        "P14",
                        index,
                        key,
                        "Reply OK.",
                        {"type": "echo", "text": "OK"},
                        params={"max_tokens": cfg["ordinary_output_cap"], key: value},
                        extra={"immutable_key": key, "immutable_value": value},
                    )
                )
    if endpoint is not None and endpoint.kind == "openrouter":
        supported = set(endpoint.metadata.get("supported_parameters", []))
        missing = next(
            (key for key in ("top_k", "parallel_tool_calls", "seed") if key not in supported),
            "trueweights_unknown_parameter",
        )
        for index in range(sizes["invariant_trials"]):
            for variant in ("required", "optional"):
                item = probe(
                    "P13",
                    index,
                    variant,
                    "Reply exactly OK.",
                    {"type": "echo", "text": "OK"},
                    params={"max_tokens": cfg["ordinary_output_cap"], missing: True},
                    extra={"allowed_statuses": [200, 400, 422], "unsupported_parameter": missing},
                )
                if variant == "optional":
                    item.payload["unsupported_probe"] = "P13"
                items.append(item)
    if codec is not None:
        for item in items:
            if item.expected["check_id"] in {"P01", "P07"}:
                item.version = "protocol-count-glm-v1"
                item.expected.update(
                    text_counter_profile=codec.profile, text_counter_revision=codec.revision
                )
    return items
