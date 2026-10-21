"""Explicit sensitivity scenarios, separate from enforceable worst-case reservations."""

from pathlib import Path
from typing import Any

import yaml
from jinja2 import TemplateError

from servegauge.core.errors import SafeError
from servegauge.core.hashing import content_hash
from servegauge.core.models import EndpointSpec
from servegauge.tokencount.counters import mock_prompt_tokens
from servegauge.tokencount.glm_counter import (
    glm_request_controls,
    glm_text_tokens,
    render_glm_chat,
)
from servegauge.tokencount.hf_counter import hf_chat_tokens


def scenario_config() -> dict[str, Any]:
    cfg = yaml.safe_load(Path("configs/cost_scenarios.yaml").read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or not cfg.get("version") or not cfg.get("rationale"):
        raise ValueError("versioned_cost_scenarios_required")
    values = cfg.get("completion_tokens")
    if (
        not isinstance(values, list)
        or not values
        or any(type(value) is not int or value < 1 for value in values)
        or type(cfg.get("continuation_prompt_tokens")) is not int
        or cfg["continuation_prompt_tokens"] < 1
        or type(cfg.get("assume_intentional_rejections_unbilled")) is not bool
    ):
        raise ValueError("positive_cost_scenario_token_counts_required")
    return dict(cfg)


def initial_input_proxy(
    body: dict[str, Any],
    endpoint: EndpointSpec,
    cache: dict[str, tuple[int, str]],
    *,
    counter_date: str,
) -> tuple[int, str]:
    reasoning = body.get("reasoning")
    default_effort = (
        "max" if endpoint.claimed_profile == "glm-5.3" and endpoint.kind != "mock" else "medium"
    )
    effort = body.get(
        "reasoning_effort",
        reasoning.get("effort", default_effort) if isinstance(reasoning, dict) else default_effort,
    )
    inputs = {
        "messages": body["messages"],
        "tools": body.get("tools"),
        "effort": effort,
        "date": counter_date,
        "mock": endpoint.kind == "mock",
        "profile": endpoint.claimed_profile,
    }
    if endpoint.claimed_profile == "glm-5.3" and endpoint.kind != "mock":
        inputs["glm_controls"] = {
            name: body[name]
            for name in ("reasoning", "reasoning_effort", "thinking", "clear_thinking")
            if name in body
        }
    key = content_hash(inputs)
    if key not in cache:
        if endpoint.kind == "mock":
            cache[key] = mock_prompt_tokens(body), "canonical mock template"
        elif endpoint.claimed_profile == "glm-5.3":
            try:
                rendered = render_glm_chat(
                    body["messages"], tools=body.get("tools"), **glm_request_controls(body)
                )
                cache[key] = (
                    glm_text_tokens(rendered),
                    "official local GLM template; native overhead unqualified",
                )
            except (SafeError, ValueError, TypeError, KeyError, TemplateError):
                cache[key] = (
                    mock_prompt_tokens(body),
                    "canonical wire proxy; GLM local template unavailable or unsupported",
                )
        else:
            try:
                if endpoint.claimed_profile not in {
                    "gpt-oss-120b",
                    "gpt-oss-20b",
                } or not isinstance(effort, str):
                    raise ValueError("unsupported_counter_profile")
                cache[key] = (
                    hf_chat_tokens(
                        body["messages"],
                        tools=body.get("tools"),
                        reasoning_effort=effort,
                        conversation_date=counter_date,
                    ),
                    "official local template; native overhead unqualified",
                )
            except (ValueError, TypeError, KeyError):
                cache[key] = mock_prompt_tokens(body), "canonical wire proxy; template unsupported"
    return cache[key]
