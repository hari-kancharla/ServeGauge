"""Pinned GLM text/template projections, distinct from provider billing semantics."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from typing import Any, TypedDict

from jinja2 import Template
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

from servegauge.tokencount.glm_assets import verified_assets


class GlmControls(TypedDict):
    reasoning_effort: str
    clear_thinking: bool


def glm_request_controls(body: dict[str, Any]) -> GlmControls:
    reasoning = body.get("reasoning")
    if "reasoning" in body and (
        not isinstance(reasoning, dict)
        or set(reasoning) - {"effort", "exclude"}
        or ("exclude" in reasoning and type(reasoning["exclude"]) is not bool)
    ):
        raise ValueError("glm_unknown_reasoning_control")
    effort = body.get(
        "reasoning_effort",
        reasoning.get("effort", "max") if isinstance(reasoning, dict) else "max",
    )
    if (
        not isinstance(effort, str)
        or effort not in {"low", "high", "max"}
        or (isinstance(reasoning, dict) and reasoning.get("effort", effort) != effort)
    ):
        raise ValueError("glm_unknown_or_conflicting_reasoning_effort")
    if "thinking" in body and body["thinking"] != {"type": "enabled"}:
        raise ValueError("glm_unknown_thinking_control")
    clear = body.get("clear_thinking", False)
    if type(clear) is not bool:
        raise ValueError("glm_boolean_clear_thinking_required")
    return {"reasoning_effort": effort, "clear_thinking": clear}


@lru_cache(maxsize=2)
def _tokenizer(raw: bytes) -> Tokenizer:
    return Tokenizer.from_str(raw.decode("utf-8"))


def glm_text_ids(text: str) -> list[int]:
    return (
        _tokenizer(verified_assets()["tokenizer.json"]).encode(text, add_special_tokens=False).ids
    )


def glm_text_tokens(text: str) -> int:
    return len(glm_text_ids(text))


def glm_decode(ids: list[int]) -> str:
    tokenizer = _tokenizer(verified_assets()["tokenizer.json"])
    vocabulary_size = tokenizer.get_vocab_size()
    if any(type(value) is not int or not 0 <= value < vocabulary_size for value in ids):
        raise ValueError("glm_valid_native_token_ids_required")
    return tokenizer.decode(ids, skip_special_tokens=False)


@lru_cache(maxsize=2)
def _template(raw: bytes) -> Template:
    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )
    # Match the official Transformers filter, including non-ASCII JSON and kwargs.
    environment.filters["tojson"] = lambda value, **kwargs: json.dumps(
        value, **{"ensure_ascii": False, **kwargs}
    )
    return environment.from_string(raw.decode("utf-8"))


def glm_template() -> Template:
    return _template(verified_assets()["chat_template.jinja"])


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("glm_duplicate_tool_argument_key")
        result[key] = value
    return result


def _invalid_number(value: str) -> None:
    raise ValueError("glm_nonfinite_tool_argument")


def _text_content(value: Any) -> str | list[Any]:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(
        isinstance(part, str)
        or (
            isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
            and set(part) <= {"type", "text"}
        )
        for part in value
    ):
        return value
    raise ValueError("glm_text_only_history_required")


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("glm_nonempty_messages_required")
    normalized = copy.deepcopy(messages)
    # Validate all numbers/objects too: no NaN, bytes, or implicit stringification.
    json.dumps(normalized, allow_nan=False)
    for message in normalized:
        if not isinstance(message, dict) or message.get("role") not in {
            "system",
            "user",
            "assistant",
            "tool",
        }:
            raise ValueError("glm_supported_message_role_required")
        if any(
            message.get(key) for key in ("audio", "images", "function_call", "refusal", "thinking")
        ):
            raise ValueError("glm_unsupported_history_channel")
        message["content"] = _text_content(message.get("content"))
        aliases: list[str] = []
        for key in ("reasoning", "reasoning_content"):
            if message.get(key) is not None:
                if not isinstance(message[key], str) or message["role"] != "assistant":
                    raise ValueError("glm_assistant_text_reasoning_required")
                aliases.append(message[key])
        details = message.get("reasoning_details")
        if details is not None:
            if not isinstance(details, list) or any(
                not isinstance(detail, dict)
                or detail.get("type") != "reasoning.text"
                or not isinstance(detail.get("text"), str)
                for detail in details
            ):
                raise ValueError("glm_raw_text_reasoning_required")
            if details:
                if message["role"] != "assistant":
                    raise ValueError("glm_assistant_text_reasoning_required")
                aliases.append("".join(detail["text"] for detail in details))
        if aliases:
            content = message["content"]
            visible = (
                content
                if isinstance(content, str)
                else "".join(part if isinstance(part, str) else part["text"] for part in content)
            )
            if any(alias != aliases[0] for alias in aliases) or (
                "<think>" in visible or "</think>" in visible
            ):
                raise ValueError("glm_conflicting_reasoning_history")
            message["reasoning_content"] = aliases[0]
        message.pop("reasoning", None)
        message.pop("reasoning_details", None)
        calls = message.get("tool_calls", [])
        if calls is None:
            calls = []
        if not isinstance(calls, list) or (calls and message["role"] != "assistant"):
            raise ValueError("glm_tool_calls_list_required")
        ids: set[str] = set()
        for call in calls:
            if not isinstance(call, dict) or call.get("type", "function") != "function":
                raise ValueError("glm_function_tool_call_required")
            function = call.get("function", call)
            if (
                not isinstance(function, dict)
                or not isinstance(function.get("name"), str)
                or not function["name"]
            ):
                raise ValueError("glm_named_function_required")
            identifier = call.get("id")
            if identifier is not None:
                if not isinstance(identifier, str) or not identifier or identifier in ids:
                    raise ValueError("glm_unique_tool_call_ids_required")
                ids.add(identifier)
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                arguments = json.loads(
                    arguments, object_pairs_hook=_object, parse_constant=_invalid_number
                )
                function["arguments"] = arguments
            if not isinstance(arguments, dict):
                raise ValueError("glm_object_tool_arguments_required")
            json.dumps(arguments, allow_nan=False)
    return normalized


def render_glm_chat(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    reasoning_effort: str = "max",
    clear_thinking: bool = False,
) -> str:
    if (
        not isinstance(reasoning_effort, str)
        or reasoning_effort not in {"low", "high", "max"}
        or type(clear_thinking) is not bool
    ):
        raise ValueError("glm_supported_explicit_template_controls_required")
    normalized = normalize_messages(messages)
    if tools is not None:
        if not isinstance(tools, list) or any(not isinstance(tool, dict) for tool in tools):
            raise ValueError("glm_function_tools_required")
        json.dumps(tools, allow_nan=False)
        for tool in tools:
            function = tool.get("function", tool)
            if (
                tool.get("type", "function") != "function"
                or not isinstance(function, dict)
                or not isinstance(function.get("name"), str)
                or not function["name"]
            ):
                raise ValueError("glm_named_function_tools_required")
    return glm_template().render(
        messages=normalized,
        tools=tools or [],
        reasoning_effort=reasoning_effort,
        clear_thinking=clear_thinking,
        add_generation_prompt=True,
    )
