"""Pinned official HF tokenizer/template, with explicit API-to-template normalization."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Template
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

from servegauge.core.errors import SafeError
from servegauge.core.json import strict_json
from servegauge.tokencount.cache import HF_FILES, configure_offline


def _checked_file(name: str) -> str:
    path = configure_offline() / "hf-gpt-oss" / name
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != HF_FILES[name]:
        raise SafeError("tokenizer_cache_missing_or_invalid; run setup-cache")
    return str(path)


@lru_cache(maxsize=1)
def hf_tokenizer() -> Tokenizer:
    return Tokenizer.from_file(_checked_file("tokenizer.json"))


def hf_text_tokens(text: str) -> int:
    return len(hf_tokenizer().encode(text, add_special_tokens=False).ids)


def _template_error(message: str) -> None:
    raise ValueError("unsupported_official_template_conversation")


@lru_cache(maxsize=1)
def _template() -> Template:
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    environment.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)
    environment.globals["raise_exception"] = _template_error
    return environment.from_string(
        Path(_checked_file("chat_template.jinja")).read_text(encoding="utf-8")
    )


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(messages)
    for message in normalized:
        message["content"] = message.get("content") or ""
        details = message.get("reasoning_details") or []
        if not isinstance(details, list) or any(
            not isinstance(block, dict) or block.get("type") != "reasoning.text"
            for block in details
        ):
            raise ValueError("opaque_or_summarized_reasoning_history_cannot_be_counted")
        if "thinking" not in message:
            if isinstance(message.get("reasoning"), str):
                message["thinking"] = message["reasoning"]
            elif details:
                if any(not isinstance(block.get("text"), str) for block in details):
                    raise ValueError("reasoning_text_required")
                message["thinking"] = "".join(block["text"] for block in details)
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list) or len(calls) > 1:
            # The pinned gpt-oss template explicitly renders tool_calls[0] only.
            raise ValueError("official_template_parallel_history_is_ambiguous")
        for call in calls:
            if not isinstance(call, dict):
                raise ValueError("official_template_tool_call_must_be_object")
            function = call.get("function", call)
            if not isinstance(function, dict):
                raise ValueError("official_template_function_must_be_object")
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                arguments = strict_json(arguments)
                function["arguments"] = arguments
            if not isinstance(arguments, dict):
                raise ValueError("official_template_tool_arguments_must_be_object")
    return normalized


def render_hf_chat(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    reasoning_effort: str = "medium",
    conversation_date: str,
) -> str:
    if not messages or reasoning_effort not in {"low", "medium", "high"}:
        raise ValueError("messages_and_known_reasoning_effort_required")
    pinned_date = date.fromisoformat(conversation_date)
    return _template().render(
        messages=normalize_messages(messages),
        tools=tools or [],
        reasoning_effort=reasoning_effort,
        add_generation_prompt=True,
        strftime_now=lambda fmt: pinned_date.strftime(fmt),
    )


def hf_chat_tokens(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    reasoning_effort: str = "medium",
    conversation_date: str,
) -> int:
    return hf_text_tokens(
        render_hf_chat(
            messages,
            tools=tools,
            reasoning_effort=reasoning_effort,
            conversation_date=conversation_date,
        )
    )
