import copy
from pathlib import Path

import pytest

from servegauge.core.errors import SafeError
from servegauge.profiles import field_verified, load_profile
from servegauge.tokencount.chat import (
    harmony_chat_tokens,
    hf_chat_tokens,
    hf_text_tokens,
    render_hf_chat,
)
from servegauge.tokencount.counters import text_tokens


def test_known_text_and_harmony_counts_offline() -> None:
    assert hf_text_tokens("Hello, world!") == 4
    assert text_tokens("Hello, world!") == 4
    assert harmony_chat_tokens([{"role": "user", "content": "Hello"}]) == 7
    assert harmony_chat_tokens([{"role": "user", "content": "Hello world"}]) == 8
    for text in ["", "café 北京 🙂", "<|start|>", "x\n\ty", "日本語 العربية русский"]:
        # Plain text special-looking markers are ordinary text in this counter.
        if "<|" not in text:
            assert hf_text_tokens(text) == text_tokens(text)


def test_official_hf_template_has_explicit_date_and_tools() -> None:
    messages = [{"role": "user", "content": "Hello"}]
    assert hf_chat_tokens(messages, conversation_date="2026-10-18") == 68
    rendered = render_hf_chat(messages, conversation_date="2026-10-18")
    assert "Current date: 2026-10-18" in rendered and rendered.endswith("<|start|>assistant")
    tool = {
        "type": "function",
        "function": {
            "name": "convert",
            "description": "Convert units",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "number"}},
                "required": ["value"],
            },
        },
    }
    tool_text = render_hf_chat(
        messages, tools=[tool], reasoning_effort="high", conversation_date="2026-10-18"
    )
    assert "namespace functions" in tool_text and "value: number" in tool_text
    assert "Reasoning: high" in tool_text
    assert hf_chat_tokens(messages, tools=[tool], conversation_date="2026-10-18") > 68
    with pytest.raises(ValueError):
        render_hf_chat([], conversation_date="2026-10-18")
    with pytest.raises(ValueError):
        render_hf_chat(messages, conversation_date="bad")
    with pytest.raises(ValueError):
        harmony_chat_tokens([{"role": "assistant", "content": None, "tool_calls": [{}]}])


def test_cache_is_fail_closed_and_profiles_are_sourced(tmp_path: Path, monkeypatch) -> None:
    from servegauge.tokencount.chat import _checked_file

    monkeypatch.setenv("SERVEGAUGE_TOKENIZER_CACHE", str(tmp_path))
    with pytest.raises(SafeError, match="cache_missing"):
        _checked_file("tokenizer.json")
    for identifier in ["gpt-oss-120b", "gpt-oss-20b"]:
        profile = load_profile(identifier)
        assert field_verified(profile, "chat_format")
        assert not field_verified(profile, "recommended_sampling")
        assert not field_verified(profile, "reasoning.api_parameter")
        assert set(profile.format_markers) >= {"<|call|>", "<|return|>"}
        assert profile.sources
    with pytest.raises(ValueError):
        load_profile("../gpt-oss-120b")


def test_wire_tool_arguments_and_full_reasoning_normalize_without_mutating_history():
    messages = [
        {"role": "user", "content": "Convert 3"},
        {
            "role": "assistant",
            "content": None,
            "reasoning": "Use the conversion tool.",
            "reasoning_details": [
                {"type": "reasoning.text", "text": "Use the conversion tool.", "index": 0}
            ],
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "convert", "arguments": '{"value":3}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "300"},
    ]
    before = copy.deepcopy(messages)
    native = copy.deepcopy(messages)
    native[1]["thinking"] = native[1].pop("reasoning")
    native[1].pop("reasoning_details")
    native[1]["tool_calls"][0]["function"]["arguments"] = {"value": 3}
    first = render_hf_chat(messages, conversation_date="2026-10-18")
    assert first == render_hf_chat(native, conversation_date="2026-10-18")
    assert '{"value": 3}' in first and messages == before
    assert hf_chat_tokens(messages, conversation_date="2026-10-18") == hf_chat_tokens(
        native, conversation_date="2026-10-18"
    )
    for failure in ("parallel", "opaque", "summary", "array_args", "invalid_call"):
        invalid = copy.deepcopy(messages)
        if failure == "parallel":
            invalid[1]["tool_calls"] *= 2
        elif failure in {"opaque", "summary"}:
            invalid[1]["reasoning_details"][0]["type"] = (
                "reasoning.encrypted" if failure == "opaque" else "reasoning.summary"
            )
        elif failure == "array_args":
            invalid[1]["tool_calls"][0]["function"]["arguments"] = "[]"
        else:
            invalid[1]["tool_calls"] = [None]
        with pytest.raises(ValueError):
            render_hf_chat(invalid, conversation_date="2026-10-18")
