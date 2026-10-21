"""Explicit official Harmony messages, without guessed server scaffolding."""

from functools import lru_cache
from typing import Any

# Upstream has annotations but no PEP 561 marker; verified by offline known answers.
from openai_harmony import (  # type: ignore[import-untyped]
    Author,
    Conversation,
    HarmonyEncoding,
    HarmonyEncodingName,
    Message,
    Role,
    load_harmony_encoding,
)

from servegauge.tokencount.counters import encoding


@lru_cache(maxsize=1)
def harmony_encoding() -> HarmonyEncoding:
    encoding()
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def harmony_chat_tokens(messages: list[dict[str, Any]]) -> int:
    rendered = []
    for raw in messages:
        if not isinstance(raw.get("content"), str) or raw.get("tool_calls"):
            raise ValueError("explicit_harmony_text_messages_required")
        message = Message.from_author_and_content(
            Author(role=Role(raw["role"]), name=raw.get("name")), raw["content"]
        )
        if raw.get("channel"):
            message = message.with_channel(raw["channel"])
        if raw.get("recipient"):
            message = message.with_recipient(raw["recipient"])
        rendered.append(message)
    return len(
        harmony_encoding().render_conversation_for_completion(
            Conversation.from_messages(rendered), Role.ASSISTANT
        )
    )
