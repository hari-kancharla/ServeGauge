"""Real offline text tokenization; model-specific chat rendering is added in M6."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any

import tiktoken

from servegauge.core.errors import SafeError
from servegauge.core.hashing import canonical_json
from servegauge.tokencount.cache import VOCAB_SHA256, configure_offline


@lru_cache(maxsize=1)
def encoding() -> tiktoken.Encoding:
    path = configure_offline() / "o200k_base.tiktoken"
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != VOCAB_SHA256:
        raise SafeError("tokenizer_cache_missing_or_invalid; run setup-cache")
    return tiktoken.get_encoding("o200k_harmony")


def encode(text: str) -> list[int]:
    return encoding().encode_ordinary(text)


def decode(tokens: list[int]) -> str:
    return encoding().decode(tokens)


def text_tokens(text: str) -> int:
    return len(encode(text))


def mock_prompt_tokens(body: dict[str, Any]) -> int:
    """The mock's declared template: canonical messages/tools plus a fixed assistant prefix."""
    prompt = {"messages": body.get("messages", [])}
    if body.get("tools"):
        prompt["tools"] = body["tools"]
    return _mock_count(canonical_json(prompt).decode())


@lru_cache(maxsize=1024)
def _mock_count(prompt: str) -> int:
    """Repeated mock draws share immutable rendered input, not a new tokenization."""
    return text_tokens(prompt + "\nassistant:")
