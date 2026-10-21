"""Recognize provider-declared failures even when HTTP transport succeeds."""

from typing import Any


def completion_error(response: dict[str, Any] | None) -> bool:
    if response is None:
        return False
    if response.get("error") is not None:
        return True
    choices = response.get("choices")
    return isinstance(choices, list) and any(
        isinstance(choice, dict) and choice.get("finish_reason") == "error" for choice in choices
    )
