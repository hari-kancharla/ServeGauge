"""Capture response bodies and allowlisted headers before any persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from servegauge.core.json import strict_json
from servegauge.core.redaction import Redactor

ALLOWED_HEADERS = {
    "content-type",
    "content-encoding",
    "date",
    "x-request-id",
    "request-id",
    "retry-after",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
}


@dataclass(frozen=True)
class CapturedResponse:
    status: int
    headers: dict[str, str]
    body: dict[str, Any] | None
    text: str


def capture_response(
    status: int, headers: dict[str, str], body: bytes, redactor: Redactor | None = None
) -> CapturedResponse:
    scrubber = redactor or Redactor.from_environment()
    clean_headers = {
        key.lower(): scrubber.text(value)
        for key, value in headers.items()
        if key.lower() in ALLOWED_HEADERS
    }
    text = body.decode("utf-8", "replace")
    try:
        parsed = strict_json(text)
    except ValueError:
        parsed = None
    clean = scrubber.redact(parsed) if isinstance(parsed, dict) else None
    return CapturedResponse(status, clean_headers, clean, scrubber.wire_text(text))
