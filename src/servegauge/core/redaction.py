"""Sanitize nested payloads, serialized SSE data, headers, URLs and environment values."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit

from servegauge.core.secrets import PATTERNS, configured_secrets

REDACTED = "[REDACTED]"
SENSITIVE = {
    "authorization",
    "proxyauthorization",
    "apikey",
    "xapikey",
    "accesstoken",
    "refreshtoken",
    "idtoken",
    "password",
    "passwd",
    "secret",
    "clientsecret",
    "cookie",
    "setcookie",
    "credentials",
    "credential",
}


def sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return normalized in SENSITIVE or normalized.endswith(("apikey", "password", "secret"))


@dataclass(frozen=True)
class Redactor:
    values: tuple[str, ...] = ()

    @classmethod
    def from_environment(cls) -> Redactor:
        return cls(tuple(sorted(set(configured_secrets()), key=len, reverse=True)))

    def with_values(self, *values: str) -> Redactor:
        return Redactor(
            tuple(sorted({*self.values, *(v for v in values if v)}, key=len, reverse=True))
        )

    def text(self, value: str) -> str:
        result = value
        for secret in self.values:
            for encoded in (
                secret,
                quote(secret, safe=""),
                quote(secret),
                json.dumps(secret)[1:-1],
            ):
                result = result.replace(encoded, REDACTED)
        result = re.sub(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            REDACTED,
            result,
            flags=re.DOTALL,
        )
        for pattern in PATTERNS[:3]:
            result = pattern.sub(REDACTED.encode(), result.encode()).decode("utf-8", "replace")
        return result

    def wire_text(self, value: str) -> str:
        """Keep original whitespace/field order while replacing JSON credential values."""
        replacements: list[tuple[int, int]] = []
        decoder = json.JSONDecoder()
        skip_until = 0
        for token in re.finditer(r'"(?:\\.|[^"\\])*"', value):
            if token.start() < skip_until:
                continue
            tail = re.match(r"\s*:\s*", value[token.end() :])
            if tail is None:
                continue
            try:
                key = json.loads(token.group())
                start = token.end() + tail.end()
                if not sensitive_key(key):
                    continue
                _, end = decoder.raw_decode(value, start)
            except ValueError:
                continue
            replacements.append((start, end))
            skip_until = end
        for start, end in reversed(replacements):
            value = value[:start] + json.dumps(REDACTED) + value[end:]
        return self.text(value)

    def url(self, value: str) -> str:
        try:
            parts = urlsplit(value)
            if parts.scheme not in {"http", "https"} or not parts.hostname:
                return "[REDACTED_URL]"
            hostname = parts.hostname
            if ":" in hostname:
                hostname = f"[{hostname}]"
            netloc = hostname + (f":{parts.port}" if parts.port else "")
            query = [
                (self.text(k), REDACTED if sensitive_key(k) or k.lower() == "key" else self.text(v))
                for k, v in parse_qsl(parts.query, keep_blank_values=True)
            ]
            return urlunsplit(
                (parts.scheme, netloc, self.text(unquote(parts.path)), urlencode(query), "")
            )
        except ValueError:
            return "[REDACTED_URL]"

    def _contains_credential(self, value: Any) -> bool:
        if isinstance(value, str):
            return self.text(value) != value
        if isinstance(value, dict):
            return any(self._contains_credential(v) for v in value.values())
        if isinstance(value, list):
            if value and all(isinstance(v, int) and 0 <= v <= 255 for v in value):
                return self._contains_credential(bytes(value).decode("utf-8", "replace"))
            fragments = "".join(str(v.get("token", "")) for v in value if isinstance(v, dict))
            byte_fragments = [
                b for v in value if isinstance(v, dict) for b in (v.get("bytes") or [])
            ]
            return (
                self._contains_credential(fragments)
                or any(self._contains_credential(v) for v in value)
                or bool(byte_fragments and self._contains_credential(byte_fragments))
            )
        return False

    def _response_guard(self, value: dict[str, Any]) -> dict[str, Any]:
        if "response" not in value and "sse_events" not in value:
            return value
        events = value.get("sse_events") or []
        fragments = "".join(
            str(event.get("data", "")) for event in events if event.get("is_comment")
        )
        if not any(
            self._contains_credential(part)
            for part in (value.get("response"), value.get("response_text"), fragments)
        ):
            return value
        guarded = dict(value)
        body = json.dumps(value.get("response"), sort_keys=True, ensure_ascii=False)
        guarded["response"] = {
            "redacted": True,
            "content_hash": hashlib.sha256(body.encode()).hexdigest(),
        }
        if value.get("response_text") is not None:
            guarded["response_text"] = REDACTED
        if events:
            guarded["sse_events"] = [
                {**event, "data": REDACTED, "raw_lines": [REDACTED]} for event in events
            ]
        guarded["metadata"] = {
            **value.get("metadata", {}),
            "sensitive_response_redacted": True,
        }
        guarded["error"] = value.get("error") or "sensitive_response_redacted"
        return guarded

    def redact(self, value: Any) -> Any:
        if isinstance(value, dict):
            value = self._response_guard(value)
            return {
                self.text(str(k)): REDACTED if sensitive_key(str(k)) else self.redact(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.redact(v) for v in value]
        if isinstance(value, str):
            if value.startswith(("https://", "http://")):
                return self.url(value)
            return self.wire_text(value)
        return value
