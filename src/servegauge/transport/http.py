"""Only a valid one-use reservation permits a Chat Completions dispatch."""

from __future__ import annotations

import asyncio
import ssl
import time
from typing import Any

import httpx

from servegauge.core.completion import completion_error
from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.json import strict_json
from servegauge.core.models import Attempt, SseEvent, Timings
from servegauge.core.redaction import Redactor
from servegauge.runner.budget import BudgetManager, Reservation
from servegauge.transport.capture import capture_response
from servegauge.transport.sse import SseParser


def wire_hash(url: str, body: dict[str, Any]) -> str:
    return content_hash({"url": url, "body": body})


def assemble_stream(events: list[SseEvent]) -> tuple[dict[str, Any], dict[str, Any]]:
    message: dict[str, Any] = {"role": "assistant", "content": ""}
    calls: dict[int, dict[str, Any]] = {}
    response: dict[str, Any] = {
        "choices": [{"index": 0, "message": message, "finish_reason": None}]
    }
    problems: dict[str, Any] = {"tool_delta_missing_index": False, "invalid_json_events": 0}
    for event in events:
        if event.is_comment or event.data == "[DONE]":
            continue
        try:
            chunk = strict_json(event.data)
            if not isinstance(chunk, dict):
                raise ValueError("chunk_not_object")
            for key in ("id", "model", "usage", "error"):
                if key in chunk:
                    if key in {"id", "model"} and key in response and response[key] != chunk[key]:
                        problems["stream_identity_conflict"] = True
                    response[key] = chunk[key]
            chunks = chunk.get("choices", [])
            if len(chunks) > 1:
                problems["stream_multiple_choices"] = True
            for choice in chunks:
                if choice.get("index", 0) != 0 or isinstance(choice.get("index"), bool):
                    problems["stream_multiple_choices"] = True
                delta = choice.get("delta", {})
                for key in ("content", "reasoning", "reasoning_content"):
                    if delta.get(key):
                        message[key] = str(message.get(key) or "") + delta[key]
                if delta.get("reasoning_details"):
                    message.setdefault("reasoning_details", []).extend(delta["reasoning_details"])
                for fragment in delta.get("tool_calls") or []:
                    index = fragment.get("index")
                    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                        problems["tool_delta_missing_index"] = True
                        index = 0
                    call = calls.setdefault(
                        index,
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if fragment.get("id"):
                        call["id"] += fragment["id"]
                    for key in ("name", "arguments"):
                        call["function"][key] += fragment.get("function", {}).get(key, "")
                if choice.get("finish_reason"):
                    response["choices"][0]["finish_reason"] = choice["finish_reason"]
        except (ValueError, TypeError, AttributeError, KeyError, IndexError):
            problems["invalid_json_events"] += 1
            continue
    if calls:
        message["tool_calls"] = [calls[index] for index in sorted(calls)]
    if not message["content"]:
        message["content"] = None
    return response, problems


async def send_reserved(
    client: httpx.AsyncClient,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    budget: BudgetManager,
    reservation: Reservation,
    attempt_n: int,
    redactor: Redactor | None = None,
    *,
    timeout_seconds: float,
    request_extensions: dict[str, Any] | None = None,
) -> Attempt:
    budget.dispatch(reservation, wire_hash(url, body))
    redactor = redactor or Redactor.from_environment()
    started, unix = time.monotonic(), time.time()
    first: float | None = None
    parser = SseParser()
    events: list[SseEvent] = []
    content = bytearray()
    status: int | None = None
    captured_headers: dict[str, str] = {}
    metadata: dict[str, Any] = {}
    response_body: dict[str, Any] | None = None
    text: str | None = None
    error: str | None = None
    try:
        async with asyncio.timeout(timeout_seconds):
            async with client.stream(
                "POST",
                url,
                content=canonical_json(body),
                headers=headers,
                extensions=request_extensions,
            ) as response:
                status = response.status_code
                streaming = "text/event-stream" in response.headers.get("content-type", "")
                captured_headers = capture_response(
                    status, dict(response.headers), b"", redactor
                ).headers
                async for chunk in response.aiter_bytes():
                    elapsed = (time.monotonic() - started) * 1000
                    if first is None:
                        first = elapsed
                    if streaming:
                        events.extend(parser.feed(chunk, elapsed))
                    else:
                        content.extend(chunk)
                if streaming:
                    events.extend(parser.finish((time.monotonic() - started) * 1000))
                    response_body, metadata = assemble_stream(events)
                    metadata.update(
                        {
                            "terminal": parser.terminal,
                            "incomplete": parser.incomplete,
                            "invalid_utf8": parser.invalid_utf8,
                        }
                    )
                else:
                    capture = capture_response(
                        status, dict(response.headers), bytes(content), redactor
                    )
                    response_body, text = capture.body, capture.text
    except (httpx.TimeoutException, TimeoutError):
        error = "transport_timeout"
    except (httpx.HTTPError, ssl.SSLError):
        error = "transport_failure"
    if response_body is None and content:
        partial = capture_response(status or 0, {}, bytes(content), redactor)
        response_body, text = partial.body, partial.text
    if response_body is None and events:
        response_body, metadata = assemble_stream(events)
        metadata.update(
            {"terminal": parser.terminal, "incomplete": True, "invalid_utf8": parser.invalid_utf8}
        )
    if error is None and status == 200 and completion_error(response_body):
        error = "provider_completion_error"
    metadata.update(timeout_scope="total_request", timeout_seconds=timeout_seconds)
    result = Attempt(
        n=attempt_n,
        status=status,
        error=error,
        timings=Timings(
            start_unix=unix, ttft_ms=first, total_ms=(time.monotonic() - started) * 1000
        ),
        response=response_body,
        response_text=text,
        sse_events=events or None,
        response_headers=captured_headers,
        metadata=metadata,
    )
    return Attempt.model_validate(redactor.redact(result.model_dump(mode="json")))
