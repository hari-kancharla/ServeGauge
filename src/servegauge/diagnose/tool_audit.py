"""Read-only tool triage. API observations do not identify a serving implementation."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from servegauge.core.hashing import content_hash
from servegauge.core.models import Exchange
from servegauge.scoring.common import calls, choice, content
from servegauge.scoring.tools import dropped


def wire_tool_calls(record: Exchange) -> list[dict[str, Any]] | None:
    """Independent reconstruction of choice-zero SSE function deltas by integer index.

    Missing indexes are ambiguous and are never guessed. This audit deliberately does
    not call the production assembler whose output it is checking.
    """
    if record.sse_events is None:
        return None
    buffers: dict[int, dict[str, Any]] = {}
    for event in record.sse_events:
        if event.is_comment or event.data == "[DONE]":
            continue
        try:
            chunk = json.loads(event.data)
            for row in chunk.get("choices", []):
                if row.get("index", 0) != 0:
                    continue
                for delta in row.get("delta", {}).get("tool_calls", []):
                    index = delta.get("index")
                    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                        return None
                    target = buffers.setdefault(
                        index,
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    for key in ("id", "type"):
                        if delta.get(key):
                            target[key] = delta[key]
                    for key in ("name", "arguments"):
                        fragment = delta.get("function", {}).get(key, "")
                        if not isinstance(fragment, str):
                            return None
                        target["function"][key] += fragment
        except (ValueError, TypeError, AttributeError):
            return None
    return [buffers[index] for index in sorted(buffers)]


def raw_logprob_bytes(record: Exchange) -> bytes | None:
    """Return only fully recorded token bytes; never invent missing raw model output."""
    container = choice(record).get("logprobs")
    rows = container.get("content") if isinstance(container, dict) else None
    if not isinstance(rows, list) or not rows:
        return None
    result = bytearray()
    for row in rows:
        value = row.get("bytes") if isinstance(row, dict) else None
        if not isinstance(value, list) or not all(
            isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= 255 for x in value
        ):
            return None
        result.extend(value)
    return bytes(result)


def audit_tools(records: list[Exchange]) -> dict[str, Any]:
    endpoints: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.suite_id != "S2":
            continue
        summary = endpoints.setdefault(
            record.endpoint_id, {"captures": 0, "counts": Counter(), "observations": []}
        )
        summary["captures"] += 1
        if record.error or record.status != 200 or record.metadata.get("request_not_sent"):
            summary["counts"]["excluded_or_error"] += 1
            continue
        try:
            actual = calls(record)
            wire = wire_tool_calls(record)
            summary["counts"][
                "stream_assembly_checked" if wire is not None else "no_unambiguous_sse_calls"
            ] += 1
            mismatch = wire is not None and content_hash(wire) != content_hash(actual)
            summary["counts"]["stream_assembly_mismatch"] += int(mismatch)
            flag = dropped(record)
            if not flag and not mismatch:
                continue
            raw = raw_logprob_bytes(record)
            observation = {
                "exchange_id": record.exchange_id,
                "item_id": record.item_id,
                "sample_idx": record.sample_idx,
                "confirmation": "-confirmation-" in record.item_id,
                "request_hash": content_hash(record.request),
                "response_hash": content_hash(record.response),
                "sse_hash": content_hash(
                    [e.model_dump(mode="json") for e in record.sse_events or []]
                ),
                "stream": bool(record.request.get("stream")),
                "finish_reason": choice(record).get("finish_reason"),
                "structured_calls": len(actual),
                "content_hash": content_hash(content(record)),
                "signature": "stream_assembly_mismatch"
                if mismatch
                else "empty_calls_with_tool_finish"
                if choice(record).get("finish_reason") == "tool_calls"
                else "call_shaped_visible_JSON",
                "raw_logprob_bytes_available": raw is not None,
                "raw_logprob_hash": content_hash(list(raw)) if raw is not None else None,
                "serving_parser_cause_verified": False,
                "alternatives": [
                    "model serialized or echoed calls as text",
                    "native template or parser translation",
                    "router normalization",
                ],
            }
            if not mismatch:
                summary["counts"][observation["signature"]] += 1
            summary["observations"].append(observation)
        except (ValueError, TypeError, KeyError, AttributeError):
            summary["counts"]["invalid_envelope"] += 1
    for summary in endpoints.values():
        summary["counts"] = dict(sorted(summary["counts"].items()))
    return {
        "reference_tier": "recorded API observations; no raw engine reference",
        "endpoints": endpoints,
    }
