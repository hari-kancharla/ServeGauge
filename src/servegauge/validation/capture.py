"""In-memory mock wire capture, with the real framing, assembler and agent executors."""

from __future__ import annotations

import copy
import json
from typing import Any

from servegauge.core.ids import exchange_id
from servegauge.core.models import Attempt, EndpointSpec, Exchange, Item, Timings
from servegauge.items.agent_tasks import execute_tool
from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.server import stream_frames
from servegauge.scoring.common import calls, message
from servegauge.scoring.toolcall_scoring import parsed_call
from servegauge.suites.registry import prepare_item
from servegauge.transport.http import assemble_stream
from servegauge.transport.sse import SseParser


def endpoint(identifier: str, seed: int, *, clean: bool) -> EndpointSpec:
    return EndpointSpec(
        id=identifier,
        kind="mock",
        base_url="http://127.0.0.1/v1",
        model="mock",
        claimed_profile="gpt-oss-120b",
        metadata={
            "context_length": 131072,
            "max_completion_tokens": 65536,
            "mock_declaration": {"seed": seed, "clean": clean},
        },
    )


def capture(
    item: Item,
    body: dict[str, Any],
    backend: ScriptedBackend,
    target: EndpointSpec,
    run_id: str,
    sample: int = 0,
) -> Exchange:
    status = 200
    if backend.rng.random() < backend.faults.error_rate:
        response: dict[str, Any] = {"error": {"code": "temporarily_unavailable"}}
        status = 503
    else:
        response = backend.respond(body)
        if "error" in response:
            status = 400
    events = None
    metadata: dict[str, Any] = {}
    if body.get("stream") and status == 200:
        parser = SseParser()
        events = []
        for frame in stream_frames(response, backend.faults):
            events.extend(parser.feed(frame, 0))
        parser.finish(0)
        response, metadata = assemble_stream(events)
        metadata.update(terminal=parser.terminal, incomplete=parser.incomplete)
    return Exchange(
        exchange_id=exchange_id(run_id, target.id, item.suite_id, item.item_id, sample),
        run_id=run_id,
        endpoint_id=target.id,
        suite_id=item.suite_id,
        item_id=item.item_id,
        sample_idx=sample,
        request=body,
        attempts=[Attempt(n=1, status=status, timings=Timings(start_unix=0, total_ms=0))],
        status=status,
        response=response,
        sse_events=events,
        metadata=metadata,
        usage=response.get("usage"),
        cost_usd=0,
        cost_source="estimate",
    )


def captures(
    items: list[Item], backend: ScriptedBackend, target: EndpointSpec, run_id: str
) -> list[Exchange]:
    records = []
    for item in items:
        decision = prepare_item(item, target)
        if decision.body is None:
            raise ValueError("lab_probe_unexpectedly_unsupported")
        for index in range(item.expected.get("samples", 1)):
            records.append(capture(item, decision.body, backend, target, run_id, index))
        if "agent_goal" not in item.expected:
            continue
        current = records[-1]
        history = copy.deepcopy(item.payload["messages"])
        state = copy.deepcopy(item.expected["agent_initial"])
        for turn in range(1, item.expected["max_turns"]):
            returned = calls(current)
            if not returned:
                break
            assistant = copy.deepcopy(message(current))
            if not item.expected["pass_reasoning"]:
                for key in ("reasoning", "reasoning_details", "reasoning_content"):
                    assistant.pop(key, None)
            history.append(assistant)
            for call in returned:
                name, args = parsed_call(call)
                state = execute_tool(state, name, args)
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(
                            {"ok": True, "state": state[name]}, ensure_ascii=False
                        ),
                    }
                )
            if state == item.expected["agent_goal"]:
                break
            child = item.model_copy(deep=True)
            child.item_id += f"-turn-{turn}"
            child.payload["messages"] = copy.deepcopy(history)
            body = prepare_item(child, target).body
            if body is None:
                raise ValueError("lab_agent_history_unsupported")
            current = capture(child, body, backend, target, run_id)
            records.append(current)
    return records
