"""Reconstruct simulated-agent success solely from recorded requests and returned calls."""

from __future__ import annotations

import copy
import json
from typing import Any

from jsonschema import ValidationError

from servegauge.core.models import Exchange, Item
from servegauge.items.agent_tasks import execute_tool
from servegauge.scoring.common import Samples, calls, choice, message
from servegauge.scoring.toolcall_scoring import parsed_call


def agent_samples(
    items: list[Item], records: list[Exchange], endpoint_id: str
) -> dict[str, Samples]:
    allowed = {
        identifier
        for item in items
        if "agent_goal" in item.expected
        for identifier in [
            item.item_id,
            *(item.item_id + f"-turn-{turn}" for turn in range(1, item.expected["max_turns"])),
        ]
    }
    by_item: dict[str, Exchange] = {}
    for incoming in records:
        if incoming.endpoint_id == endpoint_id and incoming.item_id in allowed:
            if (
                incoming.item_id in by_item
                and by_item[incoming.item_id].exchange_id != incoming.exchange_id
            ):
                raise ValueError("ambiguous_agent_run_records")
            by_item[incoming.item_id] = incoming
    output: dict[str, Samples] = {}
    for item in items:
        if "agent_goal" not in item.expected:
            continue
        sample = output.setdefault(
            item.expected["check_id"],
            Samples(
                check_id=item.expected["check_id"],
                check_class="report_only" if item.suite_id == "S8" else "comparative",
            ),
        )
        state = copy.deepcopy(item.expected["agent_initial"])
        prior_history: list[dict[str, Any]] | None = None
        success = False
        trace: dict[str, Any] = {"exchange_ids": [], "run_ids": [], "turns": 0}
        for turn in range(item.expected["max_turns"]):
            identifier = item.item_id if turn == 0 else item.item_id + f"-turn-{turn}"
            record = by_item.get(identifier)
            if record is None:
                sample.mark("ERROR", "agent continuation capture missing")
                break
            sample.evidence.append(record.exchange_id)
            trace["exchange_ids"].append(record.exchange_id)
            trace["run_ids"].append(record.run_id)
            trace["turns"] += 1
            if record.error:
                sample.mark("ERROR", "agent transport or excluded-routing capture")
                break
            if record.metadata.get("request_not_sent") or record.status != 200:
                sample.mark(
                    record.metadata.get("verdict", "NOT_SUPPORTED"), "agent capability unavailable"
                )
                break
            history = record.request.get("messages")
            if not isinstance(history, list) or (
                prior_history is not None and history != prior_history
            ):
                sample.mark("ERROR", "recorded agent history does not match actual returned calls")
                break
            try:
                if choice(record).get("finish_reason") == "length":
                    sample.mark("INCONCLUSIVE", "output cap censored agent turn")
                    break
                returned = calls(record)
                assistant = copy.deepcopy(message(record))
                if any(
                    assistant.get(key)
                    for key in ("reasoning", "reasoning_details", "reasoning_content")
                ):
                    trace["turns_with_reasoning"] = trace.get("turns_with_reasoning", 0) + 1
                if len(returned) > item.expected["max_turns"]:
                    raise ValueError("too_many_calls_for_task_budget")
                if not returned:
                    break
                if not item.expected["pass_reasoning"]:
                    for key in ("reasoning", "reasoning_details", "reasoning_content"):
                        assistant.pop(key, None)
                prior_history = [*copy.deepcopy(history), assistant]
                for call in returned:
                    name, arguments = parsed_call(call)
                    if not isinstance(call.get("id"), str) or not call["id"]:
                        raise ValueError("missing_tool_call_id")
                    state = execute_tool(state, name, arguments)
                    prior_history.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(
                                {"ok": True, "state": state[name]}, ensure_ascii=False
                            ),
                        }
                    )
                success = state == item.expected["agent_goal"]
                if success:
                    break
            except (ValueError, KeyError, TypeError, ValidationError):
                trace["invalid_returned_call"] = True
                break
        sample.values[item.item_id] = float(success)
        trace.update(final_state=state, success=success)
        sample.details.setdefault("agent_traces", {})[item.item_id] = trace
    return output
