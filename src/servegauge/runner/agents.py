"""Bounded multi-turn simulated agents; every model turn uses the reserved runner."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from jsonschema import ValidationError

from servegauge.core.hashing import content_hash
from servegauge.core.ids import safe_filename
from servegauge.core.models import Exchange, Item
from servegauge.items.agent_tasks import execute_tool
from servegauge.runner.plan import RunPlan
from servegauge.runner.scheduler import run_plan
from servegauge.scoring.common import Samples, calls, choice, message
from servegauge.scoring.toolcall_scoring import parsed_call
from servegauge.storage.raw_store import RawStore

if TYPE_CHECKING:
    from servegauge.runner.captured import CapturedCheck


async def run_agents(
    plan: RunPlan,
    items: list[Item],
    initial_records: list[Exchange],
    *,
    root: Path,
    run_id: str,
    approved: bool,
    transports: dict[str, httpx.AsyncBaseTransport],
    budget_scope: str | None = None,
    captured: CapturedCheck | None = None,
) -> tuple[dict[str, dict[str, Samples]], dict[str, Any]]:
    tasks = [item for item in items if item.expected.get("agent_goal") is not None]
    outputs: dict[str, dict[str, Samples]] = {}
    traces: dict[str, Any] = {}
    store = RawStore(root / "data/raw")
    for endpoint in plan.endpoints:
        outputs[endpoint.id] = {}
        if not tasks:
            continue
        states = {item.item_id: copy.deepcopy(item.expected["agent_initial"]) for item in tasks}
        histories = {item.item_id: copy.deepcopy(item.payload["messages"]) for item in tasks}
        pending = {item.item_id: item for item in tasks}
        current = {
            record.item_id: record
            for record in initial_records
            if record.endpoint_id == endpoint.id and record.item_id in pending
        }
        endpoint_trace: dict[str, Any] = {
            item.item_id: {"run_ids": [run_id], "exchange_ids": [], "final_state": None, "turns": 0}
            for item in tasks
        }
        for turn in range(max(item.expected["max_turns"] for item in tasks)):
            next_items = []
            next_original = {}
            for original_id, item in list(pending.items()):
                check = item.expected["check_id"]
                sample = outputs[endpoint.id].setdefault(
                    check,
                    Samples(
                        check_id=check,
                        check_class="report_only" if item.suite_id == "S8" else "comparative",
                    ),
                )
                trace = endpoint_trace[original_id]
                record = current.get(original_id)
                success = False
                finished = False
                if record is None:
                    sample.mark("ERROR", "agent turn capture missing")
                    finished = True
                else:
                    sample.evidence.append(record.exchange_id)
                    trace["exchange_ids"].append(record.exchange_id)
                    trace["turns"] += 1
                    if record.error:
                        sample.mark("ERROR", "agent transport or excluded-routing capture")
                        finished = True
                    elif record.metadata.get("request_not_sent"):
                        sample.mark(
                            record.metadata.get("verdict", "INCONCLUSIVE"),
                            "agent capability unavailable",
                        )
                        finished = True
                    elif record.status != 200:
                        sample.mark("NOT_SUPPORTED", "agent request rejected")
                        finished = True
                    else:
                        try:
                            if choice(record).get("finish_reason") == "length":
                                sample.mark("INCONCLUSIVE", "output cap censored agent turn")
                                finished = True
                            else:
                                returned = calls(record)
                                assistant = copy.deepcopy(message(record))
                                if any(
                                    assistant.get(key)
                                    for key in (
                                        "reasoning",
                                        "reasoning_details",
                                        "reasoning_content",
                                    )
                                ):
                                    trace["turns_with_reasoning"] = (
                                        trace.get("turns_with_reasoning", 0) + 1
                                    )
                                if not item.expected["pass_reasoning"]:
                                    for key in (
                                        "reasoning",
                                        "reasoning_details",
                                        "reasoning_content",
                                    ):
                                        assistant.pop(key, None)
                                histories[original_id].append(assistant)
                                if len(returned) > item.expected["max_turns"]:
                                    raise ValueError("too_many_calls_for_task_budget")
                                if not returned:
                                    finished = True
                                for call in returned:
                                    name, args = parsed_call(call)
                                    identifier = call.get("id")
                                    if not isinstance(identifier, str) or not identifier:
                                        raise ValueError("missing_tool_call_id")
                                    states[original_id] = execute_tool(
                                        states[original_id], name, args
                                    )
                                    histories[original_id].append(
                                        {
                                            "role": "tool",
                                            "tool_call_id": identifier,
                                            "content": json.dumps(
                                                {"ok": True, "state": states[original_id][name]},
                                                ensure_ascii=False,
                                            ),
                                        }
                                    )
                                success = states[original_id] == item.expected["agent_goal"]
                                finished |= success or turn + 1 >= item.expected["max_turns"]
                        except (ValueError, KeyError, TypeError, ValidationError):
                            finished = True
                            trace["invalid_returned_call"] = True
                if finished:
                    sample.values[original_id] = float(success)
                    trace["final_state"] = states[original_id]
                    trace["success"] = success
                    del pending[original_id]
                else:
                    child = item.model_copy(deep=True)
                    child.item_id = f"{original_id}-turn-{turn + 1}"
                    child.payload["messages"] = histories[original_id]
                    next_items.append(child)
                    next_original[child.item_id] = original_id
            if not next_items:
                break
            child_run = run_id + "-agent-" + content_hash(endpoint.id)[:8] + f"-{turn + 1}"
            child_plan = plan.model_copy(update={"endpoints": [endpoint]})
            # Deterministic child IDs and manifests make an interrupted continuation resumable.
            exists = (root / "data/runs" / safe_filename(child_run) / "manifest.json").exists()
            await run_plan(
                child_plan,
                root=root,
                run_id=child_run,
                resume=exists,
                approved=approved,
                transports=transports,
                items_override=next_items,
                budget_scope=budget_scope or run_id,
                captured=captured.agent(child_run[len(run_id) :]) if captured is not None else None,
            )
            current = {
                next_original[record.item_id]: record
                for record in store.exchanges(run_id=child_run)
            }
            for original_id in next_original.values():
                endpoint_trace[original_id]["run_ids"].append(child_run)
        for original_id in pending:
            check = pending[original_id].expected["check_id"]
            outputs[endpoint.id][check].values[original_id] = 0.0
            endpoint_trace[original_id].update(final_state=states[original_id], success=False)
        traces[endpoint.id] = endpoint_trace
        for check, sample in outputs[endpoint.id].items():
            sample.details["agent_traces"] = {
                key: value
                for key, value in endpoint_trace.items()
                if next(item for item in tasks if item.item_id == key).expected["check_id"] == check
            }
    return outputs, traces
