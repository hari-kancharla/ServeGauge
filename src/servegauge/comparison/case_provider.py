"""Whole-case finite external callback; raw channels only, never a detector verdict."""

from __future__ import annotations

import hashlib
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from servegauge.core.hashing import canonical_json
from servegauge.core.models import Exchange, Item
from servegauge.runner.captured import CapturedCheck, CapturedRun
from servegauge.runner.ledger import Ledger
from servegauge.runner.scheduler import run_plan
from servegauge.storage.comparison import _guard, _private_json, _relative, _require_sanitized
from servegauge.storage.comparison_program import ComparisonProgram
from servegauge.storage.workspace import initialize


def _key(check: str, item: Item) -> str:
    if check == "T06":
        return (
            item.item_id
            if item.expected["check_id"] == "tools-corpus"
            else ("structure-" + str(item.expected["trial"]))
        )
    return str(item.expected["trial"])


def _units(
    check: str, captures: list[tuple[Item, dict[str, Any]]], *, require_pairs: bool
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[Item, dict[str, Any]]]] = defaultdict(list)
    for item, record in captures:
        grouped[_key(check, item)].append((item, record))
    result = {}
    for key, rows in grouped.items():
        structural = check == "T06" and all(
            row.expected["check_id"].startswith("T04") for row, _ in rows
        )
        paired = (
            structural
            and len(rows) == 2
            and {row.payload.get("stream", False) for row, _ in rows} == {True, False}
        )
        if structural and not paired and (require_pairs or len(rows) != 1):
            raise ValueError("comparison_complete_structural_unit_required")
        result[key] = {"paired_structure": paired, "captures": [record for _, record in rows]}
    return result


def _projection(record: Exchange) -> dict[str, Any]:
    return {
        "source_address": record.metadata["captured_source"],
        "request": record.request,
        "status": record.status,
        "error": record.error,
        "request_not_sent": record.metadata.get("request_not_sent") is True,
        "response": record.response,
        "response_text": record.response_text,
        "sse_events": [event.model_dump(mode="json") for event in record.sse_events or []],
    }


async def run_raw_case(program: ComparisonProgram, *, case_id: str, root: Path) -> dict[str, Any]:
    """Consume exactly one declared case through reserved runner transports.

    Capability gates come from pinned original contexts, not a newly inferred
    grader result. Preflight, paired controls and captured confirmations are all
    consumed and inventoried; only target logical units enter the independent
    assertion. Their captured reuse supplies no independent population n.
    """
    _guard(root)
    if root.exists():
        raise ValueError("comparison_fresh_case_output_required")
    context = CapturedCheck(program, case_id=case_id)
    root.mkdir(mode=0o700, parents=True)
    initialize(root)
    primary: dict[str, Any] = {}
    confirmations = []
    support = []
    phases: list[CapturedRun] = []
    preflight = context.preflight()
    if preflight is not None:
        phases.append(preflight)
    phases.append(context.primary())
    for index, _ in enumerate(context.case.original_check.details.get("replay_batch_ids", [])):
        phases.append(context.confirmation(context.case.original_check.check_id, index))
    for index, phase in enumerate(phases):
        captured: list[Exchange] = []
        await run_plan(
            phase.local_plan(),
            root=root,
            run_id=f"captured-phase-{index}",
            approved=True,
            transports=phase.transports(),
            items_override=phase.phase.items,
            captured=phase,
            after_capture=captured.append,
        )
        mapped = {
            (row.endpoint_id, row.suite_id, row.item_id, row.sample_idx): row for row in captured
        }
        target = []
        for row in phase.phase.exchanges:
            source = row.source.record
            record = mapped[
                (source.endpoint_id, source.suite_id, source.item_id, source.sample_idx)
            ]
            projection = _projection(record)
            # Use the original address, never the replay's local run identity.
            projection["source_address"] = {
                "case_id": case_id,
                "role": row.source.role,
                "source_run_id": source.run_id,
                "endpoint_id": source.endpoint_id,
                "suite_id": source.suite_id,
                "item_id": source.item_id,
                "sample_idx": source.sample_idx,
                "exchange_id": source.exchange_id,
            }
            support.append(projection)
            if (
                row.source.role in {"primary", "confirmation"}
                and source.endpoint_id == context.case.original_check.endpoint_id
            ):
                target.append((row.item, projection))
        if target:
            units = _units(
                context.case.original_check.check_id,
                target,
                require_pairs=phase is context.primary(),
            )
            if phase is context.primary():
                primary = units
            else:
                confirmations.append(list(units.values()))
    context.tape.assert_complete()
    if set(primary) != set(context.case.logical_unit_keys):
        raise ValueError("comparison_complete_original_population_required")
    ledger = Ledger(root)
    ledger.refresh()
    if ledger.pending or ledger.development_spend() != 0:
        raise ValueError("comparison_zero_cost_ledger_required")
    result = {
        "primary": primary,
        "confirmations": confirmations,
        "support": support,
        "replay_adds_independent_n": False,
    }
    _require_sanitized(result)
    for name, value in [("finite-audit.json", context.tape.audit()), ("raw-result.json", result)]:
        descriptor = os.open(root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json(value) + b"\n")
    return result


_worker_scope: tuple[Path, str] | None = None
_worker_refused = False
_completed_outputs: dict[Path, str] = {}


async def call_api(prompt: str, options: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Actual external-tool hook; an existing case output forbids duplicate replay."""
    global _worker_scope, _worker_refused
    try:
        if _worker_refused:
            raise ValueError("comparison_provider_worker_already_refused")
        for completed, expected in _completed_outputs.items():
            _guard(completed)
            if hashlib.sha256(completed.read_bytes()).hexdigest() != expected:
                raise ValueError("comparison_completed_output_changed")
        config = options["config"]
        if (
            not isinstance(config, dict)
            or set(config) - {"program", "program_sha256", "basePath"}
            or not {"program", "program_sha256"} <= config.keys()
        ):
            raise ValueError("comparison_provider_configuration_invalid")
        if "basePath" in config and Path(config["basePath"]).absolute() != Path.cwd():
            raise ValueError("comparison_provider_base_scope_changed")
        path = _relative(Path.cwd(), config["program"])
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != config["program_sha256"]:
            raise ValueError("comparison_provider_program_changed")
        scope = (path.absolute(), digest)
        if _worker_scope is not None and _worker_scope != scope:
            raise ValueError("comparison_provider_worker_scope_changed")
        _worker_scope = scope
        program = ComparisonProgram.model_validate(_private_json(blob))
        case_id = context["vars"]["case_id"]
        if (
            not isinstance(case_id, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", case_id) is None
            or prompt != case_id
            or case_id not in {case.case_id for case in program.cases}
        ):
            raise ValueError("comparison_provider_exact_case_required")
        value = await run_raw_case(program, case_id=case_id, root=Path.cwd() / ("case-" + case_id))
        for name in ["raw-result.json", "finite-audit.json"]:
            completed = Path.cwd() / ("case-" + case_id) / name
            _guard(completed)
            _completed_outputs[completed] = hashlib.sha256(completed.read_bytes()).hexdigest()
        return {"output": value, "cost": 0}
    except (KeyError, TypeError, ValueError, OSError):
        _worker_refused = True
        return {"error": "captured_case_scope_or_channels_refused", "cost": 0}
