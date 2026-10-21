"""Compile pinned original requests; preparation never executes either comparison."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.ids import safe_filename
from servegauge.core.models import CheckResult, EndpointSpec, Item, ValueModel
from servegauge.diagnose.repro import _mock_replay_inventory
from servegauge.mock.sealed_replay import ReplayObservation
from servegauge.runner.plan import RunPlan
from servegauge.runner.retry import retryable
from servegauge.storage.comparison import (
    ComparisonSelection,
    SourcePin,
    _private_json,
    _relative,
    _require_sanitized,
)
from servegauge.storage.comparison_replay import load_replay_book
from servegauge.suites.registry import item_request, prepare_item


class CompiledExchange(ValueModel):
    source: ReplayObservation
    source_endpoint: EndpointSpec
    item: Item
    source_item_hash: str
    source_context_sha256: str
    wire_body: dict[str, Any] | None
    logical_request: dict[str, Any]
    preparation_verdict: Literal["NOT_SUPPORTED", "INCONCLUSIVE"] | None
    preparation_reason: str | None
    preparation_metadata: dict[str, Any]


class CompiledPhase(ValueModel):
    source_run_id: str
    source_plan: RunPlan
    source_context_sha256: str
    items: list[Item]
    exchanges: list[CompiledExchange]


class CompiledCase(ValueModel):
    case_id: str
    original_check: CheckResult
    population_n: int
    logical_unit_keys: list[str]
    phases: list[CompiledPhase]
    replay_adds_independent_n: Literal[False] = False


class ComparisonProgram(ValueModel):
    version: Literal["comparison-program-v1"] = "comparison-program-v1"
    state: Literal["compiled_only"] = "compiled_only"
    selection_hash: str
    cases: list[CompiledCase]
    owner_comparison_executed: Literal[False] = False
    comparison_setup_seconds: None = None

    def program_hash(self) -> str:
        return content_hash(self.model_dump(mode="json"))


def _context(snapshot: Path, pin: SourcePin, run_id: str) -> tuple[dict[str, Any], str]:
    name = f"{safe_filename(run_id)}/capture.json"
    expected = pin.files_sha256.get(name)
    if expected is None:
        raise ValueError("comparison_program_context_pin_missing")
    path = _relative(snapshot, f"sources/{safe_filename(pin.run_id)}/data/raw/{name}")
    blob = path.read_bytes()
    if sha256(blob).hexdigest() != expected:
        raise ValueError("comparison_program_context_pin_changed")
    envelope = _private_json(blob)
    if (
        not isinstance(envelope, dict)
        or set(envelope) != {"value", "hash"}
        or not isinstance(envelope["value"], dict)
        or content_hash(envelope["value"]) != envelope["hash"]
    ):
        raise ValueError("comparison_program_context_seal_invalid")
    value = dict(envelope["value"])
    _require_sanitized(value)
    if value.get("manifest", {}).get("run_id") != run_id:
        raise ValueError("comparison_program_context_identity_changed")
    return value, expected


def _compile_exchange(
    source: ReplayObservation, endpoint: EndpointSpec, item: Item, plan: RunPlan, context_hash: str
) -> CompiledExchange:
    record = source.record
    if (record.endpoint_id, record.suite_id, record.item_id) != (
        endpoint.id,
        item.suite_id,
        item.item_id,
    ):
        raise ValueError("comparison_program_source_identity_changed")
    restored = (
        _mock_replay_inventory([item], [record], endpoint.id, record.run_id)[0]
        if endpoint.kind == "mock"
        else item
    )
    decision = prepare_item(restored, endpoint)
    logical = item_request(restored, endpoint).model_dump(mode="json")
    expected_request = decision.body if decision.body is not None else logical
    if canonical_json(expected_request) != canonical_json(record.request):
        raise ValueError("comparison_program_wire_request_mismatch")
    if (decision.body is None) != (record.metadata.get("request_not_sent") is True):
        raise ValueError("comparison_program_dispatch_state_mismatch")
    if decision.body is None:
        if record.attempts or (
            record.metadata.get("verdict") != decision.verdict
            or record.metadata.get("skip_reason") != decision.reason
        ):
            raise ValueError("comparison_program_skip_decision_mismatch")
    else:
        if not record.attempts or len(record.attempts) > plan.retries + 1:
            raise ValueError("comparison_program_retry_budget_mismatch")
        allowed = item.expected.get("allowed_statuses", [200])
        if not isinstance(allowed, list) or any(type(status) is not int for status in allowed):
            raise ValueError("comparison_program_allowed_statuses_invalid")
        for index, attempt in enumerate(record.attempts):
            wants_retry = retryable(attempt) and (
                attempt.error is not None or attempt.status not in allowed
            )
            retries_remain = index < plan.retries
            if (index + 1 < len(record.attempts)) != (wants_retry and retries_remain):
                raise ValueError("comparison_program_retry_decision_mismatch")
    return CompiledExchange(
        source=source.model_copy(deep=True),
        source_endpoint=endpoint.model_copy(deep=True),
        item=restored.model_copy(deep=True),
        source_item_hash=content_hash(item.model_dump(mode="json")),
        source_context_sha256=context_hash,
        wire_body=decision.body,
        logical_request=logical,
        preparation_verdict=decision.verdict,
        preparation_reason=decision.reason,
        preparation_metadata=decision.metadata,
    )


def _compile_phase(
    snapshot: Path, pin: SourcePin, run_id: str, rows: list[ReplayObservation]
) -> CompiledPhase:
    context, digest = _context(snapshot, pin, run_id)
    plan = RunPlan.model_validate(context["manifest"]["config"])
    inventory = [Item.model_validate(value) for value in context["items"]]
    items = {(i.suite_id, i.item_id): i for i in inventory}
    endpoints = {endpoint.id: endpoint for endpoint in plan.endpoints}
    wanted = {(row.record.suite_id, row.record.item_id) for row in rows}
    if len(items) != len(inventory) or not wanted <= items.keys():
        raise ValueError("comparison_program_item_inventory_changed")
    selected = [item for item in inventory if (item.suite_id, item.item_id) in wanted]
    seen: set[str] = set()
    for item in selected:
        dependencies = item.expected.get("depends_on", [])
        if not isinstance(dependencies, list) or any(value not in seen for value in dependencies):
            raise ValueError("comparison_program_dependency_support_missing")
        seen.add(item.item_id)
    selected_endpoints = {row.record.endpoint_id for row in rows}
    if not selected_endpoints <= endpoints.keys():
        raise ValueError("comparison_program_endpoint_inventory_changed")
    for endpoint_id in selected_endpoints:
        if endpoints[endpoint_id].kind == "mock":
            _mock_replay_inventory(selected, [row.record for row in rows], endpoint_id, run_id)
    # An ordinary run executes each selected item at every selected endpoint.
    # Refuse a partial rectangle rather than add missing observations or shrink n.
    for endpoint_id in selected_endpoints:
        for item in selected:
            samples = item.expected.get("samples", 1)
            actual = sorted(
                row.record.sample_idx
                for row in rows
                if (row.record.endpoint_id, row.record.suite_id, row.record.item_id)
                == (endpoint_id, item.suite_id, item.item_id)
            )
            if type(samples) is not int or samples < 1 or actual != list(range(samples)):
                raise ValueError("comparison_program_complete_sample_support_required")
    exchanges = [
        _compile_exchange(
            row,
            endpoints[row.record.endpoint_id],
            items[(row.record.suite_id, row.record.item_id)],
            plan,
            digest,
        )
        for row in sorted(rows, key=lambda row: row.source_position)
    ]
    return CompiledPhase(
        source_run_id=run_id,
        source_plan=plan,
        source_context_sha256=digest,
        items=selected,
        exchanges=exchanges,
    )


def compile_program(snapshot: Path, *, selection_hash: str) -> ComparisonProgram:
    """Compile all captured bodies with their original adapter before any dispatch.

    The returned private source plans contain historical upstream bindings. They
    are provenance, not runnable local replay plans. Runner integration must create
    separate zero-cost literal-loopback plans and retain these source identities.
    """
    book = load_replay_book(snapshot, selection_hash=selection_hash)
    selection = ComparisonSelection.model_validate(
        _private_json(_relative(snapshot, "selection.json").read_bytes())
    )
    if content_hash(selection.model_dump(mode="json")) != selection_hash:
        raise ValueError("comparison_program_selection_changed")
    pins = {pin.run_id: pin for pin in selection.sources}
    compiled = []
    for case in book.cases:
        case_id = case["case"]["case_id"]
        rows = [row for row in book.observations if row.case_id == case_id]
        phases = [
            _compile_phase(
                snapshot,
                pins[case["case"]["run_id"]],
                run_id,
                [row for row in rows if row.record.run_id == run_id],
            )
            for run_id in sorted({row.record.run_id for row in rows})
        ]
        compiled.append(
            CompiledCase(
                case_id=case_id,
                original_check=CheckResult.model_validate(case["original_check"]),
                population_n=case["population_n"],
                logical_unit_keys=case["logical_unit_keys"],
                phases=phases,
            )
        )
    book.tape()  # Validate delivery representation without consuming a source slot.
    return ComparisonProgram(selection_hash=selection_hash, cases=compiled)
