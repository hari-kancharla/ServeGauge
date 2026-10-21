"""Separately anchored complete matched controls; never execute a comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from servegauge.core.hashing import content_hash
from servegauge.core.ids import safe_filename
from servegauge.core.models import Exchange, ModelProfile, ValueModel
from servegauge.mock.sealed_replay import FiniteReplay, ReplayObservation
from servegauge.runner.evaluate import collect, evaluate
from servegauge.storage.comparison import (
    ComparisonSelection,
    Digest,
    _load,
    _private_json,
    _relative,
    _require_sanitized,
)
from servegauge.storage.comparison_program import (
    ComparisonProgram,
    CompiledCase,
    _compile_phase,
    compile_program,
)
from servegauge.verdicts.config import DecisionPolicy


class ControlRequest(ValueModel):
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    failure_case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    endpoint_id: str = Field(min_length=1)


class SelectedControl(ControlRequest):
    check_hash: Digest


class ControlSelection(ValueModel):
    version: Literal["comparison-controls-v1"] = "comparison-controls-v1"
    failure_selection_hash: Digest
    controls: list[SelectedControl] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def distinct_controls(self) -> ControlSelection:
        if (
            len({row.case_id for row in self.controls}) != 2
            or len({row.failure_case_id for row in self.controls}) != 2
        ):
            raise ValueError("comparison_distinct_matched_controls_required")
        return self


def _derive(
    snapshot: Path,
    *,
    selection_hash: str,
    requests: list[ControlRequest],
) -> list[CompiledCase]:
    # This verifies the entire sealed source closure and the original failures.
    original = compile_program(snapshot, selection_hash=selection_hash)
    selection = ComparisonSelection.model_validate(
        _private_json(_relative(snapshot, "selection.json").read_bytes())
    )
    requests = [ControlRequest.model_validate(row.model_dump(mode="json")) for row in requests]
    _require_sanitized([row.model_dump(mode="json") for row in requests])
    failures = {case.case_id: case for case in original.cases}
    if (
        len(requests) != 2
        or {row.failure_case_id for row in requests} != failures.keys()
        or len({row.case_id for row in requests}) != 2
        or any(row.case_id in failures for row in requests)
    ):
        raise ValueError("comparison_exactly_one_distinct_control_per_failure_required")
    sources = {
        pin.run_id: _load(snapshot / "sources" / safe_filename(pin.run_id), pin)
        for pin in selection.sources
    }
    result = []
    for request in requests:
        failure = failures[request.failure_case_id]
        finding = failure.original_check
        source = sources[finding.run_id]
        if request.endpoint_id == finding.endpoint_id or request.endpoint_id not in {
            endpoint.id for endpoint in source.analysis.plan.endpoints
        }:
            raise ValueError("comparison_distinct_existing_control_endpoint_required")
        primary = next(phase for phase in failure.phases if phase.source_run_id == finding.run_id)
        slots = {
            (row.source.record.suite_id, row.source.record.item_id, row.source.record.sample_idx)
            for row in primary.exchanges
            if row.source.role == "primary"
        }
        records = [
            row
            for row in source.records.values()
            if row.run_id == finding.run_id
            and row.endpoint_id == request.endpoint_id
            and (row.suite_id, row.item_id, row.sample_idx) in slots
        ]
        if len(records) != len(slots):
            raise ValueError("comparison_complete_matched_control_support_required")
        wanted_items = {(suite, item) for suite, item, _ in slots}
        items = [item for item in primary.items if (item.suite_id, item.item_id) in wanted_items]
        plan = source.analysis.plan.model_copy(deep=True)
        plan.endpoints = [
            endpoint for endpoint in plan.endpoints if endpoint.id == request.endpoint_id
        ]
        profile = ModelProfile.model_validate(source.analysis.metadata["profile"])
        policy = DecisionPolicy.model_validate(source.analysis.metadata["policy"])
        # Fresh scores from only the control's primary raw observations avoid
        # accidentally inheriting the failed target's confirmation metadata.
        samples = collect(plan, items, records, profile, policy)
        candidates = [
            row
            for row in evaluate(
                plan, finding.run_id, items, samples, policy, target_ids=[request.endpoint_id]
            )
            if row.check_id == finding.check_id
        ]
        if len(candidates) != 1:
            raise ValueError("comparison_unique_control_aggregate_required")
        check = candidates[0]
        unit_keys = sorted(samples[request.endpoint_id][finding.check_id].values)
        if (
            check.check_class != "invariant"
            or check.verdict not in {"PASS", "INCONCLUSIVE"}
            or check.violations != 0
            or check.n != failure.population_n
            or unit_keys != failure.logical_unit_keys
            or check.ci_low is None
            or check.ci_high is None
        ):
            raise ValueError("comparison_complete_clean_matched_control_required")
        check.details["comparison_aggregate_origin"] = "raw_reconstructed_control"
        check.details["comparison_matched_failure_case_id"] = request.failure_case_id
        positions = {key: index for index, key in enumerate(source.records)}
        observed: list[tuple[Literal["primary", "preflight"], Exchange]] = [
            ("primary", row) for row in records
        ]
        cap_runs = {
            phase.source_run_id
            for phase in failure.phases
            if any(row.source.role == "preflight" for row in phase.exchanges)
        }
        observed.extend(
            ("preflight", row)
            for row in source.records.values()
            if row.run_id in cap_runs
            and row.suite_id == "S0"
            and row.endpoint_id == request.endpoint_id
        )
        observations = [
            ReplayObservation(
                case_id=request.case_id,
                role=role,
                source_position=positions[row.exchange_id],
                record=row.model_copy(deep=True),
                record_hash=content_hash(row.model_dump(mode="json")),
                request_hash=content_hash(row.request),
            )
            for role, row in observed
        ]
        FiniteReplay(observations)  # Validate delivery without consuming a slot.
        phases = [
            _compile_phase(
                snapshot, source.pin, run, [row for row in observations if row.record.run_id == run]
            )
            for run in sorted({row.record.run_id for row in observations})
        ]
        result.append(
            CompiledCase(
                case_id=request.case_id,
                original_check=check,
                population_n=check.n,
                logical_unit_keys=unit_keys,
                phases=phases,
            )
        )
    return result


def propose_controls(
    snapshot: Path,
    *,
    selection_hash: str,
    requests: list[ControlRequest],
) -> ControlSelection:
    """Return reviewable control pins; a proposal confers no owner approval."""
    cases = _derive(snapshot, selection_hash=selection_hash, requests=requests)
    return ControlSelection(
        failure_selection_hash=selection_hash,
        controls=[
            SelectedControl(
                **request.model_dump(mode="json"),
                check_hash=content_hash(case.original_check.model_dump(mode="json")),
            )
            for request, case in zip(requests, cases, strict=True)
        ],
    )


def compile_controls(
    snapshot: Path,
    selection: ControlSelection,
    *,
    selection_hash: str,
    controls_hash: str,
) -> ComparisonProgram:
    """Reconstruct matched controls under two external anchors, without execution."""
    selection = ControlSelection.model_validate(selection.model_dump(mode="json"))
    _require_sanitized(selection.model_dump(mode="json"))
    if (
        selection.failure_selection_hash != selection_hash
        or content_hash(selection.model_dump(mode="json")) != controls_hash
    ):
        raise ValueError("comparison_control_selection_changed")
    cases = _derive(
        snapshot,
        selection_hash=selection_hash,
        requests=[
            ControlRequest.model_validate(row.model_dump(exclude={"check_hash"}))
            for row in selection.controls
        ],
    )
    if any(
        content_hash(case.original_check.model_dump(mode="json")) != row.check_hash
        for case, row in zip(cases, selection.controls, strict=True)
    ):
        raise ValueError("comparison_control_aggregate_changed")
    return ComparisonProgram(selection_hash=controls_hash, cases=cases)
