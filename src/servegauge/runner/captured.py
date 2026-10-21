"""Verified finite source inputs through the ordinary zero-cost reserved runner."""

from __future__ import annotations

import copy
from typing import Any

import httpx

from servegauge.adapters.base import PreparedRequest
from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.models import Attempt, CheckResult, EndpointSpec, Exchange, Item, SseEvent
from servegauge.diagnose.repro import _mock_replay_inventory
from servegauge.mock.sealed_replay import (
    REPLAY_ADDRESS_EXTENSION,
    FiniteReplay,
    ReplayAddress,
    ReplayRefused,
    ReplayTransport,
)
from servegauge.runner.plan import RunPlan
from servegauge.storage.comparison_program import (
    ComparisonProgram,
    CompiledCase,
    CompiledExchange,
    CompiledPhase,
)
from servegauge.suites.registry import prepare_item

QueueEntry = tuple[EndpointSpec, Item, int, PreparedRequest]
Slot = tuple[str, str, str, int]


def _slot(row: CompiledExchange) -> Slot:
    record = row.source.record
    return record.endpoint_id, record.suite_id, record.item_id, record.sample_idx


def _events(events: list[SseEvent] | None) -> list[dict[str, Any]]:
    return [event.model_dump(exclude={"t_ms"}) for event in events or []]


class CapturedRun:
    def __init__(
        self,
        case: CompiledCase,
        phase: CompiledPhase,
        *,
        selection_hash: str,
        program_hash: str,
        tape: FiniteReplay,
    ) -> None:
        if phase not in case.phases or any(
            row.source.case_id != case.case_id for row in phase.exchanges
        ):
            raise ValueError("captured_replay_phase_identity_mismatch")
        self.phase = phase.model_copy(deep=True)
        self.case_id, self.selection_hash, self.program_hash = (
            case.case_id,
            selection_hash,
            program_hash,
        )
        self.tape = tape
        self._rows = {_slot(row): row for row in self.phase.exchanges}
        if len(self._rows) != len(self.phase.exchanges):
            raise ValueError("captured_replay_duplicate_slot")
        self._captured: set[Slot] = set()
        self._started = False
        self._allowed_endpoints = {
            row.source.record.endpoint_id for part in case.phases for row in part.exchanges
        }
        binding = {
            "selection_hash": selection_hash,
            "program_hash": program_hash,
            "case_id": case.case_id,
        }
        endpoints = []
        for source in phase.source_plan.endpoints:
            if source.id not in {row.source.record.endpoint_id for row in phase.exchanges}:
                continue
            local = source.model_copy(deep=True)
            local.kind, local.base_url, local.api_key_env, local.openrouter_pin = (
                "mock",
                "http://127.0.0.1:8219/v1",
                None,
                None,
            )
            local.metadata.pop("pricing", None)
            local.metadata.pop("price_provenance", None)
            local.metadata["captured_replay"] = binding.copy()
            endpoints.append(local)
        endpoints.sort(key=lambda endpoint: endpoint.id != case.original_check.endpoint_id)
        self._plan = phase.source_plan.model_copy(
            update={
                "plan_id": "comparison-" + case.case_id,
                "endpoints": endpoints,
                "suites": sorted({row.item.suite_id for row in phase.exchanges}),
                "max_usd": 0,
                "concurrency_per_endpoint": 1,
                "requests_per_minute": 1e9,
                "tokens_per_minute": None,
                "retry_base_seconds": 0,
                "retry_max_seconds": 0,
            },
            deep=True,
        )

    def local_plan(self) -> RunPlan:
        return self._plan.model_copy(deep=True)

    def transports(self) -> dict[str, httpx.AsyncBaseTransport]:
        return {
            endpoint.id: ReplayTransport(self.tape, url=endpoint.base_url + "/chat/completions")
            for endpoint in self._plan.endpoints
        }

    def validate(
        self,
        plan: RunPlan,
        items: list[Item],
        transports: dict[str, httpx.AsyncBaseTransport],
        *,
        resume: bool,
        dry_run: bool,
        approved: bool,
    ) -> None:
        actual_plan = plan.model_dump(mode="json")
        expected_plan = self._plan.model_dump(mode="json")
        for value in [actual_plan, expected_plan]:
            for endpoint_value in value["endpoints"]:
                endpoint_value["metadata"].pop("observed_capabilities", None)
        if resume or self._started or not approved or actual_plan != expected_plan:
            raise ValueError("captured_replay_local_plan_mismatch")
        if [item.model_dump(mode="json") for item in items] != [
            item.model_dump(mode="json") for item in self.phase.items
        ]:
            raise ValueError("captured_replay_item_inventory_mismatch")
        if (
            not {endpoint.id for endpoint in plan.endpoints} <= transports.keys()
            or not set(transports) <= self._allowed_endpoints
        ):
            raise ValueError("captured_replay_transport_inventory_mismatch")
        for transport in transports.values():
            if (
                not isinstance(transport, ReplayTransport)
                or transport.tape is not self.tape
                or transport.url != httpx.URL("http://127.0.0.1:8219/v1/chat/completions")
            ):
                raise ValueError("captured_replay_verified_transport_required")
        if not dry_run:
            self._started = True

    def _row(self, endpoint_id: str, item: Item, sample_idx: int) -> CompiledExchange:
        row = self._rows.get((endpoint_id, item.suite_id, item.item_id, sample_idx))
        if row is None or content_hash(item.model_dump(mode="json")) != row.source_item_hash:
            raise ReplayRefused("captured_replay_source_slot_unknown")
        return row

    def prepared(self, endpoint: EndpointSpec, item: Item) -> PreparedRequest:
        row = self._row(endpoint.id, item, 0)
        original = row.source_endpoint.model_copy(deep=True)
        if "observed_capabilities" in endpoint.metadata:
            original.metadata["observed_capabilities"] = copy.deepcopy(
                endpoint.metadata["observed_capabilities"]
            )
        restored = (
            _mock_replay_inventory(
                [item], [row.source.record], original.id, row.source.record.run_id
            )[0]
            if original.kind == "mock"
            else item
        )
        decision = prepare_item(restored, original)
        if (canonical_json(decision.body), decision.verdict, decision.reason) != (
            canonical_json(row.wire_body),
            row.preparation_verdict,
            row.preparation_reason,
        ):
            raise ValueError("captured_replay_capability_preparation_mismatch")
        return PreparedRequest(
            copy.deepcopy(row.wire_body),
            row.preparation_verdict,
            row.preparation_reason,
            copy.deepcopy(row.preparation_metadata),
        )

    def logical_request(
        self, endpoint: EndpointSpec, item: Item, sample_idx: int
    ) -> dict[str, Any]:
        return copy.deepcopy(self._row(endpoint.id, item, sample_idx).logical_request)

    def order(self, entries: list[QueueEntry]) -> list[QueueEntry]:
        indexed = {
            (endpoint.id, item.suite_id, item.item_id, sample): entry
            for entry in entries
            for endpoint, item, sample, _ in [entry]
        }
        if len(indexed) != len(entries) or set(indexed) != set(self._rows):
            raise ReplayRefused("captured_replay_queue_inventory_mismatch")
        return [indexed[_slot(row)] for row in self.phase.exchanges]

    def _address(self, row: CompiledExchange, attempt_n: int) -> ReplayAddress:
        record = row.source.record
        if (
            attempt_n < 0
            or attempt_n > len(record.attempts)
            or (attempt_n == 0) != (not record.attempts)
        ):
            raise ReplayRefused("captured_replay_attempt_budget_exhausted")
        return ReplayAddress(
            case_id=self.case_id,
            role=row.source.role,
            source_run_id=record.run_id,
            endpoint_id=record.endpoint_id,
            suite_id=record.suite_id,
            item_id=record.item_id,
            sample_idx=record.sample_idx,
            exchange_id=record.exchange_id,
            attempt_n=attempt_n,
        )

    def extensions(
        self, endpoint: EndpointSpec, item: Item, sample_idx: int, attempt_n: int
    ) -> dict[str, Any]:
        row = self._row(endpoint.id, item, sample_idx)
        return {REPLAY_ADDRESS_EXTENSION: self._address(row, attempt_n).model_dump(mode="json")}

    def skip(self, endpoint: EndpointSpec, item: Item, sample_idx: int) -> None:

        row = self._row(endpoint.id, item, sample_idx)
        self.tape.skip(self._address(row, 0), canonical_json(row.logical_request))

    def provenance(self, row: CompiledExchange, attempt_n: int | None = None) -> dict[str, Any]:
        result = {
            "address": self._address(
                row, attempt_n if attempt_n is not None else len(row.source.record.attempts)
            ).model_dump(mode="json"),
            "record_hash": row.source.record_hash,
            "request_hash": row.source.request_hash,
            "source_item_hash": row.source_item_hash,
            "selection_hash": self.selection_hash,
            "program_hash": self.program_hash,
            "replay_adds_independent_n": False,
        }
        if attempt_n:
            result["attempt_hash"] = content_hash(
                row.source.record.attempts[attempt_n - 1].model_dump(mode="json")
            )
        return result

    def annotate_attempt(
        self, endpoint: EndpointSpec, item: Item, sample_idx: int, attempt: Attempt
    ) -> None:
        row = self._row(endpoint.id, item, sample_idx)
        attempt.metadata["captured_source"] = self.provenance(row, attempt.n)

    def validate_attempt(
        self, endpoint: EndpointSpec, item: Item, sample_idx: int, attempt: Attempt
    ) -> None:
        row = self._row(endpoint.id, item, sample_idx)
        expected = row.source.record.attempts[attempt.n - 1]
        if (attempt.status, attempt.error, attempt.response, _events(attempt.sse_events)) != (
            expected.status,
            expected.error,
            expected.response,
            _events(expected.sse_events),
        ) or (
            expected.response_text is not None and expected.response_text != attempt.response_text
        ):
            raise ReplayRefused("captured_replay_attempt_observation_mismatch")

    def capture(self, record: Exchange) -> None:
        key = (record.endpoint_id, record.suite_id, record.item_id, record.sample_idx)
        row = self._rows.get(key)
        if row is None or key in self._captured:
            raise ReplayRefused("captured_replay_capture_slot_mismatch")
        expected = row.source.record
        if (
            record.request,
            record.status,
            record.error,
            record.response,
            _events(record.sse_events),
            len(record.attempts),
        ) != (
            expected.request,
            expected.status,
            expected.error,
            expected.response,
            _events(expected.sse_events),
            len(expected.attempts),
        ):
            raise ReplayRefused("captured_replay_final_observation_mismatch")
        record.metadata["captured_source"] = self.provenance(row)
        self._captured.add(key)

    def finish(self) -> None:
        if self._captured != self._rows.keys() or self.tape.audit()["refusals"]:
            raise ReplayRefused("captured_replay_phase_incomplete")


class CapturedCheck:
    """A single pinned invariant case through ordinary preflight/confirmation logic."""

    def __init__(self, program: ComparisonProgram, *, case_id: str) -> None:
        cases = [case for case in program.cases if case.case_id == case_id]
        if len(cases) != 1 or cases[0].original_check.check_class != "invariant":
            raise ValueError("captured_replay_explicit_invariant_case_required")
        self.case = cases[0].model_copy(deep=True)
        self.selection_hash, self.program_hash = program.selection_hash, program.program_hash()
        self.tape = FiniteReplay(
            [row.source for phase in self.case.phases for row in phase.exchanges]
        )
        self._runs = {
            phase.source_run_id: CapturedRun(
                self.case,
                phase,
                selection_hash=self.selection_hash,
                program_hash=self.program_hash,
                tape=self.tape,
            )
            for phase in self.case.phases
        }
        self._main = self._runs[self.case.original_check.run_id]
        caps = [
            run
            for name, run in self._runs.items()
            if name != self.case.original_check.run_id
            and any(row.source.role == "preflight" for row in run.phase.exchanges)
        ]
        if len(caps) > 1:
            raise ValueError("captured_replay_preflight_phase_ambiguous")
        self._caps = caps[0] if caps else None
        confirmations = self.case.original_check.details.get("replay_batch_ids", [])
        expected = {
            name
            for name, run in self._runs.items()
            if any(row.source.role == "confirmation" for row in run.phase.exchanges)
        }
        if (
            not isinstance(confirmations, list)
            or any(not isinstance(name, str) for name in confirmations)
            or len(set(confirmations)) != len(confirmations)
            or set(confirmations) != expected
        ):
            raise ValueError("captured_replay_confirmation_order_unknown")
        self._confirmations = [self._runs[name] for name in confirmations]
        paired = {
            row.source.record.endpoint_id
            for row in self._main.phase.exchanges
            if row.source.role == "paired"
        }
        if len(paired) > 1:
            raise ValueError("captured_replay_baseline_ambiguous")
        self.baseline_id = next(iter(paired), None)

    def local_plan(self) -> RunPlan:
        plan = self._main.local_plan()
        if self._caps is not None:
            before = {endpoint.id: endpoint for endpoint in self._caps.local_plan().endpoints}
            if set(before) != {endpoint.id for endpoint in plan.endpoints}:
                raise ValueError("captured_replay_preflight_endpoint_mismatch")
            plan.endpoints = [
                before[endpoint.id].model_copy(deep=True) for endpoint in plan.endpoints
            ]
        return plan

    def primary_items(self) -> list[Item]:
        return [item.model_copy(deep=True) for item in self._main.phase.items]

    def capability_items(self) -> list[Item]:
        return [item.model_copy(deep=True) for item in self._caps.phase.items] if self._caps else []

    def transports(self) -> dict[str, httpx.AsyncBaseTransport]:
        return self._main.transports()

    def validate_inputs(
        self,
        plan: RunPlan,
        selected: set[str],
        inventory: list[Item] | None,
        capabilities: list[Item] | None,
        baseline_id: str | None,
        *,
        requirements: Any,
        reference_pack: Any,
    ) -> None:
        if (
            plan.model_dump(mode="json") != self.local_plan().model_dump(mode="json")
            or selected != {self.case.original_check.check_id}
            or baseline_id != self.baseline_id
            or requirements is not None
            or reference_pack is not None
        ):
            raise ValueError("captured_replay_check_scope_mismatch")
        if (
            inventory is None
            or capabilities is None
            or [i.model_dump(mode="json") for i in inventory]
            != [i.model_dump(mode="json") for i in self.primary_items()]
            or [i.model_dump(mode="json") for i in capabilities]
            != [i.model_dump(mode="json") for i in self.capability_items()]
        ):
            raise ValueError("captured_replay_check_inventory_mismatch")

    def preflight(self) -> CapturedRun | None:
        return self._caps

    def primary(self) -> CapturedRun:
        return self._main

    def confirmation(self, check: str, index: int) -> CapturedRun:
        if check != self.case.original_check.check_id or not 0 <= index < len(self._confirmations):
            raise ReplayRefused("captured_replay_confirmation_budget_exhausted")
        return self._confirmations[index]

    def agent(self, suffix: str) -> CapturedRun:
        run = self._runs.get(self.case.original_check.run_id + suffix)
        if run is None or not all(row.source.role == "agent" for row in run.phase.exchanges):
            raise ReplayRefused("captured_replay_agent_source_unknown")
        return run

    def _estimated_prepare(self, item: Item, endpoint: EndpointSpec) -> PreparedRequest:
        for run in [self._caps, self._main]:
            if run is None:
                continue
            row = run._rows.get((endpoint.id, item.suite_id, item.item_id, 0))
            if row is not None and row.source_item_hash == content_hash(
                item.model_dump(mode="json")
            ):
                return PreparedRequest(
                    copy.deepcopy(row.wire_body),
                    row.preparation_verdict,
                    row.preparation_reason,
                    copy.deepcopy(row.preparation_metadata),
                )
        raise ValueError("captured_replay_estimate_source_unknown")

    def estimate(self, plan: RunPlan, items: list[Item]) -> dict[str, Any]:
        from servegauge.runner.estimate import estimate_plan

        estimate = estimate_plan(plan, items_override=items, prepare=self._estimated_prepare)
        rows = [row for phase in self.case.phases for row in phase.exchanges]
        attempts = sum(len(row.source.record.attempts) for row in rows)
        skipped = sum(not row.source.record.attempts for row in rows)
        for cost in estimate["cost_table"]:
            cost["n_requests"] = sum(
                len(row.source.record.attempts)
                for row in rows
                if row.source.record.endpoint_id == cost["endpoint"]
            )
        original = self.case.original_check
        estimate["mde_table"] = [
            {
                "check": original.check_id,
                "n": original.n,
                "reference_tier": original.reference_tier,
                "interval": [original.ci_low, original.ci_high],
                "mde": None,
                "assumptions": "Retained original units/interval; replay adds no independent n.",
            }
        ]
        estimate["captured_replay_budget"] = {
            "attempt_slots": attempts,
            "not_sent_slots": skipped,
            "source_observation_uses": len(rows),
            "population_n": original.n,
            "replay_adds_independent_n": False,
            "selection_hash": self.selection_hash,
            "program_hash": self.program_hash,
        }
        estimate["note"] = (
            "Finite captured-input reuse, including recorded confirmations/retries. Cost is zero; "
            "original provider usage/cost remains source data. No new independent n, latency "
            "or model-quality measurement."
        )
        return estimate

    def finish(self, checks: list[CheckResult]) -> dict[str, Any]:
        self.tape.assert_complete()
        original = self.case.original_check
        selected = [
            check
            for check in checks
            if (check.check_id, check.endpoint_id) == (original.check_id, original.endpoint_id)
        ]
        fields = [
            "check_class",
            "n",
            "estimate",
            "ci_low",
            "ci_high",
            "reference_tier",
            "reference_estimate",
            "delta",
            "delta_ci_low",
            "delta_ci_high",
            "margin",
            "p_value",
            "adjusted_p",
            "violations",
            "verdict",
        ]
        if len(selected) != 1 or any(
            getattr(selected[0], name) != getattr(original, name) for name in fields
        ):
            raise ReplayRefused("captured_replay_original_check_not_reproduced")
        summary = {
            "case_id": self.case.case_id,
            "selection_hash": self.selection_hash,
            "program_hash": self.program_hash,
            "original_check_hash": content_hash(original.model_dump(mode="json")),
            "population_n": original.n,
            "replay_adds_independent_n": False,
            "audit": self.tape.audit(),
        }
        selected[0].details["captured_replay"] = {
            key: value for key, value in summary.items() if key != "audit"
        }
        return summary
