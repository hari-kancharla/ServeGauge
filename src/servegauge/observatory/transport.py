"""Complete, raw-derived first-attempt batches; operating admission stays separate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from servegauge.core.hashing import content_hash
from servegauge.mock.faults import FaultConfig
from servegauge.observatory.changes import TransportPolicy
from servegauge.observatory.provenance import implementation_hash
from servegauge.observatory.rolling import RollingTransportPolicy
from servegauge.stats.intervals import wilson
from servegauge.suites.registry import prepare_item
from servegauge.suites.s1_protocol import protocol_items

if TYPE_CHECKING:
    from servegauge.observatory.admission import VerifiedAdmission
    from servegauge.storage.analysis import Analysis


@dataclass(frozen=True)
class TransportBatch:
    ready: bool
    reason: str
    run_id: str
    endpoint_id: str
    observed_units: int
    expected_units: int | None
    values: tuple[int, ...] = ()
    role_seed: int | None = None
    request_design_hash: str | None = None
    capture_hash: str | None = None
    estimate: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    reference_tier: Literal["MOCK"] | None = None
    interval_scope: Literal["conditional_iid_batch_only"] | None = None
    qualified: Literal[False] = field(default=False, init=False)
    verdict: Literal["INCONCLUSIVE"] = field(default="INCONCLUSIVE", init=False)

    @property
    def n(self) -> int:
        return len(self.values)


def transport_batch(
    analysis: Analysis,
    endpoint_id: str,
    *,
    producer_source_hash: str | None,
    resumed: bool | None,
    faults: FaultConfig,
    policy: TransportPolicy | RollingTransportPolicy,
    admission: VerifiedAdmission | None = None,
    producer_binding: dict[str, Any] | None = None,
) -> TransportBatch:
    """Extract an unqualified candidate; labels, final status and retries cannot vote."""
    plan = analysis.plan
    run_id = str(analysis.manifest["run_id"])
    items = [
        i for i in analysis.items if i.suite_id == "S1" and i.expected.get("check_id") == "P01"
    ]
    wanted = {i.item_id for i in items}
    records = [
        r
        for r in analysis.records
        if r.run_id == run_id and r.endpoint_id == endpoint_id and r.item_id in wanted
    ]
    expected_n = policy.batch_sizes.get(plan.level) if plan.level in {"core", "full"} else None

    def unavailable(reason: str) -> TransportBatch:
        return TransportBatch(False, reason, run_id, endpoint_id, len(records), expected_n)

    if admission is None:
        if producer_source_hash is None or producer_source_hash != implementation_hash():
            return unavailable("transport_source_not_current")
    elif (
        admission.current_source_hash != implementation_hash()
        or admission.policy_hash != content_hash(policy.model_dump(mode="json"))
        or producer_binding is None
        or producer_binding.get("source_hash") != producer_source_hash
        or not admission.matches_binding(producer_binding)
    ):
        return unavailable("transport_producer_not_admitted")
    if resumed is not False:
        return unavailable("transport_resume_state_unknown_or_resumed")
    if (
        analysis.metadata.get("synthetic_mock") is not True
        or plan.reference.tier != "MOCK"
        or any(e.kind != "mock" for e in plan.endpoints)
    ):
        return unavailable("transport_process_not_qualified_mock")
    endpoint = next((e for e in plan.endpoints if e.id == endpoint_id), None)
    if endpoint is None or endpoint.defaults:
        return unavailable("transport_endpoint_or_defaults_unsupported")
    if admission is not None:
        context = endpoint.metadata.get("context_length")
        if type(context) is not int or not (
            admission.min_mock_context_length <= context <= admission.max_mock_context_length
        ):
            return unavailable("transport_context_outside_admitted_scope")
    if any(
        value
        for key, value in faults.model_dump().items()
        if key not in {"seed", "fault_id", "error_rate"}
    ):
        return unavailable("transport_fault_process_unsupported")
    declaration = endpoint.metadata.get("mock_declaration", {})
    if not isinstance(declaration, dict):
        return unavailable("transport_mock_declaration_mismatch")
    role_seed = int(content_hash([plan.seed, endpoint.id])[:15], 16)
    if (
        declaration.get("backend") != "scripted"
        or declaration.get("sampling_version") != "v2"
        or declaration.get("healthy_logit_jitter") != 0
        or type(declaration.get("seed")) is not int
        or declaration.get("seed") != role_seed
        or declaration.get("clean") is not (faults.error_rate == 0)
    ):
        return unavailable("transport_mock_declaration_mismatch")
    if expected_n is None:
        return unavailable("transport_level_underpowered_or_unsupported")
    if (
        len(items) != expected_n
        or len(wanted) != expected_n
        or len(records) != expected_n
        or len({r.item_id for r in records}) != expected_n
        or len({r.exchange_id for r in records}) != expected_n
    ):
        return unavailable("transport_incomplete_or_repeated_units")
    canonical = [
        i for i in protocol_items(plan.level, plan.seed) if i.expected.get("check_id") == "P01"
    ]
    if {i.item_id: content_hash(i.model_dump(mode="json")) for i in items} != {
        i.item_id: content_hash(i.model_dump(mode="json")) for i in canonical
    }:
        return unavailable("transport_canonical_items_changed")
    by_item = {r.item_id: r for r in records}
    values = []
    for item in canonical:
        record = by_item[item.item_id]
        body = prepare_item(item, endpoint).body
        if (
            record.suite_id != "S1"
            or record.sample_idx != 0
            or body is None
            or record.request != body
            or record.request_url != endpoint.base_url.rstrip("/") + "/chat/completions"
        ):
            return unavailable("transport_request_or_unit_binding_changed")
        if (
            not record.attempts
            or record.attempts[0].n != 1
            or type(record.attempts[0].status) is not int
            or record.attempts[0].status not in {200, 503}
        ):
            return unavailable("transport_first_http_attempt_unknown")
        values.append(int(record.attempts[0].status == 200))
    low, high = wilson(sum(values), expected_n)
    return TransportBatch(
        ready=True,
        reason="transport_operating_qualification_pending",
        run_id=run_id,
        endpoint_id=endpoint_id,
        observed_units=len(records),
        expected_units=expected_n,
        values=tuple(values),
        role_seed=role_seed,
        request_design_hash=content_hash([i.model_dump(mode="json") for i in canonical]),
        capture_hash=content_hash(
            {r.exchange_id: content_hash(r.model_dump(mode="json")) for r in records}
        ),
        estimate=sum(values) / expected_n,
        ci_low=low,
        ci_high=high,
        reference_tier="MOCK",
        interval_scope="conditional_iid_batch_only",
    )
