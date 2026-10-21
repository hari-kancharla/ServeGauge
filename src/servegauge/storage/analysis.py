"""Immutable scientific inputs stored beside raw exchanges; derived results are rebuilt."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.ids import safe_filename
from servegauge.core.models import CheckResult, Exchange, Item, ModelProfile, ReferencePack
from servegauge.core.redaction import Redactor
from servegauge.diagnose.drift import annotate_context
from servegauge.diagnose.replay_expectations import mark_reproduced
from servegauge.runner.evaluate import collect, evaluate, reference_source
from servegauge.runner.plan import Requirements, RunPlan
from servegauge.scoring.agents import agent_samples
from servegauge.scoring.common import Samples, score_capabilities
from servegauge.scoring.routing import ROUTING_CHECK, routing_samples
from servegauge.scoring.tools import history_format_sample
from servegauge.storage.raw_store import RawStore
from servegauge.suites.config import suite_config
from servegauge.verdicts.config import DecisionPolicy


@dataclass
class Analysis:
    plan: RunPlan
    items: list[Item]
    records: list[Exchange]
    checks: list[CheckResult]
    samples: dict[str, dict[str, Samples]]
    metadata: dict[str, Any]
    manifest: dict[str, Any]
    reference_pack: ReferencePack | None


def capture_context(root: Path, run_id: str, value: dict[str, Any]) -> None:
    path = root / "data/raw" / safe_filename(run_id) / "capture.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = Redactor.from_environment().redact(value)
    encoded = canonical_json({"value": clean, "hash": content_hash(clean)}) + b"\n"
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError("immutable_capture_context_conflict")
    path.write_bytes(encoded)


def _read(path: Path) -> dict[str, Any]:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if content_hash(envelope["value"]) != envelope["hash"]:
        raise ValueError("scientific_context_hash_mismatch")
    return dict(envelope["value"])


def seal_analysis(
    root: Path, run_id: str, metadata: dict[str, Any], reference_pack: ReferencePack | None = None
) -> None:
    raw = root / "data/raw"
    context = _read(raw / safe_filename(run_id) / "capture.json")
    runs = {run_id, *metadata.get("replay_runs", {}).values()}
    if metadata.get("capability_run_id"):
        runs.add(metadata["capability_run_id"])
    for endpoint in metadata.get("agent_traces", {}).values():
        for trace in endpoint.values():
            runs.update(trace.get("run_ids", []))
    for batches in metadata.get("replay_batch_ids", {}).values():
        runs.update(batches)
    for checks in metadata.get("replay_by_endpoint", {}).values():
        for batches in checks.values():
            runs.update(batches)
    store = RawStore(raw)
    records = [
        record for identifier in sorted(runs) for record in store.exchanges(run_id=identifier)
    ]
    value = {
        "version": "analysis-v1",
        "run_id": run_id,
        "metadata": metadata,
        "capture_context_hash": content_hash(context),
        "record_hashes": {
            record.exchange_id: content_hash(record.model_dump(mode="json")) for record in records
        },
        "run_ids": sorted(runs),
        "reference_pack": reference_pack.model_dump(mode="json") if reference_pack else None,
    }
    path = raw / safe_filename(run_id) / "analysis.json"
    clean = Redactor.from_environment().redact(value)
    encoded = canonical_json({"value": clean, "hash": content_hash(clean)}) + b"\n"
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError("immutable_analysis_context_conflict")
    path.write_bytes(encoded)


def reconstruct(raw: Path, run_id: str) -> Analysis:
    directory = raw / safe_filename(run_id)
    sealed = _read(directory / "analysis.json")
    context = _read(directory / "capture.json")
    if content_hash(context) != sealed["capture_context_hash"]:
        raise ValueError("capture_context_mismatch")
    metadata = sealed["metadata"]
    if metadata["suite_config"] != suite_config():
        raise ValueError("captured_suite_version_requires_matching_implementation")
    plan = RunPlan.model_validate(context["manifest"]["config"])
    items = [Item.model_validate(item) for item in context["items"]]
    profile = ModelProfile.model_validate(metadata["profile"])
    policy = DecisionPolicy.model_validate(metadata["policy"])
    requirements = (
        Requirements.model_validate(metadata["requirements"])
        if metadata.get("requirements")
        else None
    )
    pack = (
        ReferencePack.model_validate(sealed["reference_pack"]) if sealed["reference_pack"] else None
    )
    reference_attachment = directory / "reference.json"
    if reference_attachment.exists():
        pack = ReferencePack.model_validate(_read(reference_attachment)["pack"])
    store = RawStore(raw)
    records = [
        record for identifier in sealed["run_ids"] for record in store.exchanges(run_id=identifier)
    ]
    if {
        record.exchange_id: content_hash(record.model_dump(mode="json")) for record in records
    } != sealed["record_hashes"]:
        raise ValueError("scientific_raw_evidence_changed_or_missing")
    primary = [record for record in records if record.run_id == run_id]
    samples = collect(plan, items, primary, profile, policy)
    caps = metadata.get("capability_run_id")
    cap_context = _read(raw / safe_filename(caps) / "capture.json") if caps else context
    cap_items = [
        Item.model_validate(item) for item in cap_context["items"] if item["suite_id"] == "S0"
    ]
    confirmed: dict[tuple[str, str], bool] = {}
    for endpoint in plan.endpoints:
        samples[endpoint.id].update(agent_samples(items, records, endpoint.id))
        if any(item.suite_id == "S2" for item in items) or metadata.get("include_T08"):
            samples[endpoint.id].setdefault("T08", history_format_sample(profile))
        routing = routing_samples(
            [*cap_items, *items] if caps else items,
            records,
            endpoint,
            run_id=run_id,
            capability_run_id=caps,
            contract=metadata.get("routing_contract"),
        )
        if routing is not None:
            samples[endpoint.id][ROUTING_CHECK] = routing
        replay_checks = metadata.get("replay_by_endpoint", {}).get(
            endpoint.id, metadata.get("replay_batch_ids", {})
        )
        for check, run_ids in replay_checks.items():
            source = samples[endpoint.id].get(check)
            if source is None:
                continue
            outcomes = []
            for identifier in run_ids:
                replay_context = _read(raw / safe_filename(identifier) / "capture.json")
                replay_items = [Item.model_validate(item) for item in replay_context["items"]]
                replay_records = [record for record in records if record.run_id == identifier]
                observed = (
                    routing_samples(
                        replay_items,
                        replay_records,
                        endpoint,
                        run_id=identifier,
                        contract=metadata.get("routing_contract"),
                    )
                    if check == ROUTING_CHECK
                    else collect(plan, replay_items, replay_records, profile, policy)[
                        endpoint.id
                    ].get(check)
                )
                if observed is not None:
                    outcomes.append(observed.model_dump(mode="json"))
                    source.evidence.extend(observed.evidence)
                    confirmed[(endpoint.id, check)] = confirmed.get(
                        (endpoint.id, check), False
                    ) or (observed.status is None and any(observed.values.values()))
            source.details.update(
                replay_run_id=run_ids[0], replay_batch_ids=run_ids, replay_outcomes=outcomes
            )
    checks = evaluate(
        plan,
        run_id,
        items,
        samples,
        policy,
        target_ids=metadata["target_ids"],
        baseline_id=metadata.get("baseline_id"),
        reference_pack=pack,
        requirements=requirements,
        replay_confirmed=confirmed,
        operational_records=records,
    )
    annotate_context(checks, plan.endpoints, policy, contract=metadata.get("drift_contract"))
    mark_reproduced(checks, metadata.get("replay_of"))
    capability_records = (
        [record for record in records if record.run_id == caps] if caps else primary
    )
    for endpoint in plan.endpoints:
        if endpoint.id in metadata["target_ids"] and cap_items:
            results, _ = score_capabilities(cap_items, capability_records, endpoint, caps or run_id)
            checks = [*results, *checks]
    return Analysis(plan, items, records, checks, samples, metadata, context["manifest"], pack)


def export_outcomes(analysis: Analysis) -> dict[str, Any]:
    return reference_source(
        analysis.plan,
        analysis.metadata["run_id"],
        analysis.items,
        analysis.samples,
        DecisionPolicy.model_validate(analysis.metadata["policy"]),
        analysis.manifest,
        analysis.metadata.get("baseline_id"),
    ).model_dump(mode="json")
