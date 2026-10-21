"""Re-run supporting checks under the original hard budget and compare actual verdicts."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from servegauge.core.hashing import content_hash
from servegauge.core.ids import safe_filename
from servegauge.core.models import Diagnosis, Exchange, Item
from servegauge.diagnose.bundle import read_bundle
from servegauge.diagnose.replay_expectations import expected_context, replay_matches
from servegauge.diagnose.rules import diagnoses
from servegauge.runner.budget import BudgetExceeded
from servegauge.runner.check import run_check
from servegauge.runner.endpoints import resolve_endpoint
from servegauge.scoring.routing import ROUTING_CHECK
from servegauge.storage.analysis import reconstruct


def _mock_replay_inventory(
    items: list[Item], records: list[Exchange], endpoint_id: str, run_id: str
) -> list[Item]:
    """Restore only captured task mapping order; never replace inventory values.

    Canonical seals sort dictionaries. A mock task is serialized inside a prompt,
    so that otherwise harmless ordering changes the actual sampling request.
    Restore its original serialization from the scoped, sealed request captures.
    """
    result = [i.model_copy(deep=True) for i in items]
    by_item: dict[tuple[str, str], list[Exchange]] = {}
    for row in records:
        if row.endpoint_id == endpoint_id and row.run_id == run_id:
            by_item.setdefault((row.suite_id, row.item_id), []).append(row)
    for item in result:
        task = item.expected.get("mock_task")
        if not isinstance(task, dict):
            continue
        wires = set()
        for row in by_item.get((item.suite_id, item.item_id), []):
            users = [m for m in row.request.get("messages", []) if m.get("role") == "user"]
            if not users:
                continue
            text = users[-1].get("content")
            if not isinstance(text, str) or "\nTW_TASK: " not in text:
                raise ValueError("mock_replay_task_mismatch")
            prefix, wire = text.rsplit("\nTW_TASK: ", 1)
            expected_users = [m for m in item.payload["messages"] if m.get("role") == "user"]
            decoded = json.loads(wire)
            if (
                not expected_users
                or prefix != expected_users[-1].get("content")
                or decoded != task
                or json.dumps(decoded, ensure_ascii=False) != wire
            ):
                raise ValueError("mock_replay_task_mismatch")
            wires.add(wire)
        if len(wires) > 1:
            raise ValueError("mock_replay_task_mismatch")
        if wires:
            item.expected["mock_task"] = json.loads(next(iter(wires)))
    return result


async def replay_bundle(
    identifier: str,
    *,
    root: Path,
    endpoint: str | None = None,
    approved: bool = False,
    transports: dict[str, httpx.AsyncBaseTransport] | None = None,
    on_estimate: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    bundle = read_bundle(identifier, root)
    diagnosis = Diagnosis.model_validate(bundle["diagnosis"])
    analysis = reconstruct(root / "data/raw", diagnosis.run_id)
    qualified = next(
        (
            value
            for value in diagnoses(analysis.checks)
            if value.diagnosis_id == diagnosis.diagnosis_id
        ),
        None,
    )
    if qualified is None:
        raise ValueError("bundle_diagnosis_not_supported_by_original_raw_evidence")
    if any(
        getattr(diagnosis, field) != getattr(qualified, field)
        for field in ("run_id", "endpoint_id", "code", "check_ids", "evidence")
    ):
        raise ValueError("bundle_witness_not_supported_by_original_raw_evidence")
    expectations = expected_context(qualified, analysis.checks)
    if bundle.get("replay_expectations", expectations) != expectations:
        raise ValueError("bundle_expectations_not_supported_by_original_raw_evidence")
    if not approved:
        raise BudgetExceeded("explicit_yes_required")
    target = next(value for value in analysis.plan.endpoints if value.id == diagnosis.endpoint_id)
    original_id = target.id
    original_kind = target.kind
    if endpoint and endpoint != target.id:
        resolved = resolve_endpoint(
            endpoint, target.model, analysis.plan.profile, api_key_env=target.api_key_env, root=root
        )
        if (
            resolved.base_url.rstrip("/") == target.base_url.rstrip("/")
            and resolved.model == target.model
            and target.kind != "openrouter"
        ):
            resolved.id = original_id
        target = resolved
    baseline_id = analysis.metadata.get("baseline_id")
    if target.id == baseline_id:
        raise ValueError("replay_target_cannot_be_its_own_baseline")
    endpoints = [target, *(value for value in analysis.plan.endpoints if value.id == baseline_id)]
    plan = analysis.plan.model_copy(update={"endpoints": endpoints})
    tests = set(diagnosis.check_ids)
    inventory = None
    history = None
    capability_inventory = None
    if (
        diagnosis.code
        in {
            "QUANTIZATION_LIKE_DRIFT",
            "DIFFERENT_MODEL_LIKE",
            "CACHE_ANOMALY",
            "ACCOUNTING_MISMATCH",
        }
        or ROUTING_CHECK in tests
        or any(value.startswith(("B01.", "S8.health.")) for value in tests)
    ):
        # Derived checks need the exact original root procedures, not regenerated traffic.
        tests = set((analysis.metadata.get("tests") or "all").split(","))
        inventory = [i for i in analysis.items if i.suite_id != "S0"]
        history = any(c.check_id == "T08" for c in analysis.checks)
        capability_run = analysis.metadata.get("capability_run_id")
        if capability_run is None:
            # A batch can legitimately have no separate preflight. Preserve
            # its exact primary inventory, including any primary S0 items.
            capability_inventory = []
            inventory = [i.model_copy(deep=True) for i in analysis.items]
        elif isinstance(capability_run, str):
            captured = json.loads(
                (root / "data/raw" / safe_filename(capability_run) / "capture.json").read_text()
            )
            if content_hash(captured["value"]) != captured["hash"]:
                raise ValueError("scientific_context_hash_mismatch")
            capability_inventory = [Item.model_validate(i) for i in captured["value"]["items"]]
        else:
            raise ValueError("replay_requires_original_capability_inventory")
    if target.kind == original_kind == "mock" and inventory is not None:
        inventory = _mock_replay_inventory(
            inventory, analysis.records, original_id, diagnosis.run_id
        )
        if capability_inventory:
            capability_inventory = _mock_replay_inventory(
                capability_inventory,
                analysis.records,
                original_id,
                analysis.metadata["capability_run_id"],
            )
    result = await run_check(
        plan,
        tests=",".join(sorted(tests)),
        root=root,
        approved=True,
        baseline_id=baseline_id,
        reference_pack=analysis.reference_pack,
        budget_scope=analysis.manifest.get("budget_scope", diagnosis.run_id),
        replay_of=expectations if target.id == original_id else None,
        inventory=inventory,
        include_history_check=history,
        capability_inventory=capability_inventory,
        transports=transports,
        on_estimate=on_estimate,
    )
    observed = {
        check.check_id: check for check in result.checks if check.check_id in diagnosis.check_ids
    }
    return {
        "diagnosis_id": diagnosis.diagnosis_id,
        "run_id": result.run_id,
        "original_endpoint_id": original_id,
        "requested_endpoint_id": target.id,
        "same_endpoint_identity": target.id == original_id,
        "budget_scope": analysis.manifest.get("budget_scope", diagnosis.run_id),
        "reproduced": replay_matches(result.checks, expectations, endpoint_id=target.id),
        "expected": expectations["expected"],
        "expected_code": diagnosis.code,
        "observed_codes": sorted(
            d.code for d in diagnoses(result.checks) if d.endpoint_id == target.id
        ),
        "observed": {name: check.model_dump(mode="json") for name, check in observed.items()},
    }
