"""Execute a static plan and its captured-state multi-turn continuations."""

import uuid
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from servegauge.config import load_budget
from servegauge.core.hashing import content_hash
from servegauge.core.ids import safe_filename
from servegauge.core.models import Item
from servegauge.diagnose.drift import DRIFT_CONTRACT
from servegauge.profiles import load_profile
from servegauge.runner.agents import run_agents
from servegauge.runner.budget import BudgetManager
from servegauge.runner.evaluate import collect
from servegauge.runner.plan import RunPlan
from servegauge.runner.scheduler import RunResult, _write_json, run_plan
from servegauge.scoring.common import score_capabilities
from servegauge.scoring.routing import ROUTING_CHECK, ROUTING_CONTRACT, routing_samples
from servegauge.storage.analysis import seal_analysis
from servegauge.storage.raw_store import RawStore
from servegauge.suites.config import suite_config
from servegauge.suites.registry import plan_items
from servegauge.verdicts.config import load_policy


async def execute_plan(
    plan: RunPlan,
    *,
    root: Path = Path("."),
    run_id: str | None = None,
    resume: bool = False,
    approved: bool = False,
    transports: dict[str, httpx.AsyncBaseTransport] | None = None,
    inventory: list[Item] | None = None,
    budget_scope: str | None = None,
) -> RunResult:
    """Execute an exact optional inventory; shared-scope spending is cumulative."""
    plan = plan.model_copy(deep=True)
    identifier = run_id or "run-" + uuid.uuid4().hex
    if inventory is not None and (
        not inventory
        or len({item.item_id for item in inventory}) != len(inventory)
        or any(not item.item_id or item.suite_id not in plan.suites for item in inventory)
    ):
        raise ValueError("unique_nonempty_declared_suite_inventory_required")
    items = (
        [item.model_copy(deep=True) for item in inventory]
        if inventory is not None
        else plan_items(plan)
    )
    if inventory is None and plan.suites == ["S0"] and "items" in plan.overrides.get("S0", {}):
        return await run_plan(
            plan,
            root=root,
            run_id=identifier,
            resume=resume,
            approved=approved,
            items_override=items,
            transports=transports,
            budget_scope=budget_scope,
        )
    caps = [item for item in items if item.suite_id == "S0"]
    caps_id = (
        identifier + "-capabilities"
        if caps and any(item.suite_id != "S0" for item in items)
        else None
    )
    primary_exists = (root / "data/runs" / safe_filename(identifier) / "manifest.json").is_file()
    caps_exists = bool(
        caps_id and (root / "data/runs" / safe_filename(caps_id) / "manifest.json").is_file()
    )
    if resume and caps_id and primary_exists and not caps_exists:
        raise ValueError("resume_preflight_manifest_missing")
    if resume and not (primary_exists or caps_exists):
        raise ValueError("resume_manifest_missing")
    if caps_id:
        await run_plan(
            plan.model_copy(update={"suites": ["S0"]}),
            root=root,
            run_id=caps_id,
            resume=resume and caps_exists,
            approved=approved,
            items_override=caps,
            transports=transports,
            budget_scope=budget_scope or identifier,
        )
        records = list(RawStore(root / "data/raw").exchanges(run_id=caps_id))
        for endpoint in plan.endpoints:
            _, observed = score_capabilities(caps, records, endpoint, caps_id)
            endpoint.metadata["observed_capabilities"] = observed
        items = [item for item in items if item.suite_id != "S0"]
    result = await run_plan(
        plan,
        root=root,
        run_id=identifier,
        resume=resume and primary_exists,
        approved=approved,
        items_override=items,
        transports=transports,
        budget_scope=budget_scope,
    )
    records = list(RawStore(root / "data/raw").exchanges(run_id=result.run_id))
    traces: dict[str, Any] = {}
    if any("agent_goal" in item.expected for item in items):
        outcomes, traces = await run_agents(
            plan,
            items,
            records,
            root=root,
            run_id=result.run_id,
            approved=approved,
            transports=transports or {},
            budget_scope=budget_scope,
        )
        directory = root / "data/runs" / safe_filename(result.run_id)
        _write_json(
            directory / "agent_outcomes.json",
            {
                endpoint: {
                    check: sample.model_dump(mode="json") for check, sample in scores.items()
                }
                for endpoint, scores in outcomes.items()
            },
        )
        _write_json(directory / "agent_traces.json", traces)
    profile, policy = load_profile(plan.profile), load_policy()
    samples = collect(plan, items, records, profile, policy)
    routing_inventory = [*caps, *items] if caps_id else items
    routing_records = [
        *records,
        *(list(RawStore(root / "data/raw").exchanges(run_id=caps_id)) if caps_id else []),
    ]
    replay_by_endpoint: dict[str, dict[str, list[str]]] = {}
    for endpoint in plan.endpoints:
        routing = routing_samples(
            routing_inventory,
            routing_records,
            endpoint,
            run_id=identifier,
            capability_run_id=caps_id,
            contract=ROUTING_CONTRACT,
        )
        assert routing is not None
        samples[endpoint.id][ROUTING_CHECK] = routing
        replay_by_endpoint[endpoint.id] = {}
        for check, sample in samples[endpoint.id].items():
            if (
                sample.check_class != "invariant"
                or sample.status
                or sum(sample.values.values()) < policy.k_fail
                or not sample.replay_candidates
            ):
                continue
            wanted = set(sample.replay_candidates[0])
            probes = [
                item
                for item in (routing_inventory if check == ROUTING_CHECK else items)
                if item.item_id in wanted
            ]
            batch = suite_config()["tool_calling"]["replay_batch"] if check.startswith("T") else 1
            limit = suite_config()["tool_calling"]["replay_limit"] if check.startswith("T") else 1
            runs = []
            for start in range(0, limit, batch):
                replay_id = (
                    identifier + "-confirmation-" + content_hash([endpoint.id, check, start])[:16]
                )
                repeated = []
                for index in range(start, min(start + batch, limit)):
                    for probe in probes:
                        clone = probe.model_copy(deep=True)
                        clone.item_id += f"-confirmation-{index}"
                        clone.expected["trial"] = str(clone.expected.get("trial")) + f"-{index}"
                        repeated.append(clone)
                child = plan.model_copy(update={"endpoints": [endpoint]})
                exists = (root / "data/runs" / safe_filename(replay_id) / "manifest.json").exists()
                await run_plan(
                    child,
                    root=root,
                    run_id=replay_id,
                    resume=exists,
                    approved=approved,
                    transports=transports,
                    items_override=repeated,
                    budget_scope=budget_scope or identifier,
                )
                runs.append(replay_id)
                captured = list(RawStore(root / "data/raw").exchanges(run_id=replay_id))
                if check == ROUTING_CHECK:
                    confirmation = routing_samples(
                        repeated,
                        captured,
                        endpoint,
                        run_id=replay_id,
                        contract=ROUTING_CONTRACT,
                    )
                    assert confirmation is not None
                else:
                    confirmation = collect(child, repeated, captured, profile, policy)[endpoint.id][
                        check
                    ]
                if confirmation.status is None and any(confirmation.values.values()):
                    break
            replay_by_endpoint[endpoint.id][check] = runs
    directory = root / "data/runs" / safe_filename(identifier)
    metadata = {
        "run_id": identifier,
        "capability_run_id": caps_id,
        "baseline_id": None,
        "target_ids": [endpoint.id for endpoint in plan.endpoints],
        "tests": "all",
        "profile": profile.model_dump(mode="json"),
        "policy": policy.model_dump(mode="json"),
        "suite_config": suite_config(),
        "agent_traces": traces,
        "replay_by_endpoint": replay_by_endpoint,
        "requirements": None,
        "synthetic_mock": all(endpoint.kind == "mock" for endpoint in plan.endpoints),
        "routing_contract": ROUTING_CONTRACT,
        "drift_contract": DRIFT_CONTRACT,
    }
    seal_analysis(root, identifier, metadata)
    _write_json(directory / "evaluation.json", metadata)
    scope = BudgetManager(
        root, budget_scope or identifier, load_budget(root), max_usd=Decimal(str(plan.max_usd))
    )
    result = replace(result, spent_usd=scope.snapshot()["run_spent"])
    _write_json(
        directory / "result.json",
        {
            "run_id": result.run_id,
            "completed": result.completed,
            "skipped": result.skipped,
            "errors": result.errors,
            "spent_usd": str(result.spent_usd),
            "unmeasured": result.unmeasured,
        },
    )
    return result
