"""Reserved, explicitly selected jobs in immutable UTC observatory windows."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any

import httpx
from filelock import FileLock, Timeout

from servegauge.config import load_budget
from servegauge.core.hashing import content_hash
from servegauge.core.ids import safe_filename
from servegauge.core.models import Item
from servegauge.mock.faults import FaultConfig
from servegauge.mock.server import create_app
from servegauge.observatory.identity import producer_identity
from servegauge.observatory.schedule import (
    CadenceWindow,
    MonitorModel,
    ObservatoryConfig,
    cadence_window,
    window_id,
)
from servegauge.profiles import load_profile
from servegauge.refs.pack import load_pack
from servegauge.runner.budget import BudgetManager
from servegauge.runner.check import _mock_metadata, normalize_tests, selected_check
from servegauge.runner.estimate import estimate_plan
from servegauge.runner.execute import execute_plan
from servegauge.runner.plan import RunPlan
from servegauge.storage.analysis import reconstruct
from servegauge.storage.monitor import MonitorJournal
from servegauge.storage.scientific import score_run
from servegauge.suites.config import suite_config
from servegauge.suites.registry import item_checks, plan_items
from servegauge.suites.s6_fingerprint import fingerprint_items
from servegauge.verdicts.config import load_policy


def execution_lock(root: Path, identifier: str) -> Path:
    return root / "data/observatory-locks" / (safe_filename(identifier) + ".lock")


def _seed(*parts: object) -> int:
    return int(content_hash(list(parts))[:15], 16)


def _transports(model: MonitorModel, seed: int, mode: str) -> dict[str, httpx.AsyncBaseTransport]:
    if mode != "mock":
        return {}
    result: dict[str, httpx.AsyncBaseTransport] = {}
    for endpoint in model.endpoints:
        role_seed = _seed(seed, endpoint.id)
        faults = (model.faults or FaultConfig()).model_copy(update={"seed": role_seed})
        result[endpoint.id] = httpx.ASGITransport(
            app=create_app(
                seed=role_seed,
                faults=faults,
                sampling_version=model.sampling_version,
                context_length=model.mock_context_length or 131072,
            )
        )
    return result


def _binding(config: ObservatoryConfig, root: Path) -> dict[str, Any]:
    # Shallow, explicit configuration directory only; never traverse other folders.
    if Path("configs").is_symlink() or any(
        p.is_symlink() for p in Path("configs").glob("*") if p.suffix in {".yaml", ".json"}
    ):
        raise ValueError("monitor_configuration_symlink")
    files = sorted(
        path
        for path in Path("configs").glob("*")
        if path.suffix in {".yaml", ".json"} and path.is_file()
    )
    packs = {}
    for model in config.models:
        if model.reference.pack:
            pack = load_pack(model.reference.pack, root / "refs")
            if pack.profile != model.profile:
                raise ValueError("monitor_reference_profile_mismatch")
            packs[model.id] = pack.model_dump(mode="json")
    return {
        "config_hash": content_hash(config.model_dump(mode="json")),
        **producer_identity(),
        "measurement_hash": content_hash(
            {
                "suites": suite_config(),
                "policy": load_policy().model_dump(mode="json"),
                "budget": load_budget(root).model_dump(mode="json"),
                "profiles": {
                    m.profile: load_profile(m.profile).model_dump(mode="json")
                    for m in config.models
                },
                "reference_packs": packs,
                "files": {p.as_posix(): sha256(p.read_bytes()).hexdigest() for p in files},
            }
        ),
    }


async def prepare_jobs(config: ObservatoryConfig, window: CadenceWindow) -> list[dict[str, Any]]:
    """Estimate exactly the filtered inventory; mock metadata stays in process."""
    setting = config.cadences[window.cadence]
    selected = normalize_tests(",".join(setting.tests))
    identifier = window_id(config.id, window)
    jobs = []
    for model in config.models:
        seed = _seed("observatory-job-v1", config.seed, identifier, model.id)
        plan = RunPlan(
            plan_id=identifier,
            profile=model.profile,
            level=setting.level,
            seed=seed,
            reference=model.reference.model_copy(deep=True),
            concurrency_per_endpoint=config.concurrency_per_endpoint,
            requests_per_minute=config.requests_per_minute,
            max_usd=config.budgets[window.cadence],
            endpoints=[e.model_copy(deep=True) for e in model.endpoints],
            suites=[f"S{i}" for i in range(9)],
            overrides=setting.overrides,
        )
        # Generate only requested suites. T05 shares the actual tools-corpus witness.
        suites = {"S0"}
        for token in selected:
            if token == "all":
                suites.update(f"S{i}" for i in range(9))
            elif token.startswith("S"):
                suites.add(token.split(".")[0])
            else:
                suites.add(
                    "S6"
                    if token.startswith("FP")
                    else {"P": "S1", "T": "S2", "J": "S3", "L": "S4", "R": "S5", "B": "S7"}[
                        token[0]
                    ]
                )
        plan.suites = sorted(suites)
        await _mock_metadata(plan, _transports(model, seed, config.mode))
        inventory = plan_items(plan)
        if "S6" in suites and setting.fingerprint_level:
            inventory = [i for i in inventory if i.suite_id != "S6"] + fingerprint_items(
                setting.fingerprint_level, seed, setting.overrides.get("S6")
            )
        inventory = [
            i
            for i in inventory
            if i.suite_id == "S0" or any(selected_check(c, selected) for c in item_checks(i))
        ]
        jobs.append(
            {
                "model_id": model.id,
                "run_id": "monitor-" + content_hash([identifier, model.id]),
                "plan": plan.model_dump(mode="json"),
                "items": [i.model_dump(mode="json") for i in inventory],
                "estimate": estimate_plan(plan, items_override=inventory),
            }
        )
    return jobs


def _verified_job(root: Path, result: dict[str, Any]) -> None:
    run_id = result["run_id"]
    analysis = reconstruct(root / "data/raw", run_id)
    if content_hash([c.model_dump(mode="json") for c in analysis.checks]) != result["checks_hash"]:
        raise ValueError("monitor_completed_evidence_changed")
    seal = root / "data/raw" / safe_filename(run_id) / "analysis.json"
    if sha256(seal.read_bytes()).hexdigest() != result["analysis_hash"]:
        raise ValueError("monitor_completed_seal_changed")


async def _execute_window(
    config: ObservatoryConfig,
    window: CadenceWindow,
    root: Path,
    resume: bool,
    binding: dict[str, Any],
    journal: MonitorJournal,
) -> dict[str, Any]:
    exists = (journal.directory / "spec.json").exists()
    if exists:
        spec = journal.spec()
        if spec.get("binding") != binding:
            raise ValueError("monitor_window_binding_changed")
        events = journal.events()
        if events and events[-1]["type"] == "window_completed":
            summary = events[-1]["result"]
            for result in summary["jobs"]:
                _verified_job(root, result)
            return {**summary, "status": "already_completed"}
        if not resume:
            raise ValueError("unfinished_monitor_window_requires_explicit_resume")
        resumed_event = {"type": "window_resumed"}
        journal.append(resumed_event)
        events.append(resumed_event)
    else:
        if resume:
            raise ValueError("monitor_resume_specification_missing")
        jobs = await prepare_jobs(config, window)
        spec = {
            "version": "observatory-window-v1",
            "window_id": window_id(config.id, window),
            "observatory_id": config.id,
            "cadence": window.cadence,
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
            "binding": binding,
            "config": config.model_dump(mode="json"),
            "jobs": jobs,
        }
        journal.create(spec)
        events = []
    identifier = spec["window_id"]
    results = []
    for job in spec["jobs"]:
        run_id = job["run_id"]
        finished = [
            e["result"] for e in events if e["type"] == "job_completed" and e["run_id"] == run_id
        ]
        if finished:
            _verified_job(root, finished[0])
            results.append(finished[0])
            continue
        started = any(e["type"] == "job_started" and e["run_id"] == run_id for e in events)
        plan = RunPlan.model_validate(job["plan"])
        if started and not any(
            (root / "data/runs" / safe_filename(candidate) / "manifest.json").is_file()
            for candidate in (run_id, run_id + "-capabilities")
        ):
            raise ValueError("monitor_resume_runner_context_missing")
        if not started:
            journal.append({"type": "job_started", "run_id": run_id})
        model = next(m for m in config.models if m.id == job["model_id"])
        captured = await execute_plan(
            plan,
            root=root,
            run_id=run_id,
            resume=started,
            approved=True,
            transports=_transports(model, plan.seed, config.mode),
            inventory=[Item.model_validate(i) for i in job["items"]],
            budget_scope=identifier,
        )
        analysis, _ = score_run(root, run_id, redact_content=True, reference=plan.reference.pack)
        result = {
            "model_id": job["model_id"],
            "run_id": run_id,
            "completed": captured.completed,
            "errors": captured.errors,
            "unmeasured": captured.unmeasured,
            "measurement_exit_code": analysis.metadata["exit_code"],
            "checks_hash": content_hash([c.model_dump(mode="json") for c in analysis.checks]),
            "analysis_hash": sha256(
                (root / "data/raw" / safe_filename(run_id) / "analysis.json").read_bytes()
            ).hexdigest(),
        }
        journal.append({"type": "job_completed", "run_id": run_id, "result": result})
        results.append(result)
    budget = BudgetManager(
        root, identifier, load_budget(root), max_usd=Decimal(str(config.budgets[window.cadence]))
    )
    summary = {
        "status": "completed",
        "window_id": identifier,
        "cadence": window.cadence,
        "start": window.start.isoformat(),
        "end": window.end.isoformat(),
        "jobs": results,
        "spent_usd": str(budget.snapshot()["run_spent"]),
        "resumed": any(e["type"] == "window_resumed" for e in events),
        "measurement_exit_code": next(
            (
                code
                for code in (5, 1, 3, 2, 0)
                if code in {r["measurement_exit_code"] for r in results}
            ),
            3,
        ),
        "change_detection": "INCONCLUSIVE: no qualified temporal channel attached",
        "limitations": [
            "Scripted mock streams do not qualify provider reliability.",
            "Interrupted mock streams restart random state; temporal independence is unqualified.",
        ],
    }
    journal.append({"type": "window_completed", "result": summary})
    return summary


async def tick(
    config: ObservatoryConfig,
    cadence: str,
    *,
    root: Path = Path("."),
    now: datetime | None = None,
    resume: bool = False,
    dry_run: bool = False,
    approved: bool = False,
) -> dict[str, Any]:
    # Revalidate even programmatically modified settings before any filesystem/transport use.
    config = ObservatoryConfig.model_validate(config.model_dump(mode="json"))
    window = cadence_window(cadence, now or datetime.now(UTC))
    if not config.enabled:
        return {"status": "disabled", "cadence": cadence}
    owner = os.environ.get("SERVEGAUGE_OWNER_RUN", os.environ.get("TRUEWEIGHTS_OWNER_RUN")) == "1"
    if config.mode == "owner" and not dry_run and not (owner and approved):
        raise ValueError("observatory_requires_explicit_owner_run_and_yes")
    identifier = window_id(config.id, window)
    BudgetManager(
        root, identifier, load_budget(root), max_usd=Decimal(str(config.budgets[cadence]))
    )
    if dry_run:
        return {
            "status": "dry_run",
            "window_id": identifier,
            "jobs": await prepare_jobs(config, window),
        }
    path = execution_lock(root, config.id)
    if any(p.is_symlink() for p in (root / "data", path.parent, path)):
        raise ValueError("monitor_execution_lock_symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(path), timeout=0):
            binding = _binding(config, root)
            journal = MonitorJournal(root / "data/raw", identifier)
            return await _execute_window(config, window, root, resume, binding, journal)
    except Timeout:
        return {"status": "in_progress", "window_id": identifier}
