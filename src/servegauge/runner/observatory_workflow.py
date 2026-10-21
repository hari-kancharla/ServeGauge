"""Prepared owner workflow with persistent private state; never publishes artifacts."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from servegauge.core.ids import safe_filename
from servegauge.observatory.schedule import CADENCES, ObservatoryConfig, load_observatory
from servegauge.observatory.tick import tick
from servegauge.storage.export import stage_export
from servegauge.storage.notices import guard_path

SCHEDULES = {
    "23 */6 * * *": "probe",
    "41 2 * * *": "daily",
    "11 3 * * 1": "weekly",
    "53 4 1 * *": "monthly",
}


def route(event_name: str, event: dict[str, Any]) -> str:
    if event_name == "schedule" and event.get("schedule") in SCHEDULES:
        return SCHEDULES[event["schedule"]]
    if event_name == "workflow_dispatch" and event.get("inputs", {}).get("cadence") in CADENCES:
        return str(event["inputs"]["cadence"])
    raise ValueError("workflow_event_or_cadence_refused")


@dataclass(frozen=True)
class WorkflowPlan:
    cadence: str
    root: Path
    config: ObservatoryConfig


def prepare_workflow(
    event_name: str, event: dict[str, Any], environment: Mapping[str, str], workspace: Path
) -> WorkflowPlan:
    if (
        environment.get("OBSERVATORY_ENABLED") != "true"
        or environment.get("SERVEGAUGE_OWNER_RUN") != "1"
    ):
        raise ValueError("workflow_disabled_or_owner_gate_missing")
    branch = event.get("repository", {}).get("default_branch")
    if not isinstance(branch, str) or environment.get("GITHUB_REF") != "refs/heads/" + branch:
        raise ValueError("workflow_default_branch_required")
    cadence = route(event_name, event)
    root = Path(environment.get("SERVEGAUGE_STATE_ROOT", ""))
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("workflow_canonical_absolute_state_required")
    guard_path(root)
    guard_path(workspace)
    try:
        root, workspace = root.resolve(strict=True), workspace.resolve(strict=True)
        overlaps = any(p.samefile(workspace) for p in [root, *root.parents]) or any(
            p.samefile(root) for p in [workspace, *workspace.parents]
        )
    except OSError:
        raise ValueError("workflow_existing_private_state_required") from None
    if overlaps or not root.is_dir():
        raise ValueError("workflow_persistent_state_outside_checkout_required")
    config_file, budget_file = root / "configs/observatory.yaml", root / "configs/budgets.yaml"
    guard_path(config_file)
    guard_path(budget_file)
    if not config_file.is_file() or not budget_file.is_file():
        raise ValueError("workflow_owner_configuration_and_budget_required")
    config = load_observatory(config_file)
    if not config.enabled or config.mode != "owner":
        raise ValueError("workflow_enabled_owner_configuration_required")
    return WorkflowPlan(cadence, root, config)


async def execute(plan: WorkflowPlan) -> dict[str, Any]:
    result = await tick(plan.config, plan.cadence, root=plan.root, approved=True)
    if result.get("status") not in {"completed", "already_completed"}:
        raise ValueError("workflow_window_not_complete")
    export = stage_export(
        plan.root,
        plan.root / "exports" / safe_filename(result["window_id"]),
        runs=[job["run_id"] for job in result["jobs"]],
    )
    return {
        "window_id": result["window_id"],
        "status": result["status"],
        "measurement_exit_code": result["measurement_exit_code"],
        "export_files": len(export["files"]),
        "publication_blockers": export["publication_blockers"],
        "published": False,
        "provider_contact_sent": False,
    }


def main() -> None:
    try:
        event_path = Path(os.environ["GITHUB_EVENT_PATH"])
        guard_path(event_path)
        event = json.loads(event_path.read_bytes())
        plan = prepare_workflow(os.environ["GITHUB_EVENT_NAME"], event, os.environ, Path.cwd())
        result = asyncio.run(execute(plan))
    except Exception:
        print("Owner workflow refused or interrupted; inspect private state and configured gates.")
        raise SystemExit(3) from None
    print(json.dumps(result, indent=2))
    raise SystemExit(result["measurement_exit_code"])
