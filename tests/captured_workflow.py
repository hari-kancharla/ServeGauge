"""Synthetic source/replay acceptance using ordinary runners and sealed raw data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from servegauge.core.hashing import content_hash
from servegauge.mock.faults import FaultConfig
from servegauge.mock.server import create_app
from servegauge.runner.captured import CapturedCheck
from servegauge.runner.check import run_check
from servegauge.runner.evaluate import evaluate
from servegauge.runner.ledger import Ledger
from servegauge.runner.plan import RunPlan
from servegauge.storage.analysis import reconstruct
from servegauge.storage.comparison import (
    CaseRequest,
    ComparisonRequest,
    _guard,
    prepare_sources,
    propose_selection,
)
from servegauge.storage.comparison_program import compile_program
from servegauge.storage.workspace import initialize
from servegauge.suites.s0_capabilities import capability_items
from servegauge.suites.s1_protocol import probe
from servegauge.suites.s2_tools import tool_items
from servegauge.validation.capture import endpoint
from servegauge.verdicts.config import load_policy


async def exercise_captured_workflow(root: Path, *, trials: int = 400) -> dict[str, Any]:
    _guard(root)
    if root.exists():
        raise ValueError("fresh_captured_workflow_directory_required")
    source = root / "source"
    initialize(source)
    requests = []
    for name, check in [("cap", "P01"), ("tools", "T06")]:
        target = endpoint(name, 31, clean=False)
        baseline = endpoint("baseline", 32, clean=True)
        plan = RunPlan(
            plan_id="captured-source-" + name,
            profile="gpt-oss-120b",
            level="core",
            seed=31,
            reference={"tier": "BASELINE"},
            max_usd=0,
            endpoints=[target, baseline],
            suites=["S0", "S1" if check == "P01" else "S2"],
            concurrency_per_endpoint=1,
            requests_per_minute=1e9,
            retry_base_seconds=0,
            retry_max_seconds=0,
        )
        text = " token" * 24
        items = (
            [
                probe(
                    "P01",
                    index,
                    "cap",
                    "Return the recorded test sequence.",
                    {"type": "echo", "text": text},
                    params={"max_tokens": 2},
                    stream=bool(index % 2),
                    extra={"full_text": text},
                )
                for index in range(trials)
            ]
            if check == "P01"
            else [
                item
                for item in tool_items(
                    "smoke", 31, {"items": 1, "structural_repeats": trials, "agent_tasks": 1}
                )
                if item.expected.get("probe") == "parallel"
            ]
        )
        caps = capability_items()
        caps.append(
            probe(
                "S0.unavailable",
                0,
                "bounded",
                "Unavailable synthetic capability.",
                {"type": "echo", "text": "unused"},
                extra={"minimum_output_tokens": 1000000, "capability": "bounded_fixture"},
            )
        )
        transports = {
            target.id: httpx.ASGITransport(
                app=create_app(
                    seed=31,
                    faults=FaultConfig(
                        ignore_max_tokens=check == "P01", stream_delta_without_index=check == "T06"
                    ),
                )
            ),
            baseline.id: httpx.ASGITransport(app=create_app(seed=32)),
        }
        result = await run_check(
            plan,
            root=source,
            approved=True,
            tests=check,
            baseline_id=baseline.id,
            inventory=items,
            capability_inventory=caps,
            transports=transports,
        )
        finding = next(row for row in result.checks if row.check_id == check)
        assert finding.verdict == "FAIL" and finding.n == trials and finding.violations == trials
        assert finding.details["replay_confirmed"]
        requests.append(
            CaseRequest(
                case_id=name,
                run_id=result.run_id,
                endpoint_id=target.id,
                suite_id=finding.suite_id,
                check_id=check,
            )
        )
    selection = propose_selection(source, ComparisonRequest(cases=requests))
    snapshot = source / ".cache/comparisons/reviewed"
    prepare_sources(source, selection, snapshot)
    program = compile_program(
        snapshot, selection_hash=content_hash(selection.model_dump(mode="json"))
    )
    rows = []
    for case in program.cases:
        context = CapturedCheck(program, case_id=case.case_id)
        replay = root / ("replay-" + case.case_id)
        initialize(replay)
        result = await run_check(
            context.local_plan(),
            root=replay,
            approved=True,
            tests=case.original_check.check_id,
            baseline_id=context.baseline_id,
            inventory=context.primary_items(),
            capability_inventory=context.capability_items(),
            transports=context.transports(),
            captured=context,
        )
        actual = next(row for row in result.checks if row.check_id == case.original_check.check_id)
        assert (
            actual.n == case.population_n == trials
            and actual.violations == trials
            and actual.verdict == "FAIL"
        )
        assert (actual.ci_low, actual.ci_high, actual.reference_tier) == (
            case.original_check.ci_low,
            case.original_check.ci_high,
            case.original_check.reference_tier,
        )
        audit = context.tape.audit()
        assert (
            audit["remaining_slots"] == 0 and not audit["refusals"] and audit["consumed_skips"] == 2
        )
        assert (
            audit["consumed_attempts"] == result.estimate["captured_replay_budget"]["attempt_slots"]
        )
        controls = []
        for workspace, identifier in [
            (source, case.original_check.run_id),
            (replay, result.run_id),
        ]:
            analysis = reconstruct(workspace / "data/raw", identifier)
            baseline = next(
                check
                for check in evaluate(
                    analysis.plan,
                    identifier,
                    analysis.items,
                    analysis.samples,
                    load_policy(),
                    target_ids=["baseline"],
                )
                if check.check_id == case.original_check.check_id
            )
            assert baseline.n == trials and baseline.violations == 0
            if trials >= 400:
                assert (
                    baseline.verdict == "PASS"
                    and baseline.ci_high is not None
                    and baseline.ci_high < 0.01
                )
            controls.append(baseline.model_dump(mode="json"))
            ledger = Ledger(workspace)
            ledger.refresh()
            assert not ledger.pending and ledger.development_spend() == 0
        rows.append(
            {
                "case_id": case.case_id,
                "check": actual.check_id,
                "n": actual.n,
                "violations": actual.violations,
                "ci_low": actual.ci_low,
                "ci_high": actual.ci_high,
                "reference_tier": actual.reference_tier,
                "attempt_slots": audit["consumed_attempts"],
                "not_sent_slots": audit["consumed_skips"],
                "confirmation_attempts": sum(
                    row["address"]["role"] == "confirmation" and not row["skipped"]
                    for row in audit["observations"]
                ),
                "source_control": controls[0],
                "replayed_control": controls[1],
            }
        )
    result = {
        "cases": rows,
        "selection_hash": program.selection_hash,
        "program_hash": program.program_hash(),
        "scope": (
            "Synthetic ordinary source/check/replay contract only; "
            "no real provider or external comparison."
        ),
        "original_counts_intervals_tiers_and_controls_retained": True,
        "ordinary_confirmation_rules_used": True,
        "owner_comparison_executed": False,
        "comparison_setup_seconds": None,
        "model_api_calls": 0,
        "new_spend_usd": 0,
    }
    (root / "acceptance.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
