"""Ordinary run_plan must reserve exact compiled source bodies without mock hints."""

from __future__ import annotations

from decimal import Decimal

import httpx
import pytest
from captured_workflow import exercise_captured_workflow
from unit.test_comparison_sources import source as source

from servegauge.core.hashing import content_hash
from servegauge.mock.sealed_replay import FiniteReplay, ReplayTransport
from servegauge.runner.captured import CapturedCheck, CapturedRun
from servegauge.runner.check import run_check
from servegauge.runner.ledger import Ledger
from servegauge.runner.scheduler import run_plan
from servegauge.storage.comparison import prepare_sources
from servegauge.storage.comparison_program import compile_program
from servegauge.storage.raw_store import RawStore
from servegauge.storage.workspace import initialize


@pytest.fixture
def compiled(source):
    root, selection, out = source
    prepare_sources(root, selection, out)
    program = compile_program(out, selection_hash=content_hash(selection.model_dump(mode="json")))
    case = program.cases[0]
    phase = next(p for p in case.phases if p.source_run_id == "source")
    tape = FiniteReplay([row.source for row in phase.exchanges])
    context = CapturedRun(
        case,
        phase,
        selection_hash=program.selection_hash,
        program_hash=program.program_hash(),
        tape=tape,
    )
    return context, tape, phase


async def test_normal_scheduler_reserves_retries_captures_and_reconciles_exact_sources(
    compiled, tmp_path
):
    initialize(tmp_path)
    context, tape, phase = compiled
    plan = context.local_plan()
    result = await run_plan(
        plan,
        root=tmp_path,
        run_id="local",
        approved=True,
        items_override=phase.items,
        transports=context.transports(),
        captured=context,
    )
    assert result.completed == len(phase.exchanges) and result.spent_usd == Decimal(0)
    records = list(RawStore(tmp_path / "data/raw").exchanges(run_id="local"))
    expected = {
        (r.source.record.endpoint_id, r.item.item_id, r.source.record.sample_idx): r
        for r in phase.exchanges
    }
    for record in records:
        row = expected[(record.endpoint_id, record.item_id, record.sample_idx)]
        assert record.request == row.source.record.request
        assert record.response == row.source.record.response
        assert len(record.attempts) == len(row.source.record.attempts)
        assert record.metadata["captured_source"]["record_hash"] == row.source.record_hash
        assert record.metadata["captured_source"]["replay_adds_independent_n"] is False
    ledger = Ledger(tmp_path)
    ledger.refresh()
    assert not ledger.pending and ledger.development_spend() == 0
    assert tape.audit()["consumed_attempts"] == sum(
        len(row.source.record.attempts) for row in phase.exchanges
    )
    tape.assert_complete()


@pytest.mark.parametrize("change", ["endpoint", "budget", "concurrency", "transport", "item"])
async def test_invalid_local_context_refuses_before_creating_a_run(compiled, tmp_path, change):
    context, _, phase = compiled
    plan = context.local_plan()
    items = [item.model_copy(deep=True) for item in phase.items]
    transports = context.transports()
    if change == "endpoint":
        plan.endpoints[0].base_url = "https://example.com/v1"
    elif change == "budget":
        plan.max_usd = 0.01
    elif change == "concurrency":
        plan.concurrency_per_endpoint = 2
    elif change == "transport":
        transports[plan.endpoints[0].id] = httpx.MockTransport(
            lambda r: httpx.Response(200, json={})
        )
    else:
        items[0].payload["messages"][0]["content"] = "changed"
    with pytest.raises(ValueError):
        await run_plan(
            plan,
            root=tmp_path,
            run_id="refused",
            approved=True,
            items_override=items,
            transports=transports,
            captured=context,
        )
    assert not (tmp_path / "data").exists()


async def test_normal_check_keeps_preflight_paired_data_confirmation_and_original_n(
    source, tmp_path
):
    root, selection, out = source
    prepare_sources(root, selection, out)
    program = compile_program(out, selection_hash=content_hash(selection.model_dump(mode="json")))
    context = CapturedCheck(program, case_id="case-a")
    initialize(tmp_path)
    result = await run_check(
        context.local_plan(),
        root=tmp_path,
        approved=True,
        tests="P01",
        baseline_id="base",
        inventory=context.primary_items(),
        capability_inventory=context.capability_items(),
        transports=context.transports(),
        captured=context,
    )
    check = next(row for row in result.checks if row.check_id == "P01")
    original = program.cases[0].original_check
    assert (check.verdict, check.n, check.violations, check.ci_low, check.ci_high) == (
        original.verdict,
        original.n,
        original.violations,
        original.ci_low,
        original.ci_high,
    )
    assert check.details["replay_confirmed"]
    assert (
        result.estimate["captured_replay_budget"]["attempt_slots"]
        == context.tape.audit()["consumed_attempts"]
    )
    assert {row["address"]["role"] for row in context.tape.audit()["observations"]} == {
        "primary",
        "paired",
        "preflight",
        "confirmation",
    }
    context.tape.assert_complete()
    ledger = Ledger(tmp_path)
    ledger.refresh()
    assert not ledger.pending and ledger.development_spend() == 0


async def test_observed_capability_rejection_cannot_be_ignored_by_source_override(
    compiled, tmp_path
):
    context, _, phase = compiled
    plan = context.local_plan()
    plan.endpoints[0].metadata["observed_capabilities"] = {"answers": False}
    with pytest.raises(ValueError, match="capability|preparation|request"):
        await run_plan(
            plan,
            root=tmp_path,
            run_id="blocked",
            approved=True,
            items_override=phase.items,
            transports=context.transports(),
            captured=context,
        )
    assert not (tmp_path / "data").exists()


async def test_tool_confirmation_and_explicit_skips_use_normal_check_rules(tmp_path):
    result = await exercise_captured_workflow(tmp_path / "workflow", trials=4)
    assert [case["confirmation_attempts"] for case in result["cases"]] == [1, 16]
    assert all(case["not_sent_slots"] == 2 and case["n"] == 4 for case in result["cases"])


@pytest.mark.slow
async def test_full_source_replay_keeps_passing_controls_and_statistical_units(tmp_path):
    result = await exercise_captured_workflow(tmp_path / "full-workflow")
    assert all(
        case["source_control"]["verdict"] == "PASS" and case["source_control"]["n"] == 400
        for case in result["cases"]
    )


async def test_substituted_response_is_a_harness_failure_and_reconciles_reservations(
    compiled, tmp_path, monkeypatch
):
    context, _, phase = compiled
    initialize(tmp_path)
    original = ReplayTransport.handle_async_request

    async def substitute(transport, request):
        response = await original(transport, request)
        await response.aclose()
        return httpx.Response(200, json={"choices": [{"message": {"content": "substituted"}}]})

    monkeypatch.setattr(ReplayTransport, "handle_async_request", substitute)
    with pytest.raises(ExceptionGroup):
        await run_plan(
            context.local_plan(),
            root=tmp_path,
            run_id="substitution",
            approved=True,
            items_override=phase.items,
            transports=context.transports(),
            captured=context,
        )
    ledger = Ledger(tmp_path)
    ledger.refresh()
    assert not ledger.pending and ledger.development_spend() == 0
    assert not list((tmp_path / "reports").glob("*/checks.json"))
