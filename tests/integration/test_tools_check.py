import json
from pathlib import Path

import httpx
import pytest

from servegauge.core.ids import safe_filename
from servegauge.items.agent_tasks import execute_tool
from servegauge.mock.faults import FaultConfig
from servegauge.mock.server import create_app
from servegauge.runner.check import run_check
from servegauge.runner.endpoints import resolve_endpoint
from servegauge.runner.evaluate import evaluate
from servegauge.runner.execute import execute_plan
from servegauge.runner.plan import RunPlan
from servegauge.scoring.agents import agent_samples
from servegauge.scoring.common import Samples
from servegauge.storage.raw_store import RawStore
from servegauge.suites.s2_tools import tool_items
from servegauge.verdicts.config import load_policy


def tool_plan():
    return RunPlan(
        plan_id="tools",
        profile="gpt-oss-120b",
        level="smoke",
        seed=11,
        reference={"tier": "BASELINE"},
        max_usd=0,
        requests_per_minute=60000,
        endpoints=[
            resolve_endpoint(
                f"http://127.0.0.1:{port}/v1", "mock", "gpt-oss-120b", identifier=str(port)
            )
            for port in (8011, 8012)
        ],
        suites=["S2"],
        overrides={"S2": {"items": 30, "structural_repeats": 2, "agent_tasks": 2}},
    )


@pytest.fixture(autouse=True)
def budget(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/budgets.yaml").write_text(Path("configs/budgets.yaml").read_text())


async def test_multi_turn_uses_returned_calls_and_records_shared_scope(tmp_path):
    plan = tool_plan()
    supplied = {
        endpoint.id: httpx.ASGITransport(app=create_app(seed=index))
        for index, endpoint in enumerate(plan.endpoints)
    }
    result = await run_check(
        plan,
        tests="T07",
        root=tmp_path,
        approved=True,
        baseline_id=plan.endpoints[1].id,
        transports=supplied,
    )
    checks = [r for r in result.checks if r.check_id.startswith("T07")]
    assert len(checks) == 2 and all(
        r.n == 2 and r.estimate == r.reference_estimate == 1 for r in checks
    )
    assert all(r.verdict == "INCONCLUSIVE" for r in checks)
    trace = next(iter(checks[0].details["agent_traces"].values()))
    assert trace["success"] and trace["turns"] >= 3
    assert len(trace["exchange_ids"]) == trace["turns"]
    for identifier in trace["run_ids"]:
        manifest = json.loads(
            (tmp_path / "data/runs" / safe_filename(identifier) / "manifest.json").read_text()
        )
        assert manifest["budget_scope"] == result.run_id
    records = list(RawStore(tmp_path / "data/raw").exchanges(run_id=trace["run_ids"][1]))
    assert any(message["role"] == "tool" for message in records[0].request["messages"])
    for check in checks:
        original_id, trace = next(iter(check.details["agent_traces"].items()))
        records = list(RawStore(tmp_path / "data/raw").exchanges(run_id=trace["run_ids"][1]))
        record = next(r for r in records if r.item_id == original_id + "-turn-1")
        assistant = next(m for m in record.request["messages"] if m["role"] == "assistant")
        assert bool(assistant.get("reasoning_details")) == check.check_id.endswith(".passback")
    reconstructed = agent_samples(
        tool_items(plan.level, plan.seed, plan.overrides["S2"]),
        list(RawStore(tmp_path / "data/raw").exchanges()),
        plan.endpoints[0].id,
    )
    for check in checks:
        sample = reconstructed[check.check_id]
        assert sample.status is None and all(sample.values.values())
        for key, trace in sample.details["agent_traces"].items():
            original = check.details["agent_traces"][key]
            assert trace["final_state"] == original["final_state"]
            assert trace["exchange_ids"] == original["exchange_ids"]


async def test_plan_execution_continuations_resume_without_duplicate_calls(tmp_path):
    plan = tool_plan()
    plan.overrides["S2"] = {"items": 2, "structural_repeats": 1, "agent_tasks": 1}
    supplied = {
        endpoint.id: httpx.ASGITransport(app=create_app(seed=index))
        for index, endpoint in enumerate(plan.endpoints)
    }
    result = await execute_plan(
        plan, root=tmp_path, run_id="agent-plan", approved=True, transports=supplied
    )
    path = tmp_path / "data/runs" / safe_filename(result.run_id) / "agent_outcomes.json"
    outcomes = json.loads(path.read_text())
    assert all(
        next(iter(sample["values"].values())) == 1
        for scores in outcomes.values()
        for sample in scores.values()
    )
    store = RawStore(tmp_path / "data/raw")
    before = list(store.exchanges())
    await execute_plan(
        plan, root=tmp_path, run_id=result.run_id, resume=True, approved=True, transports=supplied
    )
    after = list(store.exchanges())
    assert len(before) == len(after) and before == after


async def test_history_format_only_reports_profile_applicability(tmp_path):
    plan = tool_plan()
    supplied = {
        endpoint.id: httpx.ASGITransport(app=create_app(seed=index))
        for index, endpoint in enumerate(plan.endpoints)
    }
    result = await run_check(plan, tests="T08", root=tmp_path, approved=True, transports=supplied)
    finding = next(check for check in result.checks if check.check_id == "T08")
    assert finding.verdict == "NOT_APPLICABLE" and finding.n == 0 and result.code == 0
    assert not list(RawStore(tmp_path / "data/raw").exchanges(run_id=result.run_id))


def test_schema_accuracy_has_no_inference_without_shared_call_support():
    plan = tool_plan()
    first, baseline = [endpoint.id for endpoint in plan.endpoints]
    samples = {
        first: {"T02": Samples(check_id="T02", check_class="comparative", values={"a": 1})},
        baseline: {"T02": Samples(check_id="T02", check_class="comparative", values={"b": 1})},
    }
    results = evaluate(
        plan,
        "no-shared-support",
        [],
        samples,
        load_policy(),
        target_ids=[first],
        baseline_id=baseline,
    )
    assert results[0].verdict == "INCONCLUSIVE" and results[0].n == 0
    assert results[0].reason == "no observations"


async def test_fault_replay_is_actual_and_bounded_and_agent_does_not_forge_calls(tmp_path):
    plan = tool_plan()
    supplied = {
        plan.endpoints[0].id: httpx.ASGITransport(
            app=create_app(faults=FaultConfig(drop_tool_calls=1))
        ),
        plan.endpoints[1].id: httpx.ASGITransport(app=create_app(seed=2)),
    }
    result = await run_check(
        plan,
        tests="T06,T07",
        root=tmp_path,
        approved=True,
        baseline_id=plan.endpoints[1].id,
        transports=supplied,
    )
    drop = next(r for r in result.checks if r.check_id == "T06")
    assert drop.verdict == "FAIL" and drop.details["replay_confirmed"]
    assert len(drop.details["replay_batch_ids"]) == 1
    replay = drop.details["replay_outcomes"][0]
    assert len(replay["values"]) == 8
    agents = [r for r in result.checks if r.check_id.startswith("T07")]
    assert all(r.estimate == 0 and r.reference_estimate == 1 for r in agents)
    for trace in agents[0].details["agent_traces"].values():
        assert trace["turns"] == 1 and not trace["success"]
        assert all(not value for value in trace["final_state"].values())
    reconstructed = agent_samples(
        tool_items(plan.level, plan.seed, plan.overrides["S2"]),
        list(RawStore(tmp_path / "data/raw").exchanges()),
        plan.endpoints[0].id,
    )
    assert all(not any(sample.values.values()) for sample in reconstructed.values())
    with pytest.raises(ValueError, match="unknown_simulated_tool"):
        execute_tool({}, "shell", {"command": "anything"})
