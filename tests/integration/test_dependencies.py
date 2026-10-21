import asyncio
import json
from pathlib import Path

import httpx
import pytest

from servegauge.core.models import EndpointSpec, Item
from servegauge.runner.plan import RunPlan
from servegauge.runner.scheduler import run_plan
from servegauge.storage.raw_store import RawStore


@pytest.fixture(autouse=True)
def budget(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/budgets.yaml").write_text(Path("configs/budgets.yaml").read_text())


def plan():
    return RunPlan(
        plan_id="sequence",
        profile="gpt-oss-120b",
        level="smoke",
        seed=1,
        reference={"tier": "MOCK"},
        max_usd=0,
        suites=["S4"],
        requests_per_minute=60000,
        retries=0,
        endpoints=[
            EndpointSpec(
                id="e",
                kind="mock",
                model="mock",
                base_url="http://127.0.0.1/v1",
                claimed_profile="gpt-oss-120b",
            )
        ],
    )


def items():
    return [
        Item(
            item_id=phase,
            suite_id="S4",
            version="test",
            payload={
                "messages": [{"role": "user", "content": phase}],
                "params": {"max_tokens": 16},
            },
            expected={"samples": 1, "depends_on": ["prime"] if phase == "warm" else []},
        )
        for phase in ("prime", "fresh", "warm")
    ]


async def test_completed_capture_precedes_warm_dispatch_and_resume_adds_no_calls(tmp_path):
    started, release = asyncio.Event(), asyncio.Event()
    captured = False
    calls = []

    async def handler(request):
        phase = json.loads(request.content)["messages"][0]["content"]
        calls.append(phase)
        if phase == "prime":
            started.set()
            await release.wait()
        if phase == "warm":
            assert captured
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
        )

    def after(record):
        nonlocal captured
        if record.item_id == "prime":
            captured = True

    supplied = {"e": httpx.MockTransport(handler)}
    task = asyncio.create_task(
        run_plan(
            plan(),
            root=tmp_path,
            run_id="ordered",
            items_override=items(),
            transports=supplied,
            after_capture=after,
        )
    )
    await asyncio.wait_for(started.wait(), 10)
    assert "warm" not in calls
    release.set()
    result = await task
    rows = {
        row.item_id: row for row in RawStore(tmp_path / "data/raw").exchanges(run_id=result.run_id)
    }
    assert rows["warm"].metadata["completed_dependencies"] == [rows["prime"].exchange_id]
    assert sorted(calls) == ["fresh", "prime", "warm"]
    resumed = await run_plan(
        plan(),
        root=tmp_path,
        run_id=result.run_id,
        resume=True,
        items_override=items(),
        transports=supplied,
    )
    assert resumed.skipped == 3 and len(calls) == 3


async def test_failed_prerequisite_is_captured_but_warm_is_unsent(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(500, json={"error": {"code": "server_error"}})

    result = await run_plan(
        plan(),
        root=tmp_path,
        items_override=[items()[0], items()[2]],
        transports={"e": httpx.MockTransport(handler)},
    )
    rows = {
        row.item_id: row for row in RawStore(tmp_path / "data/raw").exchanges(run_id=result.run_id)
    }
    assert len(calls) == 1 and result.errors == 1 and result.unmeasured == 1
    assert not rows["warm"].attempts and rows["warm"].metadata["prerequisite_not_observed"]


async def test_invalid_dependency_graphs_fail_before_reservations(tmp_path):
    for failure in ("missing", "duplicate", "repeat"):
        inventory = items()
        if failure == "missing":
            inventory[2].expected["depends_on"] = ["unknown"]
        elif failure == "duplicate":
            inventory[1].item_id = inventory[0].item_id
        else:
            inventory[0].expected["samples"] = 2
        with pytest.raises(ValueError):
            await run_plan(plan(), root=tmp_path, items_override=inventory)
    assert not (tmp_path / "data/raw").exists()
