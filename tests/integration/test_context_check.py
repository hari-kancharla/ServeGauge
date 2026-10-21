from pathlib import Path

import httpx
import pytest

from servegauge.mock.faults import FaultConfig
from servegauge.mock.server import create_app
from servegauge.runner.check import run_check
from servegauge.runner.endpoints import resolve_endpoint
from servegauge.runner.plan import RunPlan


@pytest.fixture(autouse=True)
def budget(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/budgets.yaml").write_text(Path("configs/budgets.yaml").read_text())


async def test_context_fault_is_replayed_and_small_n_keeps_effective_context_uncertain(tmp_path):
    endpoints = [
        resolve_endpoint(
            f"http://127.0.0.1:{port}/v1", "mock", "gpt-oss-120b", identifier=str(port)
        )
        for port in (8011, 8012)
    ]
    plan = RunPlan(
        plan_id="context",
        profile="gpt-oss-120b",
        level="smoke",
        seed=8,
        reference={"tier": "BASELINE"},
        max_usd=0,
        requests_per_minute=60000,
        endpoints=endpoints,
        suites=["S4"],
        overrides={
            "S4": {"lengths": [16000, 32000], "depths": [0, 50, 100], "task_types": 2, "repeats": 2}
        },
    )
    supplied = {
        endpoints[0].id: httpx.ASGITransport(
            app=create_app(faults=FaultConfig(truncate_prompt_head_over=16384))
        ),
        endpoints[1].id: httpx.ASGITransport(app=create_app(seed=2)),
    }
    result = await run_check(
        plan,
        tests="context",
        root=tmp_path,
        approved=True,
        baseline_id=endpoints[1].id,
        transports=supplied,
    )
    checks = {check.check_id: check for check in result.checks}
    assert result.code == 1 and checks["L04"].verdict == "FAIL"
    assert checks["L04"].details["replay_confirmed"] and checks["L04"].violations == 12
    assert checks["L05"].verdict == "INCONCLUSIVE" and checks["L05.16000"].n == 12
    estimates = {row["check"]: row for row in result.estimate["mde_table"]}
    assert estimates["L04"]["n"] == 24 and estimates["L05.16000"]["n"] == 12
