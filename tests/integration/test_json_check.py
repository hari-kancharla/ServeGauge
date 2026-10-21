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


def plan():
    return RunPlan(
        plan_id="json",
        profile="gpt-oss-120b",
        level="smoke",
        seed=8,
        reference={"tier": "BASELINE"},
        max_usd=0,
        requests_per_minute=60000,
        suites=["S3"],
        overrides={"S3": {"items": 33}},
        endpoints=[
            resolve_endpoint(
                f"http://127.0.0.1:{port}/v1", "mock", "gpt-oss-120b", identifier=str(port)
            )
            for port in (8011, 8012)
        ],
    )


async def test_json_check_baseline_and_uncertainty_are_measured(tmp_path):
    config = plan()
    transports = {
        endpoint.id: httpx.ASGITransport(app=create_app(seed=index))
        for index, endpoint in enumerate(config.endpoints)
    }
    result = await run_check(
        config,
        tests="json",
        root=tmp_path,
        approved=True,
        baseline_id=config.endpoints[1].id,
        transports=transports,
    )
    checks = {check.check_id: check for check in result.checks if check.check_id.startswith("J")}
    assert checks["J01"].n == checks["J02.semantic"].n == checks["J03"].n == 33
    assert all(checks[key].reference_estimate == 1 for key in ("J01", "J02.semantic", "J03"))
    assert checks["J02.validity"].violations == checks["J05"].violations == 0
    assert checks["J04"].n == 3
    assert result.code == 2 and all(check.verdict == "INCONCLUSIVE" for check in checks.values())
    estimates = {row["check"]: row for row in result.estimate["mde_table"]}
    assert estimates["J02.semantic"]["n"] == 33 and estimates["J04"]["n"] == 3


async def test_json_cap_contract_fault_is_replayed_through_reserved_runner(tmp_path):
    config = plan()
    config.endpoints = config.endpoints[:1]
    transports = {
        config.endpoints[0].id: httpx.ASGITransport(
            app=create_app(faults=FaultConfig(ignore_max_tokens=True))
        )
    }
    result = await run_check(
        config, tests="J05", root=tmp_path, approved=True, transports=transports
    )
    finding = next(check for check in result.checks if check.check_id == "J05")
    assert result.code == 1 and finding.verdict == "FAIL" and finding.violations == 33
    assert finding.details["replay_confirmed"] and finding.details["replay_batch_ids"]
