import json
from pathlib import Path

import httpx
from tests_support import fingerprint_cards

from servegauge.core.ids import safe_filename
from servegauge.mock.server import create_app
from servegauge.runner.check import normalize_tests, run_check
from servegauge.runner.estimate import estimate_plan
from servegauge.runner.execute import execute_plan
from servegauge.runner.plan import RunPlan
from servegauge.storage.analysis import reconstruct
from servegauge.storage.raw_store import RawStore
from servegauge.storage.scientific import score_run
from servegauge.storage.tables import rebuild
from servegauge.validation.capture import endpoint


def design(tmp_path):
    (tmp_path / "configs").mkdir(exist_ok=True)
    (tmp_path / "configs/budgets.yaml").write_bytes(Path("configs/budgets.yaml").read_bytes())
    target = endpoint("mock:target", 8, clean=True)
    plan = RunPlan(
        plan_id="fingerprint-collection",
        profile="gpt-oss-120b",
        level="smoke",
        seed=77,
        reference={"tier": "MOCK"},
        endpoints=[target],
        suites=["S6"],
        max_usd=0,
        requests_per_minute=60000,
        overrides={
            "S6": {
                "methods": ["FP1", "FP2", "FP3"],
                "fp1_cells": 2,
                "fp1_samples": 3,
                "fp2_cells": 2,
                "fp2_samples": 3,
                "fp3_samples": 4,
            }
        },
    )
    return plan, {target.id: httpx.ASGITransport(app=create_app(seed=8))}


async def test_actual_s6_runner_estimate_raw_only_score_and_rebuild(tmp_path, monkeypatch):
    plan, transports = design(tmp_path)
    estimate = estimate_plan(plan)
    assert estimate["cost_table"][0]["n_requests"] == 16
    assert estimate["total_reserved_cost_usd"] == "0"
    assert all(r["mde"] is None for r in estimate["mde_table"])
    result = await execute_plan(plan, root=tmp_path, approved=True, transports=transports)
    assert result.spent_usd == 0 and not result.errors
    rows = list(RawStore(tmp_path / "data/raw").exchanges(run_id=result.run_id))
    assert len(rows) == 16 and all(r.suite_id == "S6" for r in rows)
    analysis, found = score_run(tmp_path, result.run_id)
    fingerprints = fingerprint_cards(analysis.checks)
    assert not found and {c.check_id for c in fingerprints} == {"FP1", "FP2", "FP3"}
    assert all(c.verdict == "INCONCLUSIVE" and c.estimate is None for c in fingerprints)
    assert all("observations" not in c.details for c in analysis.checks)
    (tmp_path / "configs/suites.yaml").write_bytes(Path("configs/suites.yaml").read_bytes())
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "data/runs" / safe_filename(result.run_id)
    (directory / "checks.json").write_text("corrupt derived cache")
    replayed = reconstruct(tmp_path / "data/raw", result.run_id)
    assert [c.model_dump() for c in replayed.checks] == [c.model_dump() for c in analysis.checks]
    tables = rebuild(tmp_path / "data/raw", tmp_path / "tables")
    assert tables["fingerprints"] == 3 and tables["checks"] == 4


async def test_fingerprint_alias_check_exits_inconclusive_without_calibration(tmp_path):
    plan, transports = design(tmp_path)
    assert normalize_tests("fingerprint") == {"S6"}
    result = await run_check(
        plan, tests="fingerprint", root=tmp_path, approved=True, transports=transports
    )
    assert result.code == 2
    assert {c.check_id for c in result.checks if c.suite_id == "S6"} == {"FP1", "FP2", "FP3"}
    stored = reconstruct(tmp_path / "data/raw", result.run_id)
    assert [c.model_dump() for c in stored.checks] == [c.model_dump() for c in result.checks]
    report = json.loads((result.report_dir / "checks.json").read_text())
    assert "observations" not in str(report)
