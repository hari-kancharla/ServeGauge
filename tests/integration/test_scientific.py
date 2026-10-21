import json
import shutil
from pathlib import Path

import httpx
import polars as pl
import pytest
from typer.testing import CliRunner

from servegauge.cli import app
from servegauge.core.ids import safe_filename
from servegauge.diagnose.bundle import read_bundle, write_bundle
from servegauge.diagnose.repro import replay_bundle
from servegauge.diagnose.rules import diagnoses
from servegauge.mock.faults import FaultConfig
from servegauge.mock.server import create_app
from servegauge.runner.budget import BudgetExceeded
from servegauge.runner.check import run_check
from servegauge.runner.endpoints import resolve_endpoint
from servegauge.runner.plan import RunPlan
from servegauge.storage.analysis import reconstruct
from servegauge.storage.scientific import score_run
from servegauge.storage.tables import rebuild


@pytest.fixture
def prepared(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/budgets.yaml").write_bytes(Path("configs/budgets.yaml").read_bytes())
    endpoint = resolve_endpoint("http://127.0.0.1:8011/v1", "mock", "gpt-oss-120b")
    plan = RunPlan(
        plan_id="scientific",
        profile="gpt-oss-120b",
        level="smoke",
        seed=42,
        reference={"tier": "MOCK"},
        endpoints=[endpoint],
        suites=["S1"],
        max_usd=0,
        requests_per_minute=60000,
        overrides={"S1": {"invariant_trials": 4}},
    )
    transports = {
        endpoint.id: httpx.ASGITransport(app=create_app(faults=FaultConfig(ignore_max_tokens=True)))
    }
    return plan, transports


async def test_raw_reconstruction_bundle_privacy_and_actual_reserved_replay(tmp_path, prepared):
    plan, transports = prepared
    original = await run_check(
        plan, tests="P01", root=tmp_path, approved=True, transports=transports
    )
    analysis = reconstruct(tmp_path / "data/raw", original.run_id)
    assert [check.model_dump() for check in analysis.checks] == [
        check.model_dump() for check in original.checks
    ]
    _, found = score_run(tmp_path, original.run_id)
    assert found[0]["code"] == "PARAM_IGNORED" and found[0]["confidence"] == "medium"
    identifier = found[0]["bundle_path"]
    full = read_bundle(identifier, tmp_path)
    assert len(full["exchanges"]) == 5 and full["expected_vs_observed"]
    assert (Path(identifier) / "replay.py").exists()
    with pytest.raises(BudgetExceeded):
        await replay_bundle(identifier, root=tmp_path, transports=transports)
    replay = await replay_bundle(identifier, root=tmp_path, approved=True, transports=transports)
    assert replay["reproduced"] and replay["budget_scope"] == original.run_id
    repeated = reconstruct(tmp_path / "data/raw", replay["run_id"])
    assert any(check.details.get("bundle_replay_reproduced") for check in repeated.checks)
    marker = "PRIVATE_PAYLOAD_" + "invented-only-test"
    analysis.items[0].payload["messages"][0]["content"] = marker
    analysis.records[-1].request[marker] = marker
    analysis.records[-1].response = {marker: marker}
    redacted = write_bundle(
        analysis, diagnoses(analysis.checks)[0], root=tmp_path, redact_content=True
    )
    assert all(marker not in path.read_text(encoding="utf-8") for path in redacted.iterdir())
    assert (
        "structure"
        in json.loads((redacted / "bundle.json").read_text())["evidence_hashes_and_structure"]
    )
    with pytest.raises(ValueError, match="redacted"):
        read_bundle(str(redacted), tmp_path)
    # Entire raw directory alone suffices; corrupt cached scores cannot influence inference.
    (tmp_path / "data/runs" / safe_filename(original.run_id) / "checks.json").write_text(
        "corrupt cache"
    )
    copy = tmp_path / "raw-only-copy"
    shutil.copytree(tmp_path / "data/raw", copy)
    counts = rebuild(copy, tmp_path / "tables")
    assert counts["checks"] > 0 and counts["item_scores"] > 0 and counts["diagnoses"] == 2
    table = pl.read_parquet(tmp_path / "tables/checks.parquet")
    assert any(json.loads(row)["check_id"] == "P01" for row in table["record_json"])
    sealed = copy / safe_filename(original.run_id) / "analysis.json"
    sealed.write_text('{"value":{},"hash":"bad"}')
    with pytest.raises(ValueError, match="hash_mismatch"):
        reconstruct(copy, original.run_id)


async def test_cli_offline_commands_ignore_corrupt_derived_results(tmp_path, prepared, monkeypatch):
    plan, transports = prepared
    original = await run_check(
        plan, tests="P01", root=tmp_path, approved=True, transports=transports
    )
    # Scientific version/config check deliberately uses the exact captured config.
    shutil.copyfile("configs/suites.yaml", tmp_path / "configs/suites.yaml")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    for command in ("score", "diagnose", "report"):
        response = runner.invoke(app, [command, "--run", original.run_id])
        assert response.exit_code == 0, response.output
    assert runner.invoke(app, ["score", "--run", "missing"]).exit_code == 5


async def test_agent_trace_reconstruction_keeps_observed_reasoning_turn_counts(tmp_path, prepared):
    plan, transports = prepared
    plan.suites = ["S2"]
    plan.overrides["S2"] = {"items": 2, "structural_repeats": 1, "agent_tasks": 2}
    result = await run_check(
        plan, tests="T07.passback", root=tmp_path, approved=True, transports=transports
    )
    analysis = reconstruct(tmp_path / "data/raw", result.run_id)
    assert [check.model_dump() for check in result.checks] == [
        check.model_dump() for check in analysis.checks
    ]
    traces = next(check for check in analysis.checks if check.check_id == "T07.passback").details[
        "agent_traces"
    ]
    assert all(trace["turns_with_reasoning"] == trace["turns"] for trace in traces.values())


async def test_replay_to_another_service_preserves_its_distinct_identity(tmp_path, prepared):
    plan, transports = prepared
    result = await run_check(plan, tests="P01", root=tmp_path, approved=True, transports=transports)
    _, found = score_run(tmp_path, result.run_id)
    other = resolve_endpoint("http://127.0.0.1:8063/v1", "mock", plan.profile)
    supplied = {
        other.id: httpx.ASGITransport(app=create_app(faults=FaultConfig(ignore_max_tokens=True)))
    }
    replay = await replay_bundle(
        found[0]["bundle_path"],
        root=tmp_path,
        endpoint=other.base_url,
        approved=True,
        transports=supplied,
    )
    assert replay["reproduced"] and not replay["same_endpoint_identity"]
    assert replay["requested_endpoint_id"] == other.id != replay["original_endpoint_id"]
    actual = reconstruct(tmp_path / "data/raw", replay["run_id"])
    assert all(not check.details.get("bundle_replay_reproduced") for check in actual.checks)
