"""Actual reserved failure, matching bundle, local CLI draft and raw-only tables."""

import hashlib
import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from servegauge.cli import app
from servegauge.diagnose.bundle import write_bundle
from servegauge.diagnose.rules import diagnoses
from servegauge.mock.faults import FaultConfig
from servegauge.mock.server import create_app
from servegauge.observatory.notice_workflow import prepare_notice
from servegauge.observatory.provenance import implementation_hash
from servegauge.runner.check import run_check
from servegauge.runner.endpoints import resolve_endpoint
from servegauge.runner.plan import RunPlan
from servegauge.storage.analysis import reconstruct
from servegauge.storage.tables import rebuild


@pytest.mark.slow
async def test_actual_core_failure_bundle_prepares_local_notice_and_raw_table(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/budgets.yaml").write_bytes(Path("configs/budgets.yaml").read_bytes())
    endpoint = resolve_endpoint("http://127.0.0.1:8011/v1", "mock", "gpt-oss-120b")
    plan = RunPlan(
        plan_id="notice-native",
        profile="gpt-oss-120b",
        level="core",
        seed=884,
        reference={"tier": "MOCK"},
        max_usd=0,
        endpoints=[endpoint],
        suites=["S1"],
        requests_per_minute=600000,
    )
    source = implementation_hash()
    result = await run_check(
        plan,
        tests="P01",
        root=tmp_path,
        approved=True,
        transports={
            endpoint.id: httpx.ASGITransport(
                app=create_app(
                    seed=884, faults=FaultConfig(ignore_max_tokens=True), context_length=512
                )
            )
        },
    )
    assert result.code == 1
    analysis = reconstruct(tmp_path / "data/raw", result.run_id)
    check = next(c for c in analysis.checks if c.check_id == "P01")
    assert check.verdict == "FAIL" and check.n == check.violations == 400
    diagnosis = next(d for d in diagnoses(analysis.checks) if "P01" in d.check_ids)
    bundle = write_bundle(analysis, diagnosis, root=tmp_path)
    relative = (bundle / "bundle.json").relative_to(tmp_path).as_posix()
    cli = CliRunner().invoke(
        app,
        [
            "notice",
            "prepare",
            "--run",
            result.run_id,
            "--endpoint",
            endpoint.id,
            "--check",
            "P01",
            "--bundle",
            relative,
            "--root",
            str(tmp_path),
        ],
    )
    assert cli.exit_code == 0, cli.output
    value = json.loads(cli.stdout)
    assert value["state"] == "prepared" and value["delivery_verified"] is False
    assert value["network_writes"] == 0
    assert prepare_notice(tmp_path, result.run_id, endpoint.id, "P01", relative) == value
    counts = rebuild(tmp_path / "data/raw", tmp_path / "tables")
    assert counts["notifications"] == 1
    # No assertion about a real provider is created by a native mock workflow.
    snapshots = list((tmp_path / "data/raw/notices").glob("*/event-*.json"))
    assert snapshots == []
    from servegauge.report.site.builder import build_site, publication_preflight

    site = tmp_path / "notice-site"
    build_site(tmp_path, site, runs=[result.run_id])
    note = json.loads((site / "site.json").read_text())["reply_notes"][0]
    assert note["synthetic_mock"] is True and note["assertions"] == []
    assert note["state"]["basis"] == "scripted_mock"
    assert publication_preflight(site, root=tmp_path)["owner_asserted_notices"] == 0
    import polars as pl

    from servegauge.storage.export import stage_export

    exported = tmp_path / "notice-export"
    export_result = stage_export(tmp_path, exported, runs=[result.run_id])
    assert export_result["publication_blockers"] == ["scripted_mock_not_public"]
    numeric = pl.read_parquet(next(exported.glob("data/*/notifications-*.parquet")))
    assert numeric.height == 1 and not numeric["owner_asserted"][0]
    assert not numeric["delivery_verified"][0]
    (tmp_path / "notice-native-acceptance.json").write_text(
        json.dumps(
            {
                "source_hash": source,
                "run_id": result.run_id,
                "endpoint_id": endpoint.id,
                "notice": value,
                "bundle_reference": relative,
                "bundle_sha256": hashlib.sha256((bundle / "bundle.json").read_bytes()).hexdigest(),
                "check": check.model_dump(mode="json"),
                "table_counts": counts,
                "site": {
                    "reply_notes": 1,
                    "owner_assertions": 0,
                    "eligible_basis": "scripted_mock",
                },
                "export": {
                    "notification_rows": 1,
                    "owner_assertions": 0,
                    "mock_publication_refused": True,
                },
                "model_api_calls": 0,
                "new_spend_usd": 0,
            },
            indent=2,
        )
        + "\n"
    )
    assert implementation_hash() == source
