"""Actual reserved runner and immutable raw replay for the experimental study profile."""

import json
from pathlib import Path

import httpx
from typer.testing import CliRunner

from servegauge.cli import app
from servegauge.core.hashing import content_hash
from servegauge.mock.server import create_app
from servegauge.profiles import load_profile
from servegauge.runner.estimate import estimate_plan
from servegauge.runner.execute import execute_plan
from servegauge.runner.plan import RunPlan
from servegauge.scoring.gpu_study import score_quartet
from servegauge.storage.analysis import reconstruct
from servegauge.storage.raw_store import RawStore
from servegauge.validation.capture import endpoint
from servegauge.verdicts.config import load_policy


async def test_actual_study_runner_reservations_raw_rebuild_and_label_blind_scores(tmp_path):
    config = tmp_path / "configs"
    config.mkdir()
    (config / "budgets.yaml").write_bytes(Path("configs/budgets.yaml").read_bytes())
    profile = load_profile("qwen3-0.6b-study")
    assert profile.weights == "Qwen/Qwen3-0.6B"
    assert profile.tokenizer["qualified_for_counts"] is False and not profile.reference_tiers
    samples = []
    ids = set()
    for seed in range(4):
        target = endpoint(f"mock:study-{seed}", seed + 200, clean=True)
        target.claimed_profile = profile.id
        target.metadata["fingerprint_sampling"] = {
            "independent_sessions": True,
            "matched_sampling_design": True,
            "independent_reference_captures": True,
            "independent_prompt_strata": True,
            "session_size": 1,
        }
        plan = RunPlan(
            plan_id="study-software-qa",
            profile=profile.id,
            level="smoke",
            seed=14,
            reference={"tier": "BASELINE"},
            endpoints=[target],
            suites=["S6"],
            max_usd=0,
            requests_per_minute=60000,
            overrides={
                "S6": {
                    "collection_recipe": "qwen3-study-collection-v1",
                    "methods": ["FP1", "FP2", "FP3"],
                    "fp1_cells": 2,
                    "fp2_cells": 2,
                    "fp1_samples": 4,
                    "fp2_samples": 4,
                    "fp3_samples": 4,
                }
            },
        )
        estimate = estimate_plan(plan)
        assert estimate["total_reserved_cost_usd"] == "0"
        assert estimate["cost_table"][0]["n_requests"] == 20
        result = await execute_plan(
            plan,
            root=tmp_path,
            approved=True,
            run_id=f"study-software-qa-{seed}",
            transports={
                target.id: httpx.ASGITransport(
                    app=create_app(seed=seed + 200, sampling_version="v2")
                )
            },
        )
        assert result.spent_usd == 0 and result.completed == 20 and result.errors == 0
        records = list(RawStore(tmp_path / "data/raw").exchanges(run_id=result.run_id))
        assert len(records) == 20
        assert not ids.intersection(r.exchange_id for r in records)
        ids.update(r.exchange_id for r in records)
        for record in records:
            assert "reasoning_effort" not in record.request
            assert record.request["top_p"] == 1.0
            assert "seed" not in record.request
        analysis = reconstruct(tmp_path / "data/raw", result.run_id)
        assert analysis.plan.profile == profile.id
        assert analysis.manifest["config_hash"] == content_hash(plan.model_dump(mode="json"))
        assert all(check.verdict != "FAIL" for check in analysis.checks)
        samples.append(analysis.samples[target.id])
    rows = score_quartet(samples, policy=load_policy(), seed=77)
    assert all(
        rows[m]["estimate"] is not None for m in ("FP1", "FP2", "FP3-mean", "FP3-dispersion")
    )
    assert rows["FP4"]["estimate"] is None
    assert all(row["verdict"] == "INCONCLUSIVE" for row in rows.values())
    ledger = [
        json.loads(line) for line in (tmp_path / "data/ledger/spend.jsonl").read_text().splitlines()
    ]
    for kind in ("reserve", "dispatch", "reconcile"):
        assert sum(row["type"] == kind for row in ledger) == 80
    assert all(row.get("amount", "0") == "0" for row in ledger)


def test_actual_cli_study_prepare_verify_refuses_duplicate_and_protected_paths(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    design = tmp_path / "design.json"
    design.write_text(
        json.dumps({"version": "gpu-study-v1", "study_id": "qa", "seed": 5, "replicates": 2})
    )
    runner = CliRunner()
    out = tmp_path / ".cache/gpu-studies/qa"
    result = runner.invoke(
        app, ["selfhost", "study-prepare", "--design", str(design), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    assert value["manifest"]["planned_sessions"] == 80
    verified = runner.invoke(
        app, ["selfhost", "study-verify", "--kit", str(out), "--kit-hash", value["kit_hash"]]
    )
    assert verified.exit_code == 0 and json.loads(verified.output)["verified"]
    duplicate = runner.invoke(
        app, ["selfhost", "study-prepare", "--design", str(design), "--out", str(out)]
    )
    assert duplicate.exit_code == 3
    protected = runner.invoke(
        app,
        [
            "selfhost",
            "study-prepare",
            "--design",
            str(design),
            "--out",
            str(tmp_path / "Starter Files/qa"),
        ],
    )
    assert protected.exit_code == 3
    assert not (tmp_path / "data").exists()
