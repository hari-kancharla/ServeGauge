"""Fresh reserved ASGI captures for the explicitly scoped text profile."""

import json
from pathlib import Path

import pytest
from integration.test_fingerprint_pack_workflow import capture_run
from tests_support import fingerprint_cards
from typer.testing import CliRunner

from servegauge.cli import app
from servegauge.core.hashing import content_hash
from servegauge.refs.pack import load_pack
from servegauge.storage.analysis import reconstruct
from servegauge.storage.scientific import score_run

pytestmark = [pytest.mark.slow, pytest.mark.qualified_runtime]


async def test_scoped_mock_cli_pack_actual_pass_and_raw_reconstruction(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "profiles").mkdir()
    for name in ["suites.yaml", "margins.yaml", "fingerprints.yaml", "budgets.yaml"]:
        (tmp_path / "configs" / name).write_bytes((Path("configs") / name).read_bytes())
    (tmp_path / "profiles/gpt-oss-120b.yaml").write_bytes(
        Path("profiles/gpt-oss-120b.yaml").read_bytes()
    )
    cfg = {
        "methods": ["FP1", "FP2", "FP3"],
        "fp1_cells": 40,
        "fp1_samples": 90,
        "fp2_cells": 25,
        "fp2_samples": 30,
        "fp3_samples": 400,
    }
    reference = await capture_run(
        tmp_path, "qualified-refs", [202610192701, 202610192702, 202610192703], s6=cfg, jitter=0.3
    )
    controls = [
        await capture_run(
            tmp_path, "qualified-ca", [202610192711], "other:control-a", s6=cfg, jitter=0.3
        ),
        await capture_run(
            tmp_path, "qualified-cb", [202610192712], "other:control-b", s6=cfg, jitter=0.3
        ),
    ]
    target = await capture_run(tmp_path, "qualified-target", [202610192721], s6=cfg, jitter=0.3)
    raw = tmp_path / "data/raw"
    before = content_hash([r.model_dump(mode="json") for r in reconstruct(raw, target).records])
    monkeypatch.chdir(tmp_path)
    args = [
        "refs",
        "build",
        "--profile",
        "gpt-oss-120b",
        "--tier",
        "MOCK",
        "--from-run",
        reference,
        "--control-run",
        controls[0],
        "--control-run",
        controls[1],
        "--fingerprint-qualification",
        "captured-scripted-text-v1",
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    pack = load_pack(json.loads(result.stdout)["pack_id"])
    assert pack.fingerprints["qualification"]["id"] == "captured-scripted-text-v1"
    analysis, _ = score_run(tmp_path, target, reference=pack.pack_id)
    fingerprints = fingerprint_cards(analysis.checks)
    assert {c.check_id: c.verdict for c in fingerprints} == {
        "FP1": "PASS",
        "FP2": "PASS",
        "FP3": "INFO",
    }
    assert all(c.reference_tier == "MOCK" for c in fingerprints)
    replay = reconstruct(raw, target)
    assert replay.checks == analysis.checks
    assert content_hash([r.model_dump(mode="json") for r in replay.records]) == before
    # Forging only the old boolean cannot transfer the qualification.
    from servegauge.runner.evaluate import evaluate
    from servegauge.verdicts.config import DecisionPolicy

    forged = pack.model_copy(deep=True)
    forged.fingerprints.pop("qualification")
    checks = evaluate(
        analysis.plan,
        target,
        analysis.items,
        analysis.samples,
        DecisionPolicy.model_validate(analysis.metadata["policy"]),
        target_ids=[analysis.plan.endpoints[0].id],
        reference_pack=forged,
    )
    assert all(c.verdict != "PASS" for c in checks)
