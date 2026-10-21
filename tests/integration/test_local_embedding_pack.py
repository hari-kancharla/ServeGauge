"""Actual local model plus reserved ASGI captures and sealed-raw pack workflow."""

import json
import os
from pathlib import Path

import pytest
from integration.test_fingerprint_pack_workflow import capture_run
from tests_support import fingerprint_cards
from typer.testing import CliRunner

from servegauge.cli import app
from servegauge.core.hashing import content_hash
from servegauge.core.ids import safe_filename
from servegauge.refs.pack import load_pack
from servegauge.storage.analysis import reconstruct
from servegauge.storage.scientific import score_run

pytestmark = pytest.mark.slow


async def test_real_local_fp4_reserved_capture_cli_pack_and_raw_reconstruction(
    tmp_path, monkeypatch
):
    if not os.environ.get("SERVEGAUGE_EMBED_ASSETS"):
        pytest.skip("Pinned local model assets required; never downloaded by tests")
    (tmp_path / "configs").mkdir()
    (tmp_path / "profiles").mkdir()
    for name in [
        "suites.yaml",
        "margins.yaml",
        "fingerprints.yaml",
        "fingerprints_embed.yaml",
        "embedding_minilm_v1.json",
        "budgets.yaml",
    ]:
        (tmp_path / "configs" / name).write_bytes((Path("configs") / name).read_bytes())
    (tmp_path / "profiles/gpt-oss-120b.yaml").write_bytes(
        Path("profiles/gpt-oss-120b.yaml").read_bytes()
    )
    cfg = {"methods": ["FP4"], "fp4_cells": 2, "fp4_samples": 16}
    refs = await capture_run(tmp_path, "fp4-refs", [351, 352, 353], s6=cfg)
    controls = [
        await capture_run(tmp_path, "fp4-ca", [361], "other:control-a", s6=cfg),
        await capture_run(tmp_path, "fp4-cb", [362], "other:control-b", s6=cfg),
    ]
    target = await capture_run(tmp_path, "fp4-target", [371], s6=cfg)
    raw = tmp_path / "data/raw"
    before = content_hash([r.model_dump(mode="json") for r in reconstruct(raw, target).records])
    for identifier in [refs, *controls]:
        (tmp_path / "data/runs" / safe_filename(identifier) / "outcomes.json").write_text(
            "corrupt derived cache"
        )
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "refs",
            "build",
            "--profile",
            "gpt-oss-120b",
            "--tier",
            "MOCK",
            "--from-run",
            refs,
            "--control-run",
            controls[0],
            "--control-run",
            controls[1],
        ],
    )
    assert result.exit_code == 0, result.output
    pack = load_pack(json.loads(result.stdout)["pack_id"])
    metric = pack.fingerprints["views"]["*"]["metrics"]["FP4"]
    assert metric["embedding_identity"]["device"] == "cpu"
    assert metric["n"] == 32 and metric["calibration"][0]["prompt_aggregation"] == "sum"
    analysis, _ = score_run(tmp_path, target, reference=pack.pack_id)
    fingerprints = fingerprint_cards(analysis.checks)
    assert len(fingerprints) == 1
    check = fingerprints[0]
    assert check.check_id == "FP4" and check.verdict == "INFO" and check.margin is None
    assert check.n == 32 and check.reference_tier == "MOCK" and check.adjusted_p is not None
    assert check.delta_ci_low is not None and check.delta_ci_high is not None
    (tmp_path / "data/runs" / safe_filename(target) / "checks.json").write_text(
        "corrupt derived cache"
    )
    for name in ["fingerprints.yaml", "fingerprints_embed.yaml", "embedding_minilm_v1.json"]:
        (tmp_path / "configs" / name).write_text("invalid current config")
    replay = reconstruct(raw, target)
    assert replay.checks == analysis.checks
    assert content_hash([r.model_dump(mode="json") for r in replay.records]) == before
    monkeypatch.delenv("SERVEGAUGE_EMBED_ASSETS")
    unavailable = fingerprint_cards(reconstruct(raw, target).checks)[0]
    assert unavailable.verdict == "INCONCLUSIVE" and unavailable.estimate is None
