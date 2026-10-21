"""Actual complete CLI preparation, one slot, status and raw-only report commands."""

import asyncio
import json
from pathlib import Path

from typer.testing import CliRunner
from unit.test_gpu_study_protocol import specification

from servegauge.cli import app
from servegauge.storage.workspace import initialize
from servegauge.validation.async_offline import SocketlessLoop


def test_actual_v2_cli_prepares_collects_reports_and_keeps_default_owner_inputs_incomplete(
    tmp_path, monkeypatch
):
    original_run = asyncio.run
    monkeypatch.setattr(
        asyncio, "run", lambda coro: original_run(coro, loop_factory=SocketlessLoop)
    )
    initialize(tmp_path)
    monkeypatch.setenv(
        "SERVEGAUGE_TOKENIZER_CACHE",
        str(Path(__file__).absolute().parents[2] / ".cache/tokenizers"),
    )
    monkeypatch.chdir(tmp_path)
    cli = CliRunner()
    template = cli.invoke(
        app, ["selfhost", "study-template", "--out", "configs/selfhost/design.json"]
    )
    assert template.exit_code == 0, template.output
    owner = json.loads((tmp_path / "configs/selfhost/design.json").read_bytes())
    assert owner["scope"] == "owner_gpu" and owner["within_capture_components_reviewed"] is False
    assert not owner["design"]["independence_reviewed"]
    (tmp_path / "configs/selfhost/software.json").write_text(specification().model_dump_json())
    args = [
        "selfhost",
        "study-prepare",
        "--design",
        "configs/selfhost/software.json",
        "--out",
        ".cache/gpu-studies/software",
    ]
    prepared = cli.invoke(app, args)
    assert prepared.exit_code == 0, prepared.output
    value = json.loads(prepared.output)
    assert value["manifest"]["version"] == "gpu-study-kit-v2"
    shared = ["--kit", ".cache/gpu-studies/software", "--kit-hash", value["kit_hash"]]
    matrix = json.loads((tmp_path / ".cache/gpu-studies/software/matrix.json").read_bytes())
    before = cli.invoke(app, ["selfhost", "study-status", *shared])
    assert before.exit_code == 0
    assert json.loads(before.output)["counts"]["not_started"] == 80
    selected = cli.invoke(
        app,
        [
            "selfhost",
            "study-slot",
            *shared,
            "--slot",
            matrix[0]["slot_id"],
            "--out",
            "configs/selfhost/slot.json",
        ],
    )
    assert (
        selected.exit_code == 0
        and json.loads(selected.output)["slot"]["session_id"] == matrix[0]["session_id"]
    )
    assert json.loads((tmp_path / "configs/selfhost/slot.json").read_bytes())["image"] is None
    preview = cli.invoke(
        app, ["selfhost", "study-collect", *shared, "--slot", matrix[0]["slot_id"], "--dry-run"]
    )
    assert preview.exit_code == 0 and json.loads(preview.output)["state"] == "dry_run"
    assert not (tmp_path / "data").exists()
    captured = cli.invoke(
        app, ["selfhost", "study-collect", *shared, "--slot", matrix[0]["slot_id"], "--yes"]
    )
    assert captured.exit_code == 0, captured.output
    assert json.loads(captured.output)["model_api_calls"] == 0
    status = cli.invoke(app, ["selfhost", "study-status", *shared])
    assert status.exit_code == 0 and json.loads(status.output)["counts"]["captured"] == 1
    result = cli.invoke(
        app, ["selfhost", "study-analyze", *shared, "--out", "reports/gpu-studies/partial"]
    )
    assert result.exit_code == 0, result.output
    output = json.loads(result.output)
    assert output["physical_precision_verified"] is False and output["status"] == "INCONCLUSIVE"
    assert output["capture_summary"]["missing_slots"] == 79
    assert (tmp_path / "reports/gpu-studies/partial/roc.png").is_file()
    assert cli.invoke(app, args).exit_code == 3
    assert (
        cli.invoke(
            app, ["selfhost", "study-analyze", *shared, "--out", "reports/gpu-studies/partial"]
        ).exit_code
        == 3
    )
