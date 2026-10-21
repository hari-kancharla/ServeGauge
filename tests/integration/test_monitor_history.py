"""Actual scheduled raw history is replayed without dispatch or mutable evidence."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from servegauge.cli import app
from servegauge.core.hashing import content_hash
from servegauge.mock.faults import FaultConfig
from servegauge.observatory.history import history_projection, inspect_history
from servegauge.observatory.provenance import implementation_hash
from servegauge.observatory.schedule import ObservatoryConfig
from servegauge.observatory.tick import tick
from servegauge.storage.transport_history import TransportHistoryJournal


def configuration():
    value = ObservatoryConfig.model_validate(
        {
            "id": "history-native",
            "enabled": True,
            "requests_per_minute": 600000,
            "models": [
                {
                    "id": "one",
                    "profile": "gpt-oss-120b",
                    "mock_context_length": 512,
                    "endpoints": [
                        {
                            "id": "mock:history",
                            "kind": "mock",
                            "base_url": "http://127.0.0.1:8011/v1",
                            "model": "mock",
                            "claimed_profile": "gpt-oss-120b",
                        }
                    ],
                }
            ],
        }
    )
    value.cadences["daily"].tests = ["P01"]
    value.cadences["daily"].fingerprint_level = None
    return value


def setup(root):
    (root / "configs").mkdir()
    (root / "configs/budgets.yaml").write_bytes(Path("configs/budgets.yaml").read_bytes())


def hashes(root):
    return {
        str(p.relative_to(root)): content_hash(p.read_bytes().hex())
        for p in sorted((root / "data").rglob("*"))
        if p.is_file()
    }


@pytest.mark.slow
async def test_canonical_scheduled_windows_replay_from_raw_and_inspection_is_idempotent(tmp_path):
    setup(tmp_path)
    cfg = configuration()
    source = implementation_hash()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    before_calls = []
    for day, error in [(0, 0.05), (1, 0.5)]:
        cfg.models[0].faults = FaultConfig(error_rate=error)
        result = await tick(cfg, "daily", root=tmp_path, now=start + timedelta(days=day))
        assert (
            result["status"] == "completed" and result["spent_usd"] == "0" and not result["resumed"]
        )
        before_calls.append(result)
    before = hashes(tmp_path)
    projection = history_projection(tmp_path, cfg.id)
    assert len(projection["steps"]) == 2
    assert projection["steps"][0]["status"] == "BASELINE"
    assert projection["steps"][1]["status"] == "CANDIDATE"
    assert projection["steps"][1]["event"]["ci_high"] < 0
    assert projection["qualified"] is False and projection["verdict"] == "INCONCLUSIVE"
    assert hashes(tmp_path) == before
    assert inspect_history(tmp_path, cfg.id)["status"] == "recorded"
    saved = hashes(tmp_path)
    assert inspect_history(tmp_path, cfg.id)["status"] == "unchanged" and hashes(tmp_path) == saved
    assert len(TransportHistoryJournal(tmp_path / "data/raw", cfg.id).projections()) == 1
    (tmp_path / "history-native-acceptance.json").write_text(
        json.dumps(
            {
                "source_hash": source,
                "projection": projection,
                "window_results": before_calls,
                "input_hashes": before,
                "persisted_hashes": saved,
                "new_provider_api_calls": 0,
                "new_spend_usd": 0,
                "operating_qualification_accepted": False,
            },
            indent=2,
        )
        + "\n"
    )
    assert implementation_hash() == source


def test_inspection_cli_is_read_only_when_requested_and_never_echoes_bad_context(tmp_path):
    setup(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["monitor", "inspect", "--observatory-id", "empty", "--root", str(tmp_path), "--read-only"],
    )
    assert result.exit_code == 0 and json.loads(result.stdout)["status"] == "read_only"
    assert not (tmp_path / "data").exists() and not (tmp_path / "budget").exists()
    (tmp_path / "data/raw/observatory/foreign").mkdir(parents=True)
    (tmp_path / "data/raw/observatory/foreign/spec.json").write_text("PRIVATE_BAD_HISTORY_CANARY")
    result = runner.invoke(
        app, ["monitor", "inspect", "--observatory-id", "empty", "--root", str(tmp_path)]
    )
    assert result.exit_code == 3 and "PRIVATE_BAD_HISTORY_CANARY" not in result.output


async def test_small_scheduled_window_stays_unavailable(tmp_path):
    setup(tmp_path)
    cfg = configuration()
    cfg.cadences["daily"].level = "smoke"
    cfg.cadences["daily"].overrides = {"S1": {"invariant_trials": 4}}
    result = await tick(cfg, "daily", root=tmp_path, now=datetime(2026, 1, 1, tzinfo=UTC))
    projection = history_projection(tmp_path, cfg.id)
    assert result["status"] == "completed"
    assert len(projection["steps"]) == 1 and projection["steps"][0]["status"] == "INCONCLUSIVE"
    assert projection["steps"][0]["event"] is None
