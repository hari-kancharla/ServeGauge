"""Canonical daily CLI acceptance, including cache groups and full FP1 inventory."""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from servegauge.cli import app
from servegauge.core.ids import safe_filename
from servegauge.observatory.schedule import load_observatory
from servegauge.storage.analysis import reconstruct
from servegauge.storage.monitor import MonitorJournal
from servegauge.storage.raw_store import RawStore
from servegauge.validation.async_offline import SocketlessLoop


@pytest.mark.slow
def test_canonical_daily_cli_then_same_window_makes_no_new_calls(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/budgets.yaml").write_bytes(Path("configs/budgets.yaml").read_bytes())

    def run_offline(coroutine):
        with asyncio.Runner(loop_factory=SocketlessLoop) as runner:
            return runner.run(coroutine)

    monkeypatch.setattr("servegauge.cli.asyncio.run", run_offline)
    config = Path("configs/observatory.mock.yaml")
    settings = load_observatory(config)
    assert settings.cadences["daily"].level == "core"
    assert settings.cadences["daily"].fingerprint_level == "full"
    args = [
        "monitor",
        "tick",
        "--cadence",
        "daily",
        "--config",
        str(config),
        "--root",
        str(tmp_path),
    ]
    first = CliRunner().invoke(app, args)
    assert first.exit_code in {0, 1, 2}, first.output
    result = json.loads(first.stdout)
    assert result["status"] == "completed" and result["spent_usd"] == "0"
    assert len(result["jobs"]) == 1
    job = result["jobs"][0]
    analysis = reconstruct(tmp_path / "data/raw", job["run_id"])
    journal = MonitorJournal(tmp_path / "data/raw", result["window_id"])
    inventory = journal.spec()["jobs"][0]["items"]
    assert len([i for i in inventory if i["expected"].get("check_id") == "FP1"]) == 40
    assert all(
        i["expected"]["design"]["level"] == "full" for i in inventory if i["suite_id"] == "S6"
    )
    assert any("cache" in i["version"] for i in inventory)
    assert len(analysis.records) > 16000
    assert not any(c.verdict == "ERROR" for c in analysis.checks)
    sent = list(RawStore(tmp_path / "data/raw").exchanges())
    assert len({r.exchange_id for r in sent}) == len(sent)
    before = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (tmp_path / "data").rglob("*")
        if p.is_file()
    }
    second = CliRunner().invoke(app, args)
    assert second.exit_code == 0, second.output
    assert json.loads(second.stdout)["status"] == "already_completed"
    assert before == {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (tmp_path / "data").rglob("*")
        if p.is_file()
    }
    assert list(RawStore(tmp_path / "data/raw").exchanges()) == sent
    assert (tmp_path / "data/raw" / safe_filename(job["run_id"]) / "analysis.json").is_file()
    (tmp_path / "observatory-acceptance.json").write_text(
        json.dumps(
            {
                "first": result,
                "second": json.loads(second.stdout),
                "records": len(sent),
                "artifacts": before,
            },
            indent=2,
        )
        + "\n"
    )
