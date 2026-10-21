"""Installed-style CLI preparation stays local and refuses unreviewed execution."""

import json
import os
import subprocess
import sys

import pytest
import yaml
from typer.testing import CliRunner
from unit.test_selfhost_plan import config, policy
from unit.test_selfhost_storage import SID, file_hashes

from servegauge.cli import app
from servegauge.storage.workspace import initialize


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    initialize(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs/budgets.yaml").write_text(yaml.safe_dump(policy().model_dump(mode="json")))
    source = tmp_path / "configs/selfhost/owner.json"
    source.parent.mkdir()
    source.write_text(json.dumps(config().model_dump(mode="json")))
    return tmp_path


def args():
    return [
        "selfhost",
        "up",
        "--profile",
        "gpt-oss-120b",
        "--config",
        "configs/selfhost/owner.json",
        "--max-usd",
        "2",
        "--shutdown-minutes",
        "10",
        "--session-id",
        SID,
    ]


def test_real_cli_dry_run_status_repeat_and_guarded_down_preview_are_offline(
    workspace, monkeypatch
):
    def no_command(*a, **k):
        raise AssertionError("A local dry run invoked a system command")

    monkeypatch.setattr(subprocess, "run", no_command)
    before_modal = sys.modules.get("modal")
    runner = CliRunner()
    result = runner.invoke(app, [*args(), "--dry-run"])
    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    assert value["state"] == "prepared" and value["session_id"] == SID
    assert value["plan"]["reserved_upper_usd"] == "1.02"
    before = file_hashes(workspace)
    assert runner.invoke(app, [*args(), "--dry-run"]).exit_code == 0
    status = runner.invoke(app, ["selfhost", "status", "--session-id", SID])
    assert status.exit_code == 0 and json.loads(status.output) == value
    preview = runner.invoke(app, ["selfhost", "down", "--session-id", SID, "--dry-run"])
    assert preview.exit_code == 0 and json.loads(preview.output)["network_writes"] == 0
    assert file_hashes(workspace) == before
    assert (
        not (workspace / "data/ledger").exists()
        and not (workspace / "budget/dev_spend.json").exists()
    )
    assert sys.modules.get("modal") is before_modal
    assert os.environ.get("SERVEGAUGE_OWNER_RUN") is None


@pytest.mark.parametrize("tail", [[], ["--yes"], ["--dry-run", "--yes"]])
def test_unapproved_launch_never_creates_a_reservation(workspace, tail):
    runner = CliRunner()
    runner.invoke(app, [*args(), "--dry-run"])
    before = file_hashes(workspace)
    result = runner.invoke(app, [*args(), *tail])
    assert result.exit_code == 3 and "refused" in result.output.lower()
    assert file_hashes(workspace) == before


def test_caps_and_profile_must_match_current_reviewed_plan(workspace):
    runner = CliRunner()
    assert runner.invoke(app, [*args(), "--dry-run"]).exit_code == 0
    for old, new in [("2", "3"), ("10", "11"), ("gpt-oss-120b", "gpt-oss-20b")]:
        values = [new if item == old else item for item in args()]
        assert runner.invoke(app, [*values, "--dry-run"]).exit_code == 3
    assert not (workspace / "data/ledger").exists()
    assert runner.invoke(app, [*args(), "--dry-run", "--backend", "modal"]).exit_code == 3


def test_missing_caps_unknown_session_and_protected_config_refuse(workspace):
    runner = CliRunner()
    for values in [
        ["selfhost", "up", "--profile", "gpt-oss-120b", "--dry-run"],
        ["selfhost", "status", "--session-id", SID],
        [*args(), "--dry-run", "--config", "Starter Files/owner.json"],
        [*args(), "--dry-run", "--config", ".env"],
    ]:
        assert runner.invoke(app, values).exit_code == 3
    assert not (workspace / "data/ledger").exists()


def test_template_is_incomplete_and_does_not_override_reviewed_configuration(workspace):
    runner = CliRunner()
    target = "configs/selfhost/modal.json"
    result = runner.invoke(app, ["selfhost", "template", "--backend", "modal", "--out", target])
    assert result.exit_code == 0, result.output
    value = json.loads((workspace / target).read_text())
    assert value["image"] is None and value["hourly_usd"] is None
    assert value["billing_basis_reviewed"] is False and value["inbound_cidrs"] == []
    before = file_hashes(workspace)
    assert (
        runner.invoke(
            app, ["selfhost", "template", "--backend", "modal", "--out", target]
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(app, ["selfhost", "template", "--backend", "ssh", "--out", target]).exit_code
        == 3
    )
    assert file_hashes(workspace) == before
    assert runner.invoke(app, ["selfhost", "template", "--out", ".env"]).exit_code == 3


def test_launch_that_exits_immediately_is_not_reported_as_a_ready_endpoint(workspace, monkeypatch):
    from unit.test_selfhost_controller import FakeDriver

    import servegauge.adapters.selfhosted as transport
    import servegauge.cli as cli

    runner = CliRunner()
    assert runner.invoke(app, [*args(), "--dry-run"]).exit_code == 0
    driver = FakeDriver(workspace)
    driver.running = False
    monkeypatch.setattr(transport, "driver_for", lambda *a: driver)
    monkeypatch.setattr(
        cli,
        "_selfhost_environment",
        lambda: {
            "SERVEGAUGE_OWNER_RUN": "1",
            "SELFHOST_API_KEY": "fixture-endpoint-key",
            "SELFHOST_SSH_KEY_PATH": "fixture-path",
        },
    )
    result = runner.invoke(app, [*args(), "--yes"])
    assert result.exit_code == 5, result.output
    value = json.loads(result.output)
    assert value["state"] == "reconciled" and value["engine_readiness"] == "not_probed"
    assert os.environ.get("SERVEGAUGE_OWNER_RUN") is None


@pytest.mark.parametrize("flags", [["--backend", "modal"], ["--backend=modal"]])
def test_python_launcher_refuses_backend_retargeting_before_main(monkeypatch, flags):
    import servegauge.cli as cli
    from servegauge.selfhost.entrypoints import launcher

    def forbidden():
        raise AssertionError("Mismatched backend reached CLI execution")

    monkeypatch.setattr(cli, "main", forbidden)
    monkeypatch.setattr(sys, "argv", ["ssh_launcher.py", "up", *flags])
    with pytest.raises(SystemExit) as failure:
        launcher("ssh")
    assert failure.value.code == 3
