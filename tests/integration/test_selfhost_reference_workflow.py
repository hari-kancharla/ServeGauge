"""Connect the real self-host controller/receiver contract to synthetic raw reference input."""

import json
import os
import subprocess
from types import SimpleNamespace

import yaml
from integration.test_selfhost_controls_workflow import Machine
from typer.testing import CliRunner
from unit.test_owner_reference_storage import sealed_fixture
from unit.test_selfhost_plan import config, policy
from unit.test_selfhost_storage import SID

import servegauge.cli as cli
from servegauge.refs.pack import load_pack
from servegauge.runner.ledger import Ledger
from servegauge.selfhost.remote_ssh import handle
from servegauge.storage.owner_reference import validate_pack_origin
from servegauge.storage.workspace import initialize


def test_simulated_ssh_controller_receipt_qualifies_bound_raw_and_survives_reconciliation(
    tmp_path, monkeypatch
):
    initialize(tmp_path)
    (tmp_path / "configs/budgets.yaml").write_text(yaml.safe_dump(policy().model_dump(mode="json")))
    settings = tmp_path / "configs/selfhost/owner.json"
    settings.parent.mkdir()
    settings.write_text(config().model_dump_json())
    key = tmp_path / "fixture-key"
    key.write_text("fixture")
    monkeypatch.chdir(tmp_path)
    environment = {
        "SERVEGAUGE_OWNER_RUN": "1",
        "SELFHOST_API_KEY": "native-fixture-endpoint-key",
        "SELFHOST_SSH_KEY_PATH": str(key),
    }
    monkeypatch.setattr(cli, "_selfhost_environment", lambda: dict(environment))
    monkeypatch.setattr(cli, "_reference_environment", lambda: dict(environment))
    machine = Machine()

    def ssh(args, **kwargs):
        assert args[0] == "ssh"
        result = handle(
            json.loads(kwargs["input"]), home=tmp_path / "remote-home", execute=machine.execute
        )
        return SimpleNamespace(returncode=0, stdout=json.dumps(result))

    monkeypatch.setattr(subprocess, "run", ssh)
    runner = CliRunner()
    args = [
        "selfhost",
        "up",
        "--profile",
        "gpt-oss-120b",
        "--max-usd",
        "2",
        "--shutdown-minutes",
        "10",
        "--session-id",
        SID,
    ]
    assert runner.invoke(cli.app, args + ["--dry-run"]).exit_code == 0
    started = runner.invoke(cli.app, args + ["--yes"])
    assert started.exit_code == 0, started.output
    session = json.loads(started.output)
    assert len(session["facts"]["remote_id"]) == 64
    request, profile, endpoint, run = sealed_fixture(tmp_path, "R2", observed_session=session)
    result = runner.invoke(
        cli.app,
        [
            "refs",
            "build",
            "--profile",
            profile.id,
            "--tier",
            "R2",
            "--from-run",
            run,
            "--owner-qualification",
            "configs/references/review.json",
            "--confirm",
        ],
    )
    assert result.exit_code == 0, result.output
    pack = load_pack(json.loads(result.output)["path"])
    assert pack.tier == "R2"
    validate_pack_origin(tmp_path, pack)
    stopped = runner.invoke(cli.app, ["selfhost", "down", "--session-id", SID, "--yes"])
    assert stopped.exit_code == 0, stopped.output
    assert json.loads(stopped.output)["state"] == "reconciled"
    validate_pack_origin(tmp_path, pack)
    ledger = Ledger(tmp_path)
    ledger.refresh()
    assert ledger.development_spend() == 0 and not ledger.pending
    assert (
        os.environ.get("SERVEGAUGE_OWNER_RUN") is None
        and os.environ.get("TRUEWEIGHTS_OWNER_RUN") is None
    )
