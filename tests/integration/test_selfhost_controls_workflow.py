"""Actual CLI/controller/ledger/SSH receiver flow with every remote system command simulated."""

import hashlib
import json
import os
import subprocess
from decimal import Decimal
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner
from unit.test_selfhost_plan import config, policy

import servegauge.cli as cli
from servegauge.observatory.provenance import implementation_hash
from servegauge.runner.ledger import Ledger
from servegauge.selfhost.remote_ssh import handle
from servegauge.storage.workspace import initialize


class Machine:
    def __init__(self):
        self.commands = []
        self.containers = {}
        self.fail_shutdown = False

    def execute(self, args, *, environment=None, timeout=60):
        self.commands.append(args)
        if args[:3] == ["sudo", "-n", "shutdown"]:
            if self.fail_shutdown:
                raise RuntimeError("simulated unavailable shutdown permission")
            return ""
        if args[0] == "nvidia-smi":
            return "NVIDIA H100 80GB HBM3, 570.26\n"
        if args[:3] == ["docker", "image", "inspect"]:
            return json.dumps([args[-1]])
        if args[:2] == ["docker", "run"]:
            if "--detach" not in args:
                return "0.11.0\n"
            assert environment["VLLM_API_KEY"] == "native-fixture-endpoint-key"
            name = args[args.index("--name") + 1]
            assert name not in self.containers
            labels = dict(args[i + 1].split("=", 1) for i, a in enumerate(args) if a == "--label")
            value = {
                "Id": hashlib.sha256(name.encode()).hexdigest(),
                "Config": {"Labels": labels},
                "State": {"Running": True, "ExitCode": 0},
            }
            self.containers[name] = value
            return value["Id"] + "\n"
        if args[:2] == ["docker", "inspect"]:
            return json.dumps(self.containers[args[-1]])
        if args[:2] == ["docker", "stop"]:
            self.containers[args[-1]]["State"]["Running"] = False
            return args[-1]
        raise AssertionError("Unexpected simulated remote command")


@pytest.mark.slow
def test_native_selfhost_cli_reserves_before_dispatch_and_recovers_without_real_owner_flags(
    tmp_path, monkeypatch
):
    assert os.environ.get("SERVEGAUGE_OWNER_RUN") is None
    assert os.environ.get("TRUEWEIGHTS_OWNER_RUN") is None
    initialize(tmp_path)
    (tmp_path / "configs/budgets.yaml").write_text(yaml.safe_dump(policy().model_dump(mode="json")))
    settings = tmp_path / "configs/selfhost/owner.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps(config().model_dump(mode="json")))
    key = tmp_path / "fixture-key"
    key.write_text("fixture")
    monkeypatch.chdir(tmp_path)
    owner = {
        "SERVEGAUGE_OWNER_RUN": "1",
        "SELFHOST_API_KEY": "native-fixture-endpoint-key",
        "SELFHOST_SSH_KEY_PATH": str(key),
    }
    monkeypatch.setattr(cli, "_selfhost_environment", lambda: dict(owner))
    machine = Machine()
    rpc = []

    def ssh(args, **kwargs):
        assert args[0] == "ssh"
        request = json.loads(kwargs["input"])
        ledger = Ledger(tmp_path)
        ledger.refresh()
        if request["action"] == "up":
            rows = [r for r in ledger.pending.values() if r["run_id"] == request["session_id"]]
            assert len(rows) == 1 and rows[0]["dispatched"] and rows[0]["owner"]
            assert Decimal(rows[0]["amount"]) == Decimal(request["plan"]["reserved_upper_usd"])
        assert "native-fixture-endpoint-key" not in str(args)
        rpc.append({"action": request["action"], "session_id": request["session_id"]})
        result = handle(request, home=tmp_path / "remote-home", execute=machine.execute)
        return SimpleNamespace(returncode=0, stdout=json.dumps(result))

    monkeypatch.setattr(subprocess, "run", ssh)
    runner = CliRunner()
    sid = "selfhost-" + "e" * 32

    def up(identifier, mode):
        return runner.invoke(
            cli.app,
            [
                "selfhost",
                "up",
                "--profile",
                "gpt-oss-120b",
                "--max-usd",
                "2",
                "--shutdown-minutes",
                "10",
                "--session-id",
                identifier,
                mode,
            ],
        )

    prepared = up(sid, "--dry-run")
    assert prepared.exit_code == 0, prepared.output
    assert rpc == [] and not (tmp_path / "data/ledger").exists()
    started = up(sid, "--yes")
    assert started.exit_code == 0, started.output
    running = json.loads(started.output)
    assert running["state"] == "running"
    assert "runtime_receipt_sha256" in running["facts"]
    assert up(sid, "--yes").exit_code == 3 and len(rpc) == 1
    stopped = runner.invoke(cli.app, ["selfhost", "down", "--session-id", sid, "--yes"])
    assert stopped.exit_code == 0, stopped.output
    final = json.loads(stopped.output)
    assert final["state"] == "reconciled"
    assert final["facts"]["amount_usd"] == "1.02"
    assert final["facts"]["cost_source"] == "estimate_selfhost_bound"
    before = len(rpc)
    assert runner.invoke(cli.app, ["selfhost", "status", "--session-id", sid]).exit_code == 0
    assert runner.invoke(cli.app, ["selfhost", "down", "--session-id", sid, "--yes"]).exit_code == 0
    assert len(rpc) == before
    unknown_sid = "selfhost-" + "f" * 32
    assert up(unknown_sid, "--dry-run").exit_code == 0
    machine.fail_shutdown = True
    uncertain = up(unknown_sid, "--yes")
    assert uncertain.exit_code == 5, uncertain.output
    assert json.loads(uncertain.output)["state"] == "outcome_unverified"
    assert up(unknown_sid, "--yes").exit_code == 3
    ledger = Ledger(tmp_path)
    ledger.refresh()
    assert ledger.development_spend() == 0 and ledger.by_run[sid] == Decimal("1.02")
    assert len(ledger.pending) == 1 and next(iter(ledger.pending.values()))["run_id"] == unknown_sid
    assert len(machine.containers) == 1
    files = [p for p in (tmp_path / "data").rglob("*") if p.is_file()]
    assert all(b"native-fixture-endpoint-key" not in p.read_bytes() for p in files)
    assert (
        os.environ.get("SERVEGAUGE_OWNER_RUN") is None
        and os.environ.get("TRUEWEIGHTS_OWNER_RUN") is None
    )
    receipt = {
        "source_hash": implementation_hash(),
        "scope": (
            "actual local CLI/ledger/receiver with simulated system commands; "
            "no SSH network or physical GPU"
        ),
        "rpc": rpc,
        "simulated_commands": machine.commands,
        "simulated_gpu_creates": 1,
        "successful_session": final,
        "uncertain_session_id": unknown_sid,
        "simulated_unknown_holds": 1,
        "simulated_owner_estimate_usd": "1.02",
        "actual_owner_environment_flags_unset": True,
        "runtime_observation_private": True,
        "model_api_calls": 0,
        "new_spend_usd": 0,
        "external_account_calls": 0,
    }
    (tmp_path / "selfhost-native-acceptance.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
