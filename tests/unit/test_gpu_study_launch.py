"""New recipe wiring executes only simulated systems and pure owner-environment mappings."""

import ast
import json
import os
import shlex
import subprocess
import sys
from decimal import Decimal
from types import SimpleNamespace

import pytest
import yaml
from integration.test_selfhost_controls_workflow import Machine
from typer.testing import CliRunner
from unit.test_gpu_study_recipe import reviewed
from unit.test_selfhost_adapters import environment, modal_sdk
from unit.test_selfhost_plan import config, policy
from unit.test_selfhost_storage import SID

import servegauge.cli as cli
from servegauge.adapters.selfhosted import ModalDriver, _receiver_source
from servegauge.core.models import EndpointSpec
from servegauge.refs.owner import RECIPE, Qualification, _runtime
from servegauge.runner.ledger import Ledger
from servegauge.selfhost.controller import launch_session, stop_session
from servegauge.selfhost.plan import build_plan, engine_args
from servegauge.selfhost.study_recipe import STUDY_PROFILE, VARIANTS
from servegauge.storage.selfhost import prepare_session
from servegauge.storage.workspace import initialize


@pytest.mark.parametrize("name", ["remote_ssh.py", "remote_modal.py"])
def test_bundled_receiver_is_standalone_and_uses_the_same_finite_recipe(name):
    source = _receiver_source(name)
    tree = ast.parse(source)
    assert not any(
        isinstance(node, ast.ImportFrom) and (node.module or "").startswith("servegauge")
        for node in ast.walk(tree)
    )
    namespace = {"__name__": "local_receiver_contract"}
    exec(compile(source, "local-receiver-contract", "exec"), namespace)
    for variant in VARIANTS:
        setting = reviewed(variant)
        assert namespace["study_engine_args"](setting.model_dump(mode="json")) == engine_args(
            setting
        )
    if name == "remote_ssh.py":
        assert namespace["_expected_args"](config().model_dump(mode="json")) == engine_args(
            config()
        )


def test_bundle_path_refusal_precedes_package_access(monkeypatch):
    import servegauge.adapters.selfhosted as adapters

    def forbidden(*args, **kwargs):
        raise AssertionError("a refused path must not access files")

    monkeypatch.setattr(adapters, "files", forbidden)
    for name in ["../remote_ssh.py", "Starter Files/remote_ssh.py", "unexpected.py"]:
        with pytest.raises(ValueError):
            _receiver_source(name)


@pytest.mark.parametrize("variant", VARIANTS)
def test_actual_cli_and_assembled_ssh_receiver_keep_reservations_and_shutdown(
    tmp_path, monkeypatch, variant
):
    assert os.environ.get("SERVEGAUGE_OWNER_RUN") is None
    assert os.environ.get("TRUEWEIGHTS_OWNER_RUN") is None
    initialize(tmp_path)
    (tmp_path / "configs/budgets.yaml").write_text(yaml.safe_dump(policy().model_dump(mode="json")))
    settings = reviewed(variant)
    folder = tmp_path / "configs/selfhost"
    folder.mkdir()
    (folder / "study.json").write_text(settings.model_dump_json())
    key = tmp_path / "fixture-key"
    key.write_text("fixture")
    monkeypatch.chdir(tmp_path)
    environment = {
        "SERVEGAUGE_OWNER_RUN": "1",
        "SELFHOST_API_KEY": "native-fixture-endpoint-key",
        "SELFHOST_SSH_KEY_PATH": str(key),
    }
    monkeypatch.setattr(cli, "_selfhost_environment", lambda: dict(environment))
    machine = Machine()
    original = machine.execute

    def execute(args, **kwargs):
        if args[:3] == ["docker", "run", "--rm"]:
            machine.commands.append(list(args))
            return settings.engine_version + "\n"
        return original(args, **kwargs)

    calls = []

    def ssh(args, **kwargs):
        assert args[0] == "ssh"
        request = json.loads(kwargs["input"])
        assert "native-fixture-endpoint-key" not in str(args)
        ledger = Ledger(tmp_path)
        ledger.refresh()
        if request["action"] == "up":
            rows = [r for r in ledger.pending.values() if r["run_id"] == request["session_id"]]
            assert len(rows) == 1 and rows[0]["dispatched"] and rows[0]["owner"]
        source = shlex.split(args[-1])[-1]
        namespace = {"__name__": "local_reserved_receiver"}
        exec(compile(source, "local-reserved-receiver", "exec"), namespace)
        value = namespace["handle"](request, home=tmp_path / "remote-home", execute=execute)
        calls.append(request["action"])
        return SimpleNamespace(returncode=0, stdout=json.dumps(value))

    monkeypatch.setattr(subprocess, "run", ssh)
    sid = "selfhost-" + "d" * 32
    runner = CliRunner()
    args = [
        "selfhost",
        "up",
        "--profile",
        STUDY_PROFILE,
        "--config",
        "configs/selfhost/study.json",
        "--max-usd",
        "2",
        "--shutdown-minutes",
        "10",
        "--session-id",
        sid,
    ]
    prepared = runner.invoke(cli.app, args + ["--dry-run"])
    assert prepared.exit_code == 0, prepared.output
    assert not calls and not (tmp_path / "data/ledger").exists()
    launched = runner.invoke(cli.app, args + ["--yes"])
    assert launched.exit_code == 0, launched.output
    value = json.loads(launched.output)
    assert (
        value["state"] == "running"
        and value["plan"]["reference_qualification"] == "not_a_reference_session"
    )
    assert value["plan"]["engine_args"] == engine_args(settings)
    assert runner.invoke(cli.app, args + ["--yes"]).exit_code == 3 and calls == ["up"]
    stopped = runner.invoke(cli.app, ["selfhost", "down", "--session-id", sid, "--yes"])
    assert stopped.exit_code == 0, stopped.output
    assert json.loads(stopped.output)["state"] == "reconciled"
    assert calls == ["up", "down"]
    ledger = Ledger(tmp_path)
    ledger.refresh()
    assert not ledger.pending and ledger.development_spend() == 0
    assert ledger.by_run[sid] == Decimal("1.02")
    assert os.environ.get("SERVEGAUGE_OWNER_RUN") is None
    assert os.environ.get("TRUEWEIGHTS_OWNER_RUN") is None


def test_cli_exposes_incomplete_variant_template_without_dispatch(tmp_path, monkeypatch):
    initialize(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        cli.app,
        [
            "selfhost",
            "template",
            "--profile",
            STUDY_PROFILE,
            "--variant",
            "fp8",
            "--out",
            "configs/selfhost/study.json",
        ],
    )
    assert result.exit_code == 0, result.output
    value = json.loads((tmp_path / "configs/selfhost/study.json").read_text())
    assert value["precision"] == "fp8" and value["image"] is value["hourly_usd"] is None
    assert not (tmp_path / "data/ledger").exists()


@pytest.mark.parametrize("variant", VARIANTS)
def test_actual_modal_driver_uses_bundled_recipe_and_reserved_lifetime(
    tmp_path, monkeypatch, variant
):
    settings = reviewed(variant, "modal", modal_app="servegauge-owner")
    plan = build_plan(settings, policy(), max_usd=Decimal("2"), shutdown_minutes=10)
    prepare_session(tmp_path, SID, plan)
    calls = []
    sdk, _ = modal_sdk(plan, calls)
    original = sdk.Sandbox.create

    def create(*args, **kwargs):
        ledger = Ledger(tmp_path)
        ledger.refresh()
        row = next(iter(ledger.pending.values()))
        assert row["owner"] and row["dispatched"] and row["run_id"] == SID
        namespace = {"__name__": "local_modal_receiver"}
        exec(compile(args[2], "local-modal-receiver", "exec"), namespace)
        receipt = namespace["runtime_observation"](
            SID, plan, engine_version=settings.engine_version, gpu="NVIDIA H100 80GB HBM3, 570.26"
        )
        sandbox = original(*args, **kwargs)
        sandbox.filesystem = SimpleNamespace(read_text=lambda path: json.dumps(receipt))
        return sandbox

    sdk.Sandbox.create = create
    monkeypatch.setitem(sys.modules, "modal", sdk)
    env = environment(tmp_path / "unused")
    driver = ModalDriver(tmp_path)
    value = launch_session(tmp_path, SID, policy(), driver=driver, environment=env)
    assert value["state"] == "running" and calls.count("create") == 1
    assert "runtime_receipt_sha256" in value["facts"]
    value = stop_session(tmp_path, SID, driver=driver, environment=env)
    assert value["state"] == "reconciled" and calls.count("terminate") == 1
    ledger = Ledger(tmp_path)
    ledger.refresh()
    assert not ledger.pending and ledger.development_spend() == 0
    assert os.environ.get("SERVEGAUGE_OWNER_RUN") is None
    assert os.environ.get("TRUEWEIGHTS_OWNER_RUN") is None


def test_even_a_synthetic_owner_assertion_cannot_make_an_experiment_an_r2_reference():
    plan = build_plan(reviewed(), policy(), max_usd=Decimal("2"), shutdown_minutes=10)
    request = Qualification.model_validate(
        {
            "version": "owner-reference-v1",
            "tier": "R2",
            "profile": STUDY_PROFILE,
            "profile_hash": "a" * 64,
            "endpoint_hash": "b" * 64,
            "reviewed_at": "2026-10-21T00:00:00+00:00",
            "source_url": RECIPE,
            "source_sha256": "c" * 64,
            "model_mapping_reviewed": True,
            "origin": {
                "kind": "selfhost",
                "session_id": SID,
                "plan_hash": plan["plan_hash"],
                "runtime_sha256": "d" * 64,
            },
        }
    )
    endpoint = EndpointSpec(
        id="study",
        kind="selfhosted",
        base_url="http://127.0.0.1:8000/v1",
        model=STUDY_PROFILE,
        claimed_profile=STUDY_PROFILE,
    )
    with pytest.raises(ValueError, match="experiment_session_cannot_grant_r2"):
        _runtime(request, endpoint, {"plan": plan})
