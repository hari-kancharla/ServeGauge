"""Actual adapter methods with fake SSH and Modal SDK boundaries; no external calls."""

import json
import subprocess
import sys
from decimal import Decimal
from types import SimpleNamespace

import pytest
from unit.test_selfhost_plan import config, policy
from unit.test_selfhost_remote_ssh import Commands
from unit.test_selfhost_storage import SID

from servegauge.adapters.selfhosted import ModalDriver, SSHDriver
from servegauge.runner.ledger import Ledger
from servegauge.selfhost.controller import launch_session, stop_session
from servegauge.selfhost.plan import build_plan
from servegauge.selfhost.remote_ssh import handle
from servegauge.storage.selfhost import prepare_session


def environment(key):
    return {
        "SERVEGAUGE_OWNER_RUN": "1",
        "SELFHOST_API_KEY": "fixture-endpoint-key",
        "SELFHOST_SSH_KEY_PATH": str(key),
        "MODAL_TOKEN_ID": "fixture-modal-id",
        "MODAL_TOKEN_SECRET": "fixture-modal-secret",
    }


def stage(root, backend="ssh"):
    options = (
        {}
        if backend == "ssh"
        else {
            "backend": "modal",
            "ssh_host": None,
            "ssh_key_env": None,
            "modal_app": "servegauge-owner",
            "modal_image_id": "im-fixture",
            "inbound_cidrs": ["198.51.100.7/32"],
        }
    )
    plan = build_plan(config(**options), policy(), max_usd=Decimal("2"), shutdown_minutes=10)
    prepare_session(root, SID, plan)
    return plan


def test_ssh_adapter_requires_real_shared_dispatch_before_subprocess(tmp_path, monkeypatch):
    plan = stage(tmp_path)
    key = tmp_path / "fixture-key"
    key.write_text("fixture")

    def forbidden(*a, **k):
        raise AssertionError("SSH called without a reservation")

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(ValueError):
        SSHDriver(tmp_path).launch(SID, plan, environment(key))


def test_missing_local_key_refuses_before_shared_reservation(tmp_path, monkeypatch):
    stage(tmp_path)

    def forbidden(*a, **k):
        raise AssertionError("Missing key reached SSH")

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(ValueError):
        launch_session(
            tmp_path,
            SID,
            policy(),
            environment=environment(tmp_path / "missing"),
            driver=SSHDriver(tmp_path),
        )
    assert not (tmp_path / "data/ledger").exists()


def test_adapter_rejects_a_dispatched_hold_that_does_not_cover_the_reviewed_bound(
    tmp_path, monkeypatch
):
    from servegauge.runner.budget import BudgetManager
    from servegauge.selfhost.controller import request_hash
    from servegauge.storage.selfhost import SessionJournal

    plan = stage(tmp_path)
    key = tmp_path / "fixture-key"
    key.write_text("fixture")
    env = environment(key)
    manager = BudgetManager(tmp_path, SID, policy(), max_usd=Decimal("2"), environment=env)
    request = request_hash(SID, plan)
    reserved = manager.reserve(Decimal("0.01"), "selfhost:ssh", SID, request)
    manager.dispatch(reserved, request)
    journal = SessionJournal(tmp_path, SID)
    journal.record("reserved", {"reservation_id": reserved.reservation_id})
    journal.record("dispatching", {})

    def forbidden(*a, **k):
        raise AssertionError("Under-reserved session reached SSH")

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(ValueError):
        SSHDriver(tmp_path).launch(SID, plan, env)


def test_actual_ssh_adapter_remote_job_roundtrip_stores_proof_and_stops_only_target(
    tmp_path, monkeypatch
):
    stage(tmp_path)
    key = tmp_path / "fixture-key"
    key.write_text("fixture")
    commands = Commands()
    calls = []

    def ssh(args, **kwargs):
        assert args[0] == "ssh" and args[args.index("-F") + 1] == "none"
        assert "StrictHostKeyChecking=yes" in args and "BatchMode=yes" in args
        assert "fixture-endpoint-key" not in str(args)
        calls.append(args)
        result = handle(
            json.loads(kwargs["input"]), home=tmp_path / "remote-home", execute=commands
        )
        return SimpleNamespace(returncode=0, stdout=json.dumps(result))

    monkeypatch.setattr(subprocess, "run", ssh)
    driver = SSHDriver(tmp_path)
    result = launch_session(tmp_path, SID, policy(), environment=environment(key), driver=driver)
    assert result["state"] == "running" and len(calls) == 1
    assert "runtime_receipt_sha256" in result["facts"]
    result = stop_session(tmp_path, SID, environment=environment(key), driver=driver)
    assert result["state"] == "reconciled" and len(calls) == 2
    ledger = Ledger(tmp_path)
    ledger.refresh()
    assert not ledger.pending and ledger.development_spend() == 0
    assert all(
        b"fixture-endpoint-key" not in p.read_bytes()
        for p in (tmp_path / "data").rglob("*")
        if p.is_file()
    )


def modal_sdk(plan, calls, *, identifier=SID):
    state = {"running": True, "bad_tags": False}
    receipt = {
        "version": "selfhost-runtime-v1",
        "session_id": identifier,
        "plan_hash": plan["plan_hash"],
        "backend": "modal",
        "recorded_at": "2026-10-20T13:00:00+00:00",
        "engine_version": "0.11.0",
        "image": plan["configuration"]["image"],
        "model_revision": "b" * 40,
        "gpu_requested": "H100",
        "gpu_observed": "NVIDIA H100 80GB HBM3",
        "driver_version": "570.26",
        "engine_args": plan["engine_args"],
        "shutdown_armed": True,
        "shutdown_minutes": 10,
    }

    class Sandbox:
        object_id = "sb-fixture"
        filesystem = SimpleNamespace(read_text=lambda path: json.dumps(receipt))

        def get_tags(self):
            return {
                "servegauge_session": identifier,
                "servegauge_plan": "wrong" if state["bad_tags"] else plan["plan_hash"],
            }

        def poll(self):
            return None if state["running"] else 0

        def tunnels(self):
            return {8000: SimpleNamespace(url="https://fixture.modal.test")}

        def terminate(self, *, wait):
            assert wait
            calls.append("terminate")
            state["running"] = False
            return 0

    sandbox = Sandbox()

    class Image:
        object_id = plan["configuration"]["modal_image_id"]

        def entrypoint(self, args):
            assert args == []
            calls.append("entrypoint-cleared")
            return self

    def client(token_id, token_secret):
        assert token_id == "fixture-modal-id" and token_secret == "fixture-modal-secret"
        calls.append("client")
        return "fixture-client"

    def lookup(name, *, client, create_if_missing):
        assert name == "servegauge-owner" and not create_if_missing
        calls.append("lookup")
        return "fixture-app"

    def create(*args, **kwargs):
        assert args[:2] == ("python3", "-c")
        assert kwargs["timeout"] == 600 and kwargs["gpu"] == "H100"
        assert kwargs["encrypted_ports"] == [8000]
        assert kwargs["inbound_cidr_allowlist"] == ["198.51.100.7/32"]
        assert kwargs["name"] == "servegauge-" + identifier.removeprefix("selfhost-")
        calls.append("create")
        return sandbox

    def named(app, name, **kwargs):
        assert app == "servegauge-owner" and name == "servegauge-" + identifier.removeprefix(
            "selfhost-"
        )
        assert kwargs["include_terminated"] is True
        calls.append("from_name")
        return sandbox

    sdk = SimpleNamespace(
        Client=SimpleNamespace(from_credentials=client),
        App=SimpleNamespace(lookup=lookup),
        Image=SimpleNamespace(
            from_id=lambda image, client: (
                Image()
                if image == plan["configuration"]["modal_image_id"] and client == "fixture-client"
                else None
            )
        ),
        Secret=SimpleNamespace(from_dict=lambda values: values),
        Sandbox=SimpleNamespace(create=create, from_name=named),
    )
    return sdk, state


def test_modal_adapter_has_lifetime_allowlist_named_lookup_and_targeted_cleanup(
    tmp_path, monkeypatch
):
    plan = stage(tmp_path, "modal")
    calls = []
    sdk, state = modal_sdk(plan, calls)
    monkeypatch.setitem(sys.modules, "modal", sdk)
    env = environment(tmp_path / "unused")
    driver = ModalDriver(tmp_path)
    result = launch_session(tmp_path, SID, policy(), environment=env, driver=driver)
    assert result["state"] == "running" and calls.count("create") == 1
    assert "runtime_receipt_sha256" in result["facts"]
    state["bad_tags"] = True
    assert (
        stop_session(tmp_path, SID, environment=env, driver=driver)["state"] == "outcome_unverified"
    )
    assert "terminate" not in calls
    state["bad_tags"] = False
    result = stop_session(tmp_path, SID, environment=env, driver=driver)
    assert (
        result["state"] == "reconciled"
        and calls.count("create") == 1
        and calls.count("terminate") == 1
    )
