"""Immutable owner plans and typed lifecycle facts remain private and read-only by default."""

import json
from decimal import Decimal

import pytest
from unit.test_selfhost_plan import config, policy

from servegauge.core.hashing import content_hash
from servegauge.selfhost.plan import build_plan
from servegauge.storage.selfhost import SessionJournal, prepare_session

SID = "selfhost-" + "a" * 32


def plan():
    return build_plan(config(), policy(), max_usd=Decimal("2"), shutdown_minutes=10)


def file_hashes(root):
    return {
        p.relative_to(root).as_posix(): content_hash(p.read_bytes().hex())
        for p in root.rglob("*")
        if p.is_file()
    }


def test_stage_is_immutable_repeat_is_read_only_and_lookup_does_not_create(tmp_path):
    journal = prepare_session(tmp_path, SID, plan())
    assert journal.spec()["session_id"] == SID
    assert journal.spec()["plan"] == plan()
    before = file_hashes(tmp_path)
    assert prepare_session(tmp_path, SID, plan()).spec() == journal.spec()
    assert file_hashes(tmp_path) == before
    assert SessionJournal(tmp_path, SID).status()["state"] == "prepared"
    assert file_hashes(tmp_path) == before
    with pytest.raises(ValueError):
        prepare_session(tmp_path, SID, plan() | {"max_usd": "3"})
    with pytest.raises(ValueError):
        SessionJournal(tmp_path, "selfhost-" + "b" * 32).status()
    assert file_hashes(tmp_path) == before
    assert not (tmp_path / "budget").exists()
    assert not (tmp_path / "data/ledger").exists()


@pytest.mark.parametrize("sid", ["../other", "Starter Files", "selfhost-", "selfhost-" + "A" * 32])
def test_noncanonical_ids_refuse_before_discovery(tmp_path, sid):
    with pytest.raises(ValueError):
        SessionJournal(tmp_path, sid)
    assert list(tmp_path.iterdir()) == []


def test_protected_parent_and_symlink_refuse_without_opening_them(tmp_path):
    with pytest.raises(ValueError):
        SessionJournal(tmp_path / "Starter Files", SID)
    with pytest.raises(ValueError):
        SessionJournal(tmp_path / ".git", SID)
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        prepare_session(linked, SID, plan())
    assert list(outside.iterdir()) == []


def test_secret_in_plan_refuses_instead_of_changing_reviewed_semantics(tmp_path, monkeypatch):
    monkeypatch.setenv("SELFHOST_FIXTURE_SECRET", "fixture-private-value")
    value = plan() | {"comment": "fixture-private-value"}
    with pytest.raises(ValueError):
        prepare_session(tmp_path, SID, value)
    assert not (tmp_path / "data").exists()


def test_exact_sealed_plan_and_chain_tampering_refuse(tmp_path):
    journal = prepare_session(tmp_path, SID, plan())
    path = journal.directory / "spec.json"
    value = json.loads(path.read_text())
    value["value"]["plan"]["max_usd"] = "999"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        journal.status()


def test_untyped_or_out_of_order_lifecycle_events_refuse(tmp_path):
    journal = prepare_session(tmp_path, SID, plan())
    with pytest.raises(ValueError):
        journal.append({"state": "running", "private_data": "anything"})
    with pytest.raises(ValueError):
        journal.record("running", {"remote_id": "arbitrary"})
    journal.record("reserved", {"reservation_id": "b" * 32})
    journal.record("dispatching", {})
    journal.record("outcome_unverified", {})
    assert journal.status()["state"] == "outcome_unverified"
    with pytest.raises(ValueError):
        journal.record("dispatching", {})
    with pytest.raises(ValueError):
        journal.record("running", {"endpoint_url": "https://user:credential@example.test/v1"})
    with pytest.raises(ValueError):
        journal.record("running", {"api_key": "value"})


def test_terminal_state_does_not_implicitly_release_unknown_budget(tmp_path):
    journal = prepare_session(tmp_path, SID, plan())
    journal.record("reserved", {"reservation_id": "b" * 32})
    journal.record("dispatching", {})
    journal.record(
        "running", {"remote_id": "container-fixture", "endpoint_url": "http://127.0.0.1:8000/v1"}
    )
    journal.record("stopping", {})
    journal.record("terminated", {"remote_exit_code": 0})
    assert journal.status()["billing_status"] == "reserved_unreconciled"
    with pytest.raises(ValueError):
        journal.record("reserved", {"reservation_id": "c" * 32})


def test_refresh_cannot_retarget_a_running_session_or_weaken_ssh_network_binding(tmp_path):
    journal = prepare_session(tmp_path, SID, plan())
    journal.record("reserved", {"reservation_id": "b" * 32})
    journal.record("dispatching", {})
    with pytest.raises(ValueError):
        journal.record(
            "running",
            {"remote_id": "container-fixture", "endpoint_url": "https://public.example.test/v1"},
        )
    journal.record(
        "running", {"remote_id": "container-fixture", "endpoint_url": "http://127.0.0.1:8000/v1"}
    )
    before = file_hashes(tmp_path)
    with pytest.raises(ValueError):
        journal.record(
            "running",
            {"remote_id": "another-container", "endpoint_url": "http://127.0.0.1:8000/v1"},
        )
    with pytest.raises(ValueError):
        journal.record(
            "running",
            {"remote_id": "container-fixture", "endpoint_url": "http://127.0.0.1:8001/v1"},
        )
    assert file_hashes(tmp_path) == before


@pytest.mark.parametrize("amount", [True, 1.0, "NaN", "-1", "not-a-number"])
def test_untyped_or_invalid_charge_fact_refuses(tmp_path, amount):
    journal = prepare_session(tmp_path, SID, plan())
    journal.record("reserved", {"reservation_id": "b" * 32})
    journal.record("cancelled", {})
    before = file_hashes(tmp_path)
    with pytest.raises(ValueError):
        journal.record("reconciled", {"amount_usd": amount, "cost_source": "not_dispatched"})
    assert file_hashes(tmp_path) == before
