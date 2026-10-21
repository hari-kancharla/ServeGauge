"""Owner control uses the actual shared ledger while all remote actions are faked."""

import json
import os
from decimal import Decimal

import pytest
from unit.test_selfhost_plan import config, policy
from unit.test_selfhost_storage import SID

from servegauge.runner.ledger import Ledger
from servegauge.selfhost.controller import launch_session, refresh_session, stop_session
from servegauge.selfhost.plan import build_plan
from servegauge.storage.selfhost import prepare_session


class FakeDriver:
    def __init__(self, root, failure=None):
        self.root = root
        self.failure = failure
        self.calls = []
        self.running = True

    def preflight(self, identifier, plan, environment):
        return None

    def launch(self, identifier, plan, environment):
        ledger = Ledger(self.root)
        ledger.refresh()
        pending = list(ledger.pending.values())
        assert len(pending) == 1 and pending[0]["dispatched"] is True
        assert pending[0]["run_id"] == identifier
        assert Decimal(pending[0]["amount"]) == Decimal(plan["reserved_upper_usd"])
        self.calls.append("launch")
        if self.failure:
            raise RuntimeError("fixture-private-credential and opaque transport text")
        return self.inspect(identifier, plan, environment)

    def inspect(self, identifier, plan, environment):
        self.calls.append("inspect")
        return {
            "state": "running" if self.running else "terminated",
            "remote_id": "container-fixture",
            "endpoint_url": "http://127.0.0.1:8000/v1",
            "remote_exit_code": None if self.running else 0,
        }

    def stop(self, identifier, plan, environment):
        self.calls.append("stop")
        self.running = False
        return self.inspect(identifier, plan, environment)


def stage(root):
    value = build_plan(config(), policy(), max_usd=Decimal("2"), shutdown_minutes=10)
    return prepare_session(root, SID, value)


def owner_environment():
    # Test input mapping only: process owner flags stay unset throughout the test.
    return {
        "SERVEGAUGE_OWNER_RUN": "1",
        "SELFHOST_API_KEY": "fixture-private-credential",
        "SELFHOST_SSH_KEY_PATH": "/not/read/by/fake/driver",
    }


def assert_no_real_owner_flag():
    assert os.environ.get("SERVEGAUGE_OWNER_RUN") is None
    assert os.environ.get("TRUEWEIGHTS_OWNER_RUN") is None


def test_actual_shared_reservation_precedes_dispatch_and_duplicate_launch_refuses(tmp_path):
    assert_no_real_owner_flag()
    stage(tmp_path)
    driver = FakeDriver(tmp_path)
    result = launch_session(tmp_path, SID, policy(), environment=owner_environment(), driver=driver)
    assert result["state"] == "running" and driver.calls.count("launch") == 1
    with pytest.raises(ValueError):
        launch_session(tmp_path, SID, policy(), environment=owner_environment(), driver=driver)
    assert driver.calls.count("launch") == 1
    ledger = Ledger(tmp_path)
    ledger.refresh()
    assert len(ledger.pending) == 1 and ledger.development_spend() == 0
    assert_no_real_owner_flag()


@pytest.mark.parametrize("mode", ["owner", "endpoint", "ssh"])
def test_missing_owner_or_credentials_refuses_before_any_budget_or_remote_action(tmp_path, mode):
    stage(tmp_path)
    environment = owner_environment()
    environment.pop(
        {
            "owner": "SERVEGAUGE_OWNER_RUN",
            "endpoint": "SELFHOST_API_KEY",
            "ssh": "SELFHOST_SSH_KEY_PATH",
        }[mode]
    )
    driver = FakeDriver(tmp_path)
    with pytest.raises(ValueError):
        launch_session(tmp_path, SID, policy(), environment=environment, driver=driver)
    assert driver.calls == [] and not (tmp_path / "data/ledger").exists()


def test_unknown_launch_keeps_the_hold_and_never_retries_create(tmp_path):
    stage(tmp_path)
    driver = FakeDriver(tmp_path, failure=True)
    result = launch_session(tmp_path, SID, policy(), environment=owner_environment(), driver=driver)
    assert result["state"] == "outcome_unverified"
    assert "fixture-private-credential" not in str(result)
    ledger = Ledger(tmp_path)
    ledger.refresh()
    assert len(ledger.pending) == 1 and ledger.development_spend() == 0
    with pytest.raises(ValueError):
        launch_session(tmp_path, SID, policy(), environment=owner_environment(), driver=driver)
    assert driver.calls == ["launch"]


def test_targeted_stop_reconciles_conservative_bound_without_reserving_again(tmp_path):
    journal = stage(tmp_path)
    driver = FakeDriver(tmp_path)
    launch_session(tmp_path, SID, policy(), environment=owner_environment(), driver=driver)
    result = stop_session(tmp_path, SID, environment=owner_environment(), driver=driver)
    assert result["state"] == "reconciled"
    ledger = Ledger(tmp_path)
    ledger.refresh()
    assert not ledger.pending and ledger.development_spend() == 0
    assert ledger.by_run[SID] == Decimal("1.02")
    assert result["facts"]["cost_source"] == "estimate_selfhost_bound"
    assert journal.events()[-1]["state"] == "reconciled"
    before = ledger.path.read_bytes()
    assert stop_session(tmp_path, SID, environment=owner_environment(), driver=driver) == result
    assert ledger.path.read_bytes() == before and driver.calls.count("stop") == 1


def test_refresh_never_reserves_or_creates_a_replacement_session(tmp_path):
    stage(tmp_path)
    driver = FakeDriver(tmp_path)
    launch_session(tmp_path, SID, policy(), environment=owner_environment(), driver=driver)
    driver.running = False
    result = refresh_session(tmp_path, SID, environment=owner_environment(), driver=driver)
    assert result["state"] == "reconciled" and driver.calls.count("launch") == 1
    assert json.loads((tmp_path / "budget/dev_spend.json").read_text())["spent_usd"] == 0


def test_pre_dispatch_orphan_reservation_is_adopted_without_double_holding(tmp_path):
    from servegauge.runner.budget import BudgetManager
    from servegauge.selfhost.controller import request_hash

    journal = stage(tmp_path)
    manager = BudgetManager(
        tmp_path, SID, policy(), max_usd=Decimal("2"), environment=owner_environment()
    )
    reservation = manager.reserve(
        Decimal("1.02"), "selfhost:ssh", SID, request_hash(SID, journal.spec()["plan"])
    )
    driver = FakeDriver(tmp_path)
    launch_session(tmp_path, SID, policy(), environment=owner_environment(), driver=driver)
    manager.ledger.refresh()
    assert set(manager.ledger.pending) == {reservation.reservation_id}


def test_dispatched_orphan_does_not_trigger_a_second_external_launch(tmp_path):
    from servegauge.runner.budget import BudgetManager
    from servegauge.selfhost.controller import request_hash

    journal = stage(tmp_path)
    manager = BudgetManager(
        tmp_path, SID, policy(), max_usd=Decimal("2"), environment=owner_environment()
    )
    request = request_hash(SID, journal.spec()["plan"])
    reservation = manager.reserve(Decimal("1.02"), "selfhost:ssh", SID, request)
    manager.dispatch(reservation, request)
    driver = FakeDriver(tmp_path)
    result = launch_session(tmp_path, SID, policy(), environment=owner_environment(), driver=driver)
    assert result["state"] == "outcome_unverified" and driver.calls == []


def test_current_policy_changes_refuse_launch_but_cannot_block_cleanup(tmp_path):
    stage(tmp_path)
    driver = FakeDriver(tmp_path)
    with pytest.raises(ValueError):
        launch_session(
            tmp_path,
            SID,
            policy(selfhost_usd_per_session=0),
            environment=owner_environment(),
            driver=driver,
        )
    launch_session(tmp_path, SID, policy(), environment=owner_environment(), driver=driver)
    assert (
        stop_session(tmp_path, SID, environment=owner_environment(), driver=driver)["state"]
        == "reconciled"
    )


def test_corrupt_or_missing_session_refuses_before_any_driver_action(tmp_path):
    driver = FakeDriver(tmp_path)
    with pytest.raises(ValueError):
        stop_session(tmp_path, SID, environment=owner_environment(), driver=driver)
    assert driver.calls == []


def test_stop_does_not_need_the_model_endpoint_key(tmp_path):
    stage(tmp_path)
    driver = FakeDriver(tmp_path)
    launch_session(tmp_path, SID, policy(), environment=owner_environment(), driver=driver)
    environment = owner_environment()
    environment.pop("SELFHOST_API_KEY")
    assert (
        stop_session(tmp_path, SID, environment=environment, driver=driver)["state"] == "reconciled"
    )


def test_pre_dispatch_cleanup_releases_only_a_proven_undispatched_hold(tmp_path):
    from servegauge.runner.budget import BudgetManager
    from servegauge.selfhost.controller import request_hash

    journal = stage(tmp_path)
    manager = BudgetManager(
        tmp_path, SID, policy(), max_usd=Decimal("2"), environment=owner_environment()
    )
    reservation = manager.reserve(
        Decimal("1.02"), "selfhost:ssh", SID, request_hash(SID, journal.spec()["plan"])
    )
    driver = FakeDriver(tmp_path)
    # Recover the crash before journal.record(reserved), without calling SSH.
    result = stop_session(tmp_path, SID, environment={"SERVEGAUGE_OWNER_RUN": "1"}, driver=driver)
    assert result["state"] == "reconciled" and result["facts"]["cost_source"] == "not_dispatched"
    assert result["facts"]["reservation_id"] == reservation.reservation_id
    manager.ledger.refresh()
    assert not manager.ledger.pending and manager.ledger.by_run[SID] == 0 and driver.calls == []


def test_stop_recovers_a_dispatched_reservation_before_its_journal_event(tmp_path):
    from servegauge.runner.budget import BudgetManager
    from servegauge.selfhost.controller import request_hash

    journal = stage(tmp_path)
    manager = BudgetManager(
        tmp_path, SID, policy(), max_usd=Decimal("2"), environment=owner_environment()
    )
    request = request_hash(SID, journal.spec()["plan"])
    reservation = manager.reserve(Decimal("1.02"), "selfhost:ssh", SID, request)
    manager.dispatch(reservation, request)
    journal.record("reserved", {"reservation_id": reservation.reservation_id})
    driver = FakeDriver(tmp_path)
    assert (
        stop_session(tmp_path, SID, environment=owner_environment(), driver=driver)["state"]
        == "reconciled"
    )
    assert driver.calls == ["stop", "inspect"]


def test_credentials_returned_in_driver_facts_are_never_written(tmp_path):
    stage(tmp_path)
    driver = FakeDriver(tmp_path)
    driver.inspect = lambda *args: {
        "state": "running",
        "remote_id": "fixture-private-credential",
        "endpoint_url": "http://127.0.0.1:8000/v1",
        "remote_exit_code": None,
    }
    result = launch_session(tmp_path, SID, policy(), environment=owner_environment(), driver=driver)
    assert result["state"] == "outcome_unverified"
    assert all(
        b"fixture-private-credential" not in p.read_bytes()
        for p in (tmp_path / "data").rglob("*")
        if p.is_file()
    )
