import json
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import pytest

from servegauge.runner.budget import BudgetExceeded, BudgetManager, BudgetPolicy


def manager(root: Path, run: str = "run", **kwargs: object) -> BudgetManager:
    return BudgetManager(
        root,
        run,
        BudgetPolicy(
            monthly_usd=10,
            per_run_usd=2,
            dev_live_usd=1,
            selfhost_usd_per_session=2,
            stop_at_fraction=1,
        ),
        environment={},
        **kwargs,
    )


def test_concurrent_reservations_and_reconciliation(tmp_path: Path) -> None:
    def reserve(index: int) -> tuple[str, str] | None:
        try:
            return str(index), (
                manager(tmp_path, str(index))
                .reserve(Decimal("0.2"), "endpoint", str(index), "wire-hash")
                .reservation_id
            )
        except BudgetExceeded:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        reserved = [value for value in pool.map(reserve, range(20)) if value is not None]
    assert len(reserved) == 5
    budget = manager(tmp_path)
    for run_id, rid in reserved:
        manager(tmp_path, run_id).reconcile(rid, Decimal("0.1"), "provider")
    assert budget.snapshot()["development_spent"] == Decimal("0.5")
    assert budget.snapshot()["reserved"] == 0
    assert json.loads((tmp_path / "budget/dev_spend.json").read_text())["spent_usd"] == 0.5


def test_override_cannot_raise_cap_and_owner_needs_environment_flag(tmp_path: Path) -> None:
    with pytest.raises(BudgetExceeded, match="max_usd_exceeds_cap"):
        manager(tmp_path, max_usd=Decimal("2"))
    policy = BudgetPolicy(
        monthly_usd=10,
        per_run_usd=3,
        dev_live_usd=1,
        selfhost_usd_per_session=2,
        stop_at_fraction=1,
    )
    # Simulate the owner's environment; never set the actual process variable.
    owner = BudgetManager(
        tmp_path, "owner", policy, environment={"SERVEGAUGE_OWNER_RUN": "1"}, max_usd=Decimal("2")
    )
    rid = owner.reserve(Decimal("1.5"), "endpoint", "exchange", "wire").reservation_id
    owner.reconcile(rid, Decimal("1.5"), "provider")
    assert owner.snapshot()["development_spent"] == 0
    assert owner.snapshot()["monthly_spent"] == Decimal("1.5")
    with pytest.raises(BudgetExceeded):
        BudgetManager(tmp_path, "bad", policy, environment={}, max_usd=Decimal("2"))


def test_committed_spend_survives_fresh_ledger_and_summary_lag(tmp_path: Path) -> None:
    summary = tmp_path / "budget/dev_spend.json"
    summary.parent.mkdir()
    summary.write_text('{"spent_usd":0.7,"currency":"USD"}\n')
    budget = manager(tmp_path)
    rid = budget.reserve(Decimal("0.2"), "ep", "ex", "wire").reservation_id
    budget.reconcile(rid, Decimal("0.2"), "provider")
    summary.write_text('{"spent_usd":0.7,"currency":"USD"}\n')
    assert manager(tmp_path).snapshot()["development_spent"] == Decimal("0.9")
    with pytest.raises(BudgetExceeded):
        manager(tmp_path).reserve(Decimal("0.2"), "ep", "ex2", "wire")


def test_dispatch_is_bound_to_wire_and_single_use(tmp_path: Path) -> None:
    budget = manager(tmp_path)
    reservation = budget.reserve(Decimal("0.2"), "ep", "ex", "right")
    with pytest.raises(BudgetExceeded, match="reservation_scope_mismatch"):
        manager(tmp_path, "other").dispatch(reservation, "right")
    with pytest.raises(BudgetExceeded, match="reservation_scope_mismatch"):
        manager(tmp_path, "other").reconcile(reservation.reservation_id, Decimal(0), "provider")
    with pytest.raises(BudgetExceeded, match="reservation_request_mismatch"):
        budget.dispatch(reservation, "wrong")
    budget.dispatch(reservation, "right")
    with pytest.raises(BudgetExceeded, match="reservation_already_dispatched"):
        budget.dispatch(reservation, "right")
    budget.reconcile(reservation.reservation_id, Decimal("0.1"), "provider")
    with pytest.raises(BudgetExceeded, match="reservation_not_pending"):
        budget.reconcile(reservation.reservation_id, Decimal("0.1"), "provider")


def test_unknown_inflight_cost_is_not_released_on_resume(tmp_path: Path) -> None:
    budget = manager(tmp_path)
    reservation = budget.reserve(Decimal("0.4"), "ep", "ex", "wire")
    budget.dispatch(reservation, "wire")
    resumed = manager(tmp_path)
    resumed.recover_interrupted()
    assert resumed.snapshot()["development_spent"] == Decimal("0.4")
    assert resumed.snapshot()["reserved"] == 0
    events = [
        json.loads(line) for line in (tmp_path / "data/ledger/spend.jsonl").read_text().splitlines()
    ]
    assert events[-1]["cost_source"] == "estimate_interrupted"


def test_actual_over_reservation_is_recorded_then_halts(tmp_path: Path) -> None:
    budget = manager(tmp_path)
    reservation = budget.reserve(Decimal("0.2"), "ep", "ex", "wire")
    with pytest.raises(BudgetExceeded, match="actual_exceeded_reservation"):
        budget.reconcile(reservation.reservation_id, Decimal("0.3"), "provider")
    assert budget.snapshot()["development_spent"] == Decimal("0.3")
    with pytest.raises(BudgetExceeded, match="ledger_has_overrun"):
        budget.reserve(Decimal("0.01"), "ep", "next", "wire")


def test_corrupt_ledger_fails_closed_and_free_work_remains_possible(tmp_path: Path) -> None:
    budget = manager(tmp_path)
    budget.reserve(Decimal(0), "ep", "ex", "wire")
    path = tmp_path / "data/ledger/spend.jsonl"
    with path.open("a") as handle:
        handle.write("partial")
    with pytest.raises(BudgetExceeded, match="ledger_corrupt"):
        manager(tmp_path).snapshot()
    fresh = tmp_path / "fresh"
    summary = fresh / "budget/dev_spend.json"
    summary.parent.mkdir(parents=True)
    summary.write_text('{"spent_usd":1}')
    assert manager(fresh).reserve(Decimal(0), "ep", "ex", "wire").amount == 0


def test_available_spend_accounts_for_concurrent_holds_and_reconciliation(tmp_path):
    first = manager(tmp_path, "one")
    second = manager(tmp_path, "two")
    assert first.available() == Decimal(1)
    hold = second.reserve(Decimal("0.3"), "e", "x", "h")
    assert first.available() == Decimal("0.7")
    second.reconcile(hold.reservation_id, Decimal("0.1"), "provider")
    assert first.available() == Decimal("0.9")
