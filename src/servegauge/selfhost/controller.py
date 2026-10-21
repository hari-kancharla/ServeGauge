"""Reservation-first owner control; uncertain remote outcomes never retry launch."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from filelock import FileLock

from servegauge.core.hashing import content_hash
from servegauge.core.redaction import Redactor
from servegauge.observatory.provenance import implementation_hash
from servegauge.runner.budget import BudgetManager, BudgetPolicy, Reservation
from servegauge.runner.ledger import Ledger
from servegauge.storage.notices import guard_path
from servegauge.storage.selfhost import SessionJournal
from servegauge.storage.selfhost_runtime import save_runtime


class SessionDriver(Protocol):
    def preflight(
        self, identifier: str, plan: dict[str, Any], environment: Mapping[str, str]
    ) -> None: ...

    def launch(
        self, identifier: str, plan: dict[str, Any], environment: Mapping[str, str]
    ) -> dict[str, Any]: ...

    def inspect(
        self, identifier: str, plan: dict[str, Any], environment: Mapping[str, str]
    ) -> dict[str, Any]: ...

    def stop(
        self, identifier: str, plan: dict[str, Any], environment: Mapping[str, str]
    ) -> dict[str, Any]: ...


def request_hash(identifier: str, plan: dict[str, Any]) -> str:
    return content_hash({"session_id": identifier, "plan_hash": plan["plan_hash"]})


def _authorize(
    plan: dict[str, Any], environment: Mapping[str, str], *, launch: bool, remote: bool = True
) -> None:
    if environment.get("SERVEGAUGE_OWNER_RUN", environment.get("TRUEWEIGHTS_OWNER_RUN")) != "1":
        raise ValueError("selfhost_owner_authorization_required")
    config = plan["configuration"]
    names = (
        [config["ssh_key_env"]]
        if config["backend"] == "ssh"
        else ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"]
    )
    if not remote:
        names = []
    elif launch:
        names.append(config["api_key_env"])
    if any(not environment.get(name) for name in names):
        raise ValueError("selfhost_owner_credentials_required")
    if launch:
        key = environment[config["api_key_env"]]
        if not 8 <= len(key) <= 4096 or "\0" in key:
            raise ValueError("selfhost_endpoint_credential_shape_invalid")


def _manager(
    journal: SessionJournal, policy: BudgetPolicy, environment: Mapping[str, str]
) -> BudgetManager:
    for name in [
        "data/ledger/spend.jsonl",
        "data/ledger/.budget.lock",
        "budget/dev_spend.json",
        "budget/dev_spend.next.json",
    ]:
        guard_path(journal.root / name)
    return BudgetManager(
        journal.root,
        journal.identifier,
        policy,
        max_usd=Decimal(journal.spec()["plan"]["max_usd"]),
        environment=environment,
    )


def _lock(journal: SessionJournal) -> FileLock:
    path = journal.directory.parent / (".control-" + journal.identifier + ".lock")
    guard_path(path)
    return FileLock(str(path), timeout=0)


def _pending(manager: BudgetManager, journal: SessionJournal) -> dict[str, Any] | None:
    manager.ledger.refresh()
    rows = [r for r in manager.ledger.pending.values() if r["run_id"] == journal.identifier]
    if len(rows) > 1:
        raise ValueError("selfhost_conflicting_session_reservations")
    if not rows:
        return None
    row = rows[0]
    plan = journal.spec()["plan"]
    if (
        row["owner"] is not True
        or row["request_hash"] != request_hash(journal.identifier, plan)
        or Decimal(row["amount"]) != Decimal(plan["reserved_upper_usd"])
        or row["endpoint_id"] != "selfhost:" + plan["configuration"]["backend"]
        or row["exchange_id"] != journal.identifier
    ):
        raise ValueError("selfhost_existing_reservation_binding_changed")
    return dict(row)


def _settle(journal: SessionJournal, environment: Mapping[str, str]) -> dict[str, Any]:
    status = journal.status()
    if status["state"] not in {"terminated", "cancelled"}:
        return status
    plan = status["plan"]
    manager = _manager(journal, BudgetPolicy.model_validate(plan["budget_policy"]), environment)
    row = _pending(manager, journal)
    identifier = status["facts"]["reservation_id"]
    cancelled = status["state"] == "cancelled"
    amount = Decimal(0) if cancelled else Decimal(plan["reserved_upper_usd"])
    cost_source = "not_dispatched" if cancelled else "estimate_selfhost_bound"
    if row is not None:
        if row["reservation_id"] != identifier:
            raise ValueError("selfhost_terminal_reservation_mismatch")
        if cancelled and row["dispatched"]:
            raise ValueError("selfhost_dispatched_cost_cannot_be_cancelled")
        manager.reconcile(identifier, amount, cost_source)
    else:
        # Recover a crash between the shared-ledger commit and the journal commit.
        # Missing pending data alone never proves that a charge was reconciled.
        records = [json.loads(line) for line in manager.ledger.path.read_bytes().splitlines()]
        matching = [
            r for r in records if r["type"] == "reconcile" and r["reservation_id"] == identifier
        ]
        if (
            len(matching) != 1
            or Decimal(matching[0]["actual"]) != amount
            or matching[0]["cost_source"] != cost_source
            or manager.ledger.by_run[journal.identifier] != amount
        ):
            raise ValueError("selfhost_missing_terminal_charge_proof")
    journal.record("reconciled", {"amount_usd": str(amount), "cost_source": cost_source})
    return journal.status()


def _observe(
    journal: SessionJournal, result: dict[str, Any], environment: Mapping[str, str]
) -> dict[str, Any]:
    secrets = tuple(
        value
        for name, value in environment.items()
        if len(value) >= 8
        and re.search(r"(?:^|_)(?:KEY|TOKEN|SECRET|PASSWORD)(?:$|_)", name.upper())
    )
    if Redactor(values=secrets).redact(result) != result:
        raise ValueError("selfhost_remote_observation_contains_credentials")
    required = {
        "state",
        "remote_id",
        "endpoint_url",
        "remote_exit_code",
    }
    if (
        not isinstance(result, dict)
        or not required <= result.keys()
        or not result.keys() <= required | {"runtime_receipt"}
    ):
        raise ValueError("selfhost_typed_remote_observation_required")
    receipt = result.get("runtime_receipt")
    extra = {}
    if receipt is not None:
        extra["runtime_receipt_sha256"] = save_runtime(
            journal.root, journal.identifier, journal.spec()["plan"], receipt
        )
    if result["state"] == "running" and result["remote_exit_code"] is None:
        facts = {**{k: result[k] for k in ["remote_id", "endpoint_url"]}, **extra}
        status = journal.status()
        if status["state"] != "running" or any(
            status["facts"].get(k) != v for k, v in facts.items()
        ):
            journal.record("running", facts)
    elif result["state"] == "terminated":
        # Remote resource identity remains pinned even when the first observation
        # arrives after the engine has exited; never invent an intermediate run.
        facts = {
            "remote_exit_code": result["remote_exit_code"],
            "remote_id": result["remote_id"],
            **extra,
        }
        journal.record("terminated", facts)
        return _settle(journal, environment)
    else:
        raise ValueError("selfhost_remote_outcome_unverified")
    return journal.status()


def _unverified(journal: SessionJournal) -> dict[str, Any]:
    state = journal.status()["state"]
    if state in {"dispatching", "running", "stopping"}:
        journal.record("outcome_unverified", {})
    return journal.status()


def launch_session(
    root: Path,
    identifier: str,
    policy: BudgetPolicy,
    *,
    driver: SessionDriver,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    environment = os.environ if environment is None else environment
    journal = SessionJournal(root, identifier)
    status = journal.status()
    plan = status["plan"]
    _authorize(plan, environment, launch=True)
    if (
        policy != BudgetPolicy.model_validate(plan["budget_policy"])
        or journal.spec()["implementation_hash"] != implementation_hash()
    ):
        raise ValueError("selfhost_reviewed_inputs_changed")
    with _lock(journal):
        status = journal.status()
        if status["state"] not in {"prepared", "reserved"}:
            raise ValueError("selfhost_launch_already_attempted_use_status_or_down")
        manager = _manager(journal, policy, environment)
        row = _pending(manager, journal)
        if row is None or not row["dispatched"]:
            # Local-only checks precede the shared hold. Actual account/SSH calls
            # remain forbidden until dispatch has been durably recorded below.
            driver.preflight(identifier, plan, environment)
        if row is None:
            if status["state"] != "prepared" or identifier in manager.ledger.by_run:
                raise ValueError("selfhost_existing_session_ledger_conflict")
            reservation = manager.reserve(
                Decimal(plan["reserved_upper_usd"]),
                "selfhost:" + plan["configuration"]["backend"],
                identifier,
                request_hash(identifier, plan),
            )
            row = _pending(manager, journal)
            assert row is not None
        else:
            reservation = Reservation(
                row["reservation_id"], Decimal(row["amount"]), row["request_hash"]
            )
        if status["state"] == "prepared":
            journal.record("reserved", {"reservation_id": reservation.reservation_id})
        elif status["facts"]["reservation_id"] != reservation.reservation_id:
            raise ValueError("selfhost_existing_reservation_identity_changed")
        if row["dispatched"]:
            journal.record("dispatching", {})
            return _unverified(journal)
        manager.dispatch(reservation, request_hash(identifier, plan))
        journal.record("dispatching", {})
        try:
            return _observe(journal, driver.launch(identifier, plan, environment), environment)
        except Exception:
            # Never persist exception, argv, remote stderr, environment or secrets.
            return _unverified(journal)


def _control(
    root: Path,
    identifier: str,
    driver: SessionDriver,
    environment: Mapping[str, str],
    *,
    stop: bool,
) -> dict[str, Any]:
    journal = SessionJournal(root, identifier)
    status = journal.status()
    if status["state"] == "reconciled":
        return status
    if status["state"] == "prepared":
        path = root / "data/ledger/spend.jsonl"
        guard_path(path)
        ledger = Ledger(root)
        ledger.refresh()
        if not any(r["run_id"] == identifier for r in ledger.pending.values()):
            return status
    _authorize(status["plan"], environment, launch=False, remote=False)
    with _lock(journal):
        status = journal.status()
        if status["state"] == "reconciled":
            return status
        if status["state"] in {"terminated", "cancelled"}:
            return _settle(journal, environment)
        if status["state"] in {"prepared", "reserved"}:
            manager = _manager(
                journal, BudgetPolicy.model_validate(status["plan"]["budget_policy"]), environment
            )
            row = _pending(manager, journal)
            if row is None:
                raise ValueError("selfhost_session_reservation_missing")
            if status["state"] == "prepared":
                journal.record("reserved", {"reservation_id": row["reservation_id"]})
            if not row["dispatched"]:
                if stop:
                    journal.record("cancelled", {})
                    return _settle(journal, environment)
                return journal.status()
            journal.record("dispatching", {})
            _unverified(journal)
            status = journal.status()
        _authorize(status["plan"], environment, launch=False)
        try:
            if stop:
                if status["state"] != "stopping":
                    journal.record("stopping", {})
                result = driver.stop(identifier, status["plan"], environment)
            else:
                result = driver.inspect(identifier, status["plan"], environment)
            return _observe(journal, result, environment)
        except Exception:
            return _unverified(journal)


def refresh_session(
    root: Path,
    identifier: str,
    *,
    driver: SessionDriver,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return _control(
        root, identifier, driver, os.environ if environment is None else environment, stop=False
    )


def stop_session(
    root: Path,
    identifier: str,
    *,
    driver: SessionDriver,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return _control(
        root, identifier, driver, os.environ if environment is None else environment, stop=True
    )
