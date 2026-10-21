"""Atomic reservations and caps. The CLI cannot raise configured development limits."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from filelock import FileLock
from pydantic import Field

from servegauge.core.errors import SafeError
from servegauge.core.models import ValueModel
from servegauge.core.redaction import Redactor
from servegauge.runner.ledger import Ledger, current_month


class BudgetExceeded(SafeError):
    pass


class BudgetPolicy(ValueModel):
    monthly_usd: Decimal = Field(ge=0)
    per_run_usd: Decimal = Field(ge=0)
    dev_live_usd: Decimal = Field(ge=0)
    selfhost_usd_per_session: Decimal = Field(ge=0)
    stop_at_fraction: Decimal = Field(gt=0, le=1)


@dataclass(frozen=True)
class Reservation:
    reservation_id: str
    amount: Decimal
    request_hash: str


class BudgetManager:
    def __init__(
        self,
        root: Path,
        run_id: str,
        policy: BudgetPolicy,
        *,
        max_usd: Decimal | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.root, self.run_id, self.policy = root, run_id, policy
        configured_environment = environment if environment is not None else os.environ
        self.owner = (
            configured_environment.get(
                "SERVEGAUGE_OWNER_RUN", configured_environment.get("TRUEWEIGHTS_OWNER_RUN")
            )
            == "1"
        )
        configured = min(policy.per_run_usd, policy.monthly_usd)
        if not self.owner:
            configured = min(configured, policy.dev_live_usd)
        if max_usd is not None and (not max_usd.is_finite() or max_usd < 0 or max_usd > configured):
            raise BudgetExceeded("max_usd_exceeds_cap")
        self.run_limit = configured if max_usd is None else max_usd
        self.ledger = Ledger(root)
        self.redactor = Redactor.from_environment()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        path = self.root / "data/ledger"
        path.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path / ".budget.lock")):
            try:
                self.ledger.ensure_origin()
                yield
            except (KeyError, ValueError, TypeError, InvalidOperation):
                raise BudgetExceeded("ledger_corrupt") from None

    def snapshot(self) -> dict[str, Decimal]:
        with self._locked():
            return self._snapshot()

    def available(self) -> Decimal:
        """Maximum additional enforceable spend, including every outstanding hold."""
        with self._locked():
            state = self._snapshot()
            if self.ledger.overrun:
                return Decimal(0)
            pending = list(self.ledger.pending.values())
            run_pending = sum(
                (Decimal(row["amount"]) for row in pending if row["run_id"] == self.run_id),
                Decimal(0),
            )
            limits = [
                self.run_limit * self.policy.stop_at_fraction - state["run_spent"] - run_pending,
                self.policy.monthly_usd * self.policy.stop_at_fraction
                - state["monthly_spent"]
                - state["reserved"],
            ]
            if not self.owner:
                dev_pending = sum(
                    (Decimal(row["amount"]) for row in pending if not row["owner"]), Decimal(0)
                )
                limits.append(
                    self.policy.dev_live_usd * self.policy.stop_at_fraction
                    - state["development_spent"]
                    - dev_pending
                )
            return max(Decimal(0), min(limits))

    def _snapshot(self) -> dict[str, Decimal]:
        reserved = sum((Decimal(r["amount"]) for r in self.ledger.pending.values()), Decimal(0))
        return {
            "development_spent": self.ledger.development_spend(),
            "monthly_spent": self.ledger.monthly[current_month()],
            "run_spent": self.ledger.by_run[self.run_id],
            "reserved": reserved,
        }

    def reserve(
        self, amount: Decimal, endpoint_id: str, exchange_id: str, request_hash: str
    ) -> Reservation:
        if not amount.is_finite() or amount < 0:
            raise BudgetExceeded("invalid_reservation_amount")
        with self._locked():
            state = self._snapshot()
            pending = list(self.ledger.pending.values())
            run_pending = sum(
                (Decimal(r["amount"]) for r in pending if r["run_id"] == self.run_id), Decimal(0)
            )
            dev_pending = sum((Decimal(r["amount"]) for r in pending if not r["owner"]), Decimal(0))
            fraction = self.policy.stop_at_fraction
            if amount:
                if self.ledger.overrun:
                    raise BudgetExceeded("ledger_has_overrun")
                if state["run_spent"] + run_pending + amount > self.run_limit * fraction:
                    raise BudgetExceeded("per_run_cap")
                if (
                    state["monthly_spent"] + state["reserved"] + amount
                    > self.policy.monthly_usd * fraction
                ):
                    raise BudgetExceeded("monthly_cap")
                if (
                    not self.owner
                    and state["development_spent"] + dev_pending + amount
                    > self.policy.dev_live_usd * fraction
                ):
                    raise BudgetExceeded("development_cap")
            rid = uuid.uuid4().hex
            self.ledger.append(
                {
                    "type": "reserve",
                    "reservation_id": rid,
                    "amount": str(amount),
                    "run_id": self.run_id,
                    "endpoint_id": self.redactor.text(endpoint_id),
                    "exchange_id": exchange_id,
                    "request_hash": request_hash,
                    "owner": self.owner,
                    "created_at": datetime.now(UTC).isoformat(),
                    "month": current_month(),
                }
            )
            return Reservation(rid, amount, request_hash)

    def dispatch(self, reservation: Reservation, request_hash: str) -> None:
        with self._locked():
            record = self.ledger.pending.get(reservation.reservation_id)
            if record is None:
                raise BudgetExceeded("reservation_not_pending")
            if record["run_id"] != self.run_id or record["owner"] != self.owner:
                raise BudgetExceeded("reservation_scope_mismatch")
            if record["request_hash"] != request_hash or reservation.request_hash != request_hash:
                raise BudgetExceeded("reservation_request_mismatch")
            if record["dispatched"]:
                raise BudgetExceeded("reservation_already_dispatched")
            if Decimal(record["amount"]) != reservation.amount:
                raise BudgetExceeded("reservation_amount_mismatch")
            self.ledger.append({"type": "dispatch", "reservation_id": reservation.reservation_id})

    def reconcile(self, reservation_id: str, actual: Decimal, cost_source: str) -> None:
        if not actual.is_finite() or actual < 0:
            raise BudgetExceeded("invalid_actual_cost")
        with self._locked():
            record = self.ledger.pending.get(reservation_id)
            if record is None:
                raise BudgetExceeded("reservation_not_pending")
            if record["run_id"] != self.run_id or record["owner"] != self.owner:
                raise BudgetExceeded("reservation_scope_mismatch")
            over = actual > Decimal(record["amount"])
            self.ledger.append(
                {
                    "type": "reconcile",
                    "reservation_id": reservation_id,
                    "actual": str(actual),
                    "cost_source": cost_source,
                    "month": current_month(),
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )
            self.ledger.write_summary()
            if over:
                raise BudgetExceeded("actual_exceeded_reservation")

    def recover_interrupted(
        self, known_costs: dict[str, tuple[Decimal, str]] | None = None
    ) -> None:
        with self._locked():
            for rid, record in list(self.ledger.pending.items()):
                if record["run_id"] == self.run_id:
                    if record["owner"] != self.owner:
                        raise BudgetExceeded("resume_spending_mode_changed")
                    amount, source = (known_costs or {}).get(
                        rid, (Decimal(record["amount"]), "estimate_interrupted")
                    )
                    self.ledger.append(
                        {
                            "type": "reconcile",
                            "reservation_id": rid,
                            "actual": str(amount),
                            "cost_source": source,
                            "month": current_month(),
                            "created_at": datetime.now(UTC).isoformat(),
                        }
                    )
            self.ledger.write_summary()
