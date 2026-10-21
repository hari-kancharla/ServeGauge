"""Append-only money journal. Pending reservations survive process loss."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from servegauge.core.hashing import canonical_json


class Ledger:
    def __init__(self, root: Path) -> None:
        self.path = root / "data/ledger/spend.jsonl"
        self.summary = root / "budget/dev_spend.json"
        self.offset = 0
        self.origin: Decimal | None = None
        self.development = Decimal(0)
        self.monthly: defaultdict[str, Decimal] = defaultdict(Decimal)
        self.by_run: defaultdict[str, Decimal] = defaultdict(Decimal)
        self.pending: dict[str, dict[str, Any]] = {}
        self.overrun = False

    def summary_spend(self) -> Decimal:
        if not self.summary.exists():
            return Decimal(0)
        value = Decimal(str(json.loads(self.summary.read_text(encoding="utf-8"))["spent_usd"]))
        if not value.is_finite() or value < 0:
            raise ValueError("Invalid spend summary")
        return value

    def refresh(self) -> None:
        if not self.path.exists():
            if self.offset:
                raise ValueError("Ledger disappeared")
            return
        if self.path.stat().st_size < self.offset:
            raise ValueError("Ledger truncated")
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            for line in handle:
                if not line.endswith(b"\n"):
                    raise ValueError("Incomplete ledger line")
                event = json.loads(line)
                self._apply(event)
                self.offset += len(line)

    def _apply(self, event: dict[str, Any]) -> None:
        kind = event["type"]
        if kind == "origin":
            if self.origin is not None or self.offset != 0:
                raise ValueError("Repeated ledger origin")
            self.origin = Decimal(event["development_baseline"])
        elif self.origin is None:
            raise ValueError("Missing ledger origin")
        elif kind == "reserve":
            rid = event["reservation_id"]
            if rid in self.pending:
                raise ValueError("Duplicate reservation")
            amount = Decimal(event["amount"])
            if not amount.is_finite() or amount < 0:
                raise ValueError("Invalid reservation amount")
            self.pending[rid] = {**event, "dispatched": False}
        elif kind == "dispatch":
            record = self.pending[event["reservation_id"]]
            if record["dispatched"]:
                raise ValueError("Duplicate dispatch")
            record["dispatched"] = True
        elif kind == "reconcile":
            record = self.pending.pop(event["reservation_id"])
            amount = Decimal(event["actual"])
            if not amount.is_finite() or amount < 0:
                raise ValueError("Invalid actual cost")
            if not record["owner"]:
                self.development += amount
            self.monthly[event["month"]] += amount
            self.by_run[record["run_id"]] += amount
            self.overrun |= amount > Decimal(record["amount"])
        else:
            raise ValueError("Unknown ledger event")

    def append(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = canonical_json(event) + b"\n"
        with self.path.open("ab") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        self.refresh()

    def ensure_origin(self) -> None:
        self.refresh()
        if self.origin is None:
            self.append({"type": "origin", "development_baseline": str(self.summary_spend())})

    def development_spend(self) -> Decimal:
        return max(self.summary_spend(), (self.origin or Decimal(0)) + self.development)

    def write_summary(self) -> None:
        self.summary.parent.mkdir(parents=True, exist_ok=True)
        value = {"spent_usd": float(self.development_spend()), "currency": "USD"}
        temporary = self.summary.with_suffix(".next.json")
        with temporary.open("wb") as handle:
            handle.write(canonical_json(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.summary)


def current_month() -> str:
    return datetime.now(UTC).strftime("%Y-%m")
