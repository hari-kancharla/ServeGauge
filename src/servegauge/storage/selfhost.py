"""Private immutable session plans and typed facts; never dispatch or release funds."""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from filelock import FileLock

from servegauge.core.hashing import canonical_json
from servegauge.core.ids import safe_filename
from servegauge.core.redaction import Redactor
from servegauge.core.secrets import PATTERNS
from servegauge.observatory.provenance import implementation_hash
from servegauge.selfhost.plan import validate_plan
from servegauge.storage.monitor import MonitorJournal
from servegauge.storage.notices import guard_path

TRANSITIONS = {
    "prepared": {"reserved"},
    "reserved": {"dispatching", "cancelled"},
    "dispatching": {"running", "outcome_unverified", "terminated"},
    "running": {"running", "stopping", "outcome_unverified", "terminated"},
    "outcome_unverified": {"running", "stopping", "outcome_unverified", "terminated"},
    "stopping": {"stopping", "outcome_unverified", "terminated"},
    "terminated": {"reconciled"},
    "cancelled": {"reconciled"},
    "reconciled": set(),
}
FIELDS = {
    "reserved": {"reservation_id"},
    "dispatching": set(),
    "cancelled": set(),
    "running": {"remote_id", "endpoint_url", "runtime_receipt_sha256"},
    "outcome_unverified": set(),
    "stopping": set(),
    "terminated": {"remote_id", "remote_exit_code", "runtime_receipt_sha256"},
    "reconciled": {"amount_usd", "cost_source"},
}


def _clean(value: dict[str, Any]) -> None:
    if Redactor.from_environment().redact(value) != value or any(
        pattern.search(canonical_json(value)) for pattern in PATTERNS
    ):
        raise ValueError("selfhost_private_credentials_refused")


def _facts(state: str, facts: dict[str, Any]) -> None:
    if state not in FIELDS or not facts.keys() <= FIELDS[state]:
        raise ValueError("selfhost_typed_session_facts_required")
    required = {
        "reserved": {"reservation_id"},
        "running": {"remote_id", "endpoint_url"},
        "terminated": {"remote_exit_code"},
        "reconciled": {"amount_usd", "cost_source"},
    }.get(state, set())
    if not required <= facts.keys():
        raise ValueError("selfhost_complete_session_facts_required")
    for key in ["reservation_id", "runtime_receipt_sha256"]:
        if key in facts and (
            not isinstance(facts[key], str)
            or not re.fullmatch(
                r"[a-f0-9]{32}" if key == "reservation_id" else r"[a-f0-9]{64}", facts[key]
            )
        ):
            raise ValueError("selfhost_fact_identity_invalid")
    if "remote_id" in facts and (
        not isinstance(facts["remote_id"], str)
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", facts["remote_id"])
    ):
        raise ValueError("selfhost_remote_identity_invalid")
    if "endpoint_url" in facts:
        if not isinstance(facts["endpoint_url"], str):
            raise ValueError("selfhost_endpoint_invalid")
        parts = urlsplit(facts["endpoint_url"])
        if (
            parts.scheme not in {"https", "http"}
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
            or parts.path != "/v1"
            or (parts.scheme == "http" and parts.hostname not in {"127.0.0.1", "::1"})
        ):
            raise ValueError("selfhost_private_canonical_endpoint_required")
        _ = parts.port
    if "remote_exit_code" in facts and type(facts["remote_exit_code"]) is not int:
        raise ValueError("selfhost_remote_exit_code_invalid")
    if "amount_usd" in facts:
        if not isinstance(facts["amount_usd"], str):
            raise ValueError("selfhost_charge_fact_invalid")
        try:
            amount = Decimal(facts["amount_usd"])
        except ArithmeticError:
            raise ValueError("selfhost_charge_fact_invalid") from None
        if (
            not amount.is_finite()
            or amount < 0
            or facts["cost_source"]
            not in {
                "not_dispatched",
                "estimate_selfhost_bound",
                "estimate_selfhost_elapsed",
                "provider_invoice",
            }
        ):
            raise ValueError("selfhost_charge_fact_invalid")
    _clean(facts)


class SessionJournal(MonitorJournal):
    def __init__(self, root: Path, identifier: str) -> None:
        if not re.fullmatch(r"selfhost-[a-f0-9]{32}", identifier):
            raise ValueError("selfhost_canonical_session_id_required")
        self.root = root
        self.identifier = identifier
        super().__init__(root / "data/raw", identifier)
        self.directory = self.raw / "selfhost" / safe_filename(identifier)
        self._guard()

    def _guard(self) -> None:
        if ".." in self.root.parts or any(p.casefold() == ".git" for p in self.root.parts):
            raise ValueError("selfhost_canonical_workspace_required")
        guard_path(self.root)
        guard_path(self.directory)
        super()._guard()

    def append(self, data: dict[str, Any]) -> None:
        raise ValueError("selfhost_typed_session_record_required")

    def status(self) -> dict[str, Any]:
        spec = self.spec()
        if (
            set(spec) != {"version", "session_id", "prepared_at", "implementation_hash", "plan"}
            or spec["version"] != "selfhost-session-v1"
            or spec["session_id"] != self.identifier
        ):
            raise ValueError("selfhost_session_specification_invalid")
        validate_plan(spec["plan"])
        if not re.fullmatch(r"[a-f0-9]{64}", spec["implementation_hash"]):
            raise ValueError("selfhost_preparation_implementation_invalid")
        _clean(spec)
        previous = "prepared"
        facts: dict[str, Any] = {}
        for event in self.events():
            if set(event) != {"state", "facts"} or event["state"] not in TRANSITIONS[previous]:
                raise ValueError("selfhost_invalid_session_transition")
            _facts(event["state"], event["facts"])
            self._binding(spec["plan"], facts, event["facts"])
            previous = event["state"]
            facts.update(event["facts"])
        return {
            "session_id": self.identifier,
            "state": previous,
            "state_scope": "last_persisted_session_fact",
            "engine_readiness": "not_probed",
            "plan": spec["plan"],
            "facts": facts,
            "billing_status": "reconciled_estimate_or_charge"
            if previous == "reconciled"
            else "not_reserved"
            if previous == "prepared"
            else "reserved_unreconciled",
        }

    def record(self, state: str, facts: dict[str, Any]) -> None:
        _facts(state, facts)
        self._guard()
        with FileLock(str(self.directory / ".record.lock"), timeout=0):
            current = self.status()
            if state not in TRANSITIONS[current["state"]]:
                raise ValueError("selfhost_invalid_session_transition")
            self._binding(current["plan"], current["facts"], facts)
            records = self._records()
            self._commit(
                self.directory / f"event-{len(records) + 1:08}.json",
                {
                    "version": "monitor-event-v1",
                    "sequence": len(records) + 1,
                    "spec_hash": self._read(self.directory / "spec.json")["hash"],
                    "previous_hash": records[-1]["hash"] if records else None,
                    "created_at": datetime.now(UTC).isoformat(),
                    "data": {"state": state, "facts": facts},
                },
            )

    @staticmethod
    def _binding(plan: dict[str, Any], previous: dict[str, Any], facts: dict[str, Any]) -> None:
        for key in ["reservation_id", "remote_id", "endpoint_url"]:
            if key in previous and key in facts and previous[key] != facts[key]:
                raise ValueError("selfhost_existing_session_binding_changed")
        if "endpoint_url" in facts:
            parts = urlsplit(facts["endpoint_url"])
            expected = "http" if plan["configuration"]["backend"] == "ssh" else "https"
            if parts.scheme != expected:
                raise ValueError("selfhost_backend_network_binding_changed")


def prepare_session(root: Path, identifier: str, plan: dict[str, Any]) -> SessionJournal:
    journal = SessionJournal(root, identifier)
    validate_plan(plan)
    _clean(plan)
    namespace = journal.directory.parent
    lock = namespace / (".prepare-" + identifier + ".lock")
    guard_path(lock)
    namespace.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock), timeout=0):
        descriptor = journal.directory / "spec.json"
        guard_path(descriptor)
        if descriptor.exists():
            current = journal.status()
            if current["plan"] != plan:
                raise ValueError("selfhost_reviewed_plan_conflict")
            return journal
        journal.create(
            {
                "version": "selfhost-session-v1",
                "session_id": identifier,
                "prepared_at": datetime.now(UTC).isoformat(),
                "implementation_hash": implementation_hash(),
                "plan": plan,
            }
        )
    return journal


def write_template(root: Path, out: Path, value: dict[str, Any]) -> None:
    path = out if out.is_absolute() else root / out
    if (
        ".." in path.parts
        or any(p.casefold().startswith(".env") for p in path.parts)
        or path.suffix != ".json"
        or not any(
            path.absolute().is_relative_to(root.absolute() / name)
            for name in ["configs/selfhost", "reports/selfhost"]
        )
    ):
        raise ValueError("selfhost_template_owned_json_namespace_required")
    guard_path(root)
    guard_path(path)
    _clean(value)
    blob = canonical_json(value) + b"\n"
    if path.exists():
        if not path.is_file() or path.read_bytes() != blob:
            raise ValueError("selfhost_template_will_not_replace_reviewed_configuration")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
