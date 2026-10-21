"""Promptfoo callback: finite captured transport only, with no detector or grader."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.mock.sealed_replay import (
    REPLAY_ADDRESS_EXTENSION,
    FiniteReplay,
    ReplayAddress,
    ReplayObservation,
    ReplayTransport,
)
from servegauge.runner.budget import BudgetManager, BudgetPolicy
from servegauge.storage.comparison import _guard, _private_json, _relative, _require_sanitized
from servegauge.transport.http import send_reserved, wire_hash

URL = "http://127.0.0.1:8219/v1/chat/completions"


@dataclass
class _State:
    root: Path
    path: Path
    digest: str
    tape: FiniteReplay
    budget: BudgetManager
    refused: bool = False
    audit_hash: str | None = None

    def audit(self) -> None:
        destination = self.root / "provider-audit.json"
        _guard(destination)
        if self.audit_hash is not None and (
            not destination.is_file()
            or hashlib.sha256(destination.read_bytes()).hexdigest() != self.audit_hash
        ):
            raise ValueError("captured_provider_audit_changed")
        values = self.budget.snapshot()
        value = {
            "version": "external-captured-provider-v1",
            "book_sha256": self.digest,
            "tape": self.tape.audit(),
            "pending_reservations": len(self.budget.ledger.pending),
            "new_spend_usd": float(values["development_spent"]),
            "replay_adds_independent_n": False,
            "harness_refused": self.refused,
        }
        pending = self.root / "provider-audit.pending"
        with pending.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        pending.replace(destination)
        self.audit_hash = hashlib.sha256(destination.read_bytes()).hexdigest()


_state: _State | None = None


def _load(options: dict[str, Any]) -> _State:
    global _state
    root = Path.cwd()
    _guard(root)
    config = options["config"]
    if (
        not isinstance(config, dict)
        or not {"book", "book_sha256"} <= config.keys()
        or set(config) - {"book", "book_sha256", "basePath"}
    ):
        raise ValueError("captured_provider_config_required")
    if "basePath" in config:
        if not isinstance(config["basePath"], str):
            raise ValueError("captured_provider_base_path_changed")
        base = Path(config["basePath"])
        _guard(base)
        if base.absolute() != root:
            raise ValueError("captured_provider_base_path_changed")
    if not isinstance(config["book"], str) or not isinstance(config["book_sha256"], str):
        raise ValueError("captured_provider_config_required")
    path = _relative(root, config["book"])
    digest = config["book_sha256"]
    blob = path.read_bytes()
    if len(digest) != 64 or hashlib.sha256(blob).hexdigest() != digest:
        raise ValueError("captured_provider_book_changed")
    if _state is not None:
        if (_state.root, _state.path, _state.digest) != (root, path, digest):
            raise ValueError("captured_provider_worker_scope_changed")
        return _state
    for name in ["provider-audit.json", "provider-audit.pending", "data", "budget"]:
        target = root / name
        _guard(target)
        if target.exists():
            raise ValueError("captured_provider_fresh_worker_required")
    value = _private_json(blob)
    _require_sanitized(value)
    if not isinstance(value, list) or not value:
        raise ValueError("captured_provider_nonempty_book_required")
    rows = [ReplayObservation.model_validate(row) for row in value]
    tape = FiniteReplay(rows)
    budget = BudgetManager(
        root,
        "external-captured-replay-" + content_hash({"book_sha256": digest}),
        BudgetPolicy(
            monthly_usd=Decimal(0),
            per_run_usd=Decimal(0),
            dev_live_usd=Decimal(0),
            selfhost_usd_per_session=Decimal(0),
            stop_at_fraction=Decimal(1),
        ),
        environment={},
    )
    _state = _State(root, path, digest, tape, budget)
    return _state


async def call_api(prompt: str, options: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Return source channels or an explicit harness/source error, never a verdict.

    Source addresses travel in callback context. The prompt must be the exact
    canonical original request body; reformatting/duplicate keys are not normalized.
    The parent launcher must enforce the OS boundary and immutable config/input pins.
    """
    state: _State | None = None
    try:
        state = _load(options)
        if state.refused or state.tape.audit()["refusals"]:
            raise ValueError("captured_provider_worker_refused")
        request = _private_json(prompt)
        if not isinstance(request, dict) or canonical_json(request) != prompt.encode("utf-8"):
            state.tape.refuse("captured_provider_exact_wire_bytes_required")
        address = ReplayAddress.model_validate(context["vars"]["source_address"])
        if address.attempt_n == 0:
            state.tape.skip(address, prompt.encode("utf-8"))
            return {"error": "captured_request_not_sent", "cost": 0}
        reservation = state.budget.reserve(
            Decimal(0), address.endpoint_id, address.exchange_id, wire_hash(URL, request)
        )
        try:
            async with httpx.AsyncClient(
                transport=ReplayTransport(state.tape, url=URL),
                trust_env=False,
                follow_redirects=False,
            ) as client:
                result = await send_reserved(
                    client,
                    URL,
                    request,
                    {},
                    state.budget,
                    reservation,
                    address.attempt_n,
                    timeout_seconds=30,
                    request_extensions={REPLAY_ADDRESS_EXTENSION: address.model_dump(mode="json")},
                )
        finally:
            state.budget.reconcile(reservation.reservation_id, Decimal(0), "estimate")
        if state.tape.audit()["refusals"]:
            raise ValueError("captured_provider_tape_refused")
        if result.error:
            return {"error": "captured_source_" + result.error, "cost": 0}
        return {
            "output": {
                "status": result.status,
                "response": result.response,
                "response_text": result.response_text,
                "sse_events": [event.model_dump(mode="json") for event in result.sse_events or []],
                "parse_metadata": result.metadata,
                "historical_usage": (result.response or {}).get("usage"),
                "source_address": address.model_dump(mode="json"),
                "replay_adds_independent_n": False,
            },
            "cost": 0,
            "metadata": {
                "book_sha256": state.digest,
                "request_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            },
        }
    except (ValueError, TypeError, KeyError, OSError):
        state = state or _state
        if state is not None:
            state.refused = True
        return {"error": "captured_provider_refused", "cost": 0}
    finally:
        if state is not None:
            state.audit()
