"""Finite, explicitly addressed captured observations; never a live fallback."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from hashlib import sha256
from ipaddress import ip_address
from typing import Any, Literal

import httpx
from pydantic import ConfigDict, Field, ValidationError

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.models import Attempt, Exchange, ValueModel
from servegauge.transport.sse import SseParser

REPLAY_ADDRESS_EXTENSION = "servegauge_replay_address"
Role = Literal["primary", "paired", "preflight", "confirmation", "agent"]
Lane = tuple[str, str, str, str, str]


class ReplayRefused(ValueError):
    """Static reasons only: source prompts, URLs and credentials are never echoed."""


class ReplayAddress(ValueModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, strict=True, frozen=True)
    case_id: str = Field(min_length=1)
    role: Role
    source_run_id: str = Field(min_length=1)
    endpoint_id: str = Field(min_length=1)
    suite_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    sample_idx: int = Field(ge=0)
    exchange_id: str = Field(min_length=1)
    attempt_n: int = Field(ge=0)


class ReplayObservation(ValueModel):
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    role: Role
    source_position: int = Field(ge=0)
    record_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    record: Exchange


@dataclass(frozen=True)
class ReplayResponse:
    status: int | None
    headers: dict[str, str]
    body: bytes
    error: str | None
    representation: str
    source_attempt_hash: str


def _address(row: ReplayObservation, attempt: int) -> ReplayAddress:
    record = row.record
    return ReplayAddress(
        case_id=row.case_id,
        role=row.role,
        source_run_id=record.run_id,
        endpoint_id=record.endpoint_id,
        suite_id=record.suite_id,
        item_id=record.item_id,
        sample_idx=record.sample_idx,
        exchange_id=record.exchange_id,
        attempt_n=attempt,
    )


def _lane(address: ReplayAddress) -> Lane:
    return (
        address.case_id,
        address.role,
        address.endpoint_id,
        address.source_run_id,
        address.suite_id,
    )


def _render(attempt: Attempt) -> ReplayResponse:
    if attempt.error not in {
        None,
        "transport_timeout",
        "transport_failure",
        "provider_completion_error",
    }:
        raise ReplayRefused("replay_source_error_unrepresentable")
    if attempt.status is None and (
        attempt.error not in {"transport_timeout", "transport_failure"}
        or attempt.response is not None
        or attempt.response_text is not None
        or attempt.sse_events
    ):
        raise ReplayRefused("replay_source_status_unrepresentable")
    if attempt.status is not None and (
        not 200 <= attempt.status <= 599 or 300 <= attempt.status < 400
    ):
        raise ReplayRefused("replay_source_status_unrepresentable")
    # Captures contain decoded entity bodies, not compressed socket bytes. Never
    # reapply a captured encoding to that decoded body or permit a redirect.
    headers = {
        key.lower(): value
        for key, value in attempt.response_headers.items()
        if key.lower()
        not in {"content-encoding", "content-length", "transfer-encoding", "connection"}
    }
    if "location" in headers or any("\r" in v or "\n" in v for v in headers.values()):
        raise ReplayRefused("replay_source_headers_unrepresentable")
    events = attempt.sse_events
    if events or "text/event-stream" in headers.get("content-type", ""):
        if not events or any(not event.raw_lines for event in events):
            raise ReplayRefused("replay_source_sse_lines_required")
        if attempt.metadata.get("invalid_utf8") or (
            attempt.metadata.get("incomplete") and attempt.error is None
        ):
            raise ReplayRefused("replay_source_sse_tail_unavailable")
        if any("\n" in line or "\r" in line for event in events for line in event.raw_lines):
            raise ReplayRefused("replay_source_sse_lines_invalid")
        body = b"".join(("\n".join(event.raw_lines) + "\n\n").encode() for event in events)
        parser = SseParser()
        reparsed = parser.feed(body, 0)
        parser.finish(0)
        if [event.model_dump(exclude={"t_ms"}) for event in reparsed] != [
            event.model_dump(exclude={"t_ms"}) for event in events
        ] or parser.incomplete:
            raise ReplayRefused("replay_source_sse_representation_mismatch")
        headers.setdefault("content-type", "text/event-stream")
        representation = "captured_sse_lines_lf"
    elif attempt.response_text is not None:
        body = attempt.response_text.encode()
        representation = "captured_utf8_text"
    elif attempt.response is not None:
        body = canonical_json(attempt.response)
        headers.setdefault("content-type", "application/json")
        representation = "canonical_json_projection"
    elif attempt.status is None:
        body, representation = b"", "captured_transport_error"
    else:
        raise ReplayRefused("replay_source_body_unavailable")
    return ReplayResponse(
        attempt.status,
        headers,
        body,
        attempt.error,
        representation,
        content_hash(attempt.model_dump(mode="json")),
    )


class FiniteReplay:
    """One-use attempts in captured file order within each endpoint/run/suite.

    No global concurrency, original socket framing or historical timing claim is
    possible from these captures. Reading a tape is not independent sampling.
    """

    def __init__(self, observations: Sequence[ReplayObservation]) -> None:
        if not observations:
            raise ReplayRefused("replay_observations_required")
        self._slots: dict[ReplayAddress, tuple[bytes, ReplayResponse | None, str]] = {}
        self._lanes: dict[Lane, list[ReplayAddress]] = {}
        self._positions: dict[Lane, set[int]] = {}
        self._cursors: dict[Lane, int] = {}
        self._consumed: list[dict[str, Any]] = []
        self._refusals: list[str] = []
        for row in sorted(observations, key=lambda r: r.source_position):
            row = row.model_copy(deep=True)
            record = row.record
            if (
                content_hash(record.model_dump(mode="json")) != row.record_hash
                or content_hash(record.request) != row.request_hash
            ):
                raise ReplayRefused("replay_source_hash_mismatch")
            if [a.n for a in record.attempts] != list(range(1, len(record.attempts) + 1)):
                raise ReplayRefused("replay_source_attempt_sequence_invalid")
            if not record.attempts and record.metadata.get("request_not_sent") is not True:
                raise ReplayRefused("replay_source_skip_evidence_required")
            lane = _lane(_address(row, 0))
            positions = self._positions.setdefault(lane, set())
            if row.source_position in positions:
                raise ReplayRefused("replay_duplicate_source_position")
            positions.add(row.source_position)
            attempts: list[tuple[int, ReplayResponse | None]] = [
                (a.n, _render(a)) for a in record.attempts
            ]
            if not attempts:
                attempts.append((0, None))
            for n, response in attempts:
                address = _address(row, n)
                if address in self._slots:
                    raise ReplayRefused("replay_duplicate_source_address")
                self._slots[address] = (canonical_json(record.request), response, row.record_hash)
                self._lanes.setdefault(lane, []).append(address)
            self._cursors[lane] = 0

    def refuse(self, reason: str) -> None:
        self._refusals.append(reason)
        raise ReplayRefused(reason)

    def _consume(self, address: ReplayAddress, body: bytes, *, skip: bool) -> ReplayResponse | None:
        slot = self._slots.get(address)
        if slot is None:
            self.refuse("replay_source_address_unknown")
        assert slot is not None
        lane = _lane(address)
        cursor = self._cursors[lane]
        queue = self._lanes[lane]
        if cursor >= len(queue) or address in queue[:cursor]:
            self.refuse("replay_source_exhausted")
        if queue[cursor] != address:
            self.refuse("replay_source_order_mismatch")
        if body != slot[0]:
            self.refuse("replay_wire_request_mismatch")
        if skip != (slot[1] is None):
            self.refuse("replay_source_skip_mismatch")
        self._cursors[lane] += 1
        self._consumed.append(
            {
                "address": address.model_dump(mode="json"),
                "record_hash": slot[2],
                "attempt_hash": slot[1].source_attempt_hash if slot[1] else None,
                "representation": slot[1].representation if slot[1] else "not_sent",
                "request_hash": sha256(slot[0]).hexdigest(),
                "skipped": skip,
            }
        )
        result = slot[1]
        if result is None:
            return None
        return ReplayResponse(
            result.status,
            result.headers.copy(),
            result.body,
            result.error,
            result.representation,
            result.source_attempt_hash,
        )

    def take(self, address: ReplayAddress, body: bytes) -> ReplayResponse:
        response = self._consume(address, body, skip=False)
        assert response is not None
        return response

    def skip(self, address: ReplayAddress, descriptor: bytes) -> None:
        self._consume(address, descriptor, skip=True)

    def audit(self) -> dict[str, Any]:
        # Return a copy without captured prompts/response bodies.
        return {
            "version": "finite-replay-audit-v1",
            "replay_adds_independent_n": False,
            "recorded_attempts": sum(
                response is not None for _, response, _ in self._slots.values()
            ),
            "recorded_skips": sum(response is None for _, response, _ in self._slots.values()),
            "consumed_attempts": sum(not row["skipped"] for row in self._consumed),
            "consumed_skips": sum(row["skipped"] for row in self._consumed),
            "remaining_slots": len(self._slots) - len(self._consumed),
            "observations": [dict(row, address=dict(row["address"])) for row in self._consumed],
            "refusals": list(self._refusals),
        }

    def assert_complete(self) -> None:
        if self._refusals:
            raise ReplayRefused("replay_harness_refusals_present")
        if len(self._consumed) != len(self._slots):
            raise ReplayRefused("replay_source_not_exhausted")


def _loopback_url(url: str) -> httpx.URL:
    try:
        parsed = httpx.URL(url)
        host = ip_address(parsed.host)
        if (
            parsed.scheme != "http"
            or not host.is_loopback
            or parsed.userinfo
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError
        if parsed.path != "/v1/chat/completions":
            raise ValueError
        return parsed
    except (ValueError, httpx.InvalidURL):
        raise ReplayRefused("replay_literal_loopback_target_required") from None


class _BodyStream(httpx.AsyncByteStream):
    def __init__(self, response: ReplayResponse, request: httpx.Request) -> None:
        self.response, self.request = response, request

    async def __aiter__(self) -> AsyncIterator[bytes]:
        if self.response.body:
            yield self.response.body
        if self.response.error == "transport_timeout":
            raise httpx.ReadTimeout("replay_captured_timeout", request=self.request)
        if self.response.error == "transport_failure":
            raise httpx.ReadError("replay_captured_transport_failure", request=self.request)


class ReplayTransport(httpx.AsyncBaseTransport):
    """In-process transport: no sockets, DNS, proxies, redirect or upstream code."""

    def __init__(self, tape: FiniteReplay, *, url: str) -> None:
        self.tape, self.url = tape, _loopback_url(url)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        try:
            if request.method != "POST" or request.url != self.url:
                self.tape.refuse("replay_transport_target_mismatch")
            try:
                address = ReplayAddress.model_validate(
                    request.extensions.get(REPLAY_ADDRESS_EXTENSION)
                )
            except ValidationError:
                self.tape.refuse("replay_transport_address_required")
            response = self.tape.take(address, await request.aread())
            if response.status is None:
                if response.error == "transport_timeout":
                    raise httpx.ConnectTimeout("replay_captured_timeout", request=request)
                raise httpx.ConnectError("replay_captured_transport_failure", request=request)
            return httpx.Response(
                response.status,
                headers=response.headers,
                stream=_BodyStream(response, request),
                request=request,
            )
        except ReplayRefused as error:
            raise httpx.RequestError(str(error), request=request) from None
