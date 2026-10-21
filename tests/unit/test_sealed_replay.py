"""Finite replay must retain source identity and refuse rather than pool or cycle."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.models import Attempt, Exchange, Timings
from servegauge.mock.sealed_replay import (
    REPLAY_ADDRESS_EXTENSION,
    FiniteReplay,
    ReplayAddress,
    ReplayObservation,
    ReplayRefused,
    ReplayTransport,
)
from servegauge.runner.budget import BudgetManager, BudgetPolicy
from servegauge.transport.http import assemble_stream, send_reserved, wire_hash
from servegauge.transport.sse import SseParser

URL = "http://127.0.0.1:8134/v1/chat/completions"
BODY = {"model": "source-model", "stream": False, "messages": [{"role": "user", "content": "x"}]}


def observation(
    case: str = "case-a", endpoint: str = "a", *, position: int = 0, **changes
) -> ReplayObservation:
    attempt = Attempt(
        n=1,
        status=200,
        timings=Timings(start_unix=0, total_ms=0),
        response={"choices": [{"message": {"content": endpoint}}]},
        response_headers={"content-type": "application/json", "x-request-id": endpoint},
    )
    record = Exchange(
        exchange_id="exchange-" + endpoint + "-" + str(position),
        run_id="source",
        endpoint_id=endpoint,
        suite_id="S1",
        item_id="P01-" + str(position),
        sample_idx=0,
        request=BODY.copy(),
        attempts=[attempt],
    ).model_copy(update=changes, deep=True)
    return ReplayObservation(
        case_id=case,
        role="primary",
        source_position=position,
        record_hash=content_hash(record.model_dump(mode="json")),
        request_hash=content_hash(record.request),
        record=record,
    )


def address(row: ReplayObservation, attempt: int = 1) -> ReplayAddress:
    return ReplayAddress(
        case_id=row.case_id,
        role=row.role,
        source_run_id=row.record.run_id,
        endpoint_id=row.record.endpoint_id,
        suite_id=row.record.suite_id,
        item_id=row.record.item_id,
        sample_idx=row.record.sample_idx,
        exchange_id=row.record.exchange_id,
        attempt_n=attempt,
    )


def test_same_body_scopes_never_pool_and_never_cycle():
    a, b = observation(), observation("case-b", "b")
    tape = FiniteReplay([a, b])
    assert tape.take(address(b), canonical_json(BODY)).body.endswith(b'"b"}}]}')
    assert tape.take(address(a), canonical_json(BODY)).body.endswith(b'"a"}}]}')
    assert tape.audit()["consumed_attempts"] == 2
    tape.assert_complete()
    with pytest.raises(ReplayRefused, match="exhausted"):
        tape.take(address(a), canonical_json(BODY))
    with pytest.raises(ReplayRefused, match="refusals"):
        tape.assert_complete()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("case_id", "other"),
        ("role", "paired"),
        ("source_run_id", "other"),
        ("endpoint_id", "b"),
        ("suite_id", "S2"),
        ("item_id", "other"),
        ("sample_idx", 1),
        ("exchange_id", "other"),
        ("attempt_n", 2),
    ],
)
def test_full_source_address_required(field, value):
    row = observation()
    tape = FiniteReplay([row])
    wrong = address(row).model_copy(update={field: value})
    with pytest.raises(ReplayRefused):
        tape.take(wrong, canonical_json(BODY))
    assert tape.audit()["consumed_attempts"] == 0
    tape.take(address(row), canonical_json(BODY))


@pytest.mark.parametrize(
    "body",
    [
        {**BODY, "model": "another-model"},
        {**BODY, "stream": True},
        {**BODY, "provider": {"order": ["other"]}},
    ],
)
def test_all_wire_fields_are_significant(body):
    row = observation()
    tape = FiniteReplay([row])
    with pytest.raises(ReplayRefused, match="request"):
        tape.take(address(row), canonical_json(body))
    assert tape.audit()["consumed_attempts"] == 0


def test_exact_wire_bytes_required_without_duplicate_key_normalization():
    row = observation()
    tape = FiniteReplay([row])
    for value in [b'{"model":"other","model":"source-model"}', b" " + canonical_json(BODY)]:
        with pytest.raises(ReplayRefused, match="request"):
            tape.take(address(row), value)
    tape.take(address(row), canonical_json(BODY))


def test_lane_order_and_contiguous_attempts_are_finite():
    a, b = observation(position=2), observation(position=5)
    first = a.record.attempts[0].model_copy(update={"status": 503})
    second = first.model_copy(update={"n": 2, "status": 200})
    a = observation(position=2, attempts=[first, second])
    tape = FiniteReplay([b, a])
    with pytest.raises(ReplayRefused, match="order"):
        tape.take(address(b), canonical_json(BODY))
    with pytest.raises(ReplayRefused, match="order"):
        tape.take(address(a, 2), canonical_json(BODY))
    assert tape.take(address(a), canonical_json(BODY)).status == 503
    assert tape.take(address(a, 2), canonical_json(BODY)).status == 200
    tape.take(address(b), canonical_json(BODY))
    assert tape.audit()["recorded_attempts"] == 3


def test_skipped_source_is_not_an_empty_successful_response():
    row = observation(attempts=[], metadata={"request_not_sent": True})
    tape = FiniteReplay([row])
    with pytest.raises(ReplayRefused):
        tape.take(address(row, 0), canonical_json(BODY))
    tape.skip(address(row, 0), canonical_json(BODY))
    assert tape.audit()["consumed_attempts"] == 0
    assert tape.audit()["consumed_skips"] == 1
    with pytest.raises(ReplayRefused):
        tape.skip(address(row, 0), canonical_json(BODY))


def test_copies_inputs_and_audit_does_not_expose_prompts_or_responses():
    row = observation()
    tape = FiniteReplay([row])
    row.record.request["model"] = "mutated"
    result = tape.take(address(row), canonical_json(BODY))
    assert b'"a"' in result.body
    audit = tape.audit()
    assert audit["replay_adds_independent_n"] is False
    assert "messages" not in str(audit) and "choices" not in str(audit)
    audit["consumed_attempts"] = 99
    assert tape.audit()["consumed_attempts"] == 1


@pytest.mark.parametrize(
    "mutation", ["record_hash", "request_hash", "duplicate", "attempt_gap", "unsent"]
)
def test_invalid_tape_refuses_before_any_dispatch(mutation):
    row = observation()
    rows = [row]
    if mutation in {"record_hash", "request_hash"}:
        row = row.model_copy(update={mutation: "0" * 64})
        rows = [row]
    elif mutation == "duplicate":
        rows = [row, row]
    else:
        if mutation == "attempt_gap":
            row.record.attempts[0].n = 2
        else:
            row.record.attempts = []
        row.record_hash = content_hash(row.record.model_dump(mode="json"))
    with pytest.raises(ReplayRefused):
        FiniteReplay(rows)


def test_preserves_captured_json_text_and_discloses_projection():
    row = observation()
    row.record.attempts[0].response_text = '{ "choices": [ {"message":{"content":"a"}} ] }'
    row.record_hash = content_hash(row.record.model_dump(mode="json"))
    result = FiniteReplay([row]).take(address(row), canonical_json(BODY))
    assert result.body == row.record.attempts[0].response_text.encode()
    assert result.representation == "captured_utf8_text"
    projected = FiniteReplay([observation()]).take(address(observation()), canonical_json(BODY))
    assert projected.representation == "canonical_json_projection"


def test_sse_raw_lines_comments_bytes_and_tools_round_trip():
    frames = (
        ": keep alive\n\nid: item\nevent: chunk\nretry: 5\n"
        'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c",'
        '"function":{"name":"convert","arguments":"{}"}}]},'
        '"logprobs":{"content":[{"token":"é","bytes":[195,169],"logprob":-0.1}]}}]}\n\n'
        "data: [DONE]\n\n"
    ).encode()
    parser = SseParser()
    events = parser.feed(frames, 7)
    row = observation()
    row.record.request["stream"] = True
    attempt = row.record.attempts[0]
    attempt.sse_events = events
    attempt.response = assemble_stream(events)[0]
    attempt.response_headers = {"content-type": "text/event-stream", "content-encoding": "gzip"}
    row.record_hash = content_hash(row.record.model_dump(mode="json"))
    row.request_hash = content_hash(row.record.request)
    result = FiniteReplay([row]).take(address(row), canonical_json(row.record.request))
    assert result.body == frames
    assert result.representation == "captured_sse_lines_lf"
    assert "content-encoding" not in result.headers
    assert b'"bytes":[195,169]' in result.body
    assert b'"tool_calls"' in result.body
    assert result.source_attempt_hash == content_hash(attempt.model_dump(mode="json"))


@pytest.mark.parametrize("problem", ["missing_lines", "changed_data", "redirect", "unknown_error"])
def test_unrepresentable_source_fails_closed(problem):
    row = observation()
    attempt = row.record.attempts[0]
    if problem in {"missing_lines", "changed_data"}:
        events = SseParser().feed(b'data: {"choices":[]}\n\n', 1)
        if problem == "missing_lines":
            events[0].raw_lines = []
        else:
            events[0].data = '{"choices":[{}]}'
        attempt.sse_events = events
        attempt.response_headers = {"content-type": "text/event-stream"}
    elif problem == "redirect":
        attempt.status = 302
    else:
        attempt.status, attempt.error = None, "unclassified_error"
    row.record_hash = content_hash(row.record.model_dump(mode="json"))
    with pytest.raises(ReplayRefused):
        FiniteReplay([row])


@pytest.mark.parametrize(
    "target",
    [
        "https://127.0.0.1:8134/v1/chat/completions",
        "http://localhost:8134/v1/chat/completions",
        "http://example.com/v1/chat/completions",
        URL + "?key=x",
        URL + "#fragment",
        "http://user:pass@127.0.0.1:8134/v1/chat/completions",
    ],
)
def test_transport_requires_exact_literal_loopback_target(target):
    with pytest.raises(ReplayRefused):
        ReplayTransport(FiniteReplay([observation()]), url=target)


async def test_transport_rejects_missing_context_and_different_url_without_consuming():
    row = observation()
    tape = FiniteReplay([row])
    async with httpx.AsyncClient(
        transport=ReplayTransport(tape, url=URL), trust_env=False
    ) as client:
        for url, extensions in [
            (URL, {}),
            (URL + "/other", {REPLAY_ADDRESS_EXTENSION: address(row).model_dump()}),
        ]:
            with pytest.raises(httpx.RequestError):
                await client.post(url, content=canonical_json(BODY), extensions=extensions)
        assert tape.audit()["consumed_attempts"] == 0


@pytest.mark.parametrize("error", [None, "transport_timeout", "transport_failure"])
async def test_ordinary_reserved_transport_keeps_exact_outcome(tmp_path: Path, error):
    row = observation()
    if error:
        row.record.attempts[0].status = None
        row.record.attempts[0].response = None
        row.record.attempts[0].error = error
        row.record_hash = content_hash(row.record.model_dump(mode="json"))
    tape = FiniteReplay([row])
    budget = BudgetManager(
        tmp_path,
        "synthetic-replay",
        BudgetPolicy(
            monthly_usd=0,
            per_run_usd=0,
            dev_live_usd=0,
            selfhost_usd_per_session=0,
            stop_at_fraction=1,
        ),
        environment={},
    )
    reservation = budget.reserve(Decimal(0), "a", "local-exchange", wire_hash(URL, BODY))

    async def context(request: httpx.Request) -> None:
        request.extensions[REPLAY_ADDRESS_EXTENSION] = address(row).model_dump()

    async with httpx.AsyncClient(
        transport=ReplayTransport(tape, url=URL),
        trust_env=False,
        event_hooks={"request": [context]},
    ) as client:
        result = await send_reserved(
            client, URL, BODY, {}, budget, reservation, 1, timeout_seconds=1
        )
    assert result.error == error
    assert result.status == (None if error else 200)
    if not error:
        assert result.response == row.record.attempts[0].response
    budget.reconcile(reservation.reservation_id, Decimal(0), "estimate")
    assert budget.snapshot()["reserved"] == 0
    assert budget.snapshot()["development_spent"] == 0
    tape.assert_complete()
