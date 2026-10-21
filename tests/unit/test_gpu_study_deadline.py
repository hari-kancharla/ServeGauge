"""A known expired collection window refuses transport before any network action."""

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from servegauge.storage.gpu_capture import _DeadlineTransport


async def test_expired_window_never_calls_underlying_transport():
    calls = []

    def handler(request):
        calls.append(request)
        raise AssertionError("expired study may not send a request")

    transport = _DeadlineTransport(
        httpx.MockTransport(handler), datetime.now(UTC) - timedelta(seconds=1)
    )
    with pytest.raises(httpx.RequestError, match="study_session_deadline"):
        await transport.handle_async_request(
            httpx.Request("POST", "http://127.0.0.1/v1/chat/completions")
        )
    assert calls == []
    await transport.aclose()


async def test_reviewed_window_retains_the_actual_transport_response():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, json={"error": {"code": "actual_transport_fixture"}})

    transport = _DeadlineTransport(
        httpx.MockTransport(handler), datetime.now(UTC) + timedelta(minutes=10)
    )
    request = httpx.Request("POST", "http://127.0.0.1/v1/chat/completions")
    response = await transport.handle_async_request(request)
    assert response.status_code == 503 and len(calls) == 1
    await transport.aclose()


@pytest.mark.parametrize(
    "start,total_ms,terminated,valid",
    [
        (20, 1000, None, True),
        (9, 1000, None, False),
        (99, 2000, None, False),
        (100, 0, None, True),
        (20, -1, None, False),
        (30, 1000, 30, False),
    ],
)
def test_only_observed_local_runtime_window_can_support_a_measurement(
    start, total_ms, terminated, valid
):
    from types import SimpleNamespace

    from servegauge.storage.gpu_capture import _runtime_interval_valid

    def iso(value):
        return datetime.fromtimestamp(value, UTC).isoformat()

    session = {
        "reservation": {"created_at": iso(0)},
        "running_at": iso(10),
        "collection_deadline": iso(100),
        "terminated_at": iso(terminated) if terminated is not None else None,
    }
    records = [
        SimpleNamespace(
            attempts=[SimpleNamespace(timings=SimpleNamespace(start_unix=start, total_ms=total_ms))]
        )
    ]
    assert _runtime_interval_valid(records, session) is valid
