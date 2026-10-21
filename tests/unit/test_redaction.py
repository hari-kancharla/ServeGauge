import json
from pathlib import Path
from urllib.parse import quote

import pytest

from servegauge.core.errors import SafeError
from servegauge.core.redaction import Redactor
from servegauge.storage.raw_store import RawStore
from servegauge.transport.capture import capture_response


def test_nested_encoded_and_unknown_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "key test/" + "opaque" * 5
    monkeypatch.setenv("PRIVATE_API_KEY", secret)
    redactor = Redactor.from_environment()
    value = {
        "headers": {"Authorization": "Bearer " + secret, "Content-Type": "application/json"},
        "body": {"access_token": "unknowncredential", "messages": [secret]},
        "url": "https://user:password@example.com/v1?api_key=" + quote(secret) + "&q=safe",
        "sse": json.dumps({"credentials": {"password": "unknownpassword"}, "text": secret}),
    }
    text = json.dumps(redactor.redact(value))
    for forbidden in (
        secret,
        quote(secret),
        "unknowncredential",
        "unknownpassword",
        "user:password",
    ):
        assert forbidden not in text
    assert "application/json" in text
    assert "q=safe" in text


def test_known_key_patterns_and_url_fragments() -> None:
    secret = "sk-" + "opaquevalue" * 4
    redactor = Redactor()
    assert secret not in redactor.text("Bearer " + secret)
    assert "private" not in redactor.url("https://example.com/#private")
    assert redactor.url("not a url") == "[REDACTED_URL]"
    assert redactor.redact({"safe": (1, "ok", None), "flag": True})["safe"] == [1, "ok", None]


def test_wire_text_preserves_format_and_removes_entire_private_key() -> None:
    redactor = Redactor()
    wire = ' { "content" : "hello", "api_key" : {"nested": "private"}, "value": 1 }\n'
    assert redactor.wire_text(wire) == (
        ' { "content" : "hello", "api_key" : "[REDACTED]", "value": 1 }\n'
    )
    assert redactor.wire_text('data: {"password": "opaque"}') == (
        'data: {"password": "[REDACTED]"}'
    )
    pem = "-----BEGIN " + "PRIVATE KEY-----\nprivate\n-----END PRIVATE KEY-----"
    assert redactor.text(pem) == ("[REDACTED]")


def test_no_secrets_in_capture_store_cassette_logs_or_exceptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from tests_support import exchange

    secret = "sk-" + "opaquevalue" * 4
    monkeypatch.setenv("TEST_API_KEY", secret)
    redactor = Redactor.from_environment()
    body = json.dumps({"content": secret, "api_key": "hiddenbodykey"}).encode()
    captured = capture_response(200, {"Authorization": secret, "X-Request-Id": secret}, body)
    record = exchange(request={"messages": [{"content": secret}]}, response=captured.body)
    store = RawStore(tmp_path / "raw", redactor=redactor)
    assert store.append(record)
    saved = list(store.exchanges())[0]
    cassette = tmp_path / "cassette.json"
    cassette.write_text(saved.model_dump_json())
    error = SafeError("transport_failure")
    for text in (saved.model_dump_json(), cassette.read_text(), str(error), caplog.text):
        assert secret not in text
        assert "hiddenbodykey" not in text
    assert "Authorization" not in captured.headers
    assert captured.headers["x-request-id"] == "[REDACTED]"


def test_capture_non_json_and_sensitive_string_keys() -> None:
    redactor = Redactor(("opaque credential",))
    captured = capture_response(502, {"set-cookie": "private"}, b"opaque credential", redactor)
    assert captured.body is None
    assert captured.text == "[REDACTED]"
    assert captured.headers == {}
    assert redactor.redact({"opaque credential": "x"}) == {"[REDACTED]": "x"}


def test_split_stream_and_token_bytes_cannot_reconstruct_a_credential(tmp_path: Path) -> None:
    from tests_support import exchange

    from servegauge.core.models import SseEvent

    secret = "sk-" + "opaquepart" * 4
    fragments = [secret[start : start + 7] for start in range(0, len(secret), 7)]
    events = [
        SseEvent(
            data=json.dumps({"choices": [{"delta": {"content": fragment}}]}),
            raw_lines=["data: " + fragment],
            t_ms=0,
        )
        for fragment in fragments
    ]
    record = exchange(response={"choices": [{"message": {"content": secret}}]}, sse_events=events)
    store = RawStore(tmp_path / "raw", redactor=Redactor((secret,)))
    store.append(record)
    saved = list(store.exchanges())[0]
    assert saved.error == "sensitive_response_redacted"
    assert saved.metadata["sensitive_response_redacted"]
    assert all(event.data == "[REDACTED]" for event in saved.sse_events or [])
    assert secret not in saved.model_dump_json()
    for response in (
        {"logprobs": {"content": [{"token": v} for v in fragments]}},
        {"logprobs": {"bytes": list(secret.encode())}},
    ):
        clean = Redactor((secret,)).redact({"response": response, "metadata": {}})
        assert clean["response"]["redacted"]
