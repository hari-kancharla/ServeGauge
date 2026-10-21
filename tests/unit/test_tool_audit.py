import json

from tests_support import exchange

from servegauge.core.models import SseEvent
from servegauge.diagnose.tool_audit import audit_tools, raw_logprob_bytes, wire_tool_calls


def test_independent_sse_audit_finds_missing_normalized_call():
    chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "c", "function": {"name": "f", "arguments": "{"}}
                        ]
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]},
                }
            ]
        },
    ]
    record = exchange(
        0,
        suite_id="S2",
        response={"choices": [{"finish_reason": "stop", "message": {"content": None}}]},
        sse_events=[SseEvent(t_ms=0, data=json.dumps(c)) for c in chunks],
    )
    assert wire_tool_calls(record)[0]["function"] == {"name": "f", "arguments": "{}"}
    audit = audit_tools([record])["endpoints"][record.endpoint_id]
    assert audit["counts"]["stream_assembly_mismatch"] == 1
    assert not audit["observations"][0]["serving_parser_cause_verified"]
    record.sse_events[0].data = json.dumps(
        {"choices": [{"delta": {"tool_calls": [{"function": {"name": "f"}}]}}]}
    )
    assert wire_tool_calls(record) is None


def test_call_shaped_text_does_not_prove_raw_parser_loss():
    record = exchange(
        0,
        suite_id="S2",
        response={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": '{"name":"f","arguments":{}}'},
                    "logprobs": {"content": [{"bytes": [79, 75]}]},
                }
            ]
        },
    )
    record.request["tools"] = [{"type": "function", "function": {"name": "f"}}]
    assert raw_logprob_bytes(record) == b"OK"
    observation = audit_tools([record])["endpoints"][record.endpoint_id]["observations"][0]
    assert observation["signature"] == "call_shaped_visible_JSON"
    assert not observation["serving_parser_cause_verified"]
    record.response["choices"][0]["logprobs"]["content"][0]["bytes"] = [True]
    assert raw_logprob_bytes(record) is None


def test_audit_excludes_completion_errors_and_does_not_mutate_captures():
    record = exchange(
        0,
        suite_id="S2",
        response={"choices": [{"finish_reason": "error", "message": {"content": None}}]},
    )
    before = record.model_dump()
    assert audit_tools([record])["endpoints"][record.endpoint_id]["counts"]["invalid_envelope"] == 1
    assert record.model_dump() == before
