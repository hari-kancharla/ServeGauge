import json

import pytest
from tests_support import exchange

from servegauge.core.models import EndpointSpec, SseEvent
from servegauge.items.agent_tasks import agent_item
from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.mock.server import stream_frames
from servegauge.profiles import load_profile
from servegauge.scoring.agents import agent_samples
from servegauge.scoring.tools import (
    dropped,
    fidelity,
    leaked,
    schema_accuracy,
    structural_violation,
    tool_samples,
)
from servegauge.suites.registry import prepare_item
from servegauge.suites.s2_tools import tool_items
from servegauge.transport.capture import capture_response
from servegauge.transport.http import assemble_stream
from servegauge.transport.sse import SseParser


def design():
    return tool_items("smoke", 11, {"items": 40, "structural_repeats": 2, "agent_tasks": 1})


def captured(items, faults=None):
    endpoint = EndpointSpec(
        id="e",
        kind="mock",
        base_url="http://127.0.0.1/v1",
        model="mock",
        claimed_profile="gpt-oss-120b",
    )
    engine = ScriptedBackend(8, faults or FaultConfig())
    records = []
    for item in items:
        for index in range(item.expected["samples"]):
            body = prepare_item(item, endpoint).body
            result = engine.respond(body)
            events, meta = None, {}
            if body["stream"]:
                parser = SseParser()
                events = []
                for frame in stream_frames(result, engine.faults):
                    events.extend(parser.feed(frame, 1))
                result, meta = assemble_stream(events)
                meta["terminal"] = parser.terminal
            records.append(
                exchange(
                    index,
                    endpoint_id="e",
                    item_id=item.item_id,
                    request=body,
                    response=result,
                    sse_events=events,
                    metadata=meta,
                    usage=result["usage"],
                )
            )
    return records


def test_clean_and_injected_tool_scoring_uses_actual_wire():
    items = [item for item in design() if not item.expected["check_id"].startswith("T07")]
    profile = load_profile("gpt-oss-120b")
    clean = tool_samples(items, captured(items), "e", profile)
    assert clean["T01"].metric == "f1" and len(clean["T01"].repeats) == 40
    assert all(len(row) == 2 for row in clean["T01"].repeats.values())
    assert all(value == 1 for check in ("T02", "T03") for value in clean[check].values.values())
    assert all(
        value == 0
        for check, sample in clean.items()
        if sample.check_class == "invariant"
        for value in sample.values.values()
    )
    assert clean["T08"].status == "NOT_APPLICABLE"
    for faults, key in [
        (FaultConfig(drop_tool_calls=1), "T06"),
        (FaultConfig(leak_markers=1), "T05"),
        (FaultConfig(stream_delta_without_index=True), "T04.parallel.stream"),
        (FaultConfig(plain_text_after_tool_result=1), "T03"),
    ]:
        sample = tool_samples(items, captured(items, faults), "e", profile)[key]
        assert sample.status is None
        assert any(sample.values.values()) if key != "T03" else min(sample.values.values()) == 0
        if sample.check_class == "invariant":
            assert sample.replay_candidates


def test_schema_fidelity_multiset_and_marker_contracts():
    item = next(
        item
        for item in design()
        if item.expected["check_id"] == "tools-corpus" and item.expected["should_call"]
    )
    record = captured([item])[0]
    assert schema_accuracy(item, record) and fidelity(item, record)
    altered = record.model_copy(deep=True)
    call = altered.response["choices"][0]["message"]["tool_calls"][0]
    call["function"]["arguments"] = '{"unexpected":true}'
    assert not schema_accuracy(item, altered) and not fidelity(item, altered)
    call["function"]["name"] = "not_offered"
    assert not schema_accuracy(item, altered)
    call["function"]["arguments"] = "{"
    assert not schema_accuracy(item, altered) and not fidelity(item, altered)
    profile = load_profile("gpt-oss-120b")
    altered.response["choices"][0]["message"] = {"content": "harmless", "reasoning": "<think>x"}
    assert leaked(altered, profile)
    name = record.request["tools"][0]["function"]["name"]
    altered.response["choices"][0]["message"] = {
        "content": json.dumps({"name": name, "arguments": {}})
    }
    assert dropped(altered)
    altered.response["choices"][0]["message"]["content"] = "ordinary text"
    altered.response["choices"][0]["finish_reason"] = "stop"
    assert not dropped(altered)
    profile.format_markers = ["PRIVATE_MARK"]
    profile.unverified.append("format_markers")
    altered.response["choices"][0]["message"]["content"] = "PRIVATE_MARK"
    assert not leaked(altered, profile)
    profile.tool_id_format = "unqualified"
    assert tool_samples([item], captured([item]), "e", profile)["T08"].status == "INCONCLUSIVE"
    probe = next(i for i in design() if i.expected["check_id"] == "T04.single.json")
    assert structural_violation(probe, altered)


@pytest.mark.parametrize(
    "failure", ["missing", "transport", "unsupported", "uncertain", "reject", "censored"]
)
def test_tool_capture_failures_cannot_become_quality_failures(failure):
    item = next(i for i in design() if i.expected["check_id"] == "tools-corpus")
    records = captured([item])
    if failure == "missing":
        records.pop()
    elif failure == "transport":
        records[0].error = "transport_timeout"
    elif failure in {"unsupported", "uncertain"}:
        records[0].metadata = {
            "request_not_sent": True,
            "verdict": "NOT_SUPPORTED" if failure == "unsupported" else "INCONCLUSIVE",
        }
    elif failure == "reject":
        records[0].status = 400
    else:
        records[0].response["choices"][0]["finish_reason"] = "length"
    scores = tool_samples([item], records, "e", load_profile("gpt-oss-120b"))
    assert all(
        scores[check].status in {"ERROR", "NOT_SUPPORTED", "INCONCLUSIVE"}
        for check in ("T01", "T02", "T03", "T05", "T06")
    )


def test_malformed_numbers_and_stream_shapes_are_retained_for_scoring():
    for invalid in ('{"n":NaN}', '{"n":Infinity}', '{"n":1e999}'):
        result = capture_response(200, {}, invalid.encode())
        assert result.body is None and result.text == invalid
    shapes = [
        {"choices": [None]},
        {"choices": [{"delta": {"content": {"x": 1}}}]},
        {"choices": [{"delta": {"tool_calls": [None]}}]},
        {"n": float("nan")},
    ]
    events = [SseEvent(t_ms=1, data=json.dumps(shape)) for shape in shapes]
    response, metadata = assemble_stream(events)
    assert metadata["invalid_json_events"] == len(shapes)
    assert response["choices"][0]["message"]["content"] is None
    assert len(events) == 4


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "transport",
        "unsupported",
        "rejected",
        "history",
        "censored",
        "empty_id",
        "unknown_tool",
        "invalid_args",
        "too_many",
        "no_call",
    ],
)
def test_agent_offline_reconstruction_never_substitutes_expected_calls(fault):
    item = agent_item(11, pass_reasoning=True, minimum=3, maximum=3, cap=2048, turn_headroom=2)
    records = captured([item])
    if fault == "missing":
        records = []
    else:
        record = records[0]
        returned = record.response["choices"][0]["message"]
        if fault == "transport":
            record.error = "transport_timeout"
        elif fault == "unsupported":
            record.metadata = {"request_not_sent": True, "verdict": "NOT_SUPPORTED"}
        elif fault == "rejected":
            record.status = 400
        elif fault == "history":
            record.request["messages"] = "invalid"
        elif fault == "censored":
            record.response["choices"][0]["finish_reason"] = "length"
        elif fault == "empty_id":
            returned["tool_calls"][0]["id"] = ""
        elif fault == "unknown_tool":
            returned["tool_calls"][0]["function"]["name"] = "shell"
        elif fault == "invalid_args":
            returned["tool_calls"][0]["function"]["arguments"] = '{"invalid":true}'
        elif fault == "too_many":
            returned["tool_calls"] *= 10
        else:
            returned["tool_calls"] = []
    sample = agent_samples([item], records, "e")[item.expected["check_id"]]
    assert sample.values[item.item_id] == 0
    trace = sample.details["agent_traces"][item.item_id]
    assert not trace["success"] and all(not value for value in trace["final_state"].values())
    if fault in {"missing", "transport", "history"}:
        assert sample.status == "ERROR"
    if fault in {"rejected", "unsupported"}:
        assert sample.status == "NOT_SUPPORTED"
    if fault == "censored":
        assert sample.status == "INCONCLUSIVE"


def test_nested_text_calls_and_unsupported_stream_pair():
    profile = load_profile("gpt-oss-120b")
    item = next(i for i in design() if i.expected["check_id"] == "T04.single.json")
    record = captured([item])[0]
    record.response["choices"][0]["message"] = {
        "content": '[{"function":{"name":"record_value","arguments":{}}}]'
    }
    record.response["choices"][0]["finish_reason"] = "stop"
    assert dropped(record)
    pair = [i for i in design() if i.expected["check_id"].startswith("T04.single.")]
    captures = captured(pair)
    for capture in captures:
        capture.metadata = {"request_not_sent": True, "verdict": "NOT_SUPPORTED"}
        capture.response = None
    assert tool_samples(pair, captures, "e", profile)["T06"].status == "NOT_SUPPORTED"
    parallel = [i for i in design() if i.expected["check_id"].startswith("T04.parallel.")]
    captures = captured(parallel)
    for item, capture in zip(parallel, captures, strict=True):
        if item.payload["stream"]:
            capture.response["choices"][0]["message"]["tool_calls"].pop()
    score = tool_samples(parallel, captures, "e", profile)["T06"]
    assert sum(score.values.values()) == 2 and len(score.replay_candidates) == 2
    endpoint = EndpointSpec(
        id="e",
        kind="mock",
        model="mock",
        base_url="http://127.0.0.1/v1",
        claimed_profile="gpt-oss-120b",
        defaults={"parallel_tool_calls": False},
    )
    decision = prepare_item(parallel[0], endpoint)
    assert decision.body is None and decision.verdict == "INCONCLUSIVE"
    assert decision.metadata["conflicting_default"] == "parallel_tool_calls"


def test_censored_stream_pair_is_not_a_tool_drop_observation():
    pair = [i for i in design() if i.expected["check_id"].startswith("T04.parallel.")]
    captures = captured(pair)
    for record in captures:
        if record.request["stream"]:
            record.response["choices"][0]["message"]["tool_calls"].pop()
            record.response["choices"][0]["finish_reason"] = "length"
    score = tool_samples(pair, captures, "e", load_profile("gpt-oss-120b"))["T06"]
    assert score.status == "INCONCLUSIVE"
    assert not any(score.values.values())
    assert not score.replay_candidates


def test_plain_json_for_an_unoffered_function_is_not_a_dropped_call():
    item = next(i for i in design() if i.expected["check_id"] == "T04.single.json")
    record = captured([item])[0]
    record.response["choices"][0]["message"] = {
        "content": '{"name":"ordinary_person","arguments":{"age":42}}'
    }
    record.response["choices"][0]["finish_reason"] = "stop"
    assert not dropped(record)


def test_requested_literal_marker_inside_a_correct_argument_is_not_parser_leakage():
    item = next(i for i in design() if i.expected["check_id"] == "T04.single.json").model_copy(
        deep=True
    )
    item.expected["calls"][0]["arguments"] = {"value": "<tool_call>"}
    item.expected["mock_task"]["arguments"] = {"value": "<tool_call>"}
    item.payload["tools"][0]["function"]["parameters"] = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    records = captured([item])
    assert fidelity(item, records[0]) and schema_accuracy(item, records[0])
    sample = tool_samples([item], records, "e", load_profile("gpt-oss-120b"))["T05"]
    assert not any(sample.values.values())
    records[0].response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
        '{"value":"<tool_call><tool_call>"}'
    )
    assert any(
        tool_samples([item], records, "e", load_profile("gpt-oss-120b"))["T05"].values.values()
    )
