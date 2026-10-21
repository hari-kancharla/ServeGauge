"""Independent assertions derive evidence from captured channels, never labels."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from servegauge.comparison.independent import Observation, assess_unit, get_assert, summarize


def byte_capture(text: str = "é") -> dict:
    return {
        "request": {"logprobs": True, "stream": False},
        "status": 200,
        "response": {
            "choices": [
                {
                    "message": {"content": text},
                    "logprobs": {
                        "content": [
                            {
                                "token": text,
                                "bytes": list(text.encode()),
                                "logprob": -0.1,
                                "top_logprobs": [{"token": text, "logprob": -0.1}],
                            }
                        ]
                    },
                }
            ]
        },
    }


def tool_capture(*, calls: list | None = None, content: str = "", finish: str = "stop") -> dict:
    return {
        "request": {"tools": [{"type": "function", "function": {"name": "convert"}}]},
        "status": 200,
        "response": {
            "choices": [
                {"message": {"content": content, "tool_calls": calls}, "finish_reason": finish}
            ]
        },
    }


def judge(check: str, rows: list[dict], *, paired: bool = False):
    return assess_unit(check, rows, probability_tolerance=0.001, paired_structure=paired)


def test_p06_valid_utf8_and_multi_token_byte_fragments():
    row = byte_capture()
    assert judge("P06.structure", [row]).violated is False
    logs = row["response"]["choices"][0]["logprobs"]["content"]
    logs[0]["bytes"] = [195]
    logs.append({**logs[0], "bytes": [169]})
    assert judge("P06.structure", [row]).violated is False


@pytest.mark.parametrize("problem", ["bytes", "positive", "not_finite", "sort", "sum", "missing"])
def test_p06_observed_protocol_faults_are_violations(problem: str):
    row = byte_capture()
    choice = row["response"]["choices"][0]
    token = choice["logprobs"]["content"][0]
    if problem == "bytes":
        token["bytes"] = [120]
    elif problem == "positive":
        token["logprob"] = 0.2
    elif problem == "not_finite":
        token["logprob"] = float("nan")
    elif problem == "sort":
        token["top_logprobs"] = [{"logprob": -2}, {"logprob": -1}]
    elif problem == "sum":
        token["top_logprobs"] = [{"logprob": -0.1}, {"logprob": -0.1}]
    else:
        choice.pop("logprobs")
    assert judge("P06.structure", [row]).violated is True


@pytest.mark.parametrize(
    "problem",
    ["transport", "status", "not_sent", "missing_body", "missing_stream", "truncated_stream"],
)
def test_missing_capture_channels_never_become_clean_observations(problem: str):
    row = byte_capture()
    if problem == "transport":
        row["error"] = "transport_timeout"
    elif problem == "status":
        row["status"] = 503
    elif problem == "not_sent":
        row["request_not_sent"] = True
    elif problem == "missing_body":
        row.pop("response")
    elif problem == "missing_stream":
        row["request"]["stream"] = True
    else:
        row["request"]["stream"] = True
        row["sse_events"] = [{"raw_lines": ['data: {"choices":[]}'], "data": '{"choices":[]}'}]
    value = judge("P06.structure", [row])
    assert value.available is False and value.violated is None


def test_stream_assertion_reconstructs_raw_frames_without_trusting_parsed_projection():
    row = byte_capture()
    row["request"]["stream"] = True
    row["response"]["choices"][0]["message"]["content"] = "false parsed projection"
    frame = {
        "choices": [
            {
                "index": 0,
                "delta": {"content": "é"},
                "logprobs": {
                    "content": [
                        {
                            "token": "é",
                            "bytes": [195, 169],
                            "logprob": -0.1,
                            "top_logprobs": [{"logprob": -0.1}],
                        }
                    ]
                },
            }
        ]
    }
    text = json.dumps(frame, ensure_ascii=False, separators=(",", ":"))
    row["sse_events"] = [
        {"raw_lines": [": comment"], "is_comment": True, "data": "comment"},
        {"raw_lines": ["data: " + text], "data": text},
        {"raw_lines": ["data: [DONE]"], "data": "[DONE]"},
    ]
    assert judge("P06.structure", [row]).violated is False
    row["sse_events"][1]["data"] = "changed parsed event"
    assert judge("P06.structure", [row]).available is False


@pytest.mark.parametrize(
    "content",
    [
        '{"name":"convert","arguments":{}}',
        '{"function":{"name":"convert","arguments":"{}"}}',
        '[{"name":"convert","arguments":{}}]',
        '{"tool_calls":[{"function":{"name":"convert","arguments":{}}}]}',
    ],
)
def test_t06_offered_parseable_call_in_text_without_structured_field(content: str):
    assert judge("T06", [tool_capture(content=content)]).violated is True


@pytest.mark.parametrize(
    "content",
    [
        '{"name":"unoffered","arguments":{}}',
        '{"name":"convert","arguments":"not-json"}',
        '{"name":"convert","arguments":null}',
        '{"name":"convert","name":"other","arguments":{}}',
        '{"name":{},"arguments":{}}',
        "ordinary prose",
    ],
)
def test_t06_arbitrary_json_or_prose_is_not_a_drop(content: str):
    assert judge("T06", [tool_capture(content=content)]).violated is False


def test_t06_finish_marker_and_structural_pair_are_distinct_observations():
    assert judge("T06", [tool_capture(finish="tool_calls")]).violated is True
    first = tool_capture(calls=[{"function": {"name": "convert", "arguments": "{}"}}])
    second = tool_capture()
    assert judge("T06", [first, second], paired=True).violated is True
    assert judge("T06", [first, second], paired=False).violated is False


def test_censoring_and_invalid_envelope_abstain():
    for row in [
        tool_capture(finish="length"),
        {"status": 200, "request": {}, "response": {"choices": []}},
    ]:
        assert judge("T06", [row]).available is False


def test_p06_byte_contract_still_applies_to_the_visible_capped_prefix():
    row = byte_capture()
    row["response"]["choices"][0]["finish_reason"] = "length"
    assert judge("P06.structure", [row]).violated is False
    row["response"]["choices"][0]["logprobs"]["content"][0]["bytes"] = [120]
    assert judge("P06.structure", [row]).violated is True


def test_backend_fault_labels_and_reported_verdicts_do_not_drive_assertions():
    row = byte_capture()
    row["metadata"] = {"fault_id": "F-FAKE", "verdict": "FAIL", "violations": 9000}
    assert judge("P06.structure", [row]).violated is False
    row = copy.deepcopy(row)
    row["response"]["choices"][0]["logprobs"]["content"][0]["bytes"] = [120]
    row["metadata"] = {"fault_id": None, "verdict": "PASS", "violations": 0}
    assert judge("P06.structure", [row]).violated is True


def test_independent_population_summary_preserves_power_and_reproduction_rules():
    clean, bad = (
        judge("P06.structure", [byte_capture()]),
        judge("T06", [tool_capture(finish="tool_calls")]),
    )
    arguments = dict(alpha=0.05, tolerance=0.01, k_fail=2, reference_tier=None)
    small = summarize({str(i): clean for i in range(10)}, [], **arguments)
    assert small["verdict"] == "INCONCLUSIVE" and small["n"] == 10
    assert small["ci_high"] == pytest.approx(0.2775327998628892)
    adequate = summarize({str(i): clean for i in range(400)}, [], **arguments)
    assert adequate["verdict"] == "PASS" and adequate["ci_high"] == pytest.approx(
        0.009512294334296512
    )
    primary = {str(i): bad for i in range(10)}
    assert summarize(primary, [], **arguments)["verdict"] == "INCONCLUSIVE"
    actual = summarize(primary, [[bad]], **arguments)
    assert actual["verdict"] == "FAIL" and actual["violations"] == 10
    assert actual["ci_low"] == pytest.approx(0.7224672001371106)
    assert actual["ci_high"] == 1.0
    full = summarize({str(i): bad for i in range(400)}, [[bad]], **arguments)
    assert full["ci_high"] == 1.0
    assert adequate["ci_low"] == 0.0


def test_unavailable_units_and_confirmation_batches_are_not_zero_or_fake_reproductions():
    clean = judge("P06.structure", [byte_capture()])
    bad = judge("T06", [tool_capture(finish="tool_calls")])
    unknown = judge("T06", [{"status": 503}])
    arguments = dict(alpha=0.05, tolerance=0.01, k_fail=2, reference_tier=None)
    incomplete = summarize({"a": clean, "b": unknown}, [], **arguments)
    assert incomplete["verdict"] == "ERROR" and incomplete["violations"] is None
    assert incomplete["ci_low"] is incomplete["ci_high"] is None
    assert (
        summarize({"a": bad, "b": bad}, [[bad, unknown]], **arguments)["verdict"] == "INCONCLUSIVE"
    )
    assert summarize({"a": bad, "b": bad}, [[unknown], [bad]], **arguments)["verdict"] == "FAIL"


def test_independent_module_cannot_import_a_detector_or_transport():
    import servegauge.comparison.independent as module

    tree = ast.parse(Path(module.__file__).read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert all(
        name.split(".")[0] in {"__future__", "dataclasses", "json", "math", "statistics", "typing"}
        for name in imports
    )


def test_inconsistent_observation_cannot_be_counted_as_a_clean_trial():
    for available, violated in [(True, None), (False, False), (True, 1)]:
        with pytest.raises(ValueError):
            Observation(available, violated)


def assertion_input(n=10, *, bad=False):
    keys = [str(i) for i in range(n)]
    context = {
        "vars": {
            "independent_policy": {
                "check_id": "P06.structure",
                "unit_keys": keys,
                "probability_tolerance": 0.001,
                "alpha": 0.05,
                "tolerance": 0.01,
                "k_fail": 2,
                "reference_tier": None,
            }
        }
    }
    capture = byte_capture()
    if bad:
        capture["response"]["choices"][0]["logprobs"]["content"][0]["bytes"] = [120]
    unit = {"paired_structure": False, "captures": [capture]}
    value = {
        "primary": {key: copy.deepcopy(unit) for key in keys},
        "confirmations": [[copy.deepcopy(unit)]] if bad else [],
        "replay_adds_independent_n": False,
    }
    return value, context


def test_actual_assertion_contract_retains_inconclusive_fail_and_pass_summary():
    for n, bad, verdict in [(10, False, "INCONCLUSIVE"), (10, True, "FAIL"), (400, False, "PASS")]:
        value, context = assertion_input(n, bad=bad)
        result = get_assert(json.dumps(value), context)
        summary = json.loads(result["reason"])
        assert summary["verdict"] == verdict and summary["n"] == n
        assert result["pass"] is (verdict == "PASS")
        assert summary["replay_adds_independent_n"] is False


@pytest.mark.parametrize(
    "mutation", ["omitted_unit", "extra_unit", "missing_capture", "inflated_n", "missing_policy"]
)
def test_assertion_refuses_scope_or_missing_channel_instead_of_supplying_a_clean_result(mutation):
    value, context = assertion_input()
    if mutation == "omitted_unit":
        value["primary"].pop("0")
    elif mutation == "extra_unit":
        value["primary"]["extra"] = value["primary"]["0"]
    elif mutation == "missing_capture":
        value["primary"]["0"]["captures"] = []
    elif mutation == "inflated_n":
        value["replay_adds_independent_n"] = True
    else:
        context["vars"].pop("independent_policy")
    result = get_assert(value, context)
    assert result["pass"] is False
    assert json.loads(result["reason"])["verdict"] == "ERROR"
