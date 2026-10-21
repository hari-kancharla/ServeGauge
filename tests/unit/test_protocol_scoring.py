import copy

import pytest
from tests_support import exchange

from servegauge.core.models import EndpointSpec
from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.mock.server import stream_frames
from servegauge.profiles import load_profile
from servegauge.scoring.common import (
    calls,
    choice,
    content,
    invariant_violation,
    logprob_sanity,
    message,
    protocol_samples,
    score_capabilities,
    stream_violation,
)
from servegauge.suites.registry import item_request
from servegauge.suites.s0_capabilities import capability_items
from servegauge.suites.s1_protocol import protocol_items
from servegauge.transport.http import assemble_stream
from servegauge.transport.sse import SseParser
from servegauge.verdicts.config import load_policy


def endpoint():
    return EndpointSpec(
        id="e",
        kind="mock",
        base_url="http://127.0.0.1/v1",
        model="mock",
        claimed_profile="gpt-oss-120b",
        metadata={"context_length": 512},
    )


def capture(item, engine, *, endpoint_id="e"):
    request = item_request(item, endpoint())
    body = {"model": "mock", "messages": request.messages, **request.params}
    for key in ["tools", "tool_choice", "response_format"]:
        if getattr(request, key) is not None:
            body[key] = getattr(request, key)
    result = engine.respond(body)
    status = 400 if "error" in result else 200
    meta, events = {}, None
    if request.stream and status == 200:
        parser = SseParser()
        events = []
        for frame in stream_frames(result, engine.faults):
            events.extend(parser.feed(frame, 1))
        events.extend(parser.finish(2))
        result, meta = assemble_stream(events)
        meta.update(terminal=parser.terminal, incomplete=parser.incomplete)
    return exchange(
        endpoint_id=endpoint_id,
        item_id=item.item_id,
        request=body,
        response=result,
        status=status,
        usage=result.get("usage"),
        sse_events=events,
        metadata=meta,
    )


def small_items():
    return protocol_items(
        "smoke",
        10,
        endpoint=endpoint(),
        profile=load_profile("gpt-oss-120b"),
        overrides={
            "invariant_trials": 2,
            "diversity_prompts": 2,
            "diversity_repeats": 2,
            "effort_items": 2,
            "seed_trials": 2,
        },
    )


def test_all_clean_protocol_wire_scoring_and_capabilities() -> None:
    items = small_items()
    engine = ScriptedBackend(1, FaultConfig(), context_length=512)
    records = []
    for item in items:
        for sample in range(item.expected["samples"]):
            record = capture(item, engine)
            record.sample_idx = sample
            records.append(record)
    results = protocol_samples(items, records, "e", load_profile("gpt-oss-120b"), load_policy())
    for result in results.values():
        if result.check_class == "invariant" and result.values:
            assert not any(result.values.values()), result
    assert results["P07"].values and min(results["P07"].values.values()) > 1
    assert results["P05"].values == {"0": 1.0, "1": 1.0}
    assert results["P06.calibration"].values
    assert results["P14"].status == "NOT_APPLICABLE"
    caps = capability_items()
    cap_records = [capture(item, engine) for item in caps]
    observations, flags = score_capabilities(caps, cap_records, endpoint(), "r")
    assert all(flags.values()) and all(result.n == 1 for result in observations)
    assert observations[0].verdict == "INFO"


@pytest.mark.parametrize(
    "fault,check",
    [
        ({"ignore_max_tokens": True}, "P01"),
        ({"ignore_stop": True}, "P02"),
        ({"stream_delta_without_index": True}, "P11"),
    ],
)
def test_faults_score_from_response_not_configuration(fault, check) -> None:
    items = [i for i in small_items() if i.expected["check_id"] == check]
    engine = ScriptedBackend(1, FaultConfig(**fault), context_length=512)
    result = protocol_samples(
        items, [capture(i, engine) for i in items], "e", load_profile("gpt-oss-120b"), load_policy()
    )[check]
    assert all(result.values.values()) and result.replay_candidates


def test_capability_absence_errors_missing_and_malformed_captures() -> None:
    items = capability_items()
    engine = ScriptedBackend(1, FaultConfig())
    records = [capture(i, engine) for i in items]
    records[0].error = "transport_failure"
    records[1].metadata.update(request_not_sent=True, verdict="NOT_SUPPORTED")
    records[2].response = {}
    records[3].status = 400
    results, flags = score_capabilities(items, records[:-1], endpoint(), "r")
    assert results[0].verdict == "ERROR" and results[1].verdict == "NOT_SUPPORTED"
    assert results[2].verdict == "ERROR" and flags["parallel_tool_calls"] is False
    assert results[-1].verdict == "ERROR"
    probes = [i for i in small_items() if i.expected["check_id"] == "P01"]
    record = capture(probes[0], engine)
    for method in [choice, message, calls, content]:
        malformed = copy.deepcopy(record)
        malformed.response = {"choices": [{"message": {"content": 4, "tool_calls": 2}}]}
        if method in [choice, message]:
            malformed.response = {"choices": []} if method == choice else {"choices": [{}]}
        with pytest.raises(ValueError):
            method(malformed)
    samples = protocol_samples(probes, [], "e", load_profile("gpt-oss-120b"), load_policy())
    assert samples["P01"].status == "ERROR"
    record.error = "transport_failure"
    assert (
        protocol_samples(probes[:1], [record], "e", load_profile("gpt-oss-120b"), load_policy())[
            "P01"
        ].status
        == "ERROR"
    )
    record.error = None
    record.metadata.update(request_not_sent=True, verdict="INCONCLUSIVE")
    assert (
        protocol_samples(probes[:1], [record], "e", load_profile("gpt-oss-120b"), load_policy())[
            "P01"
        ].status
        == "INCONCLUSIVE"
    )
    record.metadata["verdict"] = "NOT_SUPPORTED"
    assert (
        protocol_samples(probes[:1], [record], "e", load_profile("gpt-oss-120b"), load_policy())[
            "P01"
        ].status
        == "NOT_SUPPORTED"
    )


def test_logprobs_numeric_bounds_sort_bytes_and_stream_flags() -> None:
    item = next(i for i in small_items() if i.expected["check_id"] == "P06.structure")
    record = capture(item, ScriptedBackend(1, FaultConfig()))
    assert logprob_sanity(record, 0.001) == (False, -0.02)
    for updates in [
        {"logprob": 1},
        {"logprob": float("nan")},
        {"top_logprobs": [{"logprob": 1}]},
        {"top_logprobs": [{"logprob": -2}, {"logprob": -1}]},
        {"top_logprobs": [{"logprob": -0.01}] * 2},
        {"bytes": [255]},
        {"bytes": [999]},
    ]:
        changed = copy.deepcopy(record)
        changed.response["choices"][0]["logprobs"]["content"][0].update(updates)
        assert logprob_sanity(changed, 0.001)[0]
    changed = copy.deepcopy(record)
    changed.response["choices"][0].pop("logprobs")
    assert logprob_sanity(changed, 0.001) == (True, None)
    assert stream_violation(record)
    with pytest.raises(ValueError):
        invariant_violation(
            item.model_copy(update={"expected": {"check_id": "unknown"}}), record, load_policy()
        )


def test_censored_tool_choice_cannot_be_called_a_parameter_violation() -> None:
    item = next(
        i
        for i in small_items()
        if i.expected["check_id"] == "P09" and i.expected["mode"] == "required"
    )
    record = capture(item, ScriptedBackend(1, FaultConfig()))
    record.response["choices"][0]["finish_reason"] = "length"
    result = protocol_samples([item], [record], "e", load_profile("gpt-oss-120b"), load_policy())[
        "P09.required"
    ]
    assert result.status == "INCONCLUSIVE" and not result.values


@pytest.mark.parametrize("envelope", ["finish_error", "root_error"])
def test_http_success_with_provider_error_is_not_behavioral_evidence(envelope) -> None:
    engine = ScriptedBackend(1, FaultConfig())
    cap = next(i for i in capability_items() if i.expected.get("capability") == "tools")
    record = capture(cap, engine)
    if envelope == "finish_error":
        record.response["choices"][0].update(
            finish_reason="error", message={"role": "assistant", "content": None}
        )
    else:
        record.response["error"] = {"message": "upstream completion failed"}
    observations, flags = score_capabilities([cap], [record], endpoint(), "r")
    assert observations[0].verdict == "ERROR" and flags["tools"] is None
    assert observations[0].reason == "provider reported completion error"
    assert observations[0].estimate is None and observations[0].ci_low is None
    with pytest.raises(ValueError, match="provider_completion_error"):
        content(record)
    item = next(
        i
        for i in small_items()
        if i.expected["check_id"] == "P09" and i.expected["mode"] == "required"
    )
    record.item_id = item.item_id
    samples = protocol_samples([item], [record], "e", load_profile("gpt-oss-120b"), load_policy())
    assert samples["P09.required"].status == "ERROR"
    assert not samples["P09.required"].values


def test_protocol_trials_do_not_relabel_repeated_continuous_prompts() -> None:
    from servegauge.core.hashing import content_hash

    items = protocol_items("core", 20261018)
    for check in ["P01", "P06.structure"]:
        probes = [item for item in items if item.expected["check_id"] == check]
        assert len({content_hash(item.payload) for item in probes}) == len(probes) == 400
