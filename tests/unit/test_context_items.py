import json

from servegauge.core.models import EndpointSpec
from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.suites.registry import prepare_item
from servegauge.suites.s4_longctx import context_items
from servegauge.tokencount.counters import text_tokens


def endpoint(context=131072):
    return EndpointSpec(
        id="e",
        kind="mock",
        base_url="http://127.0.0.1/v1",
        model="mock",
        claimed_profile="gpt-oss-120b",
        metadata={"context_length": context, "max_prompt_tokens": context},
    )


def test_real_input_truncation_loses_canary_without_mock_answer_oracle():
    items = context_items(
        "core",
        8,
        [endpoint()],
        {"lengths": [4000, 16000, 32000], "depths": [0, 50, 100], "task_types": 2, "repeats": 2},
    )
    items = [item for item in items if item.expected["check_id"] == "context"]
    clean = ScriptedBackend(8, FaultConfig())
    truncated = ScriptedBackend(8, FaultConfig(truncate_prompt_head_over=16384))
    assert len({item.expected["trial"] for item in items}) == len(items)
    for item in items:
        request = prepare_item(item, endpoint()).body
        expected = item.expected["answers"]
        normal = clean.respond(request)
        assert json.loads(normal["choices"][0]["message"]["content"]) == expected
        fault = truncated.respond(request)
        actual = json.loads(fault["choices"][0]["message"]["content"])
        if item.expected["length"] <= 16000:
            assert text_tokens(request["messages"][0]["content"]) <= 16384
            assert actual == expected
        else:
            assert actual["head"] is None and actual["tail"] == expected["tail"]
            assert actual["total"] < expected["total"]


def test_unsupported_context_is_unsent_and_does_not_allocate_long_payload():
    small = endpoint(8192)
    items = context_items(
        "smoke",
        8,
        [small],
        {"lengths": [4000, 64000], "depths": [0, 50, 100], "task_types": 1, "repeats": 1},
    )
    unsupported = [item for item in items if item.expected.get("length") == 64000]
    assert len(unsupported) == 1 and len(unsupported[0].payload["messages"][0]["content"]) < 100
    decision = prepare_item(unsupported[0], small)
    assert decision.body is None and decision.verdict == "NOT_SUPPORTED"
    small.metadata = {}
    assert prepare_item(items[0], small).verdict == "INCONCLUSIVE"
