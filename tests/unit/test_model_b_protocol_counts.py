import math

import pytest
from tests_support import exchange

from servegauge.core.models import EndpointSpec, Item
from servegauge.profiles import load_profile
from servegauge.scoring.common import invariant_violation, protocol_samples
from servegauge.scoring.json_scoring import json_samples
from servegauge.suites.s1_protocol import probe
from servegauge.tokencount.model_text import TextCodec
from servegauge.verdicts.config import load_policy


@pytest.fixture
def codec(monkeypatch):
    result = TextCodec(
        profile="glm-5.3",
        revision="original-byte-dispatch-test",
        encode=lambda text: list(text.encode()),
        decode=lambda ids: bytes(ids).decode(),
    )
    monkeypatch.setattr("servegauge.scoring.glm_protocol.profile_codec", lambda profile: result)

    def forbidden(*args, **kwargs):
        raise AssertionError("GLM observations must never use a gpt-oss counter")

    monkeypatch.setattr("servegauge.scoring.common.text_tokens", forbidden)
    monkeypatch.setattr("servegauge.scoring.json_scoring.text_tokens", forbidden)
    return result


def record(item, *, text="é", reported=2, reason_count=0, finish="length"):
    return exchange(
        endpoint_id="glm",
        item_id=item.item_id,
        request={"messages": [{"role": "user", "content": "Original text"}], "max_tokens": 2},
        response={"choices": [{"index": 0, "finish_reason": finish, "message": {"content": text}}]},
        usage={
            "completion_tokens": reported,
            "completion_tokens_details": {"reasoning_tokens": reason_count},
        },
    )


def test_p01_and_json_caps_count_claimed_model_text_not_legacy_tokens(codec):
    profile = load_profile("glm-5.3")
    item = probe("P01", 0, "cap", "Original text", {}, extra={"full_text": "ééé"})
    assert not invariant_violation(item, record(item), load_policy(), profile=profile)
    assert invariant_violation(item, record(item, text="éé"), load_policy(), profile=profile)
    item = Item(
        item_id="json-cap",
        suite_id="S3",
        version="original-test",
        payload={},
        expected={"check_id": "J05", "trial": "a", "full_text": "ééé"},
    )
    endpoint = EndpointSpec(
        id="glm",
        kind="mock",
        base_url="http://127.0.0.1:8011/v1",
        model="mock",
        claimed_profile="glm-5.3",
    )
    sample = json_samples([item], [record(item, text="éé")], endpoint, profile=profile)["J05"]
    assert sample.status is None and sample.values["a"] == 1


@pytest.mark.parametrize("fault", ["missing", "boolean", "opaque", "untyped", "conflict"])
def test_incomplete_glm_cap_channels_abstain_instead_of_falling_back(codec, fault):
    item = probe("P01", 0, "cap", "Original text", {}, extra={"full_text": "ééé"})
    capture = record(item)
    message = capture.response["choices"][0]["message"]
    if fault == "missing":
        capture.usage = {}
    elif fault == "boolean":
        capture.usage["completion_tokens"] = True
    else:
        capture.usage["completion_tokens_details"]["reasoning_tokens"] = 1
        message["reasoning_content"] = "x"
        if fault != "untyped":
            message["reasoning_details"] = [
                {
                    "type": "reasoning.encrypted" if fault == "opaque" else "reasoning.text",
                    "text": "different" if fault == "conflict" else "x",
                }
            ]
    sample = protocol_samples([item], [capture], "glm", load_profile("glm-5.3"), load_policy())[
        "P01"
    ]
    assert sample.status == "INCONCLUSIVE" and not sample.values


def test_p07_raw_reasoning_uses_glm_and_disallows_mixed_count_channels(codec):
    items = [probe("P07", 0, effort, "Original reasoning task", {}) for effort in ("low", "high")]
    captures = []
    for item, text in zip(items, ("é", "ééé"), strict=True):
        capture = record(item, text="OK", reported=20, reason_count=None, finish="stop")
        capture.usage["completion_tokens_details"] = {}
        capture.response["choices"][0]["message"].update(
            reasoning_content=text, reasoning_details=[{"type": "reasoning.text", "text": text}]
        )
        captures.append(capture)
    policy = load_policy()
    sample = protocol_samples(items, captures, "glm", load_profile("glm-5.3"), policy)["P07"]
    offset = policy.entries["P07"]["token_pseudocount"]
    assert sample.status is None and sample.values["0"] == pytest.approx(
        math.log((6 + offset) / (2 + offset))
    )
    captures[0].usage["completion_tokens_details"]["reasoning_tokens"] = 2
    mixed = protocol_samples(items, captures, "glm", load_profile("glm-5.3"), policy)["P07"]
    assert mixed.status == "INCONCLUSIVE" and not mixed.values


@pytest.mark.parametrize(
    "fault", ["boolean", "negative", "too-large", "missing-message", "length", "incomplete"]
)
def test_p07_invalid_reported_reasoning_does_not_supply_a_comparison(codec, fault):
    items = [probe("P07", 0, effort, "Original task", {}) for effort in ("low", "high")]
    captures = [
        record(item, text="OK", reported=20, reason_count=2, finish="stop") for item in items
    ]
    broken = captures[1]
    if fault in {"boolean", "negative", "too-large"}:
        broken.usage["completion_tokens_details"]["reasoning_tokens"] = {
            "boolean": True,
            "negative": -1,
            "too-large": 21,
        }[fault]
    elif fault == "missing-message":
        broken.response["choices"][0].pop("message")
    elif fault == "length":
        broken.response["choices"][0]["finish_reason"] = "length"
    else:
        broken.metadata["incomplete"] = True
    sample = protocol_samples(items, captures, "glm", load_profile("glm-5.3"), load_policy())["P07"]
    assert sample.status == "INCONCLUSIVE" and not sample.values
