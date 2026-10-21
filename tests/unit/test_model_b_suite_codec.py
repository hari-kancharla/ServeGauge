"""Dispatch guards use a byte codec; native acceptance uses the real GLM tokenizer."""

import pytest

from servegauge.core.models import EndpointSpec
from servegauge.items.haystacks import context_case
from servegauge.items.longout import long_output, render_output
from servegauge.profiles import load_profile
from servegauge.suites.config import suite_config
from servegauge.suites.s1_protocol import protocol_items
from servegauge.suites.s4_longctx import context_items
from servegauge.suites.s5_longout import reasoning_items
from servegauge.tokencount.model_text import TextCodec


@pytest.fixture
def codec(monkeypatch):
    result = TextCodec(
        profile="glm-5.3",
        revision="original-byte-dispatch-test",
        encode=lambda text: list(text.encode()),
        decode=lambda ids: bytes(ids).decode(),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("GLM corpus must not use the legacy text counter")

    for module in ("items.haystacks", "items.longout", "suites.s1_protocol", "suites.s5_longout"):
        monkeypatch.setattr("servegauge." + module + ".text_tokens", forbidden)
    for module in ("suites.s1_protocol", "suites.s4_longctx", "suites.s5_longout"):
        monkeypatch.setattr("servegauge." + module + ".profile_codec", lambda profile: result)
    return result


def test_context_filler_and_marker_positions_use_the_explicit_codec(codec):
    case = context_case(
        41,
        length=2048,
        depth=50,
        task_type=0,
        aggregate_records=3,
        padding_tokens=4,
        maximum_excess=32,
        iterations=8,
        codec=codec,
    )
    assert 2048 <= case.visible_tokens <= 2080
    assert case.visible_tokens == codec.count(case.text)
    marker = case.mock_task["keys"]["single"] + "=" + case.expected["single"]
    assert case.needle_position_tokens == codec.count(case.text[: case.text.index(marker)])


@pytest.mark.parametrize("kind", ["json", "sequence"])
def test_long_output_binary_search_uses_the_requested_counter(codec, kind):
    case = long_output(41, 2048, kind, text_counter=codec.count)
    assert codec.count(case.text) >= 2048
    assert codec.count(render_output(case.start, case.count - 1, kind)) < 2048


def endpoint():
    return EndpointSpec(
        id="codec-test",
        kind="mock",
        base_url="http://127.0.0.1:8011/v1",
        model="mock",
        claimed_profile="glm-5.3",
        metadata={"context_length": 8192},
    )


def test_context_and_prefix_metadata_have_correct_native_units_and_new_versions(codec):
    items = context_items(
        "smoke",
        41,
        [endpoint()],
        {"lengths": [2048], "depths": [50], "task_types": 1, "repeats": 1},
        profile=load_profile("glm-5.3"),
    )
    for item in items:
        expected = item.expected
        assert "glm" in item.version
        assert expected["text_counter_profile"] == "glm-5.3"
        messages = item.payload["messages"]
        visible = sum(codec.count(message["content"]) for message in messages)
        assert expected["visible_prompt_tokens"] == visible
        assert (
            expected["minimum_context_tokens"]
            == visible + suite_config()["long_context"]["output_cap"]
        )


def test_long_output_headroom_and_over_context_bound_use_native_units(codec):
    cfg = suite_config()["reasoning"]
    items = reasoning_items(
        "smoke",
        41,
        {"items": 1, "samples": 1, "long_items": 1, "lengths": [cfg["allowed_lengths"][0]]},
        profile=load_profile("glm-5.3"),
    )
    item = next(item for item in items if item.expected["check_id"].startswith("R03"))
    assert "glm" in item.version
    assert item.expected["visible_output_tokens"] == codec.count(item.expected["full_text"])
    assert (
        item.expected["minimum_output_tokens"]
        == codec.count(item.expected["full_text"]) + cfg["output_headroom"]
    )
    probes = protocol_items("smoke", 41, endpoint=endpoint(), profile=load_profile("glm-5.3"))
    for item in (item for item in probes if item.expected["check_id"] == "P12"):
        assert "glm" in item.version
        assert item.expected["visible_prompt_tokens"] == codec.count(
            item.payload["messages"][0]["content"]
        )
        assert item.expected["visible_prompt_tokens"] > endpoint().metadata["context_length"]


def test_claimed_glm_generation_requires_a_profile_and_unknown_identity_never_uses_legacy(
    monkeypatch,
):
    import servegauge.tokencount.model_text as model_text

    def forbidden():
        raise AssertionError("an unverified identity must not read assets")

    monkeypatch.setattr(model_text, "verified_assets", forbidden)
    profile = load_profile("glm-5.3")
    profile.tokenizer["hf_revision"] = "unknown"
    with pytest.raises(ValueError, match="verified_glm"):
        model_text.profile_codec(profile)
    with pytest.raises(ValueError, match="requires_verified_profile"):
        context_items("smoke", 41, [endpoint()])
    with pytest.raises(ValueError, match="requires_verified_profile"):
        protocol_items("smoke", 41, endpoint=endpoint())
