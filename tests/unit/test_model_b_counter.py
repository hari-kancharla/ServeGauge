import copy

import pytest

from servegauge.profiles import field_verified, load_profile
from servegauge.tokencount.glm_counter import normalize_messages, render_glm_chat
from servegauge.tokencount.profile import counter_kind


def test_sourced_profile_has_exact_identity_and_discloses_unknown_provider_semantics():
    profile = load_profile("glm-5.3")
    assert counter_kind(profile) == "glm-5.3"
    assert profile.weights == "zai-org/GLM-5.3"
    assert profile.reasoning["levels"] == ["low", "high", "max"]
    assert field_verified(profile, "tokenizer.hf_revision")
    assert not field_verified(profile, "tokenizer.provider_overhead")
    assert not field_verified(profile, "reference_tiers")
    assert not profile.reference_tiers and not profile.immutable_params
    assert counter_kind(load_profile("gpt-oss-120b")) == "gpt-oss"


@pytest.mark.parametrize(
    "fault", ["identity", "revision", "family", "format", "unverified", "nested", "source"]
)
def test_wrong_or_unknown_model_counter_never_falls_back_to_harmony(fault):
    profile = load_profile("glm-5.3")
    if fault == "identity":
        profile.weights = "other/model"
    elif fault == "revision":
        profile.tokenizer["hf_revision"] = "other"
    elif fault == "family":
        profile.family = "gpt-oss"
    elif fault == "format":
        profile.chat_format = "harmony"
    elif fault == "source":
        profile.field_sources.pop("tokenizer.hf_revision")
    else:
        profile.unverified.append("tokenizer" if fault == "unverified" else "tokenizer.hf_revision")
    assert counter_kind(profile) is None


def history():
    return [
        {"role": "user", "content": "Convert 北京 and café"},
        {
            "role": "assistant",
            "content": None,
            "reasoning": "Use both tools.",
            "reasoning_details": [{"type": "reasoning.text", "text": "Use both tools."}],
            "tool_calls": [
                {
                    "id": "one",
                    "type": "function",
                    "function": {"name": "first", "arguments": '{"x":1}'},
                },
                {
                    "id": "two",
                    "type": "function",
                    "function": {"name": "second", "arguments": '{"label":"雨"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "two", "content": "second output"},
        {"role": "tool", "tool_call_id": "one", "content": "first output"},
    ]


def test_glm_normalizes_complete_parallel_history_without_mutating_or_duplicating_reasoning():
    original = history()
    before = copy.deepcopy(original)
    native = normalize_messages(original)
    assert original == before
    assert native[1]["reasoning_content"] == "Use both tools."
    assert native[1]["content"] == ""
    assert native[1]["tool_calls"][0]["function"]["arguments"] == {"x": 1}
    assert native[1]["tool_calls"][1]["function"]["arguments"] == {"label": "雨"}
    assert "reasoning" not in native[1] and "reasoning_details" not in native[1]
    assert normalize_messages(native) == native


def test_nullable_tool_calls_are_empty_but_conflicting_inline_text_blocks_abstain():
    messages = [{"role": "assistant", "content": "OK", "tool_calls": None}]
    assert normalize_messages(messages) == messages
    messages[0].update(
        content=[{"type": "text", "text": "<think>hidden</think>visible"}],
        reasoning_content="different",
    )
    with pytest.raises(ValueError, match="conflicting_reasoning"):
        normalize_messages(messages)


@pytest.mark.parametrize(
    "fault",
    [
        "role",
        "media",
        "block",
        "opaque",
        "summary",
        "conflict",
        "nontext",
        "nonlist",
        "badcall",
        "badfunction",
        "arrayargs",
        "duplicatekeys",
        "nonfinite",
        "calltype",
        "emptyname",
        "inlineconflict",
        "function_call",
        "duplicateids",
        "badcontent",
    ],
)
def test_unsupported_or_ambiguous_history_is_never_silently_counted(fault):
    messages = history()
    assistant = messages[1]
    call = assistant["tool_calls"][0]
    if fault == "role":
        messages[0]["role"] = "developer"
    elif fault in {"media", "block"}:
        messages[0]["content"] = [
            {"type": "image_url" if fault == "media" else "unknown", "text": "x"}
        ]
    elif fault in {"opaque", "summary"}:
        assistant["reasoning_details"][0]["type"] = "reasoning." + fault
    elif fault == "conflict":
        assistant["reasoning_content"] = "different"
    elif fault == "nontext":
        assistant["reasoning"] = 123
    elif fault == "nonlist":
        assistant["tool_calls"] = {}
    elif fault == "badcall":
        assistant["tool_calls"] = [None]
    elif fault == "badfunction":
        call["function"] = []
    elif fault in {"arrayargs", "duplicatekeys", "nonfinite"}:
        call["function"]["arguments"] = {
            "arrayargs": "[]",
            "duplicatekeys": '{"x":1,"x":2}',
            "nonfinite": '{"x":NaN}',
        }[fault]
    elif fault == "calltype":
        call["type"] = "unknown"
    elif fault == "emptyname":
        call["function"]["name"] = ""
    elif fault == "inlineconflict":
        assistant["content"] = "<think>different</think>answer"
    elif fault == "function_call":
        assistant["function_call"] = {"name": "old"}
    elif fault == "duplicateids":
        assistant["tool_calls"][1]["id"] = "one"
    elif fault == "badcontent":
        messages[0]["content"] = {"unexpected": "object"}
    with pytest.raises(ValueError):
        normalize_messages(messages)


@pytest.mark.parametrize("effort", ["low", "high", "max"])
@pytest.mark.parametrize("clear", [True, False])
def test_explicit_template_controls_are_forwarded_without_harmony_defaults(
    monkeypatch, effort, clear
):
    import servegauge.tokencount.glm_counter as counter

    class OriginalTemplate:
        def render(self, **kwargs):
            assert kwargs["reasoning_effort"] == effort
            assert kwargs["clear_thinking"] is clear
            assert kwargs["add_generation_prompt"] is True
            assert kwargs["messages"][1]["reasoning_content"] == "Use both tools."
            return "rendered"

    monkeypatch.setattr(counter, "glm_template", lambda: OriginalTemplate())
    assert render_glm_chat(history(), reasoning_effort=effort, clear_thinking=clear) == "rendered"


@pytest.mark.parametrize(
    "kwargs", [{"reasoning_effort": "medium"}, {"clear_thinking": 1}, {"reasoning_effort": None}]
)
def test_invalid_explicit_controls_abstain_before_loading_assets(kwargs):
    with pytest.raises(ValueError):
        render_glm_chat(history(), **kwargs)
