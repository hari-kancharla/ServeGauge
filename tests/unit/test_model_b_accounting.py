import json

import pytest
from tests_support import exchange

from servegauge.core.errors import SafeError
from servegauge.profiles import load_profile
from servegauge.scoring.accounting_scoring import accounting_samples
from servegauge.scoring.completion_accounting import extended_accounting_samples
from servegauge.scoring.tokenizer_hints import tokenizer_samples
from servegauge.suites.config import suite_config
from servegauge.suites.s7_accounting import accounting_items


@pytest.fixture
def counters(monkeypatch):
    """Dispatch spies only; independent native tests prove real token IDs."""
    seen = []

    def rendered(messages, **controls):
        seen.append(controls)
        return json.dumps([messages, controls], ensure_ascii=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("GLM must never use the gpt-oss counter")

    monkeypatch.setattr("servegauge.scoring.accounting_scoring.render_glm_chat", rendered)
    monkeypatch.setattr("servegauge.scoring.accounting_scoring.glm_text_tokens", len)
    monkeypatch.setattr("servegauge.scoring.completion_accounting.glm_text_tokens", len)
    monkeypatch.setattr("servegauge.scoring.accounting_scoring.render_hf_chat", forbidden)
    monkeypatch.setattr("servegauge.scoring.accounting_scoring.hf_text_tokens", forbidden)
    monkeypatch.setattr("servegauge.scoring.completion_accounting.hf_text_tokens", forbidden)
    return seen


def capture(check="B01", **params):
    item = next(
        item
        for item in accounting_items("smoke", 43, {"items": 1})
        if item.expected["check_id"] == check
    )
    record = exchange(
        endpoint_id="glm-target",
        item_id=item.item_id,
        request={"messages": item.payload["messages"], **params},
        usage={
            "prompt_tokens": 300,
            "completion_tokens": 7,
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
        response={
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "content": "OK",
                        "reasoning_content": "think",
                        "reasoning_details": [{"type": "reasoning.text", "text": "think"}],
                    },
                }
            ]
        },
    )
    return item, record


def prompt(item, records, profile=None):
    return accounting_samples(
        [item],
        records,
        "glm-target",
        profile or load_profile("glm-5.3"),
        suite_config()["accounting"],
    )


def test_prompt_and_claimed_tokenizer_hint_dispatch_to_glm_with_max_default(counters):
    for check in ("B01", "B05"):
        item, record = capture(check)
        if check == "B01":
            sample = prompt(item, [record])["B01"]
        else:
            sample = tokenizer_samples([item], [record], "glm-target", load_profile("glm-5.3"))[
                "B05"
            ]
            assert sample.details["model_identity_proof"] is False
        assert sample.values and sample.status is None
        assert sample.details["counter_version"] == "glm-hf-api-adapter-v1"
        assert sample.details["provider_template_verified"] is False
        assert sample.details["input_hashes"]
        assert counters[-1] == {"tools": None, "reasoning_effort": "max", "clear_thinking": False}


def test_reasoning_control_layouts_do_not_reuse_or_hide_a_cached_observation(counters):
    item, first = capture(reasoning_effort="high", clear_thinking=True)
    second = first.model_copy(deep=True)
    second.request["clear_thinking"] = False
    item.expected["samples"] = 2
    second.sample_idx = 1
    sample = prompt(item, [first, second])["B01"]
    assert len(sample.details["input_hashes"][item.item_id]) == 2
    assert [controls["clear_thinking"] for controls in counters] == [True, False]
    second.request.update(reasoning={"effort": "low"})
    refused = prompt(item, [first, second])["B01"]
    assert refused.status == "INCONCLUSIVE"
    assert len(refused.repeats[item.item_id]) == 1


@pytest.mark.parametrize(
    "params",
    [
        {"reasoning_effort": "medium"},
        {"clear_thinking": 1},
        {"reasoning": False},
        {"reasoning": {"exclude": "unknown"}},
        {"reasoning": {"max_tokens": 20}},
        {"reasoning": {"effort": "low"}, "reasoning_effort": "high"},
        {"thinking": {"type": "disabled"}},
        {"thinking": {"type": "enabled", "clear_thinking": True}},
        {"thinking": "enabled"},
        {"thinking": None},
    ],
)
def test_unknown_provider_controls_cannot_supply_prompt_count_evidence(
    params, counters, monkeypatch
):
    # Use the real validator/renderer before any cache read, keeping default tests asset-free.
    from servegauge.tokencount.glm_counter import render_glm_chat

    monkeypatch.setattr("servegauge.scoring.accounting_scoring.render_glm_chat", render_glm_chat)
    item, record = capture(**params)
    sample = prompt(item, [record])["B01"]
    assert sample.status == "INCONCLUSIVE" and not sample.values


def test_explicit_supported_first_party_and_router_effort_shapes(counters):
    for params in (
        {"thinking": {"type": "enabled"}, "reasoning_effort": "low"},
        {"reasoning": {"effort": "high"}},
    ):
        item, record = capture(**params)
        assert prompt(item, [record])["B01"].values
        assert counters[-1]["reasoning_effort"] in {"low", "high"}


def test_completion_uses_glm_counter_once_per_visible_and_raw_channel(counters):
    item, record = capture("B02")
    result = extended_accounting_samples([item], [record], "glm-target", load_profile("glm-5.3"))
    sample = result["B02"]
    assert sample.status is None and sample.values[item.item_id] == 0
    counts = sample.details["counts"][item.item_id][0]
    assert counts["local_visible"] == 2 and counts["local_reasoning"] == 5
    record.response["choices"][0]["message"].pop("reasoning_details")
    refused = extended_accounting_samples([item], [record], "glm-target", load_profile("glm-5.3"))[
        "B02"
    ]
    assert refused.status == "INCONCLUSIVE" and not refused.values


@pytest.mark.parametrize("check", ["B01", "B02", "B05"])
def test_missing_cache_is_an_error_and_unknown_revision_abstains(check, counters, monkeypatch):
    def unavailable(*args, **kwargs):
        raise SafeError("glm_tokenizer_cache_missing_or_invalid")

    monkeypatch.setattr("servegauge.scoring.accounting_scoring.glm_text_tokens", unavailable)
    monkeypatch.setattr("servegauge.scoring.completion_accounting.glm_text_tokens", unavailable)
    item, record = capture(check)
    profile = load_profile("glm-5.3")

    def scored():
        if check == "B01":
            return prompt(item, [record], profile)["B01"]
        if check == "B05":
            return tokenizer_samples([item], [record], "glm-target", profile)["B05"]
        return extended_accounting_samples([item], [record], "glm-target", profile)["B02"]

    assert scored().status == "ERROR" and not scored().values
    profile.tokenizer["hf_revision"] = "unqualified"
    assert scored().status == "INCONCLUSIVE" and not scored().values


def test_glm_cost_scenario_uses_scoped_controls_and_labels_unavailable_assets(monkeypatch):
    from servegauge.core.models import EndpointSpec
    from servegauge.runner.cost import prompt_reservation_tokens
    from servegauge.runner.cost_scenarios import initial_input_proxy

    seen = []

    def rendered(messages, **controls):
        seen.append(controls)
        return "GLM original rendered text"

    monkeypatch.setattr("servegauge.runner.cost_scenarios.render_glm_chat", rendered)
    monkeypatch.setattr("servegauge.runner.cost_scenarios.glm_text_tokens", len)
    endpoint = EndpointSpec(
        id="local-scenario",
        kind="openai_compat",
        base_url="http://127.0.0.1:8011/v1",
        model="synthetic",
        claimed_profile="glm-5.3",
        metadata={"context_length": 1000},
    )
    body = {"messages": [{"role": "user", "content": "Original text"}], "max_tokens": 8}
    cache = {}
    first = initial_input_proxy(body, endpoint, cache, counter_date="2026-10-18")
    assert first == (
        len("GLM original rendered text"),
        "official local GLM template; native overhead unqualified",
    )
    assert seen[-1]["reasoning_effort"] == "max"
    body["clear_thinking"] = True
    assert initial_input_proxy(body, endpoint, cache, counter_date="2026-10-18") == first
    assert len(seen) == 2 and seen[-1]["clear_thinking"] is True
    assert prompt_reservation_tokens(body, endpoint) == 1000

    def unavailable(*args, **kwargs):
        raise SafeError("glm_tokenizer_cache_missing_or_invalid")

    monkeypatch.setattr("servegauge.runner.cost_scenarios.glm_text_tokens", unavailable)
    body["clear_thinking"] = False
    count, method = initial_input_proxy(body, endpoint, {}, counter_date="2026-10-18")
    assert count > 0 and method.startswith("canonical wire proxy; GLM")
