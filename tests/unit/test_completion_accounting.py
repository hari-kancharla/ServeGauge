import copy

import pytest
from tests_support import exchange

from servegauge.profiles import load_profile
from servegauge.scoring.completion_accounting import (
    completion_channels,
    extended_accounting_samples,
)
from servegauge.suites.s7_accounting import accounting_items


def complete(**updates):
    response = {
        "id": "gen-test",
        "model": "mock",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "content": "OK",
                    "reasoning": " step step",
                    "reasoning_details": [{"type": "reasoning.text", "text": " step step"}],
                },
            }
        ],
    }
    return exchange(
        response=response,
        usage={
            "completion_tokens": 3,
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
        **updates,
    )


def test_reasoning_mirrors_count_once_and_real_repeated_chunks_are_retained():
    record = complete()
    message = record.response["choices"][0]["message"]
    message["reasoning_content"] = message["reasoning"]
    message["reasoning_details"] = [
        {"type": "reasoning.text", "text": " step", "index": 0},
        {"type": "reasoning.text", "text": " step", "index": 0},
    ]
    assert completion_channels(record) == ("OK", " step step", "raw-text")
    for key in ("reasoning", "reasoning_content"):
        message.pop(key)
    assert completion_channels(record) == ("OK", " step step", "raw-text")


@pytest.mark.parametrize(
    "fault",
    [
        "summary",
        "encrypted",
        "unknown",
        "conflict",
        "hidden",
        "unknown-hidden",
        "excluded",
        "length",
        "redacted",
        "tool",
        "multimodal",
        "refusal",
        "audio",
        "choices",
        "invalid-count",
        "invalid-details",
        "invalid-total",
        "incomplete",
        "missing-terminal",
        "completion-error",
        "nontext-reasoning",
        "untyped-reasoning",
        "multiple-stream-choices",
        "stream-identity",
        "excluded-routing",
        "boolean-n",
        "bad-choice-index",
    ],
)
def test_incomplete_completion_evidence_abstains(fault):
    record = complete()
    message = record.response["choices"][0]["message"]
    if fault in {"summary", "encrypted", "unknown"}:
        message["reasoning_details"].append({"type": "reasoning." + fault, "text": "opaque"})
    elif fault == "conflict":
        message["reasoning_content"] = "different"
    elif fault in {"hidden", "unknown-hidden"}:
        message.pop("reasoning")
        message.pop("reasoning_details")
        if fault == "unknown-hidden":
            record.usage.pop("completion_tokens_details")
    elif fault == "excluded":
        record.request["reasoning"] = {"exclude": True}
    elif fault == "length":
        record.response["choices"][0]["finish_reason"] = "length"
    elif fault == "redacted":
        from servegauge.core.redaction import REDACTED

        message["content"] = REDACTED
    elif fault == "tool":
        message["tool_calls"] = [{"function": {"name": "x"}}]
    elif fault == "multimodal":
        message["content"] = [{"type": "text", "text": "OK"}]
    elif fault == "refusal":
        message["refusal"] = "refused"
    elif fault == "audio":
        record.usage["completion_tokens_details"]["audio_tokens"] = 1
    elif fault == "choices":
        record.response["choices"].append(copy.deepcopy(record.response["choices"][0]))
    elif fault == "invalid-count":
        record.usage["completion_tokens"] = True
    elif fault == "invalid-details":
        record.usage["completion_tokens_details"] = []
    elif fault == "invalid-total":
        record.usage["completion_tokens_details"]["reasoning_tokens"] = 4
    elif fault == "incomplete":
        record.metadata["incomplete"] = True
    elif fault == "missing-terminal":
        record.request["stream"] = True
    elif fault == "completion-error":
        record.response["error"] = {"code": "failed"}
    elif fault == "nontext-reasoning":
        message["reasoning"] = 7
    elif fault == "untyped-reasoning":
        message.pop("reasoning_details")
    elif fault == "multiple-stream-choices":
        record.metadata["stream_multiple_choices"] = True
    elif fault == "stream-identity":
        record.metadata["stream_identity_conflict"] = True
    elif fault == "excluded-routing":
        record.metadata["exclude_from_scoring"] = True
    elif fault == "boolean-n":
        record.request["n"] = True
    elif fault == "bad-choice-index":
        record.response["choices"][0]["index"] = 1
    with pytest.raises(ValueError):
        completion_channels(record)


def test_confirmed_zero_reasoning_allows_visible_only_but_missing_detail_does_not():
    record = complete()
    record.response["choices"][0]["message"] = {"content": "OK"}
    record.usage["completion_tokens_details"]["reasoning_tokens"] = 0
    assert completion_channels(record) == ("OK", "", "zero-reasoning")


def test_stream_reasoning_alias_is_preserved_and_ambiguous_streams_are_flagged():
    import json

    from servegauge.core.models import SseEvent
    from servegauge.transport.http import assemble_stream

    chunks = [
        {
            "id": "gen-test",
            "model": "mock",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_content": text,
                        "reasoning_details": [{"type": "reasoning.text", "text": text}],
                    },
                }
            ],
        }
        for text in (" step", " step")
    ]
    chunks.append({"choices": [{"index": 0, "delta": {"content": "OK"}, "finish_reason": "stop"}]})
    response, meta = assemble_stream(
        [SseEvent(t_ms=i, data=json.dumps(c)) for i, c in enumerate(chunks)]
    )
    record = complete(request={"stream": True}, metadata={**meta, "terminal": True})
    record.response = response
    assert completion_channels(record) == ("OK", " step step", "raw-text")
    chunks.append({"id": "different", "choices": [{"index": 1, "delta": {"content": "other"}}]})
    _, meta = assemble_stream([SseEvent(t_ms=i, data=json.dumps(c)) for i, c in enumerate(chunks)])
    assert meta["stream_multiple_choices"] and meta["stream_identity_conflict"]


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "transport",
        "unsent",
        "rejected",
        "empty",
        "wrong-profile",
        "counter-missing",
        "wrong-counter-revision",
    ],
)
def test_accounting_group_does_not_drop_unqualified_observations(fault, monkeypatch):
    from servegauge.core.errors import SafeError

    item = next(
        i for i in accounting_items("smoke", 11, {"items": 1}) if i.expected["check_id"] == "B02"
    )
    record = complete(endpoint_id="target", item_id=item.item_id)
    profile = load_profile("gpt-oss-120b")
    if fault == "transport":
        record.error = "transport_timeout"
    elif fault == "unsent":
        record.metadata["request_not_sent"] = True
    elif fault == "rejected":
        record.status = 400
    elif fault == "empty":
        record.response["choices"][0]["message"] = {"content": ""}
        record.usage["completion_tokens_details"]["reasoning_tokens"] = 0
    elif fault == "wrong-profile":
        profile.family = "unknown"
    elif fault == "wrong-counter-revision":
        profile.tokenizer["hf_revision"] = "unqualified-revision"
    elif fault == "counter-missing":

        def unavailable(text):
            raise SafeError("fixed-safe-reason")

        monkeypatch.setattr("servegauge.scoring.completion_accounting.hf_text_tokens", unavailable)
    records = [] if fault == "missing" else [record, record] if fault == "duplicate" else [record]
    sample = extended_accounting_samples([item], records, "target", profile)["B02"]
    assert sample.status in {"ERROR", "INCONCLUSIVE"} and not sample.values


def test_extended_accounting_requires_versioned_inventory_and_generation_identity():
    items = accounting_items("smoke", 11, {"items": 1})
    item = next(item for item in items if item.expected["check_id"] == "B02")
    record = complete(
        endpoint_id="target",
        item_id=item.item_id,
        provider_meta={
            "id": "gen-test",
            "model": "mock",
            "cancelled": False,
            "tokens_prompt": 12,
            "native_tokens_prompt": 10,
            "tokens_completion": 6,
            "native_tokens_completion": 4,
        },
    )
    samples = extended_accounting_samples(items, [record], "target", load_profile("gpt-oss-120b"))
    assert samples["B02"].values[item.item_id] == 0
    assert samples["B03.prompt"].values[item.item_id] == pytest.approx(0.2)
    assert samples["B03.completion"].values[item.item_id] == pytest.approx(0.5)
    assert samples["B02"].details["counts"][item.item_id][0]["local_reasoning"] == 2
    record.provider_meta["id"] = "another-generation"
    bad = extended_accounting_samples([item], [record], "target", load_profile("gpt-oss-120b"))
    assert bad["B03.prompt"].status == "INCONCLUSIVE" and not bad["B03.prompt"].values
    legacy = items[0].model_copy(deep=True)
    assert legacy.version == "prompt-count-v1"
    assert (
        extended_accounting_samples([legacy], [record], "target", load_profile("gpt-oss-120b"))
        == {}
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("tokens_completion", None),
        ("native_tokens_completion", 0),
        ("native_tokens_completion", True),
        ("tokens_prompt", -1),
        ("model", "other"),
        ("cancelled", True),
    ],
)
def test_generation_counts_do_not_guess_nullable_or_mismatched_fields(field, value):
    item = next(
        i for i in accounting_items("smoke", 11, {"items": 1}) if i.expected["check_id"] == "B02"
    )
    meta = {
        "id": "gen-test",
        "model": "mock",
        "cancelled": False,
        "tokens_prompt": 12,
        "native_tokens_prompt": 10,
        "tokens_completion": 6,
        "native_tokens_completion": 4,
    }
    meta[field] = value
    record = complete(endpoint_id="target", item_id=item.item_id, provider_meta=meta)
    samples = extended_accounting_samples([item], [record], "target", load_profile("gpt-oss-120b"))
    affected = "B03.prompt" if field == "tokens_prompt" else "B03.completion"
    assert samples[affected].status == "INCONCLUSIVE" and not samples[affected].values
