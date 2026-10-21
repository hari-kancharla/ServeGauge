import math

import numpy as np
import pytest

from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.stats.logprob_dispersion import logprob_dispersion_test
from servegauge.stats.tests import two_sample_permutation


def sample_body():
    return {
        "messages": [
            {"role": "user", "content": 'TW_TASK: {"choices":["red","blue","green","gold"]}'}
        ],
        "temperature": 1.0,
        "logprobs": True,
        "top_logprobs": 4,
        "max_tokens": 8,
    }


def test_sampling_v2_reports_actual_distribution_and_chosen_probability():
    backend = ScriptedBackend(25, FaultConfig(), sampling_version="v2")
    body = sample_body()
    counts = dict.fromkeys(["red", "blue", "green", "gold"], 0)
    first = backend.respond(body)
    logs = first["choices"][0]["logprobs"]["content"][0]
    expected = {r["token"]: math.exp(r["logprob"]) for r in logs["top_logprobs"]}
    assert sum(expected.values()) == pytest.approx(1)
    assert expected["red"] / expected["blue"] == pytest.approx(math.exp(0.65))
    assert logs["logprob"] == pytest.approx(math.log(expected[logs["token"]]))
    assert [r["logprob"] for r in logs["top_logprobs"]] == sorted(
        [r["logprob"] for r in logs["top_logprobs"]], reverse=True
    )
    for _ in range(2000):
        counts[backend.respond(body)["choices"][0]["message"]["content"]] += 1
    assert all(abs(counts[k] / 2000 - p) < 0.04 for k, p in expected.items())
    clipped = backend.respond({**body, "top_k": 1})
    assert clipped["choices"][0]["logprobs"]["content"][0]["logprob"] == 0


def test_v2_jitter_is_seeded_and_legacy_is_unchanged():
    body = sample_body()
    a = ScriptedBackend(
        19, FaultConfig(logit_noise=0.3), sampling_version="v2", healthy_logit_jitter=0.1
    )
    b = ScriptedBackend(
        19, FaultConfig(logit_noise=0.3), sampling_version="v2", healthy_logit_jitter=0.1
    )
    assert [a.respond(body) for _ in range(10)] == [b.respond(body) for _ in range(10)]
    legacy = ScriptedBackend(19, FaultConfig()).respond(body)
    assert legacy["choices"][0]["logprobs"]["content"][0]["logprob"] == -0.02
    for kwargs in (
        {"sampling_version": "bad"},
        {"healthy_logit_jitter": -1},
        {"healthy_logit_jitter": 0.1},
    ):
        with pytest.raises(ValueError):
            ScriptedBackend(1, FaultConfig(), **kwargs)


def test_dispersion_exact_known_answer_and_side_symmetry():
    x = [{"a": -1.0, "b": -1.0}] * 6
    y = [{"a": -0.1 if i % 2 else -1.9, "b": -1.0} for i in range(6)]
    result = logprob_dispersion_test(x, y, seed=1, resamples=64)
    assert result.statistic == pytest.approx(0.81)
    assert result.p_value == pytest.approx(2 / 64)
    assert result == logprob_dispersion_test(y, x, seed=1, resamples=64)
    center = np.mean([list(r.values()) for r in x + y], axis=0)
    scalar = two_sample_permutation(
        [list(r.values()) for r in x],
        [list(r.values()) for r in y],
        lambda a, b: abs(
            float(
                np.mean(np.sum((a - center) ** 2, axis=1))
                - np.mean(np.sum((b - center) ** 2, axis=1))
            )
        ),
        seed=1,
        resamples=64,
        paired=True,
    )
    assert result.p_value == scalar.p_value
    assert logprob_dispersion_test(x, x, seed=1, resamples=64).p_value == 1


@pytest.mark.parametrize(
    "x,y",
    [
        ([], []),
        ([{"a": -1}], [{"b": -1}]),
        ([{"a": -1}], [{"a": -2}]),
        ([{"a": 1, "b": -1}] * 2, [{"a": -1, "b": -1}] * 2),
    ],
)
def test_dispersion_refuses_missing_support_or_invalid_values(x, y):
    with pytest.raises(ValueError):
        logprob_dispersion_test(x, y, seed=1, resamples=20)


def test_v2_refuses_unmodeled_or_censored_token_probabilities():
    backend = ScriptedBackend(5, FaultConfig(), sampling_version="v2")
    for choices in (["red", "red"], ["many token words", "blue"]):
        body = sample_body()
        body["messages"][0]["content"] = "TW_TASK: " + __import__("json").dumps(
            {"choices": choices}
        )
        with pytest.raises(ValueError, match="distinct_single_token"):
            backend.respond(body)
    with pytest.raises(ValueError, match="uncensored"):
        backend.respond({**sample_body(), "stop": ["red", "blue", "green", "gold"]})


def test_v2_can_sample_multitoken_text_when_token_logprobs_are_not_requested():
    import json

    choices = ["many token words", "another long option", "blue", "red"]
    body = sample_body()
    body.pop("logprobs")
    body.pop("top_logprobs")
    body["messages"][0]["content"] = "TW_TASK: " + json.dumps({"choices": choices})
    backend = ScriptedBackend(51, FaultConfig(), sampling_version="v2")
    responses = [backend.respond(body) for _ in range(8)]
    assert all(r["choices"][0]["message"]["content"] in choices for r in responses)
    assert all("logprobs" not in r["choices"][0] for r in responses)
    with pytest.raises(ValueError, match="distinct_single_token"):
        backend.respond({**body, "logprobs": True, "top_logprobs": 4})


def test_v2_collects_all_frozen_multilingual_fp1_choice_families():
    from servegauge.scoring.fingerprint_collection import fingerprint_samples
    from servegauge.suites.s6_fingerprint import fingerprint_items
    from servegauge.validation.capture import captures, endpoint

    items = fingerprint_items("core", 57, {"methods": ["FP1"], "fp1_samples": 2})
    target = endpoint("mock:multilingual", 58, clean=True)
    records = captures(
        items, ScriptedBackend(58, FaultConfig(), sampling_version="v2"), target, "all-cells"
    )
    samples = fingerprint_samples(items, records, target)["FP1"]
    assert len(items) == 40 and len(records) == 80
    assert samples.status is None and samples.details["cells"] == 40
    assert {i.expected["language"] for i in items} == {"en", "es", "fr", "de"}
