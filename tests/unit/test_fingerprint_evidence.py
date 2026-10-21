import copy
import math

import pytest

from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.scoring.fingerprint_evidence import (
    FingerprintEvidence,
    prepare_evidence,
    read_response,
    unavailable_check,
)


def envelope():
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "red"},
                "logprobs": {
                    "content": [
                        {
                            "token": "red",
                            "logprob": math.log(0.6),
                            "top_logprobs": [
                                {"token": "red", "logprob": math.log(0.6)},
                                {"token": "blue", "logprob": math.log(0.3)},
                            ],
                        }
                    ]
                },
            }
        ]
    }


def evidence(response=None, **kwargs):
    response = envelope() if response is None else response
    return FingerprintEvidence(
        reference_tier="MOCK",
        probe_id="color-v3",
        anchor_tokens=["red", "blue"],
        anchor_source="fixed-probe-configuration",
        target=[{"session_id": "s0", "draw": 0, "response": response}],
        reference=[{"session_id": "s0", "draw": 0, "response": copy.deepcopy(response)}],
        **kwargs,
    )


@pytest.mark.parametrize(
    "value,reason",
    [
        (None, "missing_response"),
        ({}, "invalid_response"),
        ({"choices": []}, "invalid_response"),
        (
            {"choices": [{"message": {"content": "red"}, "finish_reason": "length"}]},
            "censored_response",
        ),
        (
            {"choices": [{"message": {"content": "red"}, "finish_reason": "stop"}]},
            "missing_logprobs",
        ),
    ],
)
def test_missing_evidence_has_explicit_reason_without_exception(value, reason):
    result = read_response(value, tolerance=0.001)
    assert result.logprobs is None and result.reason == reason


@pytest.mark.parametrize(
    "mutation",
    [
        "null",
        "empty",
        "multi",
        "missing_top",
        "duplicate",
        "nan",
        "positive",
        "boolean",
        "mass",
        "chosen_mismatch",
        "bytes",
    ],
)
def test_invalid_probability_evidence_is_unavailable_and_text_is_retained(mutation):
    response = envelope()
    choice = response["choices"][0]
    row = choice["logprobs"]["content"][0]
    if mutation == "null":
        choice["logprobs"] = None
    elif mutation == "empty":
        choice["logprobs"]["content"] = []
    elif mutation == "multi":
        choice["logprobs"]["content"].append(copy.deepcopy(row))
    elif mutation == "missing_top":
        row.pop("top_logprobs")
    elif mutation == "duplicate":
        row["top_logprobs"].append(copy.deepcopy(row["top_logprobs"][0]))
    elif mutation in {"nan", "positive", "boolean"}:
        row["top_logprobs"][0]["logprob"] = {"nan": float("nan"), "positive": 0.1, "boolean": True}[
            mutation
        ]
    elif mutation == "mass":
        row["top_logprobs"][1]["logprob"] = math.log(0.8)
    elif mutation == "chosen_mismatch":
        row["logprob"] = -5
    elif mutation == "bytes":
        row["bytes"] = [255]
    result = read_response(response, tolerance=0.001)
    assert result.logprobs is None and result.text == "red"
    assert result.reason


def test_partial_topk_preserves_raw_probabilities_and_never_renormalizes():
    result = read_response(envelope(), tolerance=0.001)
    assert result.reason is None
    assert sum(math.exp(v) for v in result.logprobs.values()) == pytest.approx(0.9)
    prepared = prepare_evidence(evidence(), minimum_anchor_tokens=2, tolerance=0.001)
    assert prepared.logprob_reason is None
    assert prepared.first_logs == [{"red": math.log(0.6), "blue": math.log(0.3)}]


def test_missing_anchor_or_row_disables_logprob_comparison_without_row_deletion():
    sample = evidence()
    sample.target.append(sample.target[0].model_copy(update={"draw": 1, "response": None}))
    sample.reference.append(sample.reference[0].model_copy(update={"draw": 1}))
    prepared = prepare_evidence(sample, minimum_anchor_tokens=2, tolerance=0.001)
    assert prepared.logprob_reason == "incomplete_logprob_evidence"
    assert len(prepared.first) == len(prepared.second) == 2
    sample = evidence()
    sample.anchor_tokens.append("gold")
    assert (
        prepare_evidence(sample, minimum_anchor_tokens=2, tolerance=0.001).logprob_reason
        == "missing_fixed_anchor"
    )
    sample.anchor_source = None
    assert (
        prepare_evidence(sample, minimum_anchor_tokens=2, tolerance=0.001).logprob_reason
        == "fixed_probe_anchors_required"
    )


def test_actual_mock_topk_parses_and_projection_uses_only_declared_anchors():
    body = {
        "messages": [
            {"role": "user", "content": 'TW_TASK: {"choices":["red","blue","green","gold"]}'}
        ],
        "logprobs": True,
        "top_logprobs": 4,
    }
    response = ScriptedBackend(9, FaultConfig(), sampling_version="v2").respond(body)
    prepared = prepare_evidence(evidence(response), minimum_anchor_tokens=2, tolerance=0.001)
    assert prepared.logprob_reason is None
    assert set(prepared.first_logs[0]) == {"red", "blue"}


def test_unavailable_result_never_becomes_pass_fail_or_zero_probability():
    result = unavailable_check(
        evidence(), check_id="FP3-dispersion", reason="missing_fixed_anchor", groups=1
    )
    assert result.verdict == "INCONCLUSIVE" and result.n == 1
    assert result.estimate is result.p_value is result.adjusted_p is None
    assert result.details["target_requests"] == 1
