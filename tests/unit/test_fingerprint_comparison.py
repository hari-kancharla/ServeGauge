import copy

import pytest

from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.scoring.fingerprint_collection import fingerprint_samples, sampling_request_hash
from servegauge.scoring.fingerprint_comparison import collection_cells
from servegauge.suites.s6_fingerprint import fingerprint_items
from servegauge.validation.capture import captures, endpoint
from servegauge.verdicts.config import load_policy


def four_samples(method="FP1"):
    items = fingerprint_items(
        "core", 27, {"methods": [method], "fp1_cells": 2, "fp1_samples": 8, "fp3_samples": 32}
    )
    series = []
    for seed in range(4):
        ep = endpoint(f"mock:{seed}", seed, clean=True)
        ep.metadata["fingerprint_sampling"] = {
            "independent_sessions": True,
            "matched_sampling_design": True,
            "independent_reference_captures": True,
            "independent_prompt_strata": True,
            "session_size": 2,
        }
        rows = captures(
            items,
            ScriptedBackend(seed, FaultConfig(), sampling_version="v2"),
            ep,
            f"capture-{seed}",
        )
        series.append(fingerprint_samples(items, rows, ep)[method])
    return series


def test_capture_layout_becomes_matched_independent_components_without_mutation():
    samples = four_samples()
    before = copy.deepcopy(samples)
    cells = collection_cells(samples, method="FP1", policy=load_policy())
    assert len(cells) == 2 and all(len(c.a) == 4 for c in cells)
    assert samples == before


@pytest.mark.parametrize(
    "case",
    [
        "unknown",
        "prompt_dependence",
        "wrong_n",
        "wrong_request",
        "missing",
        "duplicate",
        "wrong_design",
        "bad_session",
        "cache_design",
        "overlapping_capture",
        "reference_dependence",
        "wrong_cells",
    ],
)
def test_incomplete_or_unmatched_evidence_cannot_enter_statistical_inference(case):
    samples = four_samples()
    s = samples[0]
    if case == "unknown":
        s.details["sampling_design"]["independent_sessions"] = False
    elif case == "prompt_dependence":
        s.details["sampling_design"]["independent_prompt_strata"] = False
    elif case == "wrong_n":
        s.details["observations"].pop()
    elif case == "wrong_request":
        s.details["observations"][0]["sampling_request_hash"] = "different"
    elif case == "missing":
        s.details["observations"][0]["text"] = None
    elif case == "duplicate":
        s.details["observations"].append(s.details["observations"][0])
    elif case == "wrong_design":
        s.details["design"]["output_cap"] += 1
    elif case == "bad_session":
        s.details["sampling_design"]["session_size"] = True
    elif case == "overlapping_capture":
        s.details["observations"][0]["exchange_id"] = samples[1].details["observations"][0][
            "exchange_id"
        ]
    elif case == "reference_dependence":
        s.details["sampling_design"]["independent_reference_captures"] = False
    elif case == "wrong_cells":
        for sample in samples:
            sample.details["design"]["sizes"]["fp1_cells"] += 1
    else:
        cell = s.details["observations"][0]["cell"]
        s.details["sampling_design"]["cache_groups"] = {f"{cell}/0": "x", f"{cell}/2": "x"}
    with pytest.raises(ValueError):
        collection_cells(samples, method="FP1", policy=load_policy())


def test_missing_logprobs_or_fixed_anchors_abstain_without_filtering_rows():
    samples = four_samples("FP3")
    cells = collection_cells(samples, method="FP3-mean", policy=load_policy())
    assert len(cells) == 1 and len(cells[0].a) == 16
    samples[0].details["observations"][0]["response"]["choices"][0]["logprobs"] = None
    with pytest.raises(ValueError, match="logprob"):
        collection_cells(samples, method="FP3-mean", policy=load_policy())
    assert len(samples[0].details["observations"]) == 32


def test_known_cache_links_merge_nominal_sessions_before_inference():
    samples = four_samples()
    for source in samples:
        source.details["sampling_design"]["session_size"] = 1
        cache = {}
        rows = source.details["observations"]
        for i in range(0, len(rows), 2):
            cell, draw = rows[i]["cell"], rows[i]["draw"]
            rows[i + 1]["text"] = rows[i]["text"]
            rows[i + 1]["response"] = copy.deepcopy(rows[i]["response"])
            cache[f"{cell}/{draw}"] = cache[f"{cell}/{draw + 1}"] = f"{cell}-{draw}"
        source.details["sampling_design"]["cache_groups"] = cache
    cells = collection_cells(samples, method="FP1", policy=load_policy())
    assert len(cells) == 2 and all(len(cell.a) == 4 for cell in cells)


def test_request_aliases_do_not_hide_other_behavioral_controls():
    assert sampling_request_hash(
        {"max_tokens": 100, "reasoning_effort": "low", "model": "a"}
    ) == sampling_request_hash(
        {"max_completion_tokens": 100, "reasoning": {"effort": "low"}, "model": "b"}
    )
    first = {"reasoning_effort": "low", "reasoning": {"max_tokens": 100}}
    second = {"reasoning_effort": "low", "reasoning": {"max_tokens": 200}}
    assert sampling_request_hash(first) != sampling_request_hash(second)
    assert sampling_request_hash(
        {"max_tokens": 10, "max_completion_tokens": 20}
    ) != sampling_request_hash({"max_tokens": 10, "max_completion_tokens": 30})
