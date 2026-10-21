"""Study scores reconstruct observed wire inputs, with no intervention-label input."""

import copy

import numpy as np
import pytest

from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.scoring.fingerprint_collection import fingerprint_samples
from servegauge.scoring.fingerprint_comparison import collection_cells
from servegauge.scoring.gpu_study import score_quartet
from servegauge.selfhost.study_collection import study_fingerprint_items
from servegauge.validation.capture import captures, endpoint
from servegauge.verdicts.config import load_policy


def quartet():
    items = study_fingerprint_items(
        "smoke",
        91,
        {
            "collection_recipe": "qwen3-study-collection-v1",
            "methods": ["FP1", "FP2", "FP3"],
            "fp1_cells": 2,
            "fp2_cells": 2,
            "fp1_samples": 4,
            "fp2_samples": 4,
            "fp3_samples": 4,
        },
        profile_id="qwen3-0.6b-study",
    )
    result = []
    for seed in range(4):
        target = endpoint(f"mock:{seed}", seed, clean=True)
        target.claimed_profile = "qwen3-0.6b-study"
        target.metadata["fingerprint_sampling"] = {
            "independent_sessions": True,
            "matched_sampling_design": True,
            "independent_reference_captures": True,
            "independent_prompt_strata": True,
            "session_size": 1,
        }
        records = captures(
            items,
            ScriptedBackend(seed, FaultConfig(), sampling_version="v2"),
            target,
            f"study-{seed}",
        )
        result.append(fingerprint_samples(items, records, target))
    return result


def test_all_text_and_logprob_scores_derive_from_raw_without_labels_or_mutation():
    samples = quartet()
    before = copy.deepcopy(samples)
    result = score_quartet(samples, policy=load_policy(), seed=73)
    assert set(result) == {"FP1", "FP2", "FP3-mean", "FP3-dispersion", "FP4"}
    for method in ["FP1", "FP2", "FP3-mean", "FP3-dispersion"]:
        row = result[method]
        assert row["estimate"] is not None and row["n_components"] > 0
        assert row["ci_low"] <= row["ci_high"]
        base = "FP3" if method.startswith("FP3") else method
        cells = collection_cells([s[base] for s in samples], method=method, policy=load_policy())
        expected = float(np.mean([c.distance(c.a, c.b) - c.distance(c.c, c.d) for c in cells]))
        assert row["estimate"] == pytest.approx(expected)
        assert row["reference_tier"] == "BASELINE" and row["verdict"] == "INCONCLUSIVE"
    assert result["FP4"]["estimate"] is None
    assert samples == before


def test_missing_logprobs_do_not_erase_usable_text_or_impute_a_zero():
    samples = quartet()
    samples[0]["FP3"].details["observations"][0]["response"]["choices"][0]["logprobs"] = None
    result = score_quartet(samples, policy=load_policy(), seed=73)
    assert result["FP1"]["estimate"] is not None
    assert result["FP2"]["estimate"] is not None
    for method in ["FP3-mean", "FP3-dispersion"]:
        assert result[method]["estimate"] is None
        assert result[method]["reason"] == "incomplete_fixed_anchor_logprob_evidence"


@pytest.mark.parametrize("change", ["censored", "reuse", "independence", "missing-side"])
def test_invalid_components_abstain_with_actual_counts(change):
    samples = quartet()
    if change == "censored":
        samples[0]["FP1"].details["observations"][0]["reason"] = "censored_response"
    elif change == "reuse":
        samples[0]["FP1"].details["observations"][0]["exchange_id"] = samples[1]["FP1"].details[
            "observations"
        ][0]["exchange_id"]
    elif change == "independence":
        samples[0]["FP1"].details["sampling_design"]["independent_sessions"] = False
    else:
        del samples[3]["FP1"]
    row = score_quartet(samples, policy=load_policy(), seed=73)["FP1"]
    assert row["estimate"] is None and row["verdict"] == "INCONCLUSIVE"
    assert len(row["capture_counts"]) == 4


def test_fp4_uses_the_pinned_embedding_metric_and_missing_runtime_abstains():
    from unit.test_embedding_capture import FixtureEncoder, embedding_source

    source = embedding_source("gpu-study-fixture", [31, 32, 33, 34])
    samples = list(source.fingerprints.values())
    encoder = FixtureEncoder(samples[0]["FP4"].details["design"])
    result = score_quartet(samples, policy=load_policy(), seed=73, encoder=encoder)
    assert result["FP4"]["estimate"] is not None
    assert result["FP4"]["ci_low"] <= result["FP4"]["ci_high"]
    assert result["FP4"]["reference_tier"] == "BASELINE"
    assert result["FP4"]["verdict"] == "INCONCLUSIVE"
    missing = score_quartet(samples, policy=load_policy(), seed=73)
    assert missing["FP4"]["estimate"] is None
    assert missing["FP4"]["reason"] == "matching_pinned_embedding_runtime_required"
