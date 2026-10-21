import copy
import itertools
import math

import numpy as np
import pytest
from typer.testing import CliRunner

from servegauge.cli import app
from servegauge.scoring.fingerprint_analysis import analyze_evidence, group_evidence
from servegauge.scoring.fingerprint_evidence import FingerprintEvidence, prepare_evidence
from servegauge.stats.fingerprint_blocks import block_comparisons
from servegauge.verdicts.config import load_policy


def dataset(groups=30, repeats=2):
    def rows(side):
        result = []
        for i in range(groups):
            for draw in range(repeats):
                p = 0.4 + 0.1 * ((i + draw + side) % 3)
                token = "red" if i % 2 else "blue"
                logs = [
                    {"token": "red", "logprob": math.log(p)},
                    {"token": "blue", "logprob": math.log(1 - p)},
                ]
                result.append(
                    {
                        "session_id": str(i),
                        "draw": draw,
                        "response": {
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "message": {"content": token},
                                    "logprobs": {
                                        "content": [
                                            {
                                                "token": token,
                                                "logprob": next(
                                                    r["logprob"]
                                                    for r in logs
                                                    if r["token"] == token
                                                ),
                                                "top_logprobs": logs,
                                            }
                                        ]
                                    },
                                }
                            ]
                        },
                    }
                )
        return result

    return FingerprintEvidence(
        probe_id="color-v3",
        reference_tier="MOCK",
        independent_sessions=True,
        matched_sampling_design=True,
        anchor_tokens=["red", "blue"],
        anchor_source="fixed-probe-configuration",
        target=rows(0),
        reference=rows(1),
    )


def analyze(data):
    return analyze_evidence(data, policy=load_policy(), minimum_groups=30, minimum_anchor_tokens=2)


def test_exact_permutation_swaps_whole_sessions_matches_brute_force():
    data = dataset(4, 3)
    prepared = prepare_evidence(data, minimum_anchor_tokens=2, tolerance=0.001)
    grouping = group_evidence(data, prepared)
    found = block_comparisons(
        prepared, grouping.pairs, anchors=data.anchor_tokens, seed=12, resamples=32
    )
    a = np.array([[r[t] for t in data.anchor_tokens] for r in prepared.first_logs])
    b = np.array([[r[t] for t in data.anchor_tokens] for r in prepared.second_logs])
    center = (
        np.mean([np.mean(a[x], axis=0) + np.mean(b[y], axis=0) for x, y in grouping.pairs], axis=0)
        / 2
    )

    def metric(x, y):
        return abs(
            np.mean(
                [
                    np.mean(np.sum((x[first] - center) ** 2, axis=1))
                    - np.mean(np.sum((y[second] - center) ** 2, axis=1))
                    for first, second in grouping.pairs
                ]
            )
        )

    observed = metric(a, b)
    null = []
    for signs in itertools.product([False, True], repeat=4):
        first, second = a.copy(), b.copy()
        for swap, (x, y) in zip(signs, grouping.pairs, strict=True):
            if swap:
                first[x], second[y] = b[y], a[x]
        null.append(metric(first, second))
    result = found["FP3-dispersion"].test
    assert result.n == 4 and result.permutations == 16 and result.exact
    assert result.statistic == pytest.approx(observed)
    assert result.p_value == sum(v >= observed - 1e-12 for v in null) / 16


def test_cache_lineage_merges_sessions_and_reduces_inferential_n():
    data = dataset(60, 1)
    for rows in (data.target, data.reference):
        for i in range(0, 60, 2):
            rows[i].cache_group = rows[i + 1].cache_group = str(i)
            rows[i + 1].response = copy.deepcopy(rows[i].response)
    result = analyze(data)
    assert result["quality"]["nominal_sessions"] == 60
    assert result["quality"]["effective_groups"] == 30
    assert all(c["n"] == 30 for c in result["checks"])
    # Equal text alone never merges sessions.
    data = dataset(60, 1)
    assert analyze(data)["quality"]["effective_groups"] == 60


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("unknown", "independent_sessions_required"),
        ("design", "matched_sampling_design_required"),
        ("small", "insufficient_independent_groups"),
        ("duplicate", "duplicate_sample_identity"),
        ("unpaired", "unmatched_sample_identities"),
        ("cache_mismatch", "unmatched_cache_design"),
        ("cache_invalid", "inconsistent_cache_lineage"),
    ],
)
def test_unqualified_data_is_inconclusive_without_p_values(kind, reason):
    data = dataset(30, 2)
    if kind == "unknown":
        data.independent_sessions = False
    elif kind == "design":
        data.matched_sampling_design = False
    elif kind == "small":
        data = dataset(3, 40)
    elif kind == "duplicate":
        data.target.append(data.target[0])
    elif kind == "unpaired":
        data.target.pop()
    elif kind in {"cache_mismatch", "cache_invalid"}:
        data.target[0].cache_group = data.target[1].cache_group = "cached"
        if kind == "cache_mismatch":
            data.target[1].response = copy.deepcopy(data.target[0].response)
    result = analyze(data)
    assert result["status"] == "INCONCLUSIVE"
    assert all(
        c["verdict"] == "INCONCLUSIVE" and c["reason"] == reason and c["p_value"] is None
        for c in result["checks"]
    )


def test_missing_logprobs_retains_text_and_does_not_shrink_request_count():
    data = dataset()
    data.target[0].response["choices"][0]["logprobs"] = None
    result = analyze(data)
    checks = {r["check_id"]: r for r in result["checks"]}
    assert result["status"] == "INCONCLUSIVE" and result["quality"]["target_requests"] == 60
    assert checks["FP1"]["verdict"] == "INFO"
    assert checks["FP3-mean"]["verdict"] == checks["FP3-dispersion"]["verdict"] == "INCONCLUSIVE"
    assert all(r["verdict"] not in {"PASS", "FAIL"} for r in result["checks"])


def test_offline_cli_exit_codes_and_safe_output(tmp_path):
    path = tmp_path / "evidence.json"
    data = dataset()
    path.write_text(data.model_dump_json())
    output = tmp_path / "analysis.json"
    runner = CliRunner()
    result = runner.invoke(app, ["fingerprint", "analyze", str(path), "--out", str(output)])
    assert result.exit_code == 0, result.output
    original = path.read_bytes()
    refused = runner.invoke(app, ["fingerprint", "analyze", str(path), "--out", str(path)])
    assert refused.exit_code == 3 and path.read_bytes() == original
    assert '"red"' not in output.read_text() and '"blue"' not in output.read_text()
    data.independent_sessions = False
    path.write_text(data.model_dump_json())
    result = runner.invoke(app, ["fingerprint", "analyze", str(path)])
    assert result.exit_code == 2 and "INCONCLUSIVE" in result.output
    path.write_text('{"unexpected":"private-value"}')
    result = runner.invoke(app, ["fingerprint", "analyze", str(path)])
    assert result.exit_code == 3 and "private-value" not in result.output


def test_changing_nonanchor_topk_is_usable_without_adaptive_token_selection():
    data = dataset()
    for rows in (data.target, data.reference):
        for i, sample in enumerate(rows):
            row = sample.response["choices"][0]["logprobs"]["content"][0]
            for token in row["top_logprobs"]:
                token["logprob"] += math.log(0.9)
            row["logprob"] += math.log(0.9)
            row["top_logprobs"].append(
                {"token": "gold" if i % 2 else "green", "logprob": math.log(0.05)}
            )
    result = analyze(data)
    assert result["status"] == "INFO"
    assert all(c["verdict"] == "INFO" for c in result["checks"])


def test_one_cache_component_cannot_be_counted_as_many_independent_sessions():
    data = dataset(60, 1)
    for rows in (data.target, data.reference):
        for sample in rows:
            sample.cache_group = "one-source"
            sample.response = copy.deepcopy(rows[0].response)
    result = analyze(data)
    assert result["quality"]["effective_groups"] == 1
    assert all(c["reason"] == "insufficient_independent_groups" for c in result["checks"])


def test_cache_hits_may_omit_logprobs_without_invalidating_consistent_text():
    data = dataset()
    for rows in (data.target, data.reference):
        for i in range(0, len(rows), 2):
            rows[i].cache_group = rows[i + 1].cache_group = str(i)
            rows[i + 1].response = copy.deepcopy(rows[i].response)
    data.target[1].response["choices"][0]["logprobs"] = None
    result = analyze(data)
    checks = {r["check_id"]: r for r in result["checks"]}
    assert checks["FP1"]["verdict"] == "INFO"
    assert checks["FP3-dispersion"]["verdict"] == "INCONCLUSIVE"
