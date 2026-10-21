import copy
import socket

import numpy as np
import pytest

from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.scoring.degeneration import degeneration
from servegauge.stats.bootstrap import clustered_bootstrap
from servegauge.validation.common import offline_only
from servegauge.validation.fault_matrix import clean_pass_rate, run_replicate


def test_offline_lab_refuses_any_socket():
    with offline_only(), pytest.raises(RuntimeError, match="offline_network_forbidden"):
        socket.socket()


def test_clean_gate_excludes_only_wholly_inapplicable_checks():
    irrelevant = {"check_class": "comparative", "verdict": "NOT_APPLICABLE"}
    passed = {"check_class": "comparative", "verdict": "PASS"}
    assert clean_pass_rate([irrelevant, irrelevant]) is None
    assert clean_pass_rate([passed, irrelevant])["estimate"] == 0.5
    assert clean_pass_rate([passed, {**irrelevant, "verdict": "NOT_SUPPORTED"}])["estimate"] == 0.5


def test_batched_item_bootstrap_matches_independent_original_algorithm():
    data = np.arange(15, dtype=float).reshape(5, 3)
    rng = np.random.default_rng(17)
    values = [float(np.mean(data[rng.integers(0, len(data), size=len(data))])) for _ in range(2000)]
    result = clustered_bootstrap(data, seed=17)
    assert (result.low, result.high) == tuple(np.quantile(values, [0.025, 0.975]))


def test_fault_study_routes_real_captures_through_scoring_replay_and_diagnosis():
    with offline_only():
        clean = run_replicate("P01", FaultConfig(), 11)
        bad = run_replicate("P01", FaultConfig(ignore_max_tokens=True), 12)
    assert clean["checks"][0]["verdict"] == "PASS"
    assert bad["checks"][0]["verdict"] == "FAIL"
    assert bad["checks"][0]["details"]["replay_confirmed"]
    assert {d["code"] for d in bad["diagnoses"]} == {"PARAM_IGNORED"}
    assert "ignore_max_tokens" not in str(bad["checks"])


def test_pure_caches_cannot_mutate_inputs_or_later_scores():
    body = {
        "messages": [{"role": "user", "content": 'TW_TASK: {"type":"echo","text":"hello"}'}],
        "max_tokens": 32,
    }
    original = copy.deepcopy(body)
    for faults in (FaultConfig(), FaultConfig(truncate_prompt_head_over=4)):
        ScriptedBackend(1, faults).respond(body)
        assert body == original
    first = degeneration("abcabcabc", ngram=2)
    first["compression_ratio"] = -100
    assert degeneration("abcabcabc", ngram=2)["compression_ratio"] >= 0
