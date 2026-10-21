"""New cache attribution needs the matched fresh control and complete replay witness."""

import copy

import pytest

from servegauge.core.models import CheckResult
from servegauge.diagnose.replay_expectations import expected_context, mark_reproduced
from servegauge.diagnose.rules import diagnoses
from servegauge.suites.s1_cache import cache_contract, cache_items


def fixture():
    contract = cache_contract(cache_items("smoke", 42, {"groups": 1, "draws": 2}))
    contract.update(groups=400, draws=20)
    checks = []
    for name, state, delta in [("P03.cache", "FAIL", -0.13), ("P03.cache.fresh", "PASS", 0)]:
        checks.append(
            CheckResult(
                check_id=name,
                suite_id="S1",
                check_class="comparative",
                endpoint_id="target",
                run_id="run",
                n=400,
                estimate=0.66 + delta,
                ci_low=0.65 + delta,
                ci_high=0.67 + delta,
                reference_estimate=0.66,
                reference_tier="BASELINE",
                delta=delta,
                delta_ci_low=delta - 0.01,
                delta_ci_high=delta + 0.01,
                margin=0.03,
                p_value=0.001 if state == "FAIL" else 1,
                adjusted_p=0.002 if state == "FAIL" else 1,
                verdict=state,
                reason="known numeric witness",
                evidence=[name + "-capture"],
                details={
                    "cache_contract": copy.deepcopy(contract),
                    "request_signature": name + "-request-signature",
                    "input_hashes": {str(i): [name + "-request"] * 20 for i in range(400)},
                },
            )
        )
    return checks


def cache(checks):
    return [d for d in diagnoses(checks) if d.code == "CACHE_ANOMALY"]


def test_versioned_cache_code_retains_the_passing_fresh_control_without_confidence_inflation():
    checks = fixture()
    found = cache(checks)
    assert len(found) == 1 and found[0].failure_class == "ROUTING_OR_INFRA"
    assert found[0].check_ids == ["P03.cache", "P03.cache.fresh"]
    assert found[0].confidence == "low"
    context = expected_context(found[0], checks)
    assert context["expected"] == {"P03.cache": "FAIL", "P03.cache.fresh": "PASS"}
    assert mark_reproduced(checks, context)
    assert cache(checks)[0].confidence == "medium"


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "failed",
        "uncertain",
        "other-run",
        "other-endpoint",
        "other-tier",
        "other-contract",
        "unknown-contract",
        "duplicate",
        "missing-interval",
        "missing-layout",
        "different-groups",
        "shared-capture",
    ],
)
def test_missing_failed_or_incompatible_fresh_control_withholds_new_cache_code(fault):
    checks = fixture()
    fresh = checks[1]
    if fault == "missing":
        checks.pop()
    elif fault == "failed":
        fresh.verdict = "FAIL"
    elif fault == "uncertain":
        fresh.verdict = "INCONCLUSIVE"
    elif fault == "other-run":
        fresh.run_id = "another"
    elif fault == "other-endpoint":
        fresh.endpoint_id = "another"
    elif fault == "other-tier":
        fresh.reference_tier = "R3"
    elif fault == "other-contract":
        fresh.details["cache_contract"]["draws"] = 40
    elif fault == "unknown-contract":
        for c in checks:
            c.details["cache_contract"]["version"] = "unknown"
    elif fault == "duplicate":
        checks.append(fresh.model_copy(deep=True))
    elif fault == "missing-interval":
        fresh.delta_ci_low = None
    elif fault == "missing-layout":
        fresh.details.pop("request_signature")
    elif fault == "different-groups":
        fresh.details["input_hashes"].pop("0")
    else:
        fresh.evidence = checks[0].evidence
    assert not cache(checks)


def test_unversioned_cache_lab_interpretation_is_retained_for_historical_evidence():
    checks = fixture()[:1]
    checks[0].details.pop("cache_contract")
    assert len(cache(checks)) == 1


def test_changed_control_rejects_replay_while_the_repeated_check_still_fails():
    checks = fixture()
    context = expected_context(cache(checks)[0], checks)
    checks[1].verdict = "FAIL"
    checks[1].delta = -0.4
    assert not mark_reproduced(checks, context) and not cache(checks)
