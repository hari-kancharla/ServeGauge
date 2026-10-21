"""Root-procedure health outcomes preserve retries, inputs and missing evidence."""

import copy

import pytest
from tests_support import exchange

from servegauge.core.models import Attempt, Timings
from servegauge.diagnose.rules import diagnoses
from servegauge.profiles import load_profile
from servegauge.refs.pack import build_pack, load_pack, save_pack
from servegauge.runner.endpoints import resolve_endpoint
from servegauge.runner.evaluate import collect, evaluate, reference_source
from servegauge.runner.plan import RunPlan
from servegauge.scoring.operational_health import health_samples
from servegauge.suites.s8_operations import HEALTH_CHECKS, operational_items
from servegauge.verdicts.config import load_policy


def fixture(n=2):
    endpoint = resolve_endpoint("http://127.0.0.1:8011/v1", "mock", "gpt-oss-120b")
    plan = RunPlan(
        plan_id="health",
        profile="gpt-oss-120b",
        level="smoke",
        seed=7,
        reference={"tier": "MOCK"},
        endpoints=[endpoint],
        suites=["S8"],
        max_usd=0,
        retries=1,
    )
    items = operational_items("smoke", 7, {"items": n, "agent_tasks": 0})
    records = []
    for index, item in enumerate(items):
        response = {
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"content": "ready"}}]
        }
        attempt = Attempt(
            n=1, status=200, response=response, timings=Timings(start_unix=1, total_ms=10)
        )
        records.append(
            exchange(
                run_id="run",
                endpoint_id=endpoint.id,
                item_id=item.item_id,
                exchange_id=f"health-{index}",
                request={
                    "model": "mock",
                    "messages": item.payload["messages"],
                    "stream": item.payload["stream"],
                    "max_tokens": 256,
                },
                response=response,
                attempts=[attempt],
            )
        )
    return plan, items, records


def test_complete_root_procedures_are_binary_matched_units():
    plan, items, records = fixture()
    samples = health_samples(plan, items, records, plan.endpoints[0].id, "run")
    assert set(samples) == set(HEALTH_CHECKS)
    for sample in samples.values():
        assert sample.binary and sample.check_class == "comparative" and sample.status is None
        assert sample.values == {"S8:7": 1.0, "S8:8": 1.0}
        assert set(sample.details["input_hashes"]) == set(sample.values)
        assert len(sample.evidence) == 4


def test_recovered_retry_changes_attempt_health_without_adding_independent_units():
    plan, items, records = fixture()
    failed = records[0].attempts[0].model_copy(deep=True)
    failed.status = 503
    failed.response = {"error": {"code": "unavailable"}}
    records[0].attempts.insert(0, failed)
    records[0].attempts[1].n = 2
    samples = health_samples(plan, items, records, plan.endpoints[0].id, "run")
    assert samples["S8.health.requests"].values == {"S8:7": 1.0, "S8:8": 1.0}
    assert samples["S8.health.attempts"].values == {"S8:7": 0.0, "S8:8": 1.0}
    assert len(samples["S8.health.attempts"].values) == 2


@pytest.mark.parametrize("fault", ["missing", "duplicate", "unsent", "redacted", "attempt-gap"])
def test_incomplete_or_ambiguous_procedures_abstain(fault):
    plan, items, records = fixture()
    if fault == "missing":
        records.pop()
    elif fault == "duplicate":
        duplicate = records[0].model_copy(deep=True)
        duplicate.exchange_id += "-duplicate"
        records.append(duplicate)
    elif fault == "unsent":
        records[0].metadata["request_not_sent"] = True
    elif fault == "redacted":
        records[0].request["messages"][0]["content"] = "[REDACTED]"
    else:
        records[0].attempts[0].n = 2
    samples = health_samples(plan, items, records, plan.endpoints[0].id, "run")
    assert all(s.status == "INCONCLUSIVE" and not s.values for s in samples.values())


def test_known_empty_body_is_a_failure_in_each_relevant_procedure():
    plan, items, records = fixture()
    r = records[0]
    r.response = None
    r.response_text = ""
    r.error = "invalid_response_json"
    r.attempts[0].response = None
    r.attempts[0].response_text = ""
    samples = health_samples(plan, items, records, plan.endpoints[0].id, "run")
    assert all(s.values["S8:7"] == 0 for s in samples.values())


def test_rejection_error_does_not_become_an_observable_nonempty_response():
    plan, items, records = fixture()
    records[0].status = records[0].attempts[0].status = 503
    samples = health_samples(plan, items, records, plan.endpoints[0].id, "run")
    assert samples["S8.health.requests"].values["S8:7"] == 0
    assert samples["S8.health.nonempty"].status == "INCONCLUSIVE"


def test_intentional_negative_probes_and_confirmations_cannot_create_health_alarms():
    plan, items, records = fixture()
    original = health_samples(plan, items, records, plan.endpoints[0].id, "run")
    item = items[0].model_copy(deep=True)
    item.item_id = "negative"
    item.suite_id = "S1"
    item.expected["check_id"] = "P12"
    negative = records[0].model_copy(deep=True)
    negative.item_id = item.item_id
    negative.exchange_id = "negative"
    negative.status = 400
    repeated = [r.model_copy(deep=True) for r in records]
    for record in repeated:
        record.run_id = "run-confirmation"
        record.exchange_id += "-repeat"
        record.status = 503
    actual = health_samples(
        plan, [*items, item], [*records, negative, *repeated], plan.endpoints[0].id, "run"
    )
    assert actual == original


def test_actual_root_input_hashes_bind_caps_but_allow_endpoint_model_identity():
    plan, items, records = fixture()
    original = health_samples(plan, items, records, plan.endpoints[0].id, "run")
    changed = copy.deepcopy(records)
    changed[0].request["model"] = "other-native-model"
    same = health_samples(plan, items, changed, plan.endpoints[0].id, "run")
    assert original == same
    changed[0].request["max_tokens"] = 128
    different = health_samples(plan, items, changed, plan.endpoints[0].id, "run")
    assert (
        original["S8.health.requests"].details["input_hashes"]
        != different["S8.health.requests"].details["input_hashes"]
    )


def test_legacy_observation_only_captures_do_not_gain_new_diagnoses():
    plan, items, records = fixture()
    for item in items:
        item.expected.pop("operational_health_contract")
    assert health_samples(plan, items, records, plan.endpoints[0].id, "run") == {}


@pytest.mark.parametrize("fault", [False, True])
def test_health_baseline_pack_and_input_guards_share_existing_family_decisions(tmp_path, fault):
    plan, items, records = fixture(200)
    target = plan.endpoints[0]
    baseline = target.model_copy(update={"id": "baseline"}, deep=True)
    plan.endpoints.append(baseline)
    base = [r.model_copy(deep=True) for r in records]
    for record in base:
        record.endpoint_id = baseline.id
        record.exchange_id += "-baseline"
    if fault:
        for index, record in enumerate(records):
            if index % 20 == 0:
                record.response = None
                record.response_text = ""
                record.error = "invalid_response_json"
                record.attempts[0].response = None
                record.attempts[0].response_text = ""
    all_records = [*records, *base]
    policy = load_policy()
    samples = collect(plan, items, all_records, load_profile(plan.profile), policy)
    results = evaluate(
        plan,
        "run",
        items,
        samples,
        policy,
        target_ids=[target.id],
        baseline_id=baseline.id,
        operational_records=all_records,
    )
    health = [r for r in results if r.check_id in HEALTH_CHECKS]
    assert len(health) == 3
    assert all(r.check_class == "report_only" and r.verdict == "INFO" for r in health)
    assert all(r.n == 200 and r.reference_tier == "BASELINE" and r.margin == 0.03 for r in health)
    assert all(r.details["operational_alarm"] == fault for r in health)
    assert all(r.details["comparison_verdict"] == ("FAIL" if fault else "PASS") for r in health)
    assert bool([d for d in diagnoses(results) if d.code == "INFRA_UNRELIABLE"]) == fault
    source = reference_source(
        plan,
        "run",
        items,
        samples,
        policy,
        {"item_set_hashes": {"test": "bound"}},
        baseline_id=baseline.id,
    )
    pack = load_pack(str(save_pack(build_pack(source, "BASELINE", seed=7), tmp_path)))
    packed = evaluate(
        plan,
        "run",
        items,
        samples,
        policy,
        target_ids=[target.id],
        reference_pack=pack,
        operational_records=all_records,
    )
    assert [r.model_dump() for r in packed if r.check_id in HEALTH_CHECKS] == [
        r.model_dump() for r in health
    ]
    records[0].request["max_tokens"] = 128
    mismatch = evaluate(
        plan,
        "run",
        items,
        samples,
        policy,
        target_ids=[target.id],
        reference_pack=pack,
        operational_records=all_records,
    )
    assert all(
        r.verdict == "INCONCLUSIVE" and not r.details["operational_alarm"]
        for r in mismatch
        if r.check_id in HEALTH_CHECKS
    )


@pytest.mark.parametrize("tier", ["MOCK", "R3"])
def test_fractional_pooled_health_labels_need_separate_inference_qualification(tier):
    plan, items, records = fixture(200)
    target = plan.endpoints[0]
    peers = 3 if tier == "MOCK" else 4
    for index in range(peers):
        peer = target.model_copy(deep=True)
        peer.id = f"peer-{index}"
        peer.metadata["mock_declaration"] = {"seed": index + 1, "clean": True}
        plan.endpoints.append(peer)
        copies = [r.model_copy(deep=True) for r in records[:400]]
        for record in copies:
            record.endpoint_id = peer.id
            record.exchange_id += f"-{peer.id}"
        if index < (1 if tier == "MOCK" else 2):
            record = copies[0]
            record.response = None
            record.response_text = ""
            record.error = "invalid_response_json"
            record.attempts[0].response = None
        records.extend(copies)
    policy = load_policy()
    samples = collect(plan, items, records, load_profile(plan.profile), policy)
    for endpoint in plan.endpoints:
        samples[endpoint.id].update(health_samples(plan, items, records, endpoint.id, "run"))
    source = reference_source(
        plan, "run", items, samples, policy, {"item_set_hashes": {"test": "bound"}}
    )
    pack = build_pack(source, tier, seed=7)
    checks = evaluate(
        plan,
        "run",
        items,
        samples,
        policy,
        target_ids=[target.id],
        reference_pack=pack,
        operational_records=records,
    )
    health = [c for c in checks if c.check_id in HEALTH_CHECKS]
    assert len(health) == 3
    assert all(c.verdict == "INCONCLUSIVE" and not c.details["operational_alarm"] for c in health)
    assert all("fractional" in c.reason for c in health)
