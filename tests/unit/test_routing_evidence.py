"""Provider identity needs a pin, matching generation ids and whole-root support."""

import copy

import pytest
from tests_support import exchange

from servegauge.core.models import Attempt, EndpointSpec, Item, OpenRouterPin, Timings
from servegauge.diagnose.rules import diagnoses
from servegauge.runner.evaluate import evaluate
from servegauge.runner.plan import RunPlan
from servegauge.scoring.routing import ROUTING_CHECK, ROUTING_CONTRACT, routing_samples
from servegauge.verdicts.config import load_policy


def fixture(n=2):
    endpoint = EndpointSpec(
        id="route:fixture",
        kind="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="example/model",
        claimed_profile="gpt-oss-120b",
        openrouter_pin=OpenRouterPin(tag="provider-a/fp4/us"),
        labels={"provider_name": "Provider A"},
    )
    items, records = [], []
    for index in range(n):
        for variant in ("json", "stream"):
            identifier = f"route-{index}-{variant}"
            items.append(
                Item(
                    item_id=identifier,
                    suite_id="S1",
                    version="fixture-v1",
                    payload={},
                    expected={"check_id": "P05", "trial": str(index), "samples": 1},
                )
            )
            body = {
                "id": f"gen-{index}-{variant}",
                "choices": [{"message": {"content": "ready"}, "finish_reason": "stop"}],
            }
            metadata = {
                "generation_record_status": "available",
                "generation_record": {"id": body["id"], "provider_name": "Provider A"},
                "candidate_fallback": True,
                "fallback_schema_verified": False,
            }
            records.append(
                exchange(
                    run_id="run",
                    endpoint_id=endpoint.id,
                    item_id=identifier,
                    exchange_id=identifier,
                    request={
                        "model": endpoint.model,
                        "messages": [{"role": "user", "content": "ready"}],
                        "provider": {
                            "only": [endpoint.openrouter_pin.tag],
                            "allow_fallbacks": False,
                        },
                    },
                    response=body,
                    metadata=metadata,
                    attempts=[
                        Attempt(
                            n=1,
                            status=200,
                            response=body,
                            metadata=copy.deepcopy(metadata),
                            timings=Timings(start_unix=1, total_ms=2),
                        )
                    ],
                )
            )
    return endpoint, items, records


def sample(endpoint, items, records):
    return routing_samples(items, records, endpoint, run_id="run", contract=ROUTING_CONTRACT)


def result(endpoint, items, records, confirmed=False):
    value = sample(endpoint, items, records)
    plan = RunPlan(
        plan_id="routing",
        profile=endpoint.claimed_profile,
        level="core",
        seed=9,
        reference={"tier": "BASELINE"},
        endpoints=[endpoint],
        suites=["S1"],
        max_usd=0,
    )
    return next(
        c
        for c in evaluate(
            plan,
            "run",
            items,
            {endpoint.id: {ROUTING_CHECK: value}},
            load_policy(),
            target_ids=[endpoint.id],
            replay_confirmed={(endpoint.id, ROUTING_CHECK): confirmed},
        )
        if c.check_id == ROUTING_CHECK
    )


def mismatch(record):
    record.metadata["generation_record"]["provider_name"] = "Provider B"
    record.attempts[-1].metadata["generation_record"]["provider_name"] = "Provider B"
    record.error = "ROUTING_MISMATCH"
    record.metadata["exclude_from_scoring"] = True


def test_provider_routing_legacy_and_non_openrouter_contract_guards():
    endpoint, items, records = fixture()
    assert routing_samples(items, records, endpoint, run_id="run") is None
    with pytest.raises(ValueError, match="routing_contract"):
        routing_samples(items, records, endpoint, run_id="run", contract={"version": "unknown"})
    endpoint.kind = "openai_compat"
    assert sample(endpoint, items, records).status == "NOT_APPLICABLE"


def test_variants_and_repeated_captures_are_one_root_not_independent_failures():
    endpoint, items, records = fixture(1)
    for record in records:
        mismatch(record)
    value = sample(endpoint, items, records)
    assert value.values == {"S1:P05:0": 1.0}
    assert value.replay_candidates == [[i.item_id for i in items]]
    check = result(endpoint, items, records, confirmed=True)
    assert check.verdict == "INCONCLUSIVE" and check.violations == 1
    assert not check.details["operational_alarm"]


def test_mismatches_need_existing_k_fail_and_actual_reserved_reproduction():
    endpoint, items, records = fixture(2)
    mismatch(records[0])
    mismatch(records[2])
    unconfirmed = result(endpoint, items, records)
    assert unconfirmed.verdict == "INCONCLUSIVE"
    confirmed = result(endpoint, items, records, confirmed=True)
    assert confirmed.verdict == "INFO" and confirmed.check_class == "report_only"
    assert confirmed.details["comparison_verdict"] == "FAIL"
    assert confirmed.details["operational_alarm"] and confirmed.violations == 2
    assert confirmed.n == 2 and confirmed.margin == 0.01 and confirmed.reference_estimate == 0
    found = diagnoses([confirmed])
    assert len(found) == 1 and found[0].code == "ROUTING_INSTABILITY"
    assert found[0].failure_class == "ROUTING_OR_INFRA" and found[0].evidence


def test_provider_name_does_not_verify_exact_tag_or_candidate_fallback():
    endpoint, items, records = fixture(400)
    check = result(endpoint, items, records)
    assert check.n == 400 and check.violations == 0
    assert check.details["comparison_verdict"] == "PASS" and check.verdict == "INFO"
    assert not check.details["operational_alarm"]
    assert not check.details["tag_verified"] and not check.details["fallback_qualified"]
    assert not diagnoses([check])


@pytest.mark.parametrize(
    "fault",
    [
        "missing-meta",
        "wrong-generation-id",
        "wrong-attempt-id",
        "malformed-generation-id",
        "changed-meta",
        "wrong-pin",
        "fallback-enabled",
        "missing-provider",
        "redacted",
        "unsent",
        "http-error",
        "duplicate",
        "missing-capture",
    ],
)
def test_unknown_or_ambiguous_unit_cannot_become_a_clean_pass(fault):
    endpoint, items, records = fixture(400)
    record = records[0]
    if fault == "missing-meta":
        record.metadata.pop("generation_record")
    elif fault == "wrong-generation-id":
        record.metadata["generation_record"]["id"] = "gen-other"
        record.attempts[-1].metadata = copy.deepcopy(record.metadata)
    elif fault == "wrong-attempt-id":
        record.attempts[-1].response["id"] = "gen-other"
    elif fault == "malformed-generation-id":
        record.response["id"] = ["invalid"]
    elif fault == "changed-meta":
        record.attempts[-1].metadata["generation_record"]["provider_name"] = "Provider B"
    elif fault == "wrong-pin":
        record.request["provider"]["only"] = ["provider-b"]
    elif fault == "fallback-enabled":
        record.request["provider"]["allow_fallbacks"] = True
    elif fault == "missing-provider":
        record.metadata["generation_record"].pop("provider_name")
        record.attempts[-1].metadata = copy.deepcopy(record.metadata)
    elif fault == "redacted":
        record.metadata["generation_record"]["provider_name"] = "[REDACTED]"
        record.attempts[-1].metadata = copy.deepcopy(record.metadata)
    elif fault == "unsent":
        record.metadata["request_not_sent"] = True
    elif fault == "http-error":
        record.status = 503
    elif fault == "duplicate":
        duplicate = record.model_copy(deep=True)
        duplicate.exchange_id += "-duplicate"
        records.append(duplicate)
    else:
        records.pop(0)
    check = result(endpoint, items, records)
    assert check.verdict == "INCONCLUSIVE" and not check.details["operational_alarm"]
    assert check.n == 399 and check.details["incomplete_trials"]


def test_partial_observability_preserves_confirmed_positive_evidence_but_not_pass():
    endpoint, items, records = fixture(3)
    mismatch(records[0])
    mismatch(records[2])
    records[-1].metadata.pop("generation_record")
    check = result(endpoint, items, records, confirmed=True)
    assert check.n == 2 and check.violations == 2 and check.details["incomplete_trials"]
    assert check.details["operational_alarm"]


def test_capability_and_primary_records_keep_explicit_run_lineage():
    endpoint, items, records = fixture(2)
    for item in items[:2]:
        item.suite_id = "S0"
    for record in records[:2]:
        record.run_id = "run-capabilities"
        record.suite_id = "S0"
    ignored = records[2].model_copy(deep=True)
    ignored.run_id = "run-replay-confirmation"
    ignored.exchange_id += "-replay"
    mismatch(ignored)
    value = routing_samples(
        items,
        [*records, ignored],
        endpoint,
        run_id="run",
        capability_run_id="run-capabilities",
        contract=ROUTING_CONTRACT,
    )
    assert (
        value.values == {"S0:P05:0": 0.0, "S1:P05:1": 0.0}
        and not value.details["incomplete_trials"]
    )


@pytest.mark.parametrize("identity", ["exchange", "generation", "suite"])
def test_reused_capture_identity_cannot_manufacture_independent_support(identity):
    endpoint, items, records = fixture(3)
    for record in records:
        mismatch(record)
    if identity == "exchange":
        records[2].exchange_id = records[0].exchange_id
    elif identity == "generation":
        identifier = records[0].response["id"]
        records[2].response["id"] = identifier
        records[2].attempts[-1].response["id"] = identifier
        records[2].metadata["generation_record"]["id"] = identifier
        records[2].attempts[-1].metadata["generation_record"]["id"] = identifier
    else:
        records[2].suite_id = "S0"
    value = sample(endpoint, items, records)
    assert value.details["incomplete_trials"]
    assert len(value.values) == (2 if identity == "suite" else 1)


def test_routing_flags_alone_cannot_create_a_provider_contradiction():
    endpoint, items, records = fixture(400)
    for record in records:
        record.error = "ROUTING_MISMATCH"
        record.metadata["exclude_from_scoring"] = True
        record.metadata["routing_status"] = "ROUTING_MISMATCH"
    check = result(endpoint, items, records, confirmed=True)
    assert check.violations == 0 and not check.details["operational_alarm"]


def test_entirely_unsent_procedures_are_separate_from_unknown_sent_identity():
    endpoint, items, records = fixture(401)
    for record in records[-2:]:
        record.metadata["request_not_sent"] = True
    check = result(endpoint, items, records)
    assert check.n == 400 and check.details["unsent_units"] == 1
    assert not check.details["incomplete_trials"]
    assert check.details["comparison_verdict"] == "PASS"
    for record in records:
        record.metadata["request_not_sent"] = True
    check = result(endpoint, items, records)
    assert check.verdict == "INCONCLUSIVE" and check.n == 0


@pytest.mark.parametrize("field", ["missing-label", "empty-pin", "missing-pin"])
def test_unbound_expected_provider_or_pin_cannot_drive_a_verdict(field):
    endpoint, items, records = fixture()
    for record in records:
        mismatch(record)
    if field == "missing-label":
        endpoint.labels.clear()
    elif field == "empty-pin":
        endpoint.openrouter_pin.tag = ""
        for record in records:
            record.request["provider"]["only"] = [""]
    else:
        endpoint.openrouter_pin = None
    value = sample(endpoint, items, records)
    assert value.status == "INCONCLUSIVE" and not value.values


@pytest.mark.parametrize("fault", ["empty", "duplicate", "wrong-suite", "collision"])
async def test_routing_capability_replay_inventory_rejects_invalid_inputs_before_dispatch(fault):
    from servegauge.runner.check import run_check

    endpoint, items, _ = fixture(1)
    cap = items[0].model_copy(deep=True)
    cap.suite_id = "S0"
    caps = [cap]
    if fault == "empty":
        caps = []
    elif fault == "duplicate":
        caps = [cap, cap.model_copy(deep=True)]
    elif fault == "wrong-suite":
        caps = [items[0]]
    plan = RunPlan(
        plan_id="routing-guard",
        profile=endpoint.claimed_profile,
        level="core",
        seed=9,
        reference={"tier": "BASELINE"},
        endpoints=[endpoint],
        suites=["S1"],
        max_usd=0,
    )
    with pytest.raises(ValueError, match="inventory_required"):
        await run_check(
            plan,
            tests=ROUTING_CHECK,
            dry_run=True,
            inventory=items if fault == "collision" else [],
            capability_inventory=caps,
        )


def test_routing_requirements_cannot_add_an_undeclared_sla_margin():
    from servegauge.runner.check import expand_requirements, normalize_tests
    from servegauge.runner.plan import Requirements

    assert normalize_tests(ROUTING_CHECK) == {ROUTING_CHECK}
    plain = Requirements(app="fixture", requires=[{"id": "identity", "tests": [ROUTING_CHECK]}])
    assert expand_requirements(plain).requires[0].tests == [ROUTING_CHECK]
    bounded = Requirements(
        app="fixture", requires=[{"id": "identity", "tests": [ROUTING_CHECK], "margin": 0.5}]
    )
    with pytest.raises(ValueError, match="not_SLA_gates"):
        expand_requirements(bounded)
