import pytest
from tests_support import exchange

from servegauge.core.hashing import content_hash
from servegauge.core.models import EndpointSpec
from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.scoring.longctx_scoring import context_samples, effective_context
from servegauge.suites.registry import prepare_item
from servegauge.suites.s4_longctx import context_items
from servegauge.verdicts.config import load_policy
from servegauge.verdicts.engine import Measurement, decide_family


def cases():
    endpoint = EndpointSpec(
        id="e",
        kind="mock",
        model="mock",
        base_url="http://127.0.0.1/v1",
        claimed_profile="gpt-oss-120b",
        metadata={"context_length": 131072},
    )
    items = context_items(
        "smoke",
        8,
        [endpoint],
        {"lengths": [16000, 32000], "depths": [0, 50, 100], "task_types": 2, "repeats": 2},
    )
    return endpoint, items


def captured(endpoint, items, faults=None):
    engine = ScriptedBackend(8, faults or FaultConfig())
    output = []
    for item in items:
        body = prepare_item(item, endpoint).body
        response = engine.respond(body)
        output.append(
            exchange(
                endpoint_id=endpoint.id,
                item_id=item.item_id,
                request=body,
                response=response,
                usage=response["usage"],
                exchange_id=content_hash(item.item_id),
                metadata={
                    "completed_dependencies": [
                        content_hash(value) for value in item.expected.get("depends_on", [])
                    ]
                },
            )
        )
    return output


def test_canary_retrieval_tracking_and_aggregation_score_observed_output():
    endpoint, items = cases()
    clean = context_samples(items, captured(endpoint, items), endpoint.id, load_policy())
    fault = context_samples(
        items,
        captured(endpoint, items, FaultConfig(truncate_prompt_head_over=16384)),
        endpoint.id,
        load_policy(),
    )
    assert all(sample.status is None for sample in clean.values())
    assert len(clean["L04"].values) == 24 and not any(clean["L04"].values.values())
    assert all(clean["L05.16000"].values.values()) and all(clean["L05.32000"].values.values())
    assert all(fault["L05.16000"].values.values()) and not any(fault["L05.32000"].values.values())
    assert sum(fault["L04"].values.values()) == 12 and len(fault["L04"].replay_candidates) == 12
    assert not any(fault["L02.32000"].values.values())
    assert not any(fault["L03.32000"].values.values())


def test_effective_context_is_a_discrete_grid_summary_with_actual_per_length_CIs():
    measurements = [
        Measurement(
            check_id=f"L05.{length}",
            suite_id="S4",
            endpoint_id="e",
            run_id="r",
            check_class="comparative",
            outcomes=[float(length <= 16000)] * 112,
            reference=[1.0] * 112,
            reference_tier="BASELINE",
            noise_band=0,
            seed=8,
        )
        for length in (4000, 16000, 32000, 64000)
    ]
    results = decide_family(measurements, load_policy())
    summary = effective_context(results)[0]
    assert summary.estimate == 16000 and summary.details["grid_interval"] == [16000, 32000]
    assert summary.ci_low is None and summary.ci_high is None  # No invented parameter 95% CI.
    assert summary.reference_estimate == 64000 and summary.reference_tier == "BASELINE"
    assert all(row["interval"][0] is not None for row in summary.details["per_length"])
    for measurement in measurements:
        measurement.outcomes = [1.0] * 112
    all_good = effective_context(decide_family(measurements, load_policy()))[0]
    assert all_good.estimate == 64000 and all_good.details["right_censored"]
    for measurement in measurements:
        measurement.reference = None
        measurement.reference_tier = None
    absent = effective_context(decide_family(measurements, load_policy()))[0]
    assert absent.verdict == "INCONCLUSIVE" and absent.estimate is None


@pytest.mark.parametrize(
    "failure", ["missing", "transport", "unsent", "reject", "censored", "invalid", "envelope"]
)
def test_unobserved_context_trials_cannot_turn_into_a_canary_pass(failure):
    endpoint, items = cases()
    items = items[:1]
    records = captured(endpoint, items)
    if failure == "missing":
        records = []
    elif failure == "transport":
        records[0].error = "transport_timeout"
    elif failure == "unsent":
        records[0].metadata = {"request_not_sent": True, "verdict": "NOT_SUPPORTED"}
    elif failure == "reject":
        records[0].status = 400
    elif failure == "censored":
        records[0].response["choices"][0]["finish_reason"] = "length"
    elif failure == "invalid":
        records[0].response["choices"][0]["message"]["content"] = "not a JSON object"
    else:
        records[0].response = {"unexpected": True}
    samples = context_samples(items, records, endpoint.id, load_policy())
    assert not samples["L04"].values
    assert samples["L04"].status in {"ERROR", "INCONCLUSIVE"}
    measurement = Measurement(
        check_id="L04",
        suite_id="S4",
        endpoint_id="e",
        run_id="r",
        check_class="invariant",
        outcomes=[0.0] * 500,
        seed=8,
        details={"incomplete_trials": 1},
    )
    assert decide_family([measurement], load_policy())[0].verdict == "INCONCLUSIVE"
