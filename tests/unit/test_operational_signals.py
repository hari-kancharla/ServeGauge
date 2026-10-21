import copy
import json
import math

import numpy as np
import pytest
from tests_support import exchange

from servegauge.core.models import Attempt, SseEvent, Timings
from servegauge.runner.check import expand_requirements
from servegauge.runner.endpoints import resolve_endpoint
from servegauge.runner.estimate import estimate_plan
from servegauge.runner.plan import Requirements, RunPlan
from servegauge.scoring.operations import first_output_window, operational_results
from servegauge.stats.operational import bounded_rate_interval
from servegauge.suites.s8_operations import operational_items
from servegauge.verdicts.config import load_policy


def test_rate_interval_known_answer_and_unknown_variability_is_not_zero():
    low, high = bounded_rate_interval([0] * 100, [1] * 100, [1] * 100, alpha=0.05)
    h = math.sqrt(50 * math.log(80))
    assert low == 0 and high == pytest.approx(h / (100 - h))
    assert high > 0
    with pytest.raises(ValueError):
        bounded_rate_interval([2], [1], [1], alpha=0.05)
    with pytest.raises(ValueError):
        bounded_rate_interval([0], [0], [1], alpha=0.05)


def test_rate_interval_covers_independent_units_with_correlated_repeats_and_random_denominators():
    rng = np.random.default_rng(30911)
    bounds = np.tile([1, 4, 8, 2], 16)
    probabilities = np.linspace(0.01, 0.8, len(bounds))
    truth = float(np.sum(bounds * probabilities) / np.sum(bounds))
    covered = 0
    for _ in range(2000):
        den = bounds * (rng.random(len(bounds)) < 0.6)
        num = den * (rng.random(len(bounds)) < probabilities)
        low, high = bounded_rate_interval(num, den, bounds, alpha=0.05)
        covered += low <= truth <= high
    assert covered >= 1950


def streamed_record():
    chunks = [
        (5, None),
        (10, {"role": "assistant"}),
        (20, {"content": "one two"}),
        (40, {"content": " three four"}),
    ]
    events = [
        SseEvent(t_ms=t, data="keepalive", is_comment=True)
        if delta is None
        else SseEvent(t_ms=t, data=json.dumps({"choices": [{"index": 0, "delta": delta}]}))
        for t, delta in chunks
    ]
    events.append(SseEvent(t_ms=50, data="[DONE]"))
    response = {
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"content": "one two three four"}}
        ]
    }
    attempt = Attempt(
        n=1,
        status=200,
        response=response,
        sse_events=events,
        timings=Timings(start_unix=1, ttft_ms=5, total_ms=100),
        metadata={"terminal": True},
    )
    return exchange(
        run_id="run",
        request={"stream": True},
        response=response,
        usage={"completion_tokens": 4},
        attempts=[attempt],
        sse_events=events,
        timings=attempt.timings,
        metadata={"terminal": True},
    )


def test_first_output_ignores_comments_roles_and_preserves_historical_first_body_field():
    record = streamed_record()
    assert first_output_window(record) == (20, 40)
    assert record.timings.ttft_ms == 5
    record.request["stream"] = False
    assert first_output_window(record) is None


@pytest.mark.parametrize(
    "fault", ["incomplete", "nonmonotone", "bad-json", "multi-choice", "missing-terminal"]
)
def test_bad_stream_observation_does_not_supply_model_output_timing(fault):
    record = streamed_record()
    if fault == "incomplete":
        record.metadata["incomplete"] = True
    elif fault == "nonmonotone":
        record.sse_events[3].t_ms = 1
    elif fault == "bad-json":
        record.sse_events[2].data = "invalid"
    elif fault == "multi-choice":
        record.sse_events[2].data = json.dumps(
            {"choices": [{"index": 1, "delta": {"content": "other"}}]}
        )
    elif fault == "missing-terminal":
        record.metadata.pop("terminal")
    assert first_output_window(record) is None


def plan_and_items():
    ep = resolve_endpoint("http://127.0.0.1:8011/v1", "mock", "gpt-oss-120b")
    plan = RunPlan(
        plan_id="ops",
        profile="gpt-oss-120b",
        level="smoke",
        seed=4,
        reference={"tier": "MOCK"},
        endpoints=[ep],
        suites=["S8"],
        max_usd=0,
        retries=1,
    )
    items = operational_items("smoke", 4, {"items": 2, "agent_tasks": 0})
    return plan, items


def test_request_and_attempt_rates_keep_retries_clustered_and_use_correct_denominators():
    plan, items = plan_and_items()
    records = []
    for index, item in enumerate(items):
        record = streamed_record()
        record.endpoint_id = plan.endpoints[0].id
        record.item_id = item.item_id
        record.exchange_id = f"exchange-{index}"
        if index == 0:
            failed = record.attempts[0].model_copy(deep=True)
            failed.status = 503
            record.attempts.insert(0, failed)
            record.attempts[1].n = 2
        records.append(record)
    results = operational_results(plan, items, records, plan.endpoints[0].id, "run", load_policy())
    by = {r.check_id: r for r in results}
    assert by["S8.request_error_rate"].estimate == 0
    assert by["S8.attempt_error_rate"].estimate == pytest.approx(1 / 5)
    assert by["S8.attempt_error_rate"].n == 2
    assert by["S8.attempt_error_rate"].details["observed_denominator"] == 5
    assert by["S8.attempt_error_rate"].ci_high > 1 / 5
    assert by["S8.ttft_ms"].estimate == 20
    assert by["S8.output_tokens_per_second"].estimate == 40
    assert all(r.check_class == "report_only" and r.verdict != "FAIL" for r in results)


def test_missing_capture_and_one_cluster_do_not_report_a_qualified_measurement():
    plan, items = plan_and_items()
    record = streamed_record()
    record.endpoint_id = plan.endpoints[0].id
    record.item_id = items[0].item_id
    results = operational_results(plan, items, [record], plan.endpoints[0].id, "run", load_policy())
    assert all(r.verdict == "INCONCLUSIVE" for r in results)
    legacy = [i.model_copy(update={"version": "legacy"}) for i in items]
    assert (
        operational_results(plan, legacy, [record], plan.endpoints[0].id, "run", load_policy())
        == []
    )


def test_estimate_operational_design_covers_selected_traffic_and_negative_strata():
    plan, items = plan_and_items()
    extra = [i.model_copy(deep=True) for i in items]
    for index, item in enumerate(extra):
        item.item_id = f"protocol-{index}"
        item.suite_id = "S1"
        item.expected["check_id"] = "P01" if index < 2 else "P12"
    plan.suites = ["S8", "S1"]
    table = {
        d["check"]: d for d in estimate_plan(plan, items_override=[*items, *extra])["mde_table"]
    }
    assert table["S8.request_error_rate"]["n"] == 3
    assert table["S8.request_error_rate.negative_probe"]["n"] == 1
    assert table["S8.ttft_ms"]["n"] == 3
    assert table["S8.request_error_rate"]["reference_tier"] is None
    assert table["S8.request_error_rate"]["mde"] is None


def test_operational_requirements_reject_undeclared_sla_margin_before_dispatch():
    requirements = Requirements(
        app="test",
        requires=[
            {
                "id": "ops",
                "tests": ["operations"],
                "margin": 0.03,
            }
        ],
    )
    with pytest.raises(ValueError, match="report_only_not_SLA"):
        expand_requirements(requirements)


def test_timeout_empty_and_incomplete_are_visible_with_conditional_denominators():
    plan, items = plan_and_items()
    records = []
    for index, item in enumerate(items):
        record = streamed_record()
        record.endpoint_id = plan.endpoints[0].id
        record.item_id = item.item_id
        record.exchange_id = f"rates-{index}"
        if index == 0:
            record.error = "transport_timeout"
            record.attempts[0].error = record.error
        elif index == 1:
            record.response["choices"][0]["message"]["content"] = ""
            record.attempts[0].response = copy.deepcopy(record.response)
            record.sse_events = [record.sse_events[i] for i in (0, 1, 4)]
            record.attempts[0].sse_events = copy.deepcopy(record.sse_events)
        elif index == 2:
            record.metadata["incomplete"] = True
            record.attempts[0].metadata["incomplete"] = True
        records.append(record)
    by = {
        r.check_id: r
        for r in operational_results(
            plan, items, records, plan.endpoints[0].id, "run", load_policy()
        )
    }
    assert by["S8.request_error_rate"].estimate == 0.5
    assert by["S8.attempt_error_rate"].estimate == 0.5
    assert by["S8.request_timeout_rate"].estimate == 0.25
    assert by["S8.attempt_timeout_rate"].estimate == 0.25
    assert by["S8.empty_response_rate"].estimate == 0.5
    assert by["S8.empty_response_rate"].details["observed_denominator"] == 2
    assert by["S8.ttft_ms"].details["excluded_observations"] == 3


def test_captured_stream_refusal_is_output_even_if_legacy_assembler_omits_that_field():
    plan, items = plan_and_items()
    records = []
    for index, item in enumerate(items):
        record = streamed_record()
        record.endpoint_id = plan.endpoints[0].id
        record.item_id = item.item_id
        record.exchange_id = f"refusal-{index}"
        record.response["choices"][0]["message"]["content"] = None
        for event in record.sse_events[2:4]:
            event.data = json.dumps({"choices": [{"index": 0, "delta": {"refusal": "declined"}}]})
        records.append(record)
    by = {
        r.check_id: r
        for r in operational_results(
            plan, items, records, plan.endpoints[0].id, "run", load_policy()
        )
    }
    assert by["S8.empty_response_rate"].estimate == 0


def test_unsent_and_unknown_usage_do_not_become_zero_observations():
    plan, items = plan_and_items()
    records = []
    for index, item in enumerate(items):
        record = streamed_record()
        record.endpoint_id = plan.endpoints[0].id
        record.item_id = item.item_id
        record.exchange_id = f"conditional-{index}"
        record.usage = {"completion_tokens": True if index % 2 else None}
        if index == 0:
            record.metadata["request_not_sent"] = True
        records.append(record)
    by = {
        r.check_id: r
        for r in operational_results(
            plan, items, records, plan.endpoints[0].id, "run", load_policy()
        )
    }
    assert by["S8.request_error_rate"].details["observed_denominator"] == 3
    assert by["S8.request_error_rate"].details["exposure"]["unsent_requests"] == 1
    assert by["S8.output_tokens_per_second"].estimate is None
    assert by["S8.output_tokens_per_second"].verdict == "INCONCLUSIVE"


def test_intentional_negative_probe_errors_are_separate():
    plan, items = plan_and_items()
    negative = items[0].model_copy(deep=True)
    negative.item_id = "negative"
    negative.suite_id = "S1"
    negative.expected.update(check_id="P12")
    negative2 = negative.model_copy(deep=True)
    negative2.item_id = "negative2"
    negative2.expected["trial"] = "next"
    items += [negative, negative2]
    records = []
    for index, item in enumerate(items):
        record = streamed_record()
        record.endpoint_id = plan.endpoints[0].id
        record.item_id = item.item_id
        record.exchange_id = f"negative-{index}"
        if item.suite_id == "S1":
            record.status = record.attempts[0].status = 400
        records.append(record)
    by = {
        r.check_id: r
        for r in operational_results(
            plan, items, records, plan.endpoints[0].id, "run", load_policy()
        )
    }
    assert by["S8.request_error_rate"].estimate == 0
    assert by["S8.request_error_rate.negative_probe"].estimate == 1
    assert by["S8.request_error_rate.negative_probe"].n == 2


def test_confirmation_runs_cannot_duplicate_operational_primary_traffic():
    plan, items = plan_and_items()
    records = []
    for index, item in enumerate(items):
        record = streamed_record()
        record.endpoint_id = plan.endpoints[0].id
        record.item_id = item.item_id
        record.exchange_id = f"primary-{index}"
        records.append(record)
    original = operational_results(plan, items, records, plan.endpoints[0].id, "run", load_policy())
    repeated = [r.model_copy(deep=True) for r in records]
    for record in repeated:
        record.run_id = "run-confirmation-p01"
        record.exchange_id += "-repeated"
        record.error = "transport_timeout"
    assert (
        operational_results(
            plan, items, [*records, *repeated], plan.endpoints[0].id, "run", load_policy()
        )
        == original
    )


def test_zero_action_tasks_have_failure_observations_but_no_fabricated_tokens_per_action():
    plan, _ = plan_and_items()
    items = operational_items("smoke", 4, {"items": 2, "agent_tasks": 2})
    records = []
    for index, item in enumerate(items):
        record = streamed_record()
        record.endpoint_id = plan.endpoints[0].id
        record.item_id = item.item_id
        record.exchange_id = f"zero-action-{index}"
        record.request["messages"] = item.payload["messages"]
        records.append(record)
    by = {
        r.check_id: r
        for r in operational_results(
            plan, items, records, plan.endpoints[0].id, "run", load_policy()
        )
    }
    assert by["S8.agent_task_success_rate"].n == 2
    assert by["S8.agent_task_success_rate"].estimate == 0
    assert by["S8.completion_tokens_per_action"].estimate is None
    assert by["S8.completion_tokens_per_action"].details["excluded_observations"] == 2


@pytest.mark.parametrize("kind", ["opaque", "one-frame", "bad-count", "malformed-envelope"])
def test_unobservable_delivery_throughput_abstains(kind):
    plan, items = plan_and_items()
    records = []
    for index, item in enumerate(items):
        record = streamed_record()
        record.endpoint_id = plan.endpoints[0].id
        record.item_id = item.item_id
        record.exchange_id = f"delivery-{index}"
        record.usage["completion_tokens_details"] = {"reasoning_tokens": 0}
        if kind == "opaque":
            record.usage["completion_tokens_details"]["reasoning_tokens"] = 1
        elif kind == "one-frame":
            record.sse_events[3].t_ms = 20
        elif kind == "bad-count":
            record.usage["completion_tokens"] = True
        else:
            record.response = {"choices": []}
        records.append(record)
    by = {
        r.check_id: r
        for r in operational_results(
            plan, items, records, plan.endpoints[0].id, "run", load_policy()
        )
    }
    assert by["S8.delivery_tokens_per_second"].estimate is None
    assert by["S8.delivery_tokens_per_second"].verdict == "INCONCLUSIVE"
    if kind == "malformed-envelope":
        assert by["S8.request_error_rate"].estimate == 1


@pytest.mark.parametrize(
    "delta",
    [
        {"reasoning_content": "step"},
        {"reasoning_details": [{"type": "reasoning.text", "text": "step"}]},
        {"tool_calls": [{"function": {"name": "convert"}}]},
        {"refusal": "declined"},
    ],
)
def test_first_output_accepts_reasoning_tool_and_refusal_channels(delta):
    record = streamed_record()
    record.sse_events[2].data = json.dumps({"choices": [{"index": 0, "delta": delta}]})
    assert first_output_window(record) == (20, 40)
