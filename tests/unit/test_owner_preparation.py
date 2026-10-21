import copy
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from servegauge.config import load_budget
from servegauge.core.models import Item
from servegauge.runner.cost import prompt_reservation_tokens
from servegauge.runner.cost_scenarios import initial_input_proxy, scenario_config
from servegauge.runner.estimate import estimate_plan
from servegauge.runner.owner_plan import prepare_plan, select_endpoints
from servegauge.runner.plan import RunPlan
from servegauge.suites.registry import prepare_item


def public_snapshot():
    paths = Path("configs/endpoints").glob("openai_gpt-oss-120b-*.yaml")
    return yaml.safe_load(next(paths).read_text(encoding="utf-8"))


def test_owner_plan_is_active_diverse_bounded_and_deterministic():
    snapshot = public_snapshot()
    first = prepare_plan(snapshot, load_budget(Path(".")))
    again = prepare_plan(snapshot, load_budget(Path(".")))
    assert (
        first == again
        and len(first.endpoints) == 10
        and first.max_usd == load_budget(Path(".")).per_run_usd
    )
    assert first.reference.tier == "R3" and first.suites == ["S0", "S1", "S2", "S3", "S4", "S7"]
    assert all(endpoint.metadata["status"] == 0 for endpoint in first.endpoints)
    active_labels = {
        row["labels"]["quantization"]
        for row in snapshot["endpoints"]
        if row["metadata"]["status"] == 0
    }
    assert {endpoint.labels["quantization"] for endpoint in first.endpoints} == active_labels
    assert len({endpoint.labels["provider_name"] for endpoint in first.endpoints}) == 10
    assert first.endpoints[0].metadata["context_length"] == max(
        endpoint.metadata["context_length"] for endpoint in first.endpoints
    )
    for invalid in (7, 11):
        with pytest.raises(ValueError):
            select_endpoints(snapshot, count=invalid)
    too_few = copy.deepcopy(snapshot)
    too_few["endpoints"] = too_few["endpoints"][:3]
    with pytest.raises(ValueError):
        select_endpoints(too_few)
    snapshot["model"] = "other/model"
    with pytest.raises(ValueError, match="Model_A"):
        prepare_plan(snapshot, load_budget(Path(".")))


def test_scenario_counts_are_explicit_proxies_and_profile_cache_is_scoped():
    endpoint = select_endpoints(public_snapshot())[0]
    body = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 128}
    cache = {}
    count, method = initial_input_proxy(body, endpoint, cache, counter_date="2026-10-18")
    assert count > 0 and method.startswith("official local template")
    assert initial_input_proxy(body, endpoint, cache, counter_date="2026-10-18") == (count, method)
    endpoint.claimed_profile = "unknown"
    fallback, method = initial_input_proxy(body, endpoint, cache, counter_date="2026-10-18")
    assert fallback > 0 and method.startswith("canonical wire proxy")
    assert scenario_config()["completion_tokens"] == [64, 256, 1024]
    assert load_budget(Path(".")).per_run_usd == Decimal(0)


def test_P12_never_fails_an_endpoint_using_a_smaller_probe_bound():
    endpoint = select_endpoints(public_snapshot())[0]
    item = Item(
        item_id="p12",
        suite_id="S1",
        version="test",
        payload={"messages": [{"role": "user", "content": "short"}], "params": {"max_tokens": 16}},
        expected={"check_id": "P12", "context_length": 1000},
    )
    decision = prepare_item(item, endpoint)
    assert decision.body is None and decision.verdict == "INCONCLUSIVE"


def test_conditional_rejection_scenario_does_not_change_money_reservations():
    endpoint = select_endpoints(public_snapshot())[0]
    endpoint.metadata["max_prompt_tokens"] = 1000
    item = Item(
        item_id="reject",
        suite_id="S1",
        version="test",
        payload={
            "messages": [{"role": "user", "content": "wide " * 500}],
            "params": {"max_tokens": 16},
        },
        expected={
            "check_id": "P12",
            "samples": 1,
            "trial": 0,
            "context_length": endpoint.metadata["context_length"],
        },
    )
    plan = RunPlan(
        plan_id="cost",
        profile="gpt-oss-120b",
        level="core",
        seed=42,
        reference={"tier": "R3"},
        endpoints=[endpoint],
        suites=["S1"],
        max_usd=25,
    )
    estimate = estimate_plan(plan, items_override=[item])
    assert estimate["cost_table"][0]["assumed_unbilled_rejection_requests"] == 1
    assert all(Decimal(cost) == 0 for cost in estimate["total_scenario_costs_usd"].values())
    assert all(
        Decimal(cost) > 0 for cost in estimate["total_all_attempt_scenario_costs_usd"].values()
    )
    assert Decimal(estimate["total_reserved_cost_usd"]) > 0
    assert (
        prompt_reservation_tokens({"messages": [{"role": "user", "content": "x" * 1200}]}, endpoint)
        > 1200
    )
