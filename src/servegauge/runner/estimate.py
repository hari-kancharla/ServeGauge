"""Estimate the actual inventory and conservative reservations before dispatch."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from typing import Any

from servegauge.adapters.base import PreparedRequest
from servegauge.core.models import EndpointSpec, Item
from servegauge.runner.cost import prices, request_fee, worst_case_cost
from servegauge.runner.cost_scenarios import initial_input_proxy, scenario_config
from servegauge.runner.plan import RunPlan
from servegauge.scoring.routing import ROUTING_CHECK, routing_unit
from servegauge.stats.intervals import wilson
from servegauge.stats.power import paired_mde
from servegauge.suites.config import suite_config
from servegauge.suites.registry import item_checks, plan_items, prepare_item
from servegauge.suites.s7_accounting import (
    billing_group,
    cost_contract,
    cost_group,
    extended_contract,
    extended_groups,
)
from servegauge.suites.s8_operations import (
    HEALTH_CHECKS,
    OPERATIONAL_CHECKS,
    operational_health_contract,
    operational_suffix,
    operational_unit,
    operations_contract,
)
from servegauge.verdicts.classes import check_class
from servegauge.verdicts.config import load_policy


def estimate_plan(
    plan: RunPlan,
    *,
    items_override: list[Item] | None = None,
    prepare: Callable[[Item, EndpointSpec], PreparedRequest] | None = None,
) -> dict[str, Any]:
    items = plan_items(plan) if items_override is None else items_override
    prepare_for_endpoint = prepare or prepare_item
    rows = []
    scenarios = scenario_config()
    counter_date = suite_config()["accounting"]["counter_date"]
    proxy_cache: dict[str, tuple[int, str]] = {}
    for endpoint in plan.endpoints:
        excluded = endpoint.metadata.get("status", 0) != 0
        decisions = (
            [(item, prepare_for_endpoint(item, endpoint)) for item in items] if not excluded else []
        )
        subtotal = (
            sum(
                (
                    worst_case_cost(decision.body, endpoint)
                    * int(item.expected.get("samples", 1))
                    * int(item.expected.get("max_turns", 1))
                    for item, decision in decisions
                    if decision.body is not None
                ),
                Decimal(0),
            )
            if not excluded
            else Decimal(0)
        )
        proxy_tokens, all_proxy_tokens, fallback_requests, assumed_rejections = 0, 0, 0, 0
        output_tokens = {str(average): 0 for average in scenarios["completion_tokens"]}
        all_output_tokens = dict(output_tokens)
        all_requests, billed_requests = 0, 0
        for item, decision in decisions:
            if decision.body is None:
                continue
            draws = int(item.expected.get("samples", 1))
            turns = int(item.expected.get("max_turns", 1))
            all_requests += draws * turns
            count, method = initial_input_proxy(
                decision.body, endpoint, proxy_cache, counter_date=counter_date
            )
            input_tokens = draws * (count + (turns - 1) * scenarios["continuation_prompt_tokens"])
            all_proxy_tokens += input_tokens
            rejected = (
                scenarios["assume_intentional_rejections_unbilled"]
                and endpoint.kind == "openrouter"
                and item.expected.get("check_id") == "P12"
            )
            assumed_rejections += draws * int(rejected)
            if not rejected:
                proxy_tokens += input_tokens
                billed_requests += draws * turns
            fallback_requests += draws * int(method.startswith("canonical wire proxy"))
            cap = int(
                decision.body.get("max_completion_tokens", decision.body.get("max_tokens", 0))
            )
            for average in scenarios["completion_tokens"]:
                tokens = draws * turns * min(average, cap)
                all_output_tokens[str(average)] += tokens
                if not rejected:
                    output_tokens[str(average)] += tokens
        prompt_price, output_price = prices(endpoint) if not excluded else (Decimal(0), Decimal(0))
        fee = request_fee(endpoint) if not excluded else Decimal(0)
        rows.append(
            {
                "endpoint": endpoint.id,
                "n_requests": 0
                if excluded
                else sum(
                    int(i.expected.get("samples", 1)) * int(i.expected.get("max_turns", 1))
                    for i, decision in decisions
                    if decision.body is not None
                ),
                "reserved_cost_usd": str(subtotal),
                "excluded_status": excluded,
                "scenario_input_proxy_tokens": proxy_tokens,
                "unsupported_template_initial_requests": fallback_requests,
                "assumed_unbilled_rejection_requests": assumed_rejections,
                "scenario_costs_usd": {
                    average: str(
                        prompt_price * proxy_tokens + output_price * count + fee * billed_requests
                    )
                    for average, count in output_tokens.items()
                },
                "all_attempt_scenario_costs_usd": {
                    average: str(
                        prompt_price * all_proxy_tokens + output_price * count + fee * all_requests
                    )
                    for average, count in all_output_tokens.items()
                },
                "unmeasured": [
                    {
                        "item_id": item.item_id,
                        "verdict": decision.verdict,
                        "reason": decision.reason,
                    }
                    for item, decision in decisions
                    if decision.body is None
                ],
            }
        )
    groups: dict[str, set[str]] = {}
    contract = extended_contract(items) if "S7" in plan.suites else None
    costs = cost_contract(items) if "S7" in plan.suites else None
    for item in items:
        for check in item_checks(item):
            if item.suite_id == "S8":
                continue
            if check == "T03" and item.expected.get("should_call") is False:
                continue
            groups.setdefault(check, set()).add(str(item.expected.get("trial", item.item_id)))
        if "S7" in plan.suites and item.suite_id != "S0":
            check, unit = billing_group(item)
            groups.setdefault(check, set()).add(unit)
            if contract is not None:
                for check, unit in extended_groups(item):
                    groups.setdefault(check, set()).add(unit)
            if costs is not None:
                check, unit = cost_group(item)
                groups.setdefault(check, set()).add(unit)
    if operations_contract(items) is not None:
        for item in items:
            if item.suite_id == "S0":
                continue
            for check in OPERATIONAL_CHECKS:
                if (
                    check in {"S8.completion_tokens_per_action", "S8.agent_task_success_rate"}
                    and "agent_goal" not in item.expected
                ):
                    continue
                if check in {
                    "S8.ttft_ms",
                    "S8.delivery_tokens_per_second",
                } and not item.payload.get("stream"):
                    continue
                groups.setdefault(check + operational_suffix(item), set()).add(
                    operational_unit(item)
                )
    if operational_health_contract(items) is not None:
        for item in items:
            if item.suite_id != "S0" and not operational_suffix(item):
                for check in HEALTH_CHECKS:
                    groups.setdefault(check, set()).add(operational_unit(item))
    if any(endpoint.kind == "openrouter" for endpoint in plan.endpoints):
        groups[ROUTING_CHECK] = {routing_unit(item) for item in items}
    family_size = sum(
        check_class(check) == "comparative" or check in HEALTH_CHECKS for check in groups
    )
    policy = load_policy()
    designs = []
    for check, trials in groups.items():
        n = len(trials)
        is_fingerprint = check.startswith("FP")
        is_comparative = (
            check_class(check) == "comparative" or check in HEALTH_CHECKS
        ) and not is_fingerprint
        variance = 4.0 if check in {"P03", "P04.top_p", "P04.top_k"} else 1.0
        illustrative = check in {"P06.calibration", "P07", "T01"} or check.startswith(
            ("B", "R04", "R05")
        )
        designs.append(
            {
                "check": check,
                "n": n,
                "reference_tier": None if check.startswith("S8.") else plan.reference.tier,
                "reference_estimate": 0 if check == ROUTING_CHECK else None,
                "interval": None,
                "mde": paired_mde(
                    n,
                    variance=variance,
                    family_size=family_size,
                    alpha=policy.alpha,
                    power=policy.target_power,
                    two_sided=check in {"P06.calibration", "P13"} or check.startswith("B"),
                )
                if is_comparative
                else None,
                "paired_variance": variance if is_comparative else None,
                "worst_case_holm_alpha": policy.alpha / family_size if is_comparative else None,
                "zero_violations_planned_upper_bound": wilson(0, n, alpha=policy.alpha)[1]
                if check_class(check) == "invariant" or check == ROUTING_CHECK
                else None,
                "assumptions": "planned independent suite/trial units; conditional observations "
                "may reduce n; repeats, retries and task turns stay clustered; no SLA gate"
                if check.startswith("S8.")
                else "fingerprint prompt cells; request repetitions do not establish "
                "independence; distribution tests need matched reference calibration"
                if is_fingerprint
                else "paired pilot variance assumed 1 in metric units; "
                "replace with measured variance"
                if illustrative
                else "paired bounded-outcome worst-case variance; worst-case Holm alpha"
                if is_comparative
                else "report_only"
                if check_class(check) == "report_only"
                else "invariant violation trials",
            }
        )
    return {
        "plan_id": plan.plan_id,
        "level": plan.level,
        "cost_table": rows,
        "total_reserved_cost_usd": str(
            sum((Decimal(row["reserved_cost_usd"]) for row in rows), Decimal(0))
        ),
        "retry_worst_case_usd": str(
            sum((Decimal(row["reserved_cost_usd"]) for row in rows), Decimal(0))
            * (plan.retries + 1)
        ),
        "mde_table": designs,
        "cost_scenarios": scenarios,
        "total_scenario_costs_usd": {
            str(average): str(
                sum((Decimal(row["scenario_costs_usd"][str(average)]) for row in rows), Decimal(0))
            )
            for average in scenarios["completion_tokens"]
        },
        "total_all_attempt_scenario_costs_usd": {
            str(average): str(
                sum(
                    (Decimal(row["all_attempt_scenario_costs_usd"][str(average)]) for row in rows),
                    Decimal(0),
                )
            )
            for average in scenarios["completion_tokens"]
        },
        "note": "Design estimates, not measured endpoint results. Agent requests include all "
        "bounded continuation turns; T02 n is an upper bound conditional on observed calls. "
        "Confirmation replays are additional, remain within the same hard budget, and stop "
        "after reproduction (S2 at most 64 repeat trials per failing invariant). "
        "B01 n is an upper bound conditional on valid usage and qualified matching inputs; "
        "capability traffic is excluded from derived B01 groups.",
    }
