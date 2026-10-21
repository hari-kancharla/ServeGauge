"""Deterministic preparation only: disclosed labels guide coverage, never diagnosis."""

import math
from decimal import Decimal
from typing import Any

from servegauge.core.models import EndpointSpec
from servegauge.runner.budget import BudgetPolicy
from servegauge.runner.cost import prices
from servegauge.runner.plan import ReferenceSpec, RunPlan


def select_endpoints(snapshot: dict[str, Any], *, count: int = 10) -> list[EndpointSpec]:
    if not 8 <= count <= 10:
        raise ValueError("M15_lite_requires_8_to_10_endpoints")
    candidates = [
        EndpointSpec.model_validate(row)
        for row in snapshot["endpoints"]
        if type(row.get("metadata", {}).get("status")) is int and row["metadata"]["status"] == 0
    ]
    if len(candidates) < count or len({endpoint.id for endpoint in candidates}) != len(candidates):
        raise ValueError("insufficient_unique_active_endpoints")
    selected: list[EndpointSpec] = []
    providers: set[str] = set()
    labels: set[str] = set()

    def rank(endpoint: EndpointSpec) -> tuple[bool, bool, float, Decimal, str]:
        try:
            uptime = float(endpoint.metadata["uptime_last_1d"])
        except (KeyError, ValueError, TypeError):
            uptime = -1.0
        if not math.isfinite(uptime) or not 0 <= uptime <= 100:
            uptime = -1.0
        return (
            endpoint.labels.get("quantization", "unknown") not in labels,
            endpoint.labels.get("provider_name", "unknown") not in providers,
            uptime,
            -sum(prices(endpoint), Decimal(0)),
            endpoint.id,
        )

    while len(selected) < count:
        best = max(candidates, key=rank)
        selected.append(best)
        candidates.remove(best)
        providers.add(best.labels.get("provider_name", "unknown"))
        labels.add(best.labels.get("quantization", "unknown"))
    # P12's common over-context item is generated from the first endpoint's bound.
    return sorted(
        selected,
        key=lambda endpoint: (-int(endpoint.metadata.get("context_length") or 0), endpoint.id),
    )


def prepare_plan(snapshot: dict[str, Any], budgets: BudgetPolicy) -> RunPlan:
    endpoints = select_endpoints(snapshot)
    if snapshot["model"] != "openai/gpt-oss-120b":
        raise ValueError("Model_A_snapshot_required")
    return RunPlan(
        plan_id="real_core_m15_lite",
        profile="gpt-oss-120b",
        level="core",
        seed=20261018,
        reference=ReferenceSpec(tier="R3", pack=None),
        concurrency_per_endpoint=4,
        requests_per_minute=60,
        max_usd=float(budgets.per_run_usd),
        endpoints=endpoints,
        suites=["S0", "S1", "S2", "S3", "S4", "S7"],
    )
