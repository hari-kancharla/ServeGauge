"""Matched binary procedure health, using the same captured S8 observation units."""

from servegauge.core.models import Exchange, Item
from servegauge.runner.plan import RunPlan
from servegauge.scoring.common import Samples
from servegauge.scoring.operations import operational_observations
from servegauge.suites.s8_operations import operational_health_contract

METRICS = {
    "S8.health.requests": "S8.request_error_rate",
    "S8.health.attempts": "S8.attempt_error_rate",
    "S8.health.nonempty": "S8.empty_response_rate",
}


def health_samples(
    plan: RunPlan, items: list[Item], records: list[Exchange], endpoint: str, run_id: str
) -> dict[str, Samples]:
    contract = operational_health_contract(items)
    if contract is None:
        return {}
    observed = operational_observations(plan, items, records, endpoint, run_id)
    if observed is None:
        raise ValueError("health_requires_sealed_operational_observation_contract")
    units = set(observed.bounds.get("", {}))
    output = {}
    for check, metric in METRICS.items():
        sample = Samples(
            check_id=check,
            check_class="comparative",
            binary=True,
            details={
                "operational_health_contract": contract,
                "operational_diagnostic": True,
                "metric": "fraction of complete root procedures with no observed adverse event",
                "source_metric": metric,
                "independent_unit": "suite/trial procedure; variants/retries/task turns clustered",
                "independence_assumption": "independent roots; cross-unit dependence unqualified",
                "population": "ordinary primary/task procedures; excludes negative probes",
                "input_hashes": dict(observed.inputs.get("", {})),
                "exposure": dict(observed.exposure.get("", {})),
                "planned_units": len(units),
                "operational_alarm": False,
            },
        )
        output[check] = sample
        rows = observed.metrics.get(metric)
        if (
            observed.missing
            or observed.redacted
            or not units
            or rows is None
            or rows.excluded
            or set(rows.units) != units
            or observed.exposure.get("", {}).get("unsent_requests", 0)
        ):
            sample.mark("INCONCLUSIVE", "complete unredacted observable procedure cohort required")
            if rows is not None:
                sample.evidence = sorted(set(rows.evidence))
                sample.details["excluded_observations"] = rows.excluded
            continue
        failures = {u: 0.0 for u in units}
        for (numerator, _), unit in zip(rows.rows, rows.units, strict=True):
            failures[unit] += numerator
        sample.values = {u: float(failures[u] == 0) for u in sorted(units)}
        sample.evidence = sorted(set(rows.evidence))
        sample.details["observed_opportunities"] = len(rows.rows)
    return output
