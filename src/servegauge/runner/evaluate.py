"""Orchestrate pure wire scoring, matched references and decisions; persist outside scorers."""

from __future__ import annotations

from typing import Any

from servegauge.core.hashing import content_hash
from servegauge.core.models import (
    CheckResult,
    Exchange,
    Item,
    ModelProfile,
    ReferencePack,
    Tier,
)
from servegauge.refs.calibrate import calibrate
from servegauge.refs.pack import ReferenceEndpoint, ReferenceRun, reference_for
from servegauge.runner.plan import Requirements, RunPlan
from servegauge.scoring.accounting_scoring import accounting_samples
from servegauge.scoring.cache_comparison import cache_samples
from servegauge.scoring.common import Samples, protocol_samples
from servegauge.scoring.completion_accounting import extended_accounting_samples
from servegauge.scoring.cost_accounting import cost_samples
from servegauge.scoring.fingerprint_collection import fingerprint_samples
from servegauge.scoring.fingerprint_results import fingerprint_results
from servegauge.scoring.json_scoring import json_samples
from servegauge.scoring.longctx_scoring import context_samples, effective_context
from servegauge.scoring.longout_scoring import reasoning_samples
from servegauge.scoring.operational_health import health_samples
from servegauge.scoring.operations import operational_results
from servegauge.scoring.routing import ROUTING_CHECK
from servegauge.scoring.tokenizer_hints import tokenizer_samples
from servegauge.scoring.toolcall_scoring import f1_reference_statistic
from servegauge.scoring.tools import tool_samples
from servegauge.storage.accounting_identity import accounting_counter_identity
from servegauge.suites.config import suite_config
from servegauge.suites.registry import item_checks
from servegauge.suites.s1_cache import VERSION as CACHE_VERSION
from servegauge.suites.s1_cache import cache_contract, cache_request_signatures
from servegauge.suites.s7_accounting import (
    billing_group,
    cost_contract,
    cost_group,
    extended_contract,
    extended_groups,
    tokenizer_contract,
)
from servegauge.suites.s8_operations import (
    HEALTH_CHECKS,
    operational_health_contract,
    operational_suffix,
    operations_contract,
)
from servegauge.verdicts.config import DecisionPolicy
from servegauge.verdicts.engine import Measurement, decide_family, map_requirements


def evaluation_configuration(
    plan: RunPlan, items: list[Item], policy: DecisionPolicy
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    contract = extended_contract(items) if "S7" in plan.suites else None
    costs = cost_contract(items) if "S7" in plan.suites else None
    hint = tokenizer_contract(items) if "S7" in plan.suites else None
    health = operational_health_contract(items)
    caches = cache_contract(items)
    cache_requests = cache_request_signatures(items, plan.endpoints)
    for item in items:
        for check in item_checks(item):
            groups.setdefault(check, []).append(item.model_dump(mode="json"))
        if "S7" in plan.suites and item.expected.get("check_id") != "B01":
            groups.setdefault(billing_group(item)[0], []).append(item.model_dump(mode="json"))
        if contract is not None:
            for check, _ in extended_groups(item):
                if check not in item_checks(item):
                    groups.setdefault(check, []).append(item.model_dump(mode="json"))
        if costs is not None:
            check, _ = cost_group(item)
            if check not in item_checks(item):
                groups.setdefault(check, []).append(item.model_dump(mode="json"))
    if health is not None:
        ordinary = [
            i.model_dump(mode="json")
            for i in items
            if i.suite_id != "S0" and not operational_suffix(i)
        ]
        for check in HEALTH_CHECKS:
            groups[check] = ordinary
    counter = suite_config()["accounting"]
    local_counter = accounting_counter_identity(plan.profile)
    if local_counter is not None:
        counter = {**counter, "counter_version": local_counter["counter_version"]}
    return {
        "profile": plan.profile,
        "level": plan.level,
        "seed": plan.seed,
        "item_set_hash": content_hash([item.model_dump(mode="json") for item in items]),
        "policy_hash": content_hash(policy.model_dump(mode="json")),
        "checks": {
            check: {
                "item_set_hash": content_hash(values),
                **(
                    {"counter": {key: counter[key] for key in ("counter_date", "counter_version")}}
                    if check.startswith(("B01", "B05"))
                    else {}
                ),
                **({"accounting_contract": contract} if check.startswith(("B02", "B03")) else {}),
                **(
                    {"local_counter": local_counter}
                    if local_counter is not None
                    and (
                        check in {"P01", "P07", "P12", "J05"}
                        or check.startswith(("B01", "B02", "B05", "L0", "R03", "R04"))
                    )
                    else {}
                ),
                **({"cost_contract": costs} if check.startswith("B04") else {}),
                **({"tokenizer_contract": hint} if check == "B05" else {}),
                **(
                    {"cache_contract": caches, "request_signature": cache_requests[check]}
                    if check in cache_requests
                    else {}
                ),
                **(
                    {
                        "operational_health_contract": health,
                        "procedure_policy": {
                            k: getattr(plan, k)
                            for k in (
                                "retries",
                                "timeout_seconds",
                                "retry_base_seconds",
                                "retry_max_seconds",
                            )
                        },
                    }
                    if check in HEALTH_CHECKS
                    else {}
                ),
            }
            for check, values in groups.items()
        },
    }


def collect(
    plan: RunPlan,
    items: list[Item],
    records: list[Exchange],
    profile: ModelProfile,
    policy: DecisionPolicy,
) -> dict[str, dict[str, Samples]]:
    ordinary_protocol = [i for i in items if i.suite_id == "S1" and i.version != CACHE_VERSION]
    output = {
        endpoint.id: {
            **(
                protocol_samples(ordinary_protocol, records, endpoint.id, profile, policy)
                if ordinary_protocol
                else {}
            ),
            **cache_samples(items, records, endpoint),
            **(
                tool_samples(items, records, endpoint.id, profile)
                if any(item.suite_id == "S2" for item in items)
                else {}
            ),
            **(
                json_samples(items, records, endpoint, profile=profile)
                if any(item.suite_id == "S3" for item in items)
                else {}
            ),
            **(
                context_samples(items, records, endpoint.id, policy)
                if any(item.suite_id == "S4" for item in items)
                else {}
            ),
            **(
                accounting_samples(
                    items, records, endpoint.id, profile, cfg=suite_config()["accounting"]
                )
                if "S7" in plan.suites
                else {}
            ),
            **(
                extended_accounting_samples(items, records, endpoint.id, profile)
                if "S7" in plan.suites
                else {}
            ),
            **(cost_samples(items, records, endpoint.id) if "S7" in plan.suites else {}),
            **(
                tokenizer_samples(items, records, endpoint.id, profile)
                if "S7" in plan.suites
                else {}
            ),
            **(
                reasoning_samples(items, records, endpoint.id)
                if any(item.suite_id == "S5" for item in items)
                else {}
            ),
            **fingerprint_samples(items, records, endpoint, policy),
        }
        for endpoint in plan.endpoints
    }
    common = cache_request_signatures(items, plan.endpoints)
    for scores in output.values():
        for check, expected in common.items():
            sample = scores[check]
            if sample.status is None and (
                expected is None or sample.details.get("request_signature") != expected
            ):
                sample.mark("INCONCLUSIVE", "cache comparison requires identical request layouts")
                sample.values.clear()
    return output


def evaluate(
    plan: RunPlan,
    run_id: str,
    items: list[Item],
    samples: dict[str, dict[str, Samples]],
    policy: DecisionPolicy,
    *,
    target_ids: list[str],
    baseline_id: str | None = None,
    reference_pack: ReferencePack | None = None,
    requirements: Requirements | None = None,
    replay_confirmed: dict[tuple[str, str], bool] | None = None,
    operational_records: list[Exchange] | None = None,
) -> list[CheckResult]:
    configuration = evaluation_configuration(plan, items, policy)
    has_operations = operations_contract(items) is not None
    output = []
    if reference_pack is not None and reference_pack.profile != plan.profile:
        raise ValueError("reference_profile_mismatch")
    for endpoint in plan.endpoints:
        samples[endpoint.id].update(
            health_samples(plan, items, operational_records or [], endpoint.id, run_id)
        )
    for target in target_ids:
        measurements = []
        for check, sample in samples[target].items():
            if sample.metric == "fingerprint" or (
                has_operations
                and check.startswith("S8.")
                and check not in {*HEALTH_CHECKS, ROUTING_CHECK}
            ):
                continue
            if sample.details.get("cache_contract") is not None:
                expected = configuration["checks"][check].get("request_signature")
                if sample.status is None and (
                    expected is None or sample.details.get("request_signature") != expected
                ):
                    sample = sample.model_copy(deep=True)
                    sample.mark(
                        "INCONCLUSIVE", "cache comparison requires identical request layouts"
                    )
                    sample.values.clear()
            labels = None
            reference = None
            calibration: dict[str, Any] | None = None
            keys = sorted(sample.values)
            reference_tier: Tier | None = None
            reference_error = None
            reference_repeats = None
            if sample.check_class == "comparative" and keys and sample.status is None:
                if baseline_id is not None:
                    baseline = samples[baseline_id].get(check)
                    if (
                        check.startswith(("B", "S8.health."))
                        and baseline is not None
                        and (
                            not sample.details.get("input_hashes")
                            or sample.details.get("input_hashes")
                            != baseline.details.get("input_hashes")
                        )
                    ):
                        baseline = None
                        reference_error = (
                            "operational reference requires identical input layouts"
                            if check in HEALTH_CHECKS
                            else "accounting reference requires identical input layouts"
                        )
                    if check == "T02" and baseline is not None and baseline.status is None:
                        shared = sorted(set(keys) & set(baseline.values))
                        sample.details["schema_denominator"] = {
                            "definition": "paired shared tool-call response support",
                            "endpoint_call_items": len(keys),
                            "baseline_call_items": len(baseline.values),
                            "shared_items": len(shared),
                            "trigger_failures_measured_by": "T01; known-call fidelity by T03",
                        }
                        keys = shared
                    if baseline and not baseline.status and set(keys) <= set(baseline.values):
                        reference = [baseline.values[key] for key in keys]
                        if sample.details.get("binary_repeats"):
                            reference_repeats = [baseline.repeats[key] for key in keys]
                        if sample.metric == "f1":
                            if any(len(baseline.repeats.get(key, [])) < 2 for key in keys):
                                reference = None
                                reference_error = "independent baseline repeat unavailable"
                            else:
                                labels = [baseline.repeats[key][0] for key in keys]
                                reference = [baseline.repeats[key][1] for key in keys]
                        if reference:
                            calibration = calibrate(
                                [[a, b] for a, b in zip(labels, reference, strict=True)]
                                if labels is not None
                                else reference,
                                seed=int(
                                    content_hash(
                                        [plan.seed, "*", check]
                                        if check in HEALTH_CHECKS
                                        else [plan.seed, check, "calibration"]
                                    )[:16],
                                    16,
                                ),
                                resamples=policy.bootstrap_resamples,
                                statistic=f1_reference_statistic if labels is not None else None,
                            )
                            reference_tier = "BASELINE"
                    else:
                        reference_error = "baseline check unavailable at matched items and n"
                elif reference_pack is not None:
                    if reference_pack.profile != plan.profile:
                        raise ValueError("reference_profile_mismatch")
                    try:
                        reference, calibration = reference_for(
                            reference_pack,
                            target=target,
                            check=check,
                            item_ids=keys,
                            configuration=configuration,
                            input_hashes=sample.details.get("input_hashes"),
                        )
                        reference_tier = reference_pack.tier
                        if sample.details.get("binary_repeats"):
                            reference_repeats = calibration.get("binary_repeats")
                            if reference_repeats is None:
                                reference = None
                                reference_error = "reference binary repeat layout unavailable"
                        if sample.metric == "f1":
                            labels = calibration.get("f1_anchor")
                            reference = calibration.get("f1_repeat")
                    except (ValueError, LookupError):
                        reference_error = "reference does not match items, n or configuration"
            if (
                check in HEALTH_CHECKS
                and reference is not None
                and any(value not in {0.0, 1.0} for value in reference)
            ):
                reference = None
                calibration = None
                reference_error = (
                    "fractional pooled health reference lacks binary-procedure qualification"
                )
            measurements.append(
                Measurement(
                    check_id=check,
                    suite_id="S8"
                    if check in {*HEALTH_CHECKS, ROUTING_CHECK}
                    else {"P": "S1", "T": "S2", "J": "S3", "L": "S4", "R": "S5", "B": "S7"}.get(
                        check[0], "S0"
                    ),
                    endpoint_id=target,
                    run_id=run_id,
                    check_class=sample.check_class,
                    outcomes=[sample.values[key] for key in keys],
                    metric="f1" if sample.metric == "f1" else "mean",
                    labels=labels,
                    reference=reference,
                    reference_tier=reference_tier,
                    binary=sample.binary,
                    two_sided=sample.two_sided,
                    noise_band=float(calibration["noise_band"]) if calibration else None,
                    replay_confirmed=(replay_confirmed or {}).get((target, check), False),
                    supported=sample.status != "NOT_SUPPORTED",
                    applicable=sample.status != "NOT_APPLICABLE",
                    verified=sample.status != "INCONCLUSIVE",
                    error=sample.reason if sample.status == "ERROR" else None,
                    seed=int(content_hash([plan.seed, target, check, "inference"])[:16], 16),
                    evidence=sorted(set(sample.evidence)),
                    details={
                        **sample.details,
                        "item_keys": keys,
                        "reference_calibration": calibration,
                        "reference_unavailable": reference_error,
                        "binary_repeats": [sample.repeats[key] for key in keys]
                        if sample.details.get("binary_repeats")
                        else None,
                        "reference_binary_repeats": reference_repeats,
                    },
                )
            )
        fingerprints = fingerprint_results(
            samples[target],
            endpoint_id=target,
            run_id=run_id,
            pack=reference_pack,
            policy=policy,
            seed=plan.seed,
            endpoint=next(e for e in plan.endpoints if e.id == target),
        )
        operations = operational_results(
            plan, items, operational_records or [], target, run_id, policy
        )
        if requirements:
            measurements = map_requirements(
                requirements,
                measurements,
                policy,
                available_checks={r.check_id for r in [*fingerprints, *operations]},
            )
            for result in fingerprints:
                matching = [r for r in requirements.requires if result.check_id in r.tests]
                result.requirement_ids = [r.id for r in matching]
                requested = [r.margin for r in matching if r.margin is not None]
                if requested:
                    if result.check_id == "FP3":
                        raise ValueError("FP3_has_no_declared_log_unit_practical_margin")
                    if result.check_id == "FP4":
                        raise ValueError("FP4_has_no_declared_embedding_energy_practical_margin")
                    result.margin = min(policy.margin(result.check_id, m) for m in requested)
                result.details["requirement_params"] = {r.id: r.params for r in matching}
            for result in operations:
                matching = [r for r in requirements.requires if result.check_id in r.tests]
                if any(r.margin is not None for r in matching):
                    raise ValueError("operational_metrics_are_report_only_not_SLA_gates")
                result.requirement_ids = [r.id for r in matching]
        results = decide_family(measurements, policy, external=fingerprints)
        for result in results:
            source = samples[target][result.check_id]
            if source.status and source.reason:
                result.reason = source.reason
            elif result.details.get("reference_unavailable") and result.verdict == "INCONCLUSIVE":
                result.reason = str(result.details["reference_unavailable"])
            if result.check_id in {*HEALTH_CHECKS, ROUTING_CHECK}:
                comparison = result.verdict
                result.details.update(
                    comparison_verdict=comparison, operational_alarm=comparison == "FAIL"
                )
                result.check_class = "report_only"
                if comparison in {"PASS", "FAIL"}:
                    result.verdict = "INFO"
                    result.reason = (
                        "provider metadata invariant: "
                        if result.check_id == ROUTING_CHECK
                        else "calibrated procedure health comparison: "
                    ) + result.reason
        output.extend(results)
        output.extend(operations)
    return [*output, *effective_context(output)]


def reference_source(
    plan: RunPlan,
    run_id: str,
    items: list[Item],
    samples: dict[str, dict[str, Samples]],
    policy: DecisionPolicy,
    manifest: dict[str, Any],
    baseline_id: str | None = None,
) -> ReferenceRun:
    configuration = evaluation_configuration(plan, items, policy)
    checks: dict[str, dict[str, dict[str, list[float]]]] = {}
    inputs: dict[str, dict[str, dict[str, list[str]]]] = {}
    for endpoint, scores in samples.items():
        for check, score in scores.items():
            if score.check_class == "comparative" and score.values and score.status is None:
                if score.details.get("cache_contract") is not None:
                    expected = configuration["checks"][check].get("request_signature")
                    if expected is None or score.details.get("request_signature") != expected:
                        continue
                checks.setdefault(check, {})[endpoint] = {
                    key: score.repeats.get(key, [value]) for key, value in score.values.items()
                }
                if check.startswith(("B", "S8.health.")):
                    inputs.setdefault(check, {})[endpoint] = score.details.get("input_hashes", {})
    return ReferenceRun(
        run_id=run_id,
        profile=plan.profile,
        configuration=configuration,
        item_set_hashes=manifest["item_set_hashes"],
        manifest={
            **manifest,
            "endpoint_metadata": [e.model_dump(mode="json") for e in plan.endpoints],
        }
        if any(s.metric == "fingerprint" for scores in samples.values() for s in scores.values())
        else manifest,
        baseline_endpoint_id=baseline_id,
        checks=checks,
        inputs=inputs,
        fingerprints={
            e: {m: s.model_copy(deep=True) for m, s in scores.items() if s.metric == "fingerprint"}
            for e, scores in samples.items()
            if any(s.metric == "fingerprint" for s in scores.values())
        },
        fingerprint_policy=policy.model_copy(deep=True)
        if any(s.metric == "fingerprint" for scores in samples.values() for s in scores.values())
        else None,
        endpoints=[
            ReferenceEndpoint(
                id=e.id,
                profile=e.claimed_profile,
                kind=e.kind,
                seed=e.metadata.get("mock_declaration", {}).get("seed"),
                clean=e.metadata.get("mock_declaration", {}).get("clean", False),
            )
            for e in plan.endpoints
        ],
    )


def exit_code(results: list[CheckResult], required: set[str]) -> int:
    selected = [result for result in results if result.check_id in required]
    if not selected:
        return 3
    if any(result.verdict == "ERROR" for result in selected):
        return 5
    if any(result.verdict == "FAIL" for result in selected):
        return 1
    if any(result.verdict in {"INCONCLUSIVE", "NOT_SUPPORTED"} for result in selected):
        return 2
    return 0
