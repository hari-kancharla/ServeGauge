"""S8 observational client metrics from captured attempts and reconstructed tasks."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from servegauge.core.completion import completion_error
from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.json import strict_json
from servegauge.core.models import CheckResult, Exchange, Item
from servegauge.core.redaction import REDACTED
from servegauge.runner.plan import RunPlan
from servegauge.scoring.agents import agent_samples
from servegauge.scoring.common import calls, message
from servegauge.scoring.completion_accounting import completion_channels
from servegauge.stats.bootstrap import clustered_bootstrap
from servegauge.stats.common import FloatArray
from servegauge.stats.operational import bounded_rate_interval
from servegauge.suites.s8_operations import (
    OPERATIONAL_CHECKS,
    operational_suffix,
    operational_unit,
    operations_contract,
)
from servegauge.verdicts.config import DecisionPolicy


def _output(delta: dict[str, Any]) -> bool:
    if any(
        isinstance(delta.get(k), str) and delta[k]
        for k in ("content", "reasoning", "reasoning_content", "refusal")
    ):
        return True
    if any(
        isinstance(d, dict)
        and d.get("type") == "reasoning.text"
        and isinstance(d.get("text"), str)
        and d["text"]
        for d in delta.get("reasoning_details", []) or []
    ):
        return True
    return any(
        isinstance(c, dict)
        and isinstance(c.get("function"), dict)
        and any(
            isinstance(c["function"].get(k), str) and c["function"][k]
            for k in ("name", "arguments")
        )
        for c in delta.get("tool_calls", []) or []
    )


def first_output_window(record: Exchange) -> tuple[float, float] | None:
    """First/last substantive SSE output events, never HTTP body or role events."""
    if (
        not record.request.get("stream")
        or not record.sse_events
        or record.error
        or record.status != 200
        or not record.attempts
        or not record.metadata.get("terminal")
        or any(
            record.metadata.get(k)
            for k in (
                "incomplete",
                "invalid_utf8",
                "invalid_json_events",
                "stream_multiple_choices",
                "stream_identity_conflict",
            )
        )
    ):
        return None
    times = []
    prior = 0.0
    total = record.attempts[-1].timings.total_ms
    try:
        for event in record.sse_events:
            if not math.isfinite(event.t_ms) or not prior <= event.t_ms <= total:
                return None
            prior = event.t_ms
            if event.is_comment or event.data == "[DONE]":
                continue
            data = strict_json(event.data)
            if not isinstance(data, dict) or data.get("error"):
                return None
            choices = data.get("choices", [])
            if not isinstance(choices, list) or len(choices) > 1:
                return None
            for chosen in choices:
                if (
                    not isinstance(chosen, dict)
                    or type(chosen.get("index", 0)) is not int
                    or chosen.get("index", 0) != 0
                ):
                    return None
                delta = chosen.get("delta", {})
                if not isinstance(delta, dict):
                    return None
                if _output(delta):
                    times.append(event.t_ms)
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
    return (times[0], times[-1]) if times else None


def _error(
    status: int | None,
    error: str | None,
    response: dict[str, Any] | None,
    metadata: dict[str, Any],
) -> float | None:
    if error:
        return 1.0
    if type(status) is not int or not 100 <= status <= 599:
        return None
    choices = (response or {}).get("choices")
    valid = (
        isinstance(choices, list)
        and len(choices) == 1
        and isinstance(choices[0], dict)
        and isinstance(choices[0].get("message"), dict)
    )
    return float(
        status != 200
        or completion_error(response)
        or not valid
        or any(metadata.get(k) for k in ("incomplete", "invalid_utf8", "invalid_json_events"))
    )


def _tokens(record: Exchange) -> int | None:
    count = (record.usage or {}).get("completion_tokens")
    return count if type(count) is int and count >= 0 else None


def _complete(record: Exchange) -> bool:
    return bool(
        record.status == 200
        and not record.error
        and not record.metadata.get("request_not_sent")
        and not record.metadata.get("incomplete")
        and _error(record.status, record.error, record.response, record.metadata) == 0
    )


def _empty(record: Exchange) -> float | None:
    if record.status != 200 or record.metadata.get("incomplete"):
        return None
    if record.response is None:
        return (
            1.0
            if record.error in {None, "invalid_response_json"}
            and record.response_text is not None
            and not record.response_text.strip()
            else None
        )
    if record.error or completion_error(record.response):
        return None
    if first_output_window(record) is not None:
        return 0.0
    try:
        returned = message(record)
        return float(
            not any(
                returned.get(k)
                for k in (
                    "content",
                    "tool_calls",
                    "function_call",
                    "reasoning",
                    "reasoning_content",
                    "reasoning_details",
                    "refusal",
                    "audio",
                    "images",
                )
            )
        )
    except (ValueError, TypeError, KeyError):
        return None


def _ratio(data: FloatArray) -> float:
    return float(np.sum(data[:, 0]) / np.sum(data[:, 1]))


@dataclass
class _Metric:
    rows: list[tuple[float, float]] = field(default_factory=list)
    units: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    excluded: int = 0

    def add(self, num: float | None, den: float, unit: str, evidence: list[str]) -> None:
        if num is None or not math.isfinite(num) or not math.isfinite(den) or den <= 0:
            self.excluded += 1
            return
        self.rows.append((num, den))
        self.units.append(unit)
        self.evidence.extend(evidence)


@dataclass
class OperationalObservations:
    contract: dict[str, Any]
    metrics: dict[str, _Metric]
    bounds: dict[str, dict[str, float]]
    exposure: dict[str, dict[str, int]]
    inputs: dict[str, dict[str, list[str]]]
    missing: bool
    redacted: bool


def operational_observations(
    plan: RunPlan,
    items: list[Item],
    records: list[Exchange],
    endpoint: str,
    run_id: str,
) -> OperationalObservations | None:
    contract = operations_contract(items)
    if contract is None:
        return None
    allowed_runs = {
        run_id,
        *(
            run_id + "-agent-" + content_hash(endpoint)[:8] + f"-{turn}"
            for item in items
            if "agent_goal" in item.expected
            for turn in range(1, int(item.expected["max_turns"]))
        ),
    }
    by_item: dict[str, list[Exchange]] = defaultdict(list)
    redacted = False
    for record in records:
        if record.endpoint_id == endpoint and record.run_id in allowed_runs:
            by_item[record.item_id].append(record)
            redacted |= REDACTED in canonical_json(record.request).decode() or bool(
                record.metadata.get("sensitive_response_redacted")
            )
    metrics: dict[str, _Metric] = {}
    bounds: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    details: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    inputs: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    missing = False
    sources = [i for i in items if i.suite_id != "S0"]
    for item in sources:
        suffix, unit = operational_suffix(item), operational_unit(item)
        draws = int(item.expected.get("samples", 1))
        turns = int(item.expected.get("max_turns", 1))
        bounds[suffix][unit] += draws * turns
        available = sorted(by_item[item.item_id], key=lambda r: r.sample_idx)
        if [r.sample_idx for r in available] != list(range(draws)):
            missing = True
        for record in available:
            inputs[suffix][unit].append(
                content_hash(
                    {
                        "item_id": item.item_id,
                        "version": item.version,
                        "sample_idx": record.sample_idx,
                        "request": {k: v for k, v in record.request.items() if k != "model"},
                        "task_protocol": {
                            k: item.expected[k]
                            for k in ("agent_initial", "agent_goal", "max_turns", "pass_reasoning")
                            if k in item.expected
                        },
                    }
                )
            )
        task_records = available
        if "agent_goal" in item.expected:
            sample = agent_samples([item], records, endpoint)[item.expected["check_id"]]
            trace = sample.details.get("agent_traces", {}).get(item.item_id, {})
            used = set(trace.get("exchange_ids", []))
            identifiers = {item.item_id, *(item.item_id + f"-turn-{t}" for t in range(1, turns))}
            task_records = [r for name in sorted(identifiers) for r in by_item[name]]
            if any(
                k in sample.reason for k in ("capture missing", "recorded agent history")
            ) or any(r.exchange_id not in used for r in task_records):
                missing = True
            attempted = bool(task_records) and not all(
                r.metadata.get("request_not_sent") for r in task_records
            )
            if attempted:
                metrics.setdefault("S8.agent_task_success_rate" + suffix, _Metric()).add(
                    sample.values.get(item.item_id) if sample.status != "NOT_SUPPORTED" else None,
                    1,
                    unit,
                    list(used),
                )
            count = None
            actions = 0
            if sample.status is None and not trace.get("invalid_returned_call"):
                try:
                    token_values = [_tokens(r) for r in task_records]
                    actions = sum(len(calls(r)) for r in task_records)
                    if all(v is not None for v in token_values):
                        count = float(sum(v for v in token_values if v is not None))
                except (ValueError, TypeError, KeyError):
                    pass
            metrics.setdefault("S8.completion_tokens_per_action" + suffix, _Metric()).add(
                count, float(actions), unit, list(used)
            )
        for record in task_records:
            if record.metadata.get("request_not_sent"):
                details[suffix]["unsent_requests"] += 1
                continue
            details[suffix]["sent_requests"] += 1
            ev = [record.exchange_id]
            error = _error(record.status, record.error, record.response, record.metadata)
            for key, value in (
                ("S8.request_error_rate", error),
                (
                    "S8.request_timeout_rate",
                    None if error is None else float(record.error == "transport_timeout"),
                ),
                ("S8.empty_response_rate", _empty(record)),
            ):
                metrics.setdefault(key + suffix, _Metric()).add(value, 1, unit, ev)
            details[suffix]["attempts"] += len(record.attempts)
            if (
                not record.attempts
                or [a.n for a in record.attempts] != list(range(1, len(record.attempts) + 1))
                or len(record.attempts) > plan.retries + 1
            ):
                missing = True
            for attempt in record.attempts:
                failed = _error(attempt.status, attempt.error, attempt.response, attempt.metadata)
                for key, value in (
                    ("S8.attempt_error_rate", failed),
                    (
                        "S8.attempt_timeout_rate",
                        None if failed is None else float(attempt.error == "transport_timeout"),
                    ),
                ):
                    metrics.setdefault(key + suffix, _Metric()).add(value, 1, unit, ev)
            window = first_output_window(record)
            if _complete(record) and record.attempts:
                timing = record.attempts[-1].timings
                tokens = _tokens(record)
                metrics.setdefault("S8.ttft_ms" + suffix, _Metric()).add(
                    window[0] if window else None, 1, unit, ev
                )
                first = timing.ttft_ms
                metrics.setdefault("S8.first_body_ms" + suffix, _Metric()).add(
                    first if first is not None and 0 <= first <= timing.total_ms else None,
                    1,
                    unit,
                    ev,
                )
                metrics.setdefault("S8.output_tokens_per_second" + suffix, _Metric()).add(
                    float(tokens) if tokens is not None else None,
                    timing.total_ms / 1000,
                    unit,
                    ev,
                )
                returned = False
                try:
                    completion_channels(record)
                    returned = True
                except (ValueError, TypeError, KeyError):
                    pass
                metrics.setdefault("S8.delivery_tokens_per_second" + suffix, _Metric()).add(
                    float(tokens) if returned and tokens is not None else None,
                    (window[1] - window[0]) / 1000 if window else 0,
                    unit,
                    ev,
                )
            else:
                for key in (
                    "S8.ttft_ms",
                    "S8.first_body_ms",
                    "S8.output_tokens_per_second",
                    "S8.delivery_tokens_per_second",
                ):
                    metrics.setdefault(key + suffix, _Metric()).excluded += 1
    return OperationalObservations(contract, metrics, bounds, details, inputs, missing, redacted)


def operational_results(
    plan: RunPlan,
    items: list[Item],
    records: list[Exchange],
    endpoint: str,
    run_id: str,
    policy: DecisionPolicy,
) -> list[CheckResult]:
    observations = operational_observations(plan, items, records, endpoint, run_id)
    if observations is None:
        return []
    contract, metrics, bounds = observations.contract, observations.metrics, observations.bounds
    details, missing = observations.exposure, observations.missing
    output = []
    for suffix in bounds:
        for base in OPERATIONAL_CHECKS:
            key = base + suffix
            metric = metrics.get(key, _Metric())
            n = len(set(metric.units))
            result = CheckResult(
                check_id=key,
                check_class="report_only",
                suite_id="S8",
                endpoint_id=endpoint,
                run_id=run_id,
                n=n,
                reference_tier=None,
                verdict="INCONCLUSIVE",
                reason="no observable independent units",
                evidence=sorted(set(metric.evidence)),
                details={
                    "operations_contract": contract,
                    "reference_context": "observed only; no comparison or alarm",
                    "independent_unit": "suite/trial; repeats and task turns stay clustered",
                    "independence_assumption": "independent roots; other dependence unqualified",
                    "scope": "primary/task traffic; excludes capability/confirmation calls",
                    "traffic_class": "intentional negative probes"
                    if suffix
                    else "ordinary planned traffic",
                    "excluded_observations": metric.excluded,
                    "observed_rows": len(metric.rows),
                    "planned_units": len(bounds[suffix]),
                    "exposure": dict(details[suffix]),
                    "timing_basis": "final-attempt dispatch; excludes retries/backoff",
                    "throughput_basis": "reported tokens per client duration; no hardware claim",
                    "action_basis": "executed simulated calls; positive-action tasks, known usage",
                    "error_basis": "transport/HTTP/provider failures; invalid/incomplete envelope",
                    "confidence_level": 1 - policy.alpha,
                    "operational_alarm": False,
                },
            )
            if metric.rows:
                data = np.asarray(metric.rows, dtype=float)
                result.estimate = _ratio(data)
                result.details.update(
                    observed_numerator=float(np.sum(data[:, 0])),
                    observed_denominator=float(np.sum(data[:, 1])),
                )
                if not missing and n >= 2:
                    if base.endswith("_rate"):
                        units = sorted(bounds[suffix])
                        num = {u: 0.0 for u in units}
                        den = {u: 0.0 for u in units}
                        for (a, b), u in zip(metric.rows, metric.units, strict=True):
                            num[u] += a
                            den[u] += b
                        multiplier = plan.retries + 1 if base.startswith("S8.attempt_") else 1
                        limits = [bounds[suffix][u] * multiplier for u in units]
                        result.ci_low, result.ci_high = bounded_rate_interval(
                            [num[u] for u in units],
                            [den[u] for u in units],
                            limits,
                            alpha=policy.alpha,
                        )
                        result.details["interval_method"] = (
                            "bounded-sum Hoeffding union bound; fixed design caps"
                        )
                    else:
                        interval = clustered_bootstrap(
                            data,
                            metric.units,
                            seed=int(
                                content_hash([plan.seed, endpoint, key, "ops-interval"])[:16], 16
                            ),
                            resamples=policy.bootstrap_resamples,
                            alpha=policy.alpha,
                            statistic=_ratio,
                        )
                        result.ci_low, result.ci_high = interval.low, interval.high
                        result.details["interval_method"] = (
                            "whole-unit bootstrap; empirical distribution, no tail guarantee"
                        )
                    result.verdict = "INFO"
                    result.reason = "observational client metric; no operational fitness decision"
                elif n == 1:
                    result.reason = "one independent unit; descriptive observation only"
            if missing:
                result.reason = (
                    "planned capture or task lineage missing/ambiguous; no qualified aggregate"
                )
            output.append(result)
    return output
