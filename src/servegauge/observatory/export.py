"""Pure, versioned numeric export projections; never expose captured text or metadata."""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any

import polars as pl

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.diagnose.rules import diagnoses
from servegauge.observatory.notices import aware
from servegauge.storage.analysis import Analysis

VERSION = "observatory-export-v1"
BASE = {"run_token": pl.String, "endpoint_token": pl.String}
NUMBERS = [
    "estimate",
    "ci_low",
    "ci_high",
    "reference_estimate",
    "delta",
    "delta_ci_low",
    "delta_ci_high",
    "margin",
    "p_value",
    "adjusted_p",
    "mde",
]
CHECK_SCHEMA = {
    **BASE,
    "check_id": pl.String,
    "suite_id": pl.String,
    "check_class": pl.String,
    "verdict": pl.String,
    "n": pl.Int64,
    "violations": pl.Int64,
    "reference_tier": pl.String,
    **dict.fromkeys(NUMBERS, pl.Float64),
    "ratio": pl.Float64,
    "ratio_ci_low": pl.Float64,
    "ratio_ci_high": pl.Float64,
    "noise_band": pl.Float64,
}
SCHEMAS: dict[str, dict[str, Any]] = {
    "runs": {
        "run_token": pl.String,
        "captured_at": pl.String,
        "logical_records": pl.Int64,
        "sent_records": pl.Int64,
        "attempts": pl.Int64,
        "check_count": pl.Int64,
        "excluded_external_or_unknown_records": pl.Int64,
    },
    "endpoints": {
        **BASE,
        "kind": pl.String,
        "declared_model": pl.String,
        "declared_provider": pl.String,
        "profile_token": pl.String,
    },
    "checks": CHECK_SCHEMA,
    "fingerprints": CHECK_SCHEMA,
    "exchanges": {
        **BASE,
        "exchange_token": pl.String,
        "capture_run_token": pl.String,
        "item_token": pl.String,
        "suite_id": pl.String,
        "sample_idx": pl.Int64,
        "status": pl.Int64,
        "first_status": pl.Int64,
        "attempts": pl.Int64,
        "total_ms": pl.Float64,
        "ttft_ms": pl.Float64,
    },
    "item_scores": {
        **BASE,
        "check_id": pl.String,
        "item_token": pl.String,
        "outcome": pl.Float64,
        "repeats": pl.List(pl.Float64),
    },
    "costs": {
        **BASE,
        "exchange_token": pl.String,
        "cost_usd": pl.Float64,
        "cost_source": pl.String,
        "attempts": pl.Int64,
    },
    "diagnoses": {
        **BASE,
        "diagnosis_token": pl.String,
        "failure_class": pl.String,
        "code": pl.String,
        "confidence": pl.String,
        "check_ids": pl.List(pl.String),
    },
    "change_events": {
        **BASE,
        "event_token": pl.String,
        "input_runs": pl.List(pl.String),
        "observation_start": pl.String,
        "observation_end": pl.String,
        "left_successes": pl.Int64,
        "right_successes": pl.Int64,
        "left_n": pl.Int64,
        "right_n": pl.Int64,
        "difference": pl.Float64,
        "ci_low": pl.Float64,
        "ci_high": pl.Float64,
        "reference_tier": pl.String,
        "quality_verdict": pl.String,
        "interval_scope": pl.String,
        "qualification_sha256": pl.String,
    },
    "notifications": {
        **BASE,
        "check_id": pl.String,
        "subject_token": pl.String,
        "assertion_type": pl.String,
        "claimed_at": pl.String,
        "owner_asserted": pl.Boolean,
        "delivery_verified": pl.Boolean,
        "eligible_at_snapshot": pl.Boolean,
    },
}


def token(kind: str, value: str) -> str:
    return kind + "-" + content_hash(value)


def number(value: Any) -> float | None:
    return float(value) if type(value) in {int, float} and math.isfinite(value) else None


def capture_time(analysis: Analysis) -> datetime:
    return aware(datetime.fromisoformat(str(analysis.manifest["created_at"])))


def numeric_tables(
    analyses: list[Analysis], context: dict[str, Any]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Each month follows primary capture UTC time, without pooling statistical units."""
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}

    def tables(month: str) -> dict[str, list[dict[str, Any]]]:
        return result.setdefault(month, {name: [] for name in SCHEMAS})

    by_run = {str(a.manifest["run_id"]): a for a in analyses}
    for run, analysis in sorted(by_run.items()):
        rows = tables(capture_time(analysis).strftime("%Y-%m"))
        run_token = token("run", run)
        allowed = {
            i.item_id
            for i in analysis.items
            if i.source == "generated" and i.license == "CC-BY-4.0"
        }
        rows["runs"].append(
            {
                "run_token": run_token,
                "captured_at": capture_time(analysis).isoformat(),
                "logical_records": len(analysis.records),
                "sent_records": sum(bool(r.attempts) for r in analysis.records),
                "attempts": sum(len(r.attempts) for r in analysis.records),
                "check_count": len(analysis.checks),
                "excluded_external_or_unknown_records": sum(
                    r.item_id not in allowed for r in analysis.records
                ),
            }
        )

        def base(endpoint: str, run_token: str = run_token) -> dict[str, str]:
            return {"run_token": run_token, "endpoint_token": token("endpoint", endpoint)}

        for endpoint in analysis.plan.endpoints:
            model = provider = None
            if endpoint.kind == "openrouter" and endpoint.openrouter_pin:
                model, provider = endpoint.model, endpoint.openrouter_pin.tag
                if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model) or not re.fullmatch(
                    r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", provider
                ):
                    raise ValueError("export_public_route_identifier_invalid")
            rows["endpoints"].append(
                {
                    **base(endpoint.id),
                    "kind": endpoint.kind,
                    "declared_model": model,
                    "declared_provider": provider,
                    "profile_token": token("profile", analysis.plan.profile),
                }
            )
        for check in analysis.checks:
            bounds = check.details.get("ratio_interval")
            bounds = bounds if isinstance(bounds, list) and len(bounds) == 2 else [None, None]
            row = {
                **base(check.endpoint_id),
                "check_id": check.check_id,
                "suite_id": check.suite_id,
                "check_class": check.check_class,
                "verdict": check.verdict,
                "n": check.n,
                "violations": check.violations,
                "reference_tier": check.reference_tier,
                **{k: number(getattr(check, k)) for k in NUMBERS},
                "ratio": number(check.details.get("geometric_mean_ratio")),
                "ratio_ci_low": number(bounds[0]),
                "ratio_ci_high": number(bounds[1]),
                "noise_band": number(check.details.get("noise_band")),
            }
            rows["checks"].append(row)
            if check.check_id.startswith("FP"):
                rows["fingerprints"].append(dict(row))
        for record in analysis.records:
            if record.item_id not in allowed:
                continue
            exchange = token("exchange", record.exchange_id)
            rows["exchanges"].append(
                {
                    **base(record.endpoint_id),
                    "exchange_token": exchange,
                    "capture_run_token": token("run", record.run_id),
                    "item_token": token("item", record.item_id),
                    "suite_id": record.suite_id,
                    "sample_idx": record.sample_idx,
                    "status": record.status,
                    "first_status": record.attempts[0].status if record.attempts else None,
                    "attempts": len(record.attempts),
                    "total_ms": number(record.timings.total_ms) if record.timings else None,
                    "ttft_ms": number(record.timings.ttft_ms) if record.timings else None,
                }
            )
            rows["costs"].append(
                {
                    **base(record.endpoint_id),
                    "exchange_token": exchange,
                    "cost_usd": number(record.cost_usd),
                    "cost_source": record.cost_source,
                    "attempts": len(record.attempts),
                }
            )
        for endpoint_id, samples in analysis.samples.items():
            for check_id, sample in samples.items():
                for item, value in sorted(sample.values.items()):
                    if item in allowed:
                        rows["item_scores"].append(
                            {
                                **base(endpoint_id),
                                "check_id": check_id,
                                "item_token": token("item", item),
                                "outcome": number(value),
                                "repeats": [number(v) for v in sample.repeats.get(item, [])],
                            }
                        )
        for diagnosis in diagnoses(analysis.checks):
            rows["diagnoses"].append(
                {
                    **base(diagnosis.endpoint_id),
                    "diagnosis_token": token("diagnosis", diagnosis.diagnosis_id),
                    "failure_class": diagnosis.failure_class,
                    "code": diagnosis.code,
                    "confidence": diagnosis.confidence,
                    "check_ids": diagnosis.check_ids,
                }
            )
    for event in context["monitor_events"]:
        run = event["run_ids"][-1]
        rows = tables(capture_time(by_run[run]).strftime("%Y-%m"))
        contrast = event["contrast"]
        rows["change_events"].append(
            {
                "run_token": token("run", run),
                "endpoint_token": token("endpoint", event["endpoint_id"]),
                "event_token": token("event", event["event_id"]),
                "input_runs": [token("run", r) for r in event["run_ids"]],
                **{
                    k: contrast[k]
                    for k in [
                        "observation_start",
                        "observation_end",
                        "left_successes",
                        "right_successes",
                        "left_n",
                        "right_n",
                        "difference",
                        "ci_low",
                        "ci_high",
                    ]
                },
                **{k: event[k] for k in ["reference_tier", "quality_verdict", "interval_scope"]},
                "qualification_sha256": event["qualification"]["sha256"],
            }
        )
    for note in context["reply_notes"]:
        rows = tables(capture_time(by_run[note["run_id"]]).strftime("%Y-%m"))
        for assertion in note["assertions"] or [
            {"type": "prepared_or_missing", "at": None, "owner_asserted": False}
        ]:
            rows["notifications"].append(
                {
                    "run_token": token("run", note["run_id"]),
                    "endpoint_token": token("endpoint", note["endpoint_id"]),
                    "check_id": note["check_id"],
                    "subject_token": token("subject", note["subject_id"]),
                    "assertion_type": assertion["type"],
                    "claimed_at": assertion["at"],
                    "owner_asserted": assertion["owner_asserted"],
                    "delivery_verified": False,
                    "eligible_at_snapshot": note["state"]["eligible"],
                }
            )
    if not result:
        tables("empty")
    for month in result.values():
        for values in month.values():
            values.sort(key=canonical_json)
    return result
