"""Rebuild tables exclusively from raw captures; compare content rather than file bytes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from servegauge.core.hashing import canonical_json
from servegauge.core.models import Exchange
from servegauge.diagnose.rules import diagnoses
from servegauge.storage.analysis import _read, reconstruct
from servegauge.storage.catalog import build_catalog
from servegauge.storage.observatory_rows import observatory_rows
from servegauge.storage.raw_store import RawStore

TABLE_NAMES = (
    "endpoints",
    "runs",
    "exchanges",
    "item_scores",
    "checks",
    "fingerprints",
    "diagnoses",
    "change_events",
    "costs",
    "notifications",
)
EXCHANGE_SCHEMA = {
    "exchange_id": pl.String,
    "run_id": pl.String,
    "endpoint_id": pl.String,
    "suite_id": pl.String,
    "item_id": pl.String,
    "sample_idx": pl.Int64,
    "status": pl.Int64,
    "cost_usd": pl.Float64,
    "cost_source": pl.String,
    "error": pl.String,
    "request_json": pl.String,
    "response_json": pl.String,
    "attempts_json": pl.String,
    "sse_events_json": pl.String,
    "usage_json": pl.String,
    "provider_meta_json": pl.String,
    "record_json": pl.String,
}


def exchange_row(record: Exchange) -> dict[str, Any]:
    data = record.model_dump(mode="json")
    row = {key: data[key] for key in EXCHANGE_SCHEMA if key in data}
    for key in ("request", "response", "attempts", "sse_events", "usage", "provider_meta"):
        row[f"{key}_json"] = canonical_json(data[key]).decode()
    row["record_json"] = canonical_json(data).decode()
    return row


def rebuild(raw: Path, out: Path = Path("data/tables")) -> dict[str, int]:
    records = sorted(RawStore(raw).exchanges(), key=lambda e: e.exchange_id)
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "exchanges": pl.DataFrame([exchange_row(e) for e in records], schema=EXCHANGE_SCHEMA),
        "runs": pl.DataFrame(
            {"run_id": sorted({e.run_id for e in records})}, schema={"run_id": pl.String}
        ),
        "endpoints": pl.DataFrame(
            {"endpoint_id": sorted({e.endpoint_id for e in records})},
            schema={"endpoint_id": pl.String},
        ),
        "costs": pl.DataFrame(
            [
                {
                    "exchange_id": e.exchange_id,
                    "run_id": e.run_id,
                    "endpoint_id": e.endpoint_id,
                    "cost_usd": e.cost_usd,
                    "cost_source": e.cost_source,
                }
                for e in records
            ],
            schema={
                "exchange_id": pl.String,
                "run_id": pl.String,
                "endpoint_id": pl.String,
                "cost_usd": pl.Float64,
                "cost_source": pl.String,
            },
        ),
    }
    # Populated by later scoring/diagnosis milestones; empty tables are explicit, not fake results.
    for name in TABLE_NAMES:
        tables.setdefault(
            name, pl.DataFrame(schema={"record_id": pl.String, "record_json": pl.String})
        )
    derived: dict[str, list[dict[str, str]]] = {
        name: [] for name in ("item_scores", "checks", "diagnoses", "fingerprints")
    }
    for snapshot in sorted(raw.rglob("analysis.json")):
        run_id = _read(snapshot)["run_id"]
        analysis = reconstruct(raw, run_id)
        for endpoint, scores in analysis.samples.items():
            for check, sample in scores.items():
                if sample.metric == "fingerprint":
                    derived["fingerprints"].append(
                        {
                            "record_id": f"{run_id}/{endpoint}/{check}",
                            "record_json": canonical_json(
                                {
                                    "run_id": run_id,
                                    "endpoint_id": endpoint,
                                    "check_id": check,
                                    **sample.details,
                                }
                            ).decode(),
                        }
                    )
                for item, value in sample.values.items():
                    row = {
                        "endpoint_id": endpoint,
                        "run_id": run_id,
                        "check_id": check,
                        "item_id": item,
                        "outcome": value,
                        "repeats": sample.repeats.get(item, []),
                    }
                    derived["item_scores"].append(
                        {
                            "record_id": f"{run_id}/{endpoint}/{check}/{item}",
                            "record_json": canonical_json(row).decode(),
                        }
                    )
        for check_result in analysis.checks:
            derived["checks"].append(
                {
                    "record_id": f"{run_id}/{check_result.endpoint_id}/{check_result.check_id}",
                    "record_json": canonical_json(check_result.model_dump(mode="json")).decode(),
                }
            )
        for diagnosis in diagnoses(analysis.checks):
            derived["diagnoses"].append(
                {
                    "record_id": diagnosis.diagnosis_id,
                    "record_json": canonical_json(diagnosis.model_dump(mode="json")).decode(),
                }
            )
    derived.update(observatory_rows(raw, records))
    for name, rows in derived.items():
        tables[name] = pl.DataFrame(
            rows, schema={"record_id": pl.String, "record_json": pl.String}
        ).sort("record_id")
    for name, table in tables.items():
        temporary = out / f"{name}.next.parquet"
        table.write_parquet(temporary)
        temporary.replace(out / f"{name}.parquet")
    build_catalog(out, TABLE_NAMES)
    return {name: tables[name].height for name in TABLE_NAMES}
