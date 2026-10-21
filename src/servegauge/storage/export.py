"""Immutable selected monthly numeric artifacts and current publication preflight."""

from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.secrets import PATTERNS, configured_secrets
from servegauge.observatory.export import SCHEMAS, VERSION, numeric_tables
from servegauge.observatory.notice_workflow import publication_status
from servegauge.observatory.notices import aware, failure_subject
from servegauge.observatory.provenance import implementation_hash
from servegauge.report.site.observatory import observatory_context
from servegauge.storage.analysis import Analysis, reconstruct
from servegauge.storage.notices import guard_path

LOCAL = "local-selection.json"
DATASHEET = """# ServeGauge observatory datasheet

This is an explicit selection of recorded endpoint checks, not a representative
provider sample or a provider ranking. Each primary run remains separate. Monthly
partitions use primary capture time in UTC; event observation intervals are separate
columns. Missing/nonfinite measurements are null, not zero or passing evidence.
Empty selection produces schema-only tables and no discovered private runs.

The license for this numeric dataset is CC BY 4.0. No prompt, response, tool arguments,
benchmark text, headers, error messages, account identifiers, support-channel paths,
opaque metadata or generation records are exported. Only self-generated CC BY 4.0
items have per-item/exchange/cost rows; all external or unknown items are aggregate-only.
Raw item/run/exchange and direct endpoint identifiers become stable hash tokens.
These tokens are linkable and are not a guarantee of anonymity. Declared OpenRouter
model and route slugs remain; they do not verify model weights or hardware.

Checks retain recorded n, estimates, intervals, margins, reference tiers and outcomes.
Fingerprint rows are selected FP check summaries, not vectors or new qualification.
Diagnoses are nonunique hypotheses, not claims of intent. Use the corresponding
local reports/bundles for alternatives and replay. Repeated and shared captures
are not new independent observations; exchange tokens permit identifying reuse.
Cost rows are retained final-record amounts with their source, not all-attempt
invoices. Estimates and unknown amounts remain explicitly distinguished.

Events retain their original proof hash, counts and fixed-family scope; an incomplete
selection emits no event. Absence does not establish stability. MOCK events cover
their declared scripted process only. All MOCK selections are local-only and blocked
from public upload. No provider or production monitoring guarantee follows.

Notifications are exact-revision local owner assertions, never independent delivery
or public-source verification. Missing/draft records do not start a waiting period.
Changed/withdrawn notices or proof bundles require a fresh artifact. Publication
rechecks current source evidence and notice policy before freezing upload bytes.
No contact, schedule activation, inference call or upload follows a dry run.

The private local-selection.json binds raw selection and input hashes locally and
is excluded from the upload inventory. The public manifest enumerates only the
reviewed numeric files and documentation. Public uploads require owner confirmation,
the allow-publish variable, enabled destination configuration and credentials.
Monthly files, datasheets and manifests have content-bound snapshot paths. Older
snapshot files remain retained; only the root dataset card selects the latest
reviewed selection. The viewer does not pool old and new snapshots of captures.
"""


def _clean(blob: bytes) -> None:
    if any(p.search(blob) for p in PATTERNS) or any(
        s.encode() in blob for s in configured_secrets()
    ):
        raise ValueError("export_secret_scan_failed")


def _inventory(out: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    pending = [out]
    while pending:
        directory = pending.pop()
        guard_path(directory)
        for path in directory.iterdir():
            guard_path(path)
            if path.is_dir():
                pending.append(path)
            elif path.is_file():
                result[path.relative_to(out).as_posix()] = path.read_bytes()
            else:
                raise ValueError("export_foreign_entry")
    return result


def verify_export(out: Path) -> dict[str, Any]:
    guard_path(out)
    local = out / LOCAL
    guard_path(local)
    value = json.loads(local.read_bytes())
    if not isinstance(value, dict) or value.get("version") != VERSION:
        raise ValueError("export_manifest_invalid")
    files = value.get("files")
    if not isinstance(files, dict) or "README.md" not in files:
        raise ValueError("export_inventory_invalid")
    if (
        not isinstance(value.get("runs"), list)
        or any(not isinstance(r, str) for r in value["runs"])
        or value["runs"] != sorted(set(value["runs"]))
        or not isinstance(value.get("as_of"), str)
        or not isinstance(value.get("input_hash"), str)
        or not re.fullmatch(r"[a-f0-9]{64}", value["input_hash"])
        or any(
            not isinstance(k, str) or not isinstance(v, str) or not re.fullmatch(r"[a-f0-9]{64}", v)
            for k, v in files.items()
        )
    ):
        raise ValueError("export_manifest_invalid")
    aware(datetime.fromisoformat(value["as_of"]))
    snapshots = {
        name.split("/")[1]
        for name in files
        if re.fullmatch(r"snapshots/[a-f0-9]{64}/(?:DATASHEET\.md|manifest\.json)", name)
    }
    if len(snapshots) != 1:
        raise ValueError("export_single_snapshot_required")
    snapshot = snapshots.pop()
    documents = {
        "README.md",
        f"snapshots/{snapshot}/DATASHEET.md",
        f"snapshots/{snapshot}/manifest.json",
    }
    if not documents <= files.keys():
        raise ValueError("export_snapshot_documents_required")
    for name in files:
        if name not in documents and not re.fullmatch(
            r"data/(?:[0-9]{4}-(?:0[1-9]|1[0-2])|empty)/(?:"
            + "|".join(SCHEMAS)
            + ")-"
            + snapshot
            + r"\.parquet",
            name,
        ):
            raise ValueError("export_inventory_invalid")
    partitions = {name.split("/")[1] for name in files if name.startswith("data/")}
    if (
        not partitions
        or ("empty" in partitions and len(partitions) != 1)
        or any(
            f"data/{month}/{table}-{snapshot}.parquet" not in files
            for month in partitions
            for table in SCHEMAS
        )
    ):
        raise ValueError("export_complete_table_schemas_required")
    actual = _inventory(out)
    if set(actual) != {*files, LOCAL} or any(
        hashlib.sha256(actual[name]).hexdigest() != digest for name, digest in files.items()
    ):
        raise ValueError("export_inventory_changed")
    return dict(value)


def _blockers(root: Path, analyses: list[Analysis]) -> list[str]:
    if not analyses:
        return ["no_selected_runs"]
    blockers = set()
    for analysis in analyses:
        if (
            analysis.metadata.get("synthetic_mock") is True
            or analysis.plan.reference.tier == "MOCK"
            or any(e.kind == "mock" for e in analysis.plan.endpoints)
            or any(c.reference_tier == "MOCK" for c in analysis.checks)
        ):
            blockers.add("scripted_mock_not_public")
        for check in analysis.checks:
            if (
                check.verdict == "FAIL"
                and not publication_status(
                    root, failure_subject(analysis, check.endpoint_id, check.check_id)
                )["eligible"]
            ):
                blockers.add("right_of_reply_required")
    return sorted(blockers)


def _projection(
    root: Path, runs: list[str], stamp: datetime
) -> tuple[dict[str, bytes], str, list[str]]:
    analyses = [reconstruct(root / "data/raw", run) for run in runs]
    if [a.manifest["run_id"] for a in analyses] != runs:
        raise ValueError("export_selection_identity_changed")
    context = observatory_context(root, analyses, as_of=stamp)
    tables = numeric_tables(analyses, context)
    _clean(canonical_json(tables))
    payloads: dict[str, bytes] = {}
    counts = {}
    for month, rows in sorted(tables.items()):
        for name, schema in SCHEMAS.items():
            buffer = io.BytesIO()
            frame = pl.DataFrame(rows[name], schema=schema, strict=True)
            frame.write_parquet(buffer)
            relative = f"data/{month}/{name}.parquet"
            payloads[relative] = buffer.getvalue()
            counts[relative] = frame.height
    snapshot = content_hash(
        {
            "as_of": stamp.isoformat(),
            "source": implementation_hash(),
            "datasheet": DATASHEET,
            "files": {n: hashlib.sha256(b).hexdigest() for n, b in sorted(payloads.items())},
        }
    )
    payloads = {
        name.removesuffix(".parquet") + "-" + snapshot + ".parquet": blob
        for name, blob in payloads.items()
    }
    counts = {
        name.removesuffix(".parquet") + "-" + snapshot + ".parquet": count
        for name, count in counts.items()
    }
    configs = [
        {
            "config_name": name,
            "data_files": [
                {
                    "split": "observations",
                    "path": sorted(
                        p for p in payloads if p.endswith(f"/{name}-{snapshot}.parquet")
                    ),
                }
            ],
        }
        for name in SCHEMAS
    ]
    import yaml

    card = (
        "---\n"
        + yaml.safe_dump({"license": "cc-by-4.0", "configs": configs}, sort_keys=False)
        + "---\n\n# ServeGauge recorded numeric observations\n\n"
        f"See [the datasheet](snapshots/{snapshot}/DATASHEET.md) for scope, units, privacy, "
        "missingness, licensing and publication gates. Older snapshot files remain "
        "retained; this viewer selects only the current reviewed capture selection.\n"
    )
    payloads["README.md"] = card.encode()
    payloads[f"snapshots/{snapshot}/DATASHEET.md"] = DATASHEET.encode()
    payloads[f"snapshots/{snapshot}/manifest.json"] = (
        canonical_json(
            {
                "version": VERSION,
                "snapshot_id": snapshot,
                "as_of": stamp.isoformat(),
                "license": "cc-by-4.0",
                "row_counts": counts,
                "schemas": {
                    name: {k: str(v) for k, v in schema.items()} for name, schema in SCHEMAS.items()
                },
                "files": {
                    name: hashlib.sha256(blob).hexdigest()
                    for name, blob in sorted(payloads.items())
                },
            }
        )
        + b"\n"
    )
    for blob in payloads.values():
        _clean(blob)
    source = [
        {
            "manifest": a.manifest,
            "metadata": a.metadata,
            "plan": a.plan.model_dump(mode="json"),
            "checks": [c.model_dump(mode="json") for c in a.checks],
            "records": {r.exchange_id: content_hash(r.model_dump(mode="json")) for r in a.records},
            "items": [i.model_dump(mode="json") for i in a.items],
            "samples": {
                ep: {name: s.model_dump(mode="json") for name, s in values.items()}
                for ep, values in a.samples.items()
            },
            "reference": a.reference_pack.model_dump(mode="json") if a.reference_pack else None,
        }
        for a in analyses
    ]
    digest = content_hash(
        {
            "source": source,
            "context": context,
            "implementation": implementation_hash(),
            "polars": pl.__version__,
            "files": {
                name: hashlib.sha256(blob).hexdigest() for name, blob in sorted(payloads.items())
            },
        }
    )
    return payloads, digest, _blockers(root, analyses)


def stage_export(root: Path, out: Path, *, runs: list[str] | None = None) -> dict[str, Any]:
    if ".." in out.parts:
        raise ValueError("export_canonical_output_required")
    guard_path(root)
    guard_path(out)
    if any(
        out.absolute().is_relative_to(root.absolute() / name)
        for name in ["data", ".git", "src", "tests", "configs", "profiles", "refs", "budget"]
    ):
        raise ValueError("export_output_overlaps_private_or_source_tree")
    identifiers = sorted(set(runs or []))
    previous = verify_export(out) if out.exists() else None
    stamp = aware(datetime.fromisoformat(previous["as_of"])) if previous else datetime.now(UTC)
    payloads, digest, blockers = _projection(root, identifiers, stamp)
    value = {
        "version": VERSION,
        "runs": identifiers,
        "as_of": stamp.isoformat(),
        "input_hash": digest,
        "files": {
            name: hashlib.sha256(blob).hexdigest() for name, blob in sorted(payloads.items())
        },
    }
    local = canonical_json(value) + b"\n"
    _clean(local)
    if previous:
        if previous != value or any(
            (out / name).read_bytes() != blob for name, blob in payloads.items()
        ):
            raise ValueError("export_inputs_changed_use_fresh_output")
    else:
        out.mkdir(parents=True, exist_ok=False)
        for name, blob in payloads.items():
            path = out / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        (out / LOCAL).write_bytes(local)
        verify_export(out)
    return {
        "out": str(out.absolute()),
        "dry_run": True,
        "files": sorted(payloads),
        "datasheet": next(n for n in payloads if n.endswith("/DATASHEET.md")),
        "publication_blockers": blockers,
        "network_writes": 0,
        "published": False,
        "model_api_calls": 0,
    }


def publication_preflight(root: Path, out: Path) -> dict[str, bytes]:
    """Reconstruct current selection, then freeze only verified public upload bytes."""
    guard_path(root)
    local = verify_export(out)
    stamp = aware(datetime.fromisoformat(local["as_of"]))
    if stamp > datetime.now(UTC):
        raise ValueError("export_publication_future_snapshot")
    payloads, digest, blockers = _projection(root, local["runs"], stamp)
    if (
        blockers
        or digest != local["input_hash"]
        or any((out / name).read_bytes() != blob for name, blob in payloads.items())
    ):
        raise ValueError("export_publication_requires_current_reviewed_evidence")
    return payloads
