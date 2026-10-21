"""Strict, local HTML/Markdown reports without rewriting scientific source data."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from importlib.resources import files
from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined, select_autoescape

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.redaction import Redactor
from servegauge.diagnose.bundle import write_bundle
from servegauge.diagnose.issue import markdown_text
from servegauge.diagnose.rules import diagnoses
from servegauge.diagnose.summary import content_summary
from servegauge.report.charts import chart_payload
from servegauge.report.source_provenance import source_provenance
from servegauge.storage.analysis import Analysis

VERSION = "endpoint-report-v1"


def _href(value: str | None) -> str | None:
    if value is None:
        return None
    if (
        not value
        or "\\" in value
        or any(
            not re.fullmatch(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*", part)
            for part in value.split("/")
        )
    ):
        raise ValueError("unsafe_report_link")
    return value


def _number(value: Any) -> str:
    if type(value) not in {int, float} or not math.isfinite(value):
        return "unavailable"
    return f"{value:.6g}"


def _interval(low: Any, high: Any) -> str:
    values = [_number(low), _number(high)]
    return "unavailable" if "unavailable" in values else "[" + ", ".join(values) + "]"


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _document(analysis: Analysis, redact_content: bool) -> dict[str, Any]:
    scrubber = Redactor.from_environment()
    provenance = source_provenance(analysis.manifest, analysis.records)
    checks = []
    for check in analysis.checks:
        row = _json_safe(check.model_dump(mode="json"))
        if redact_content:
            row["details"] = content_summary(row["details"])
        checks.append(row)
    return dict(
        scrubber.redact(
            {
                "metadata": {
                    "report_version": VERSION,
                    "run_id": analysis.manifest["run_id"],
                    "profile": analysis.plan.profile,
                    "redact_content": redact_content,
                    "synthetic_mock": all(e.kind == "mock" for e in analysis.plan.endpoints),
                    "logical_records": len(analysis.records),
                    "sent_records": sum(bool(r.attempts) for r in analysis.records),
                    "recorded_attempts": sum(len(r.attempts) for r in analysis.records),
                    "unsent_records": sum(not r.attempts for r in analysis.records),
                    "captured_source": provenance["context"]["source_tree_hash"],
                    "git_sha": provenance["context"]["git_sha"],
                    "source_provenance": provenance,
                    "capture_config_hash": analysis.manifest.get("config_hash"),
                },
                "checks": checks,
                "diagnoses": [d.model_dump(mode="json") for d in diagnoses(analysis.checks)],
            }
        )
    )


def endpoint_context(
    analysis: Analysis,
    *,
    redact_content: bool = False,
    bundle_hrefs: dict[str, str] | None = None,
    methods_href: str | None = None,
) -> dict[str, Any]:
    """Format recorded numbers; never recompute a statistic or alter a verdict."""
    methods_href = _href(methods_href)
    links = {key: _href(value) for key, value in (bundle_hrefs or {}).items()}
    scrubber = Redactor.from_environment()
    by_id = {e.id: e for e in analysis.plan.endpoints}
    observed = sorted({c.endpoint_id for c in analysis.checks})
    labels = {
        key: scrubber.text(
            (by_id[key].labels.get("provider_name") or key)
            if key in by_id and not redact_content
            else key
        )
        for key in observed
    }
    endpoints = [{"id": key, "label": labels[key]} for key in observed]
    payload = chart_payload(analysis.checks, endpoints)
    counts: Counter[str] = Counter(c.verdict for c in analysis.checks)
    summaries = [
        {"verdict": key, "filter": key, "label": label, "count": counts[key], "description": text}
        for key, label, text in [
            ("PASS", "Pass", "Supported within the recorded check's scope."),
            ("FAIL", "Fail", "Recorded requirement/reference deviation."),
            ("ERROR", "Error", "Evidence or harness could not complete the check."),
            ("INCONCLUSIVE", "Inconclusive", "Evidence is insufficient for a decision."),
        ]
    ]
    summaries.append(
        {
            "verdict": "INFO",
            "filter": "other",
            "label": "Other",
            "count": sum(counts[k] for k in ["INFO", "NOT_SUPPORTED", "NOT_APPLICABLE"]),
            "description": "Information, unavailable features or inapplicable checks.",
        }
    )
    rows = []
    for index, check in enumerate(analysis.checks):
        ratio = check.details.get("geometric_mean_ratio")
        raw_bounds = check.details.get("ratio_interval")
        ratio_bounds: list[Any] = raw_bounds if isinstance(raw_bounds, list) else []
        has_ratio = type(ratio) in {int, float} and len(ratio_bounds) == 2
        rows.append(
            {
                "row_id": "check-" + content_hash([check.endpoint_id, check.check_id, index])[:16],
                "check_id": check.check_id,
                "suite_id": check.suite_id,
                "check_class": check.check_class,
                "endpoint_id": check.endpoint_id,
                "endpoint_label": labels[check.endpoint_id],
                "verdict": check.verdict,
                "verdict_label": check.verdict
                + (" · operational alarm" if check.details.get("operational_alarm") else ""),
                "n": check.n,
                "requirement_ids": check.requirement_ids,
                "estimate_text": ("ratio " + _number(ratio))
                if has_ratio
                else _number(check.estimate),
                "interval_text": _interval(*ratio_bounds)
                if has_ratio
                else _interval(check.ci_low, check.ci_high),
                "interval_label": str(check.details.get("interval_kind", "95% CI"))
                if check.estimate is not None
                else "",
                "delta_text": _number(check.delta),
                "delta_interval_text": _interval(check.delta_ci_low, check.delta_ci_high),
                "reference_tier": check.reference_tier
                or ("invariant expectation" if check.check_class == "invariant" else "unavailable"),
                "reference_text": _number(check.reference_estimate),
                "margin_text": _number(check.margin),
                "adjusted_p_text": _number(check.adjusted_p),
                "p_text": _number(check.p_value),
                "mde_text": _number(check.mde),
                "reason": scrubber.text(check.reason),
                "evidence_count": len(check.evidence),
                "evidence_ids": check.evidence[:5],
                "search": " ".join(
                    [check.check_id, check.suite_id, scrubber.text(check.reason)]
                ).lower(),
            }
        )
    found = []
    for d in diagnoses(analysis.checks):
        found.append(
            {
                **d.model_dump(mode="json"),
                "endpoint_label": labels[d.endpoint_id],
                "evidence_count": len(d.evidence),
                "bundle_href": links.get(d.diagnosis_id),
            }
        )
    metadata = _document(analysis, redact_content)["metadata"]
    provenance = metadata["source_provenance"]
    scope_labels = {
        "loaded-package-python-v1": "Loaded package Python source",
        "legacy-cwd-git-inventory": "Legacy working-directory inventory",
        "unknown": "Unknown code inventory",
    }
    provenance_rows = [
        {
            "label": key.replace("_", " ").capitalize(),
            "value": str(value) if value is not None else "unavailable",
        }
        for key, value in metadata.items()
        if key != "source_provenance"
    ]
    provenance_rows.extend(
        [
            {
                "label": "Context inventory scope",
                "value": scope_labels[provenance["context"]["scope"]],
            },
            {
                "label": "Attempt provenance",
                "value": (
                    f"{provenance['loaded_package_attempts']} of "
                    f"{provenance['recorded_attempts']} attempts with package-source identity; "
                    f"{provenance['legacy_attempts']} legacy; "
                    f"{provenance['unknown_attempts']} unknown"
                ),
            },
        ]
    )
    provenance_rows.extend(
        {
            "label": f"{scope_labels[row['scope']]} · {row['attempts']} attempts",
            "value": row["source_tree_hash"],
        }
        for row in provenance["sources"]
    )
    synthetic = metadata["synthetic_mock"]
    return dict(
        scrubber.redact(
            {
                "title": "Recorded endpoint checks",
                "profile": analysis.plan.profile,
                "description": f"{metadata['recorded_attempts']} recorded attempts; "
                f"{metadata['unsent_records']} unsent logical records. "
                "Missing measurements remain unavailable.",
                "captured_at": str(analysis.manifest.get("created_at", "unavailable")),
                "run_id": analysis.manifest["run_id"],
                "short_run_id": analysis.manifest["run_id"][:24],
                "scope_label": "Scripted mock evidence"
                if synthetic
                else "Recorded endpoint evidence",
                "scope_note": "Results apply to the recorded probes, references and sampling "
                "settings. No provider intent, model-weight or hardware certification.",
                "total_checks": len(rows),
                "endpoint_count": len(observed),
                "capture_count": metadata["sent_records"],
                "summaries": summaries,
                "reference_label": ", ".join(
                    sorted({c.reference_tier for c in analysis.checks if c.reference_tier})
                )
                or "invariant expectations",
                "legend": payload["legend"],
                "counts": payload["counts"],
                "chart_payload": payload,
                "endpoints": endpoints,
                "checks": rows,
                "diagnoses": found,
                "asset_prefix": "assets/",
                "methods_href": methods_href,
                "provenance_note": "Reconstructed from sealed raw inputs. Hashes detect content "
                "changes; they do not attest an external server.",
                "provenance": provenance_rows,
            }
        )
    )


def _markdown(context: dict[str, Any]) -> str:
    def text(value: Any) -> str:
        return markdown_text(value)

    lines = [
        "# ServeGauge recorded checks",
        "",
        text(context["scope_label"]),
        "",
        "Run: " + text(context["run_id"]),
        "",
        text(context["description"]),
        "",
        "| Check | Endpoint | Outcome | n | Estimate / interval | Delta / interval | Reference "
        "| Margin / adjusted p | Reason |",
        "|---|---|---|---:|---|---|---|---|---|",
    ]
    for row in context["checks"]:
        values = [
            row["check_id"],
            row["endpoint_label"],
            row["verdict_label"],
            row["n"],
            row["estimate_text"] + " " + row["interval_text"],
            row["delta_text"] + " " + row["delta_interval_text"],
            row["reference_tier"] + " / " + row["reference_text"],
            row["margin_text"] + " / " + row["adjusted_p_text"],
            row["reason"],
        ]
        lines.append("| " + " | ".join(text(value) for value in values) + " |")
    for d in context["diagnoses"]:
        lines.extend(
            [
                "",
                "## " + text(d["code"]),
                "",
                text(d["failure_class"]),
                "",
                text(d["summary"]),
                "",
                "Confidence: " + text(d["confidence"]),
                "",
                "Alternatives: " + "; ".join(text(v) for v in d["alternatives"]),
                "",
                "Falsifiers: " + "; ".join(text(v) for v in d["falsifiers"]),
            ]
        )
        if d["bundle_href"]:
            lines.extend(["", "[Local evidence bundle](" + d["bundle_href"] + ")"])
    lines.extend(
        [
            "",
            "Confidence is an evidence rule, not a probability of cause.",
            "No certification of weights, hardware, quantization or provider intent.",
            "",
        ]
    )
    return "\n".join(lines)


def _sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _output_files(out: Path) -> dict[str, str]:
    result = {}
    for path in sorted(out.rglob("*")):
        if path.is_symlink():
            raise ValueError("existing_report_output_symlink")
        if path.is_file() and path != out / "report-manifest.json":
            result[path.relative_to(out).as_posix()] = _sha(path)
    return result


def write_endpoint_report(
    analysis: Analysis,
    out: Path,
    *,
    redact_content: bool = False,
    methods_path: Path | None = None,
) -> dict[str, Any]:
    """Create a fresh artifact or verify an identical one; never erase old output."""
    if out.is_symlink():
        raise ValueError("existing_report_output_symlink")
    out = out.absolute()
    resource = files("servegauge").joinpath("report")
    vendor = json.loads(resource.joinpath("vendor/manifest.json").read_text(encoding="utf-8"))
    assets = {
        name: resource.joinpath("site", name).read_bytes() for name in ["report.js", "report.css"]
    }
    for name, entry in vendor["files"].items():
        data = resource.joinpath("vendor", name).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"] or len(data) != entry["bytes"]:
            raise ValueError("report_vendor_checksum_mismatch")
        assets[name] = data
    template = resource.joinpath("templates/report.html.jinja").read_text(encoding="utf-8")
    methods = methods_path.read_bytes() if methods_path is not None else None
    document = _document(analysis, redact_content)
    # Bind every input used by local bundles as well as the visible report. Only
    # this digest is public; private payloads never enter the report manifest.
    source = {
        "plan": analysis.plan.model_dump(mode="json"),
        "items": [item.model_dump(mode="json") for item in analysis.items],
        "records": {
            record.exchange_id: content_hash(record.model_dump(mode="json"))
            for record in analysis.records
        },
        "manifest": analysis.manifest,
        "metadata": analysis.metadata,
        "reference": analysis.reference_pack.model_dump(mode="json")
        if analysis.reference_pack
        else None,
    }
    digest = content_hash(
        {
            "document": document,
            "source_hash": content_hash(source),
            "template": template,
            "assets": {k: hashlib.sha256(v).hexdigest() for k, v in assets.items()},
            "methods": hashlib.sha256(methods).hexdigest() if methods is not None else None,
        }
    )
    result = {
        "run_id": analysis.manifest["run_id"],
        "checks": len(analysis.checks),
        "out": str(out),
        "html": str(out / "index.html"),
        "diagnoses": document["diagnoses"],
        "redact_content": redact_content,
    }
    if out.exists():
        manifest = out / "report-manifest.json"
        if not out.is_dir() or not manifest.is_file() or manifest.is_symlink():
            raise ValueError("existing_report_output_conflict")
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            not isinstance(previous, dict)
            or previous.get("version") != VERSION
            or previous.get("input_hash") != digest
            or previous.get("files") != _output_files(out)
        ):
            raise ValueError("existing_report_output_conflict")
        return result
    out.mkdir(parents=True, exist_ok=False)
    links = {}
    for d in diagnoses(analysis.checks):
        path = write_bundle(analysis, d, root=out, redact_content=redact_content)
        links[d.diagnosis_id] = str(_href((path / "README.md").relative_to(out).as_posix()))
    context = endpoint_context(
        analysis,
        redact_content=redact_content,
        bundle_hrefs=links,
        methods_href="methods.md" if methods is not None else None,
    )
    environment = Environment(
        undefined=StrictUndefined,
        autoescape=select_autoescape(["html", "jinja"], default_for_string=True),
    )
    rendered = environment.from_string(template).render(context)
    (out / "assets").mkdir()
    for name, data in assets.items():
        (out / "assets" / name).write_bytes(data)
    if methods is not None:
        (out / "methods.md").write_bytes(methods)
    (out / "index.html").write_text(rendered, encoding="utf-8")
    (out / "checks.md").write_text(_markdown(context), encoding="utf-8")
    (out / "checks.json").write_bytes(canonical_json(document) + b"\n")
    manifest_data = {
        "version": VERSION,
        "input_hash": digest,
        "redact_content": redact_content,
        "run_id": analysis.manifest["run_id"],
        "files": _output_files(out),
    }
    (out / "report-manifest.json").write_bytes(canonical_json(manifest_data) + b"\n")
    return result
