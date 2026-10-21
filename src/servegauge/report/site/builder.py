"""Build an explicit, content-removed selection without exporting the raw store."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from importlib.resources import files
from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.redaction import Redactor
from servegauge.core.secrets import scan_paths
from servegauge.observatory.notice_workflow import publication_status
from servegauge.observatory.notices import aware, failure_subject
from servegauge.report.endpoint_report import (
    _interval,
    _number,
    endpoint_context,
    write_endpoint_report,
)
from servegauge.report.site.observatory import observatory_context
from servegauge.storage.analysis import Analysis, reconstruct

VERSION = "static-site-v1"
VALIDATION = (
    "offline-fault-acceptance",
    "offline-studies-acceptance",
    "robust-statistical-rerun",
    "product-fingerprint-wire",
    "product-fingerprint-qualification-runner",
    "product-M11-completion",
    "product-report-renderer",
)


def artifact_hashes(out: Path) -> dict[str, str]:
    """Inventory only this owned artifact; refuse links rather than follow them."""
    if out.is_symlink():
        raise ValueError("site_output_symlink")
    result = {}
    for path in sorted(out.rglob("*")):
        if path.is_symlink():
            raise ValueError("site_output_symlink")
        if path.is_file() and path != out / "site-manifest.json":
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            result[path.relative_to(out).as_posix()] = digest.hexdigest()
    return result


def verify_site(out: Path) -> dict[str, Any]:
    """Validate a completed content-removed inventory; this is not publication approval."""
    manifest = out / "site-manifest.json"
    if out.is_symlink() or manifest.is_symlink() or not manifest.is_file():
        raise ValueError("site_output_manifest_required")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("version") != VERSION
        or value.get("redact_content") is not True
        or value.get("files") != artifact_hashes(out)
        or not {"index.html", "site.json", "methods.html", "validation.html"}.issubset(
            value["files"]
        )
    ):
        raise ValueError("site_output_integrity_mismatch")
    return dict(value)


def _band(value: Any) -> str:
    return _interval(*value) if isinstance(value, list) and len(value) == 2 else "unavailable"


def publication_preflight(out: Path, *, root: Path = Path(".")) -> dict[str, Any]:
    """Revalidate current selected failures and owner records; never publish."""
    manifest = verify_site(out)
    index = json.loads((out / "site.json").read_text(encoding="utf-8"))
    try:
        stamp = aware(datetime.fromisoformat(index["publication_as_of"]))
        if stamp > datetime.now(UTC):
            raise ValueError("site_publication_future_snapshot")
        analyses = [reconstruct(root / "data/raw", run) for run in manifest["runs"]]
        expected = site_context(analyses)
        expected.update(observatory_context(root, analyses, as_of=stamp))
        if index != expected:
            raise ValueError("site_publication_stale_selection")
        assertions = 0
        for analysis in analyses:
            for check in analysis.checks:
                if check.verdict != "FAIL":
                    continue
                subject = failure_subject(analysis, check.endpoint_id, check.check_id)
                state = publication_status(root, subject)
                if not state["eligible"]:
                    raise ValueError("site_publication_requires_right_of_reply_workflow")
                assertions += state.get("owner_asserted") is True
    except (KeyError, TypeError, OSError, ValueError) as error:
        raise ValueError("site_publication_requires_current_evidence_and_right_of_reply") from error
    if scan_paths([out / name for name in manifest["files"]]):
        raise ValueError("site_publication_secret_scan_failed")
    return {
        "eligible_scope": "Explicit recorded selection under current local notice policy",
        "files": len(manifest["files"]),
        "owner_asserted_notices": assertions,
        "delivery_verified": False,
        "network_writes": 0,
        "published": False,
    }


def site_context(analyses: list[Analysis]) -> dict[str, Any]:
    """Project existing numeric cards; do not pool runs or infer change events."""
    endpoints = []
    for analysis in sorted(analyses, key=lambda a: a.manifest["run_id"]):
        run = analysis.manifest["run_id"]
        full = endpoint_context(analysis, redact_content=True)
        config = {endpoint.id: endpoint for endpoint in analysis.plan.endpoints}
        for endpoint_id in sorted({check.endpoint_id for check in analysis.checks}):
            declared = config[endpoint_id]
            checks = [check for check in analysis.checks if check.endpoint_id == endpoint_id]
            rows = [row for row in full["checks"] if row["endpoint_id"] == endpoint_id]
            records = [record for record in analysis.records if record.endpoint_id == endpoint_id]
            paid = [
                record
                for record in records
                if record.attempts
                and record.cost_usd is not None
                and record.cost_source in {"provider", "generation_record"}
            ]
            fingerprints = []
            for check in checks:
                if not check.check_id.startswith("FP"):
                    continue
                available = check.details.get("fingerprint_metrics")
                metrics = available.get(check.check_id) if isinstance(available, dict) else None
                metrics = metrics if isinstance(metrics, dict) else {}
                controls = metrics.get("control_positions")
                controls = controls if isinstance(controls, list) else []
                fingerprints.append(
                    {
                        **next(row for row in rows if row["check_id"] == check.check_id),
                        "position": check.details.get("fingerprint_position", "unavailable"),
                        "noise_text": _number(check.details.get("noise_band")),
                        "different_text": _band(metrics.get("different_model_band")),
                        "controls": [
                            {
                                "model": entry.get("model_identity", "unavailable"),
                                "position": entry.get("position", "unavailable"),
                                "band": _band(entry.get("different_model_band")),
                            }
                            for entry in controls
                            if isinstance(entry, dict)
                        ],
                    }
                )
            endpoints.append(
                {
                    "endpoint_id": endpoint_id,
                    "model": declared.model,
                    "profile": analysis.plan.profile,
                    "run_id": run,
                    "captured_at": str(analysis.manifest.get("created_at", "unavailable")),
                    "synthetic_mock": declared.kind == "mock",
                    "href": "endpoints/" + content_hash([run, endpoint_id])[:24] + ".html",
                    "report_href": "runs/" + content_hash(run)[:24] + "/index.html",
                    "counts": dict(Counter(check.verdict for check in checks)),
                    "suites": [
                        {
                            "id": suite,
                            "counts": dict(
                                Counter(
                                    check.verdict for check in checks if check.suite_id == suite
                                )
                            ),
                        }
                        for suite in sorted({check.suite_id for check in checks})
                    ],
                    "checks": rows,
                    "capabilities": [row for row in rows if row["suite_id"] == "S0"],
                    "fingerprints": fingerprints,
                    "cost": {
                        "reported_usd": str(
                            sum((Decimal(str(record.cost_usd)) for record in paid), Decimal(0))
                        )
                        if paid
                        else None,
                        "known_records": len(paid),
                        "sent_records": sum(bool(record.attempts) for record in records),
                        "attempts": sum(len(record.attempts) for record in records),
                        "unsent_records": sum(not record.attempts for record in records),
                    },
                    "change_events": None,
                }
            )
    models = sorted({endpoint["model"] for endpoint in endpoints})
    return dict(
        Redactor.from_environment().redact(
            {
                "run_count": len(analyses),
                "endpoints": endpoints,
                "models": [
                    {"id": model, "href": "models/" + content_hash(model)[:24] + ".html"}
                    for model in models
                ],
                "redact_content": True,
                "publication_ready": False,
                "scope": "Selected recorded captures; counts are not provider rankings. "
                "No cross-run pooling or weight/hardware attestation.",
            }
        )
    )


def _validations(root: Path) -> list[dict[str, Any]]:
    selected = []
    for name in VALIDATION:
        path = root / "docs/validation" / (name + ".json")
        if not path.is_file():
            continue
        data = path.read_bytes()
        value = json.loads(data)
        selected.append(
            {
                "name": name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "scope": value.get("accepted_component")
                or value.get("accepted_milestone")
                or value.get("scope")
                or value.get("study")
                or "Historical scoped receipt; see methods and repository evidence.",
                "tests": value.get("source_tests_passed", value.get("default_source_tests_passed")),
                "source_hash": value.get("source_hash"),
            }
        )
    return list(Redactor.from_environment().redact(selected))


def _sections(text: str) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for line in text.splitlines():
        if line.startswith(("# ", "## ", "### ")):
            result.append({"heading": line.lstrip("# "), "text": ""})
        else:
            if not result:
                result.append({"heading": "Methods", "text": ""})
            result[-1]["text"] += line + "\n"
    return result


def build_site(root: Path, out: Path, *, runs: list[str] | None = None) -> dict[str, Any]:
    """Create a fresh site or verify the same artifact; never erase prior output."""
    if out.is_symlink():
        raise ValueError("site_output_symlink")
    out = out.absolute()
    identifiers = sorted(set(runs or []))
    analyses = [reconstruct(root / "data/raw", run) for run in identifiers]
    context = site_context(analyses)
    previous = verify_site(out) if out.exists() else None
    stamp = (
        aware(datetime.fromisoformat(previous["publication_as_of"]))
        if previous and "publication_as_of" in previous
        else datetime.now(UTC)
    )
    context.update(observatory_context(root, analyses, as_of=stamp))
    methods_path = root / "docs/METHODS.md"
    methods = (
        methods_path.read_text(encoding="utf-8")
        if methods_path.is_file()
        else ("# Methods unavailable\nNo local docs/METHODS.md was supplied with this build.")
    )
    methods = Redactor.from_environment().text(methods)
    validation = _validations(root)
    resource = files("servegauge").joinpath("report")
    template = resource.joinpath("templates/site.html.jinja").read_text(encoding="utf-8")
    assets = {
        name: resource.joinpath("site", name).read_bytes() for name in ["site.css", "site.js"]
    }
    digest = content_hash(
        {
            "version": VERSION,
            "context": context,
            "methods": methods,
            "validation": validation,
            "template": template,
            "assets": {key: hashlib.sha256(value).hexdigest() for key, value in assets.items()},
            "source": [
                {
                    "manifest": a.manifest,
                    "metadata": a.metadata,
                    "plan": a.plan.model_dump(mode="json"),
                    "checks": [c.model_dump(mode="json") for c in a.checks],
                    "records": {
                        r.exchange_id: content_hash(r.model_dump(mode="json")) for r in a.records
                    },
                    "items": [i.model_dump(mode="json") for i in a.items],
                    "reference": a.reference_pack.model_dump(mode="json")
                    if a.reference_pack
                    else None,
                }
                for a in analyses
            ],
            "report_resources": {
                name: hashlib.sha256(resource.joinpath(name).read_bytes()).hexdigest()
                for name in [
                    "endpoint_report.py",
                    "charts.py",
                    "templates/report.html.jinja",
                    "site/report.js",
                    "site/report.css",
                    "vendor/manifest.json",
                    "vendor/vega-6.4.0.min.js",
                    "vendor/vega-LICENSE.txt",
                ]
            },
        }
    )
    result = {
        "out": str(out),
        "html": str(out / "index.html"),
        "runs": len(identifiers),
        "endpoints": len(context["endpoints"]),
        "redact_content": True,
        "publication_ready": False,
    }
    if out.exists():
        previous = verify_site(out)
        if previous["input_hash"] != digest:
            raise ValueError("site_output_inputs_changed")
        return result
    out.mkdir(parents=True, exist_ok=False)
    for analysis in analyses:
        write_endpoint_report(
            analysis,
            out / "runs" / content_hash(analysis.manifest["run_id"])[:24],
            redact_content=True,
            methods_path=methods_path if methods_path.is_file() else None,
        )
    (out / "assets").mkdir()
    for name, payload in assets.items():
        (out / "assets" / name).write_bytes(payload)
    environment = Environment(undefined=StrictUndefined, autoescape=True)
    compiled = environment.from_string(template)

    def render(path: str, page: str, title: str, prefix: str = "", **extra: Any) -> None:
        target = out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            compiled.render(
                **context, page=page, title=title, prefix=prefix, validation=validation, **extra
            ),
            encoding="utf-8",
        )

    render("index.html", "index", "Recorded endpoint evidence")
    for model in context["models"]:
        render(model["href"], "model", model["id"], "../", selected_model=model["id"])
    for endpoint in context["endpoints"]:
        render(endpoint["href"], "endpoint", endpoint["endpoint_id"], "../", endpoint=endpoint)
    render("methods.html", "methods", "Methods and limits", sections=_sections(methods))
    render("validation.html", "validation", "Validation evidence and limits")
    render("timeline.html", "timeline", "Capture history and change events")
    render("replies.html", "replies", "Right of reply")
    (out / "methods.md").write_text(methods, encoding="utf-8")
    (out / "site.json").write_bytes(canonical_json(context) + b"\n")
    (out / "validation.json").write_bytes(canonical_json(validation) + b"\n")
    (out / ".nojekyll").write_bytes(b"")
    manifest = {
        "version": VERSION,
        "input_hash": digest,
        "redact_content": True,
        "publication_ready": False,
        "runs": identifiers,
        "publication_as_of": stamp.isoformat(),
        "files": artifact_hashes(out),
    }
    (out / "site-manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    verify_site(out)
    return result
