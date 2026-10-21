"""Reconstruct complete selected quartets before joining labels or rendering a report."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.embeddings.local import LocalEncoder
from servegauge.embeddings.runtime import get_encoder
from servegauge.scoring.common import Samples
from servegauge.scoring.gpu_study import METHODS, score_quartet
from servegauge.selfhost.study_recipe import VARIANTS
from servegauge.stats.gpu_roc import roc_analysis
from servegauge.storage.gpu_capture import (
    _entry,
    _read,
    _source_pin,
    capture_view,
    load_selected_slot,
    study_context,
)
from servegauge.storage.gpu_study import _paths, _write
from servegauge.storage.source_identity import capture_source
from servegauge.verdicts.config import DecisionPolicy


def _measurement_seed(samples: list[dict[str, Samples]]) -> int:
    """Keep raw provenance intact without letting checkout state change resampling."""
    observed = []
    for side in samples:
        methods = {}
        for method, sample in side.items():
            if sample.metric != "fingerprint":
                continue
            methods[method] = {
                "design_hash": sample.details.get("design_hash"),
                "sampling_design": sample.details.get("sampling_design"),
                "observations": [
                    {
                        **row,
                        "metadata": {
                            key: value
                            for key, value in row.get("metadata", {}).items()
                            if key != "execution_source"
                        },
                    }
                    for row in sample.details.get("observations", [])
                ],
            }
        observed.append(methods)
    return int(content_hash(observed)[:16], 16)


def _encoding(
    values: list[dict[str, Any]], encoder: LocalEncoder | None
) -> tuple[LocalEncoder | None, str | None]:
    if encoder is not None:
        return encoder, None
    selected = next(
        (value["samples"]["FP4"] for value in values if "FP4" in value["samples"]), None
    )
    if selected is None:
        return None, "no_selected_FP4_capture"
    source = os.environ.get("SERVEGAUGE_EMBED_ASSETS")
    if source is not None:
        _paths(Path(source), Path(os.environ.get("SERVEGAUGE_EMBED_CACHE", ".cache/embeddings")))
    try:
        return get_encoder(selected), None
    except ValueError:
        return None, "pinned_local_embedding_runtime_unavailable"


def analyze_study(
    root: Path, kit: Path, kit_hash: str, *, encoder: LocalEncoder | None = None
) -> dict[str, Any]:
    context = study_context(root, kit, kit_hash)
    tier = "MOCK" if context.spec.scope == "mock_software_qa" else "BASELINE"
    policy = DecisionPolicy.model_validate(context.assets["policy"])
    with capture_view(context):
        captured: dict[str, dict[str, Any]] = {}
        pending: list[str] = []
        seen_ids: set[str] = set()
        seen_files: set[str] = set()
        selected_hashes: dict[str, str] = {}
        for slot in context.matrix:
            value = load_selected_slot(context, slot)
            if value is None:
                intent = _entry(context, slot, "intent")
                if intent.exists():
                    record = _read(intent)
                    if (
                        record.get("kit_hash") != kit_hash
                        or record.get("slot") != slot
                        or record.get("version") != "gpu-study-intent-v1"
                    ):
                        raise ValueError("study_pending_intent_binding_changed")
                    pending.append(slot["slot_id"])
                continue
            ids = value["capture_ids"]
            files = set(value["binding"]["source"]["files_sha256"])
            if seen_ids.intersection(ids) or seen_files.intersection(files):
                raise ValueError("study_capture_reuse_across_quartets_or_partitions")
            seen_ids.update(ids)
            seen_files.update(files)
            captured[slot["slot_id"]] = value
            selected_hashes[slot["slot_id"]] = value["binding_hash"]
        encoder, embedding_reason = _encoding(list(captured.values()), encoder)
        runtime_cohorts = sorted(
            {
                (
                    value["observed_runtime"]["gpu_observed"],
                    value["observed_runtime"]["driver_version"],
                )
                for value in captured.values()
                if value["observed_runtime"] is not None
            }
        )
        matched_runtime = len(runtime_cohorts) <= 1
        # Measurement phase: condition labels are not passed to the scorer or its seed.
        measurements: dict[str, dict[str, Any]] = {}
        groups: dict[str, list[dict[str, Any]]] = {}
        for slot in context.matrix:
            groups.setdefault(slot["unit_id"], []).append(slot)
        for unit_id, slots in groups.items():
            samples = [
                captured[slot["slot_id"]]["samples"] if slot["slot_id"] in captured else {}
                for slot in slots
            ]
            seed = _measurement_seed(samples)
            metrics = score_quartet(samples, policy=policy, seed=seed, encoder=encoder)
            complete = all(slot["slot_id"] in captured for slot in slots)
            for row in metrics.values():
                row["reference_tier"] = tier
                if not complete:
                    row["reason"] = "missing_capture_binding"
                if not matched_runtime:
                    row.update(
                        estimate=None,
                        ci_low=None,
                        ci_high=None,
                        n_components=0,
                        reason="unmatched_observed_runtime_cohort",
                    )
            measurements[unit_id] = {
                "measurement_seed": seed,
                "metrics": metrics,
                "complete_capture_bindings": complete,
            }
        # Label join is separate and occurs only after every raw-derived score exists.
        units = [
            {
                "unit_id": unit_id,
                "partition": slots[0]["partition"],
                "condition": slots[0]["condition"],
                "replicate": slots[0]["replicate"],
                "slot_ids": [slot["slot_id"] for slot in slots],
                **measurements[unit_id],
            }
            for unit_id, slots in groups.items()
        ]
        curves: dict[str, dict[str, Any]] = {}
        for condition in VARIANTS:
            if condition == "bf16":
                continue
            curves[condition] = {}
            for method in METHODS:
                rows = [
                    {
                        "unit_id": unit["unit_id"],
                        "partition": unit["partition"],
                        "positive": unit["condition"] == condition,
                        "score": unit["metrics"][method]["estimate"],
                    }
                    for unit in units
                    if unit["condition"] in {"bf16", condition}
                ]
                result = roc_analysis(
                    rows,
                    alpha=policy.alpha,
                    independent=context.spec.design.independence_reviewed
                    and context.spec.iid_condition_quartets_reviewed,
                )
                result["reference_tier"] = tier
                curves[condition][method] = result
        # All records have been reconstructed under the exact prepared source/assets.
        source = capture_source()
        records = [record for value in captured.values() for record in value["analysis"].records]
        value = {
            "version": "gpu-study-results-v1",
            "kit_hash": kit_hash,
            "scope": context.spec.scope,
            "source_hash": source["source_tree_hash"],
            "reference_tier": tier,
            "status": "INCONCLUSIVE",
            "model_repository": context.assets["profile"]["weights"],
            "model_revision": context.assets["profile"]["tokenizer"]["hf_revision"],
            "physical_precision_verified": False,
            "provider_qualified": False,
            "capture_summary": {
                "planned_slots": len(context.matrix),
                "bound_slots": len(captured),
                "pending_intents": len(pending),
                "missing_slots": len(context.matrix) - len(captured),
                "planned_quartets": len(units),
                "complete_quartets": sum(unit["complete_capture_bindings"] for unit in units),
                "unique_stored_capture_records": len(seen_ids),
                "recorded_endpoint_attempts": sum(len(r.attempts) for r in records),
            },
            "capture_statuses": {
                slot["slot_id"]: {
                    "bound": slot["slot_id"] in captured,
                    "pending_intent": slot["slot_id"] in pending,
                    "runtime_interval_valid": captured[slot["slot_id"]]["runtime_interval_valid"]
                    if slot["slot_id"] in captured
                    else None,
                    "methods": {
                        method: {
                            "status": sample.status,
                            "reason": sample.reason,
                            "expected_requests": sample.details.get("expected_requests"),
                            "requests": sample.details.get("requests"),
                            "stored_records": sample.details.get("captures"),
                            "missing_responses": sample.details.get("missing_responses"),
                            "missing_logprobs": sample.details.get("missing_logprobs"),
                            "censored_responses": sample.details.get("censored_responses"),
                        }
                        for method, sample in captured[slot["slot_id"]]["samples"].items()
                        if sample.metric == "fingerprint"
                    }
                    if slot["slot_id"] in captured
                    else {},
                }
                for slot in context.matrix
            },
            "complete_quartets_scope": (
                "four sealed capture bindings; not a guarantee of all methods, "
                "independence or power"
            ),
            "runtime_cohort": {
                "applicable": context.spec.scope == "owner_gpu",
                "matched": matched_runtime,
                "observations": runtime_cohorts,
                "scope": (
                    "observed GPU/driver strings; image/version pins checked per capture, "
                    "not a hardware certificate"
                ),
            },
            "assumptions": {
                "independent_quartets": context.spec.design.independence_reviewed,
                "within_capture_components": context.spec.within_capture_components_reviewed,
                "IID_condition_quartets": context.spec.iid_condition_quartets_reviewed,
                "basis": "explicit_operator_declaration; not_independently_verified_for_providers",
            },
            "embedding": {
                "available": encoder is not None,
                "unavailable_reason": embedding_reason,
                "model": context.assets["embedding_model"],
                "runtime": encoder.identity if encoder is not None else None,
            },
            "runtime": {
                "python": platform.python_version(),
                **{name: version(name) for name in ("numpy", "scipy")},
            },
            "between_curve_scope": (
                "Shared BF16 negative units; no independence or simultaneous coverage across curves"
            ),
            "interval_scope": (
                "Complete-case pointwise conditional Wilson and independent-unit "
                "bounded-differences AUC; not universal coverage"
            ),
            "inference_scope": (
                "Configured experiment contrasts only; no serving PASS/FAIL, hardware "
                "precision or weight certificate"
            ),
            "pending_slot_ids": pending,
            "selected_binding_hashes": selected_hashes,
            "selected_raw_sources": {
                slot_id: value["binding"]["source"] for slot_id, value in captured.items()
            },
            "units": units,
            "comparisons": curves,
        }
        # Detect edits while the selected immutable data were being scored.
        for slot in context.matrix:
            if slot["slot_id"] in captured and (
                content_hash(_read(_entry(context, slot, "binding")))
                != selected_hashes[slot["slot_id"]]
                or _source_pin(root, slot["run_id"])
                != captured[slot["slot_id"]]["binding"]["source"]
            ):
                raise ValueError("study_selected_source_changed_during_analysis")
        study_context(root, kit, kit_hash)
        return value


def _number(value: Any) -> str:
    return "NA" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)


def _csv_rows(value: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    fields = [
        "condition",
        "method",
        "reference_tier",
        "auc",
        "auc_low",
        "auc_high",
        "n_negative",
        "n_positive",
        "missing_negative",
        "missing_positive",
        "threshold",
        "fpr",
        "fpr_low",
        "fpr_high",
        "tpr",
        "tpr_low",
        "tpr_high",
        "scope",
    ]
    rows = []
    for condition, methods in value["comparisons"].items():
        for method, curve in methods.items():
            auc, missing, point = (
                curve["auc"],
                curve["missing"]["evaluation"],
                curve["operating_point"],
            )
            threshold = (
                "unavailable"
                if point is None
                else "never_flag"
                if point["threshold"] is None
                else point["threshold"]
            )
            fpr = point["fpr"] if point is not None else {}
            tpr = point["tpr"] if point is not None else {}
            rows.append(
                [
                    condition,
                    method,
                    value["reference_tier"],
                    auc["estimate"],
                    auc["ci_low"],
                    auc["ci_high"],
                    auc["n_negative"],
                    auc["n_positive"],
                    missing["negative"],
                    missing["positive"],
                    threshold,
                    fpr.get("estimate"),
                    fpr.get("ci_low"),
                    fpr.get("ci_high"),
                    tpr.get("estimate"),
                    tpr.get("ci_low"),
                    tpr.get("ci_high"),
                    value["scope"],
                ]
            )
    return fields, rows


def _plots(value: dict[str, Any], root: Path) -> dict[str, bytes]:
    _paths(Path(os.environ.get("MPLCONFIGDIR", str(root / ".cache/matplotlib"))))
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"svg.hashsalt": "servegauge-gpu-study-v1"}):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        for axis, (condition, methods) in zip(axes.flat, value["comparisons"].items(), strict=True):
            available = False
            for method, curve in methods.items():
                if curve["auc"]["estimate"] is None:
                    continue
                points = [
                    point
                    for point in curve["curve"]
                    if point["fpr"]["estimate"] is not None and point["tpr"]["estimate"] is not None
                ]
                (line,) = axis.plot(
                    [p["fpr"]["estimate"] for p in points],
                    [p["tpr"]["estimate"] for p in points],
                    marker=".",
                    label=f"{method}; n={curve['auc']['n_negative']}/{curve['auc']['n_positive']}",
                )
                point = curve["operating_point"]
                if (
                    point is not None
                    and point["fpr"]["ci_low"] is not None
                    and point["tpr"]["ci_low"] is not None
                ):
                    x, y = point["fpr"]["estimate"], point["tpr"]["estimate"]
                    axis.errorbar(
                        x,
                        y,
                        xerr=[
                            [max(0, x - point["fpr"]["ci_low"])],
                            [max(0, point["fpr"]["ci_high"] - x)],
                        ],
                        yerr=[
                            [max(0, y - point["tpr"]["ci_low"])],
                            [max(0, point["tpr"]["ci_high"] - y)],
                        ],
                        alpha=0.45,
                        capsize=2,
                        color=line.get_color(),
                        fmt="none",
                    )
                available = True
            axis.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.7)
            axis.set(
                xlim=(-0.03, 1.03),
                ylim=(-0.03, 1.03),
                title=condition,
                xlabel="False-positive rate",
                ylabel="True-positive rate",
            )
            if available:
                axis.legend(fontsize=7)
            else:
                axis.text(
                    0.5,
                    0.5,
                    "Unavailable: incomplete selected quartets",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                    wrap=True,
                )
        label = (
            "SOFTWARE QA - no physical GPU result"
            if value["scope"] == "mock_software_qa"
            else "Configured self-host study - precision unverified"
        )
        fig.suptitle(
            label + "\nCalibration-grid ROC; operating-point intervals are pointwise", fontsize=12
        )
        result = {}
        for suffix in ("svg", "png"):
            stream = io.BytesIO()
            fig.savefig(
                stream, format=suffix, metadata={"Date": None} if suffix == "svg" else None, dpi=140
            )
            result["roc." + suffix] = stream.getvalue()
        plt.close(fig)
        return result


def write_study_report(root: Path, value: dict[str, Any], *, out: Path) -> dict[str, Any]:
    _paths(root, out)
    namespace = root.absolute() / "reports/gpu-studies"
    if not out.absolute().is_relative_to(namespace) or out.absolute() == namespace or out.exists():
        raise ValueError("study_fresh_private_report_directory_required")
    fields, rows = _csv_rows(value)
    csv_file = io.StringIO(newline="")
    writer = csv.writer(csv_file, lineterminator="\n")
    writer.writerow(fields)
    writer.writerows([_number(item) for item in row] for row in rows)
    title = (
        "SOFTWARE QA: GPU study workflow"
        if value["scope"] == "mock_software_qa"
        else "Configured self-host study"
    )
    intro = (
        "No physical GPU measurements are present."
        if value["scope"] == "mock_software_qa"
        else (
            "Observed owner-supplied runtime records do not independently certify "
            "weights or physical precision."
        )
    )
    summary = value["capture_summary"]
    lines = [
        "# " + title,
        "",
        intro,
        "",
        "Status: INCONCLUSIVE. Descriptive contrasts do not grant a serving PASS/FAIL.",
        f"Reference tier: {value['reference_tier']}. "
        "Unit: one exclusive target/B/C/D capture quartet.",
        f"Bound capture slots: {summary['bound_slots']}/{summary['planned_slots']}; "
        f"quartets with four bound captures: {summary['complete_quartets']}/"
        f"{summary['planned_quartets']}.",
        "",
        (
            "Calibration alone selects thresholds and the Youden operating point. "
            "Evaluation is held out."
        ),
        (
            "Wilson rates assume IID condition-specific quartet scores; the AUC "
            "bound requires independent units."
        ),
        "Intervals are pointwise and complete-case. Missingness may depend on the intervention.",
        (
            "BF16 negatives are shared across curves; no joint coverage or "
            "independent-curve claim is made."
        ),
        (
            "Unavailable measurements remain NA. Missing logprobs or local "
            "embeddings cannot become a zero score."
        ),
        (
            "Small samples and undeclared assumptions remain visible. Neither "
            "operator declarations nor engine flags attest numeric precision."
        ),
        "",
        "![ROC curves](roc.svg)",
        "",
        "| Condition | Method | AUC (interval) | n negative/positive | Missing negative/positive |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row[0]} | {row[1]} | {_number(row[3])} "
            f"[{_number(row[4])}, {_number(row[5])}] | {row[6]}/{row[7]} | {row[8]}/{row[9]} |"
        )
    lines.extend(
        [
            "",
            (
                "See curves.csv for every operating-point interval and analysis.json "
                "for per-method capture counts, source pins, assumptions and "
                "missing-unit reasons."
            ),
            "",
        ]
    )
    payloads = {
        "README.md": "\n".join(lines).encode(),
        "curves.csv": csv_file.getvalue().encode(),
        "analysis.json": canonical_json(value) + b"\n",
        **_plots(value, root),
    }
    from servegauge.core.redaction import Redactor

    if Redactor.from_environment().redact(value) != value:
        raise ValueError("study_private_report_values_refused")
    out.mkdir(parents=True, mode=0o700)
    for name, blob in payloads.items():
        path = out / name
        _paths(path)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
    manifest = {
        "version": "gpu-study-report-v1",
        "kit_hash": value["kit_hash"],
        "scope": value["scope"],
        "physical_precision_verified": False,
        "files_sha256": {name: hashlib.sha256(blob).hexdigest() for name, blob in payloads.items()},
    }
    _write(out / "report.json", manifest)
    return manifest
