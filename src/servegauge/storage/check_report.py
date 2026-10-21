"""A compact inspectable JSON/Markdown report, extended to full cards in M13."""

from pathlib import Path
from typing import Any

from servegauge.core.hashing import canonical_json
from servegauge.core.models import CheckResult
from servegauge.core.redaction import Redactor


def write_check_report(
    directory: Path, results: list[CheckResult], *, metadata: dict[str, Any]
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    value = {"metadata": metadata, "checks": [result.model_dump(mode="json") for result in results]}
    value = Redactor.from_environment().redact(value)
    (directory / "checks.json").write_bytes(canonical_json(value) + b"\n")
    lines = [
        "# ServeGauge requirements check",
        "",
        "Synthetic mock evidence."
        if metadata.get("synthetic_mock")
        else "Endpoint observations against the stated requirements/reference.",
        "",
        f"Run: {metadata['run_id']}",
        "",
        "| Check | Verdict | n | Estimate and interval | Reference tier / estimate | "
        "Reason | Class / code / alternatives |",
        "|---|---|---:|---|---|---|---|",
    ]
    for result in results:
        ratio = result.details.get("geometric_mean_ratio")
        interval = (
            "unavailable"
            if result.estimate is None
            else f"{result.estimate:.6g} [{result.ci_low}, {result.ci_high}]"
        )
        if result.estimate is not None:
            if "grid_interval" in result.details:
                interval = f"{result.estimate:.6g} {result.details['grid_interval']}"
            interval += " (" + result.details.get("interval_kind", "95% CI") + ")"
        reference_estimate = result.reference_estimate
        if ratio is not None:
            interval = f"ratio {ratio:.6g} {result.details['ratio_interval']}"
            reference_estimate = result.details["reference_geometric_mean_ratio"]
        fields = [
            result.check_id,
            result.verdict,
            str(result.n),
            interval,
            f"{result.reference_tier or 'none'} / {reference_estimate}",
            result.reason,
        ]
        matching = [
            value
            for value in metadata.get("diagnoses", [])
            if value["endpoint_id"] == result.endpoint_id and result.check_id in value["check_ids"]
        ]
        attribution = "; ".join(
            value["failure_class"]
            + " / "
            + value["code"]
            + " / "
            + "; ".join(value["alternatives"])
            for value in matching
        )
        if result.verdict == "FAIL" and not attribution:
            attribution = (
                "unassigned in fast track; alternatives: "
                "model behavior, API behavior, harness mismatch"
            )
        fields.append(attribution or "-")
        lines.append(
            "| "
            + " | ".join(field.replace("|", "\\|").replace("\n", " ") for field in fields)
            + " |"
        )
    lines.extend(
        [
            "",
            "Counts refer to independent items/trials; repeated requests remain clustered.",
            "Full deltas, intervals, margins, raw/adjusted p-values, MDE "
            "and exchange links are in checks.json.",
            "Fast track does not assess model fingerprints or quantization.",
            "",
        ]
    )
    for diagnosis in metadata.get("diagnoses", []):
        lines.extend(
            [
                "",
                diagnosis["summary"],
                f"Confidence: {diagnosis['confidence']}. Bundle: {diagnosis.get('bundle_path')}",
                "Falsifiers: " + "; ".join(diagnosis["falsifiers"]),
            ]
        )
    (directory / "checks.md").write_text(
        Redactor.from_environment().text("\n".join(lines)), encoding="utf-8"
    )
