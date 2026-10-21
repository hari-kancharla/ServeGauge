"""Pure local Markdown drafts; payloads and external communication are absent."""

from __future__ import annotations

import html
from collections.abc import Callable
from typing import Any

from servegauge.core.models import CheckResult, Diagnosis


def markdown_text(value: Any) -> str:
    text = html.escape(str(value), quote=False).replace("\n", " ").replace("\r", " ")
    text = text.replace("\\", "\\\\")
    for character in "`*_{}[]()#+.!|":
        text = text.replace(character, "\\" + character)
    return text


def _number(value: float | int | None) -> str:
    return "unavailable" if value is None else f"{value:.6g}"


def _interval(low: float | None, high: float | None) -> str:
    return "unavailable" if low is None or high is None else f"[{_number(low)}, {_number(high)}]"


def issue_markdown(
    diagnosis: Diagnosis,
    checks: list[CheckResult],
    *,
    redact_content: bool = False,
    sanitize: Callable[[str], str] | None = None,
) -> str:
    def safe(value: Any) -> str:
        text = str(value)
        return markdown_text(sanitize(text) if sanitize else text)

    support = [
        c
        for c in checks
        if c.run_id == diagnosis.run_id
        and c.endpoint_id == diagnosis.endpoint_id
        and c.check_id in diagnosis.check_ids
    ]
    if (
        not diagnosis.check_ids
        or len(support) != len(diagnosis.check_ids)
        or {c.check_id for c in support} != set(diagnosis.check_ids)
    ):
        raise ValueError("complete_issue_witness_required")
    lines = [
        "# Draft: " + safe(diagnosis.code),
        "",
        "Local draft for owner review.",
        "",
        "Endpoint: " + safe(diagnosis.endpoint_id),
        "Run: " + safe(diagnosis.run_id),
        "Failure class: " + safe(diagnosis.failure_class),
        "Confidence: " + safe(diagnosis.confidence),
        "",
        safe(diagnosis.summary),
        "",
        "## Captured witness",
        "",
        "| Check | State | n | Violations | Estimate / interval | Delta / interval | "
        "Reference tier / estimate | Margin | Raw / adjusted p |",
        "|---|---|---:|---:|---|---|---|---|---|",
    ]
    for check in sorted(support, key=lambda c: c.check_id):
        tier = check.reference_tier or (
            "invariant expectation" if check.check_class == "invariant" else "unavailable"
        )
        fields = [
            check.check_id,
            check.verdict,
            str(check.n),
            _number(check.violations),
            _number(check.estimate) + " / " + _interval(check.ci_low, check.ci_high),
            _number(check.delta) + " / " + _interval(check.delta_ci_low, check.delta_ci_high),
            tier + " / " + _number(check.reference_estimate),
            _number(check.margin),
            _number(check.p_value) + " / " + _number(check.adjusted_p),
        ]
        lines.append("| " + " | ".join(safe(v) for v in fields) + " |")
    lines.extend(
        [
            "",
            "States and intervals are the original captured measurements. Required passing and "
            "INCONCLUSIVE controls remain part of the witness; "
            "they do not add positive confidence.",
            "",
            "## Alternatives",
            "",
            *("- " + safe(v) for v in diagnosis.alternatives),
            "",
            "## Falsifiers",
            "",
            *("- " + safe(v) for v in diagnosis.falsifiers),
            "",
            "## Evidence and reproduction",
            "",
            "[bundle.json](bundle.json) · [replay.py](replay.py) · [Instructions](README.md)",
            "",
            "Payloads are removed; replay requires the original private raw store."
            if redact_content
            else "The private bundle retains replay material; the reserved runner also "
            "requires the original raw store and budget ledger.",
            "Observed behavior and competing explanations; "
            "no hardware or model-weight attestation.",
            "",
        ]
    )
    return "\n".join(lines)
