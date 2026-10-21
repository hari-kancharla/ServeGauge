"""Exact finite-audit verification and honest external-result normalization."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.mock.sealed_replay import FiniteReplay, ReplayAddress
from servegauge.storage.comparison import _files, _guard, _private_json
from servegauge.storage.comparison_program import CompiledCase


def validate_audit(case: CompiledCase, audit: dict[str, Any]) -> None:
    """Verify addresses/order/hashes in memory; no transport, dispatch or new n."""
    try:
        tape = FiniteReplay([row.source for phase in case.phases for row in phase.exchanges])
        if not isinstance(audit.get("observations"), list):
            raise ValueError
        for row in audit["observations"]:
            address = ReplayAddress.model_validate(row["address"])
            body = tape._slots[address][0]
            if type(row["skipped"]) is not bool:
                raise ValueError
            if row["skipped"]:
                tape.skip(address, body)
            else:
                tape.take(address, body)
        tape.assert_complete()
        # Canonical bytes also distinguish booleans, strings and integer fields.
        if canonical_json(tape.audit()) != canonical_json(audit):
            raise ValueError
    except (ValueError, TypeError, KeyError):
        raise ValueError("comparison_complete_unchanged_finite_audit_required") from None


def _error(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "original_population_n": case["population_n"],
        "n": None,
        "observed_n": None,
        "violations": None,
        "ci_low": None,
        "ci_high": None,
        "reference_tier": case["original_check"]["reference_tier"],
        "verdict": "ERROR",
        "reason": "external_scope_or_result_unavailable",
        "replay_adds_independent_n": False,
    }


def _probability(value: Any) -> float:
    if type(value) not in {int, float}:
        raise ValueError
    number = float(value)
    if not math.isfinite(number):
        raise ValueError
    return number


def normalize_external(value: Any, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve explicit verdicts; Boolean assertion failures are never counts."""
    try:
        rows = value["results"]["results"]
        known = {case["case_id"] for case in cases}
        if not isinstance(rows, list) or any(
            not isinstance(row, dict)
            or row["testCase"]["description"] not in known
            or row["testCase"]["vars"]["case_id"] != row["testCase"]["description"]
            for row in rows
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        return [_error(case) for case in cases]
    result = []
    for case in cases:
        selected = [row for row in rows if row["testCase"]["description"] == case["case_id"]]
        normalized = _error(case)
        try:
            if len(selected) != 1:
                raise ValueError
            components = selected[0]["gradingResult"]["componentResults"]
            if not isinstance(components, list) or len(components) != 1:
                raise ValueError
            summary = _private_json(components[0]["reason"])
            if (
                not isinstance(summary, dict)
                or summary.get("replay_adds_independent_n") is not False
            ):
                raise ValueError
            verdict = summary.get("verdict")
            if verdict not in {"PASS", "FAIL", "INCONCLUSIVE", "ERROR"}:
                raise ValueError
            n = summary.get("n")
            if n is not None and (type(n) is not int or n != case["population_n"]):
                raise ValueError
            if summary.get("reference_tier") != case["original_check"]["reference_tier"]:
                raise ValueError
            if verdict == "ERROR":
                if any(summary.get(key) is not None for key in ["violations", "ci_low", "ci_high"]):
                    raise ValueError
            else:
                count = summary.get("violations")
                low, high = (
                    _probability(summary.get("ci_low")),
                    _probability(summary.get("ci_high")),
                )
                if (
                    n is None
                    or type(count) is not int
                    or not 0 <= count <= n
                    or not 0 <= low <= high <= 1
                ):
                    raise ValueError
            normalized = {
                **summary,
                "case_id": case["case_id"],
                "original_population_n": case["population_n"],
            }
        except (KeyError, TypeError, ValueError):
            pass
        result.append(normalized)
    return result


def verify_results(directory: Path, *, result_hash: str) -> dict[str, Any]:
    """Bind all private output evidence to the separately retained result hash."""
    _guard(directory)
    path = directory / "comparison-results.json"
    _guard(path)
    value = _private_json(path.read_bytes())
    if (
        not isinstance(value, dict)
        or value.get("version") != "comparison-results-v1"
        or content_hash(value) != result_hash
    ):
        raise ValueError("comparison_external_result_anchor_changed")
    actual = {
        item.relative_to(directory).as_posix(): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in _files(directory)
        if item != path
    }
    if actual != value.get("files_sha256"):
        raise ValueError("comparison_result_evidence_changed")
    return dict(value)


def render_comparison(value: dict[str, Any]) -> str:
    """Private draft only; retain errors, original n and unavailable measurements."""
    lines = [
        "# Captured failure comparison",
        "",
        f"Scope: {value['mode']} captured-data reproduction. Replay adds no independent n.",
        "",
        "Total setup: unmeasured. Prebuilt compiler, providers and assertions are reused.",
        "",
        "| Case | Procedure | Verdict | Original n | n used | Violations | Interval | Reference |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for procedure in value["procedures"]:
        for row in procedure["cases"]:
            interval = (
                f"[{row['ci_low']:.8g}, {row['ci_high']:.8g}]"
                if row.get("ci_low") is not None and row.get("ci_high") is not None
                else "unavailable"
            )
            count = str(row["violations"]) if row.get("violations") is not None else "unknown"
            n = str(row["n"]) if row.get("n") is not None else "unknown"
            lines.append(
                f"| {row['case_id']} | {procedure['procedure']} | {row['verdict']} | "
                f"{row['original_population_n']} | {n} | {count} | {interval} | "
                f"{row.get('reference_tier') or 'no reference'} |"
            )
    lines += [
        "",
        "INCONCLUSIVE remains underpowered or unresolved under the captured policy.",
        "ERROR has unknown counts/intervals; it is not a clean observation.",
        "",
        value["root_cause_limit"],
        "",
        "Prepared execution timings are single executions, with no timing interval or reference.",
    ]
    for procedure in value["procedures"]:
        lines.append(
            f"- {procedure['procedure']}: {procedure['execution_seconds']:.3f} s; "
            f"first valid result: {procedure.get('first_valid_result_utc') or 'unavailable'}."
        )
        setup = procedure.get("setup", {})
        elapsed = setup.get("owner_started_to_first_valid_seconds")
        lines.append(
            "  Owner-declared start to first valid result: "
            + (
                f"{elapsed:.3f} s (prebuilt engineering excluded)."
                if elapsed is not None
                else "unmeasured."
            )
        )
        for step in setup.get("manual_steps") or []:
            lines.append("  Recorded manual step: " + step.replace("\n", " "))
    lines += [
        "",
        "Review the sealed kit, raw outputs, finite audits and errors before interpretation.",
        "This private draft is not a provider, model-weight or hardware qualification.",
        "",
    ]
    return "\n".join(lines)
