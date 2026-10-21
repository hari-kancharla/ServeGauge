"""Guarded distribution/accuracy pattern; no inference of physical quantization."""

from __future__ import annotations

import copy
import math
from typing import Any

from servegauge.core.models import CheckResult, EndpointSpec
from servegauge.core.redaction import REDACTED
from servegauge.verdicts.config import DecisionPolicy

REQUIRED_CONTROLS = ("P01", "P02", "P03", "P04.top_p", "P07", "P11", "T01", "T02", "T05", "T06")
DRIFT_CONTRACT = {
    "version": "guarded-drift-v1",
    "required_controls": list(REQUIRED_CONTROLS),
    "accuracy": "whole-negative-interval-within-existing-practical-margin",
    "fingerprints": "two-qualified-methods-between-all-control-bands; at-least-one-FAIL",
    "disclosure": "captured-declaration-only; no-hardware-verification",
}


def _finite(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(value)


def _disclosure(endpoint: EndpointSpec) -> dict[str, Any]:
    result = {
        "status": "UNVERIFIED",
        "label": None,
        "source": "unavailable",
        "hardware_verified": False,
    }
    values = []
    for source, mapping in (
        ("endpoint metadata", endpoint.metadata),
        ("configured label", endpoint.labels),
    ):
        if "quantization" not in mapping:
            continue
        value = mapping["quantization"]
        if value is not None and (not isinstance(value, str) or REDACTED in value):
            return result
        normal = value.strip() if isinstance(value, str) else None
        normal = None if not normal or normal.casefold() == "unknown" else normal
        values.append((source, normal))
    if not values or len({v.casefold() if v else None for _, v in values}) != 1:
        return result
    source, label = values[0]
    return {
        "status": "DISCLOSED" if label else "UNDISCLOSED",
        "label": label,
        "source": source,
        "hardware_verified": False,
    }


def annotate_context(
    checks: list[CheckResult],
    endpoints: list[EndpointSpec],
    policy: DecisionPolicy,
    *,
    contract: dict[str, Any] | None = None,
) -> None:
    """Only new sealed analyses acquire context; no statistics or verdicts change."""
    if contract is None:
        return
    if contract != DRIFT_CONTRACT:
        raise ValueError("unsupported_drift_contract")
    by_id = {e.id: e for e in endpoints}
    for check in checks:
        if check.check_id not in {"FP1", "FP2", "FP3", "FP4"}:
            continue
        if check.endpoint_id not in by_id:
            raise ValueError("drift_endpoint_context_missing")
        check.details["drift_context"] = {
            "contract": copy.deepcopy(contract),
            "alpha": policy.alpha,
            "disclosure": _disclosure(by_id[check.endpoint_id]),
        }


def _between(check: CheckResult) -> bool:
    if (
        check.check_id not in {"FP1", "FP2", "FP3", "FP4"}
        or check.check_class != "comparative"
        or check.verdict not in {"FAIL", "INCONCLUSIVE"}
        or check.n <= 0
        or not check.evidence
        or check.reference_tier is None
        or check.details.get("fingerprint_position") != "between-bands"
        or check.details.get("equivalence_qualified") is not True
    ):
        return False
    context = check.details.get("drift_context", {})
    qualification = check.details.get("qualification")
    metrics = check.details.get("fingerprint_metrics")
    metric = metrics.get(check.check_id) if isinstance(metrics, dict) else None
    disclosure = context.get("disclosure") if isinstance(context, dict) else None
    if (
        not isinstance(context, dict)
        or context.get("contract") != DRIFT_CONTRACT
        or not _finite(context.get("alpha"))
        or not 0 < context["alpha"] < 1
        or not isinstance(disclosure, dict)
        or disclosure.get("status") not in {"DISCLOSED", "UNDISCLOSED", "UNVERIFIED"}
        or disclosure.get("hardware_verified") is not False
        or not isinstance(disclosure.get("source"), str)
        or not (disclosure.get("label") is None or isinstance(disclosure["label"], str))
        or not isinstance(qualification, dict)
        or any(
            not isinstance(qualification.get(k), str) or not qualification[k]
            for k in ("id", "binding")
        )
        or not isinstance(check.details.get("reference_pack_id"), str)
        or not check.details["reference_pack_id"]
        or not isinstance(metric, dict)
        or metric.get("equivalence_qualified") is not True
        or metric.get("position") != "between-bands"
        or metric.get("n") != check.n
        or any(
            metric.get(k) != getattr(check, k) for k in ("delta", "delta_ci_low", "delta_ci_high")
        )
    ):
        return False
    values = [
        check.delta,
        check.delta_ci_low,
        check.delta_ci_high,
        check.margin,
        check.p_value,
        check.adjusted_p,
        metric.get("noise_band"),
        check.estimate,
        check.reference_estimate,
        check.ci_low,
        check.ci_high,
    ]
    if not all(_finite(v) for v in values):
        return False
    assert (
        check.delta is not None
        and check.delta_ci_low is not None
        and check.delta_ci_high is not None
    )
    assert check.margin is not None and check.p_value is not None and check.adjusted_p is not None
    if not (
        0 <= check.p_value <= check.adjusted_p <= context["alpha"]
        and 0 <= metric["noise_band"] < check.delta_ci_low <= check.delta <= check.delta_ci_high
        and check.margin >= 0
        and (check.delta > check.margin) == (check.verdict == "FAIL")
    ):
        return False
    controls = metric.get("control_positions")
    if not isinstance(controls, list) or len(controls) < 2:
        return False
    identities = set()
    captures = set()
    for control in controls:
        if (
            not isinstance(control, dict)
            or control.get("bands_separated") is not True
            or control.get("position") != "between-bands"
            or any(
                not isinstance(control.get(k), str) or not control[k]
                for k in ("control_run_id", "control_endpoint_id", "model_identity", "capture_hash")
            )
        ):
            return False
        interval = control.get("different_model_band")
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or not all(_finite(v) for v in interval)
            or not check.delta_ci_high < interval[0] <= interval[1]
        ):
            return False
        identities.add(control["model_identity"])
        captures.add(control["capture_hash"])
    return len(identities) >= 2 and len(captures) >= 2


def _small_loss(check: CheckResult) -> bool:
    if (
        check.check_id.split(".")[0] not in {"R01", "R02"}
        or check.check_class != "comparative"
        or check.verdict != "PASS"
        or check.n <= 0
        or not check.evidence
        or check.reference_tier is None
    ):
        return False
    if not all(
        _finite(v)
        for v in [
            check.reference_estimate,
            check.delta,
            check.delta_ci_low,
            check.delta_ci_high,
            check.margin,
        ]
    ):
        return False
    assert (
        check.margin is not None
        and check.delta_ci_low is not None
        and check.delta is not None
        and check.delta_ci_high is not None
    )
    return -check.margin < check.delta_ci_low <= check.delta <= check.delta_ci_high < 0


def drift_witness(checks: list[CheckResult]) -> list[CheckResult]:
    """Return the complete conditional witness, preserving each original verdict."""
    if len({(c.run_id, c.endpoint_id) for c in checks}) != 1 or len(
        {c.check_id for c in checks}
    ) != len(checks):
        return []
    if any(
        (c.check_id.startswith(("P", "T")) and c.verdict in {"FAIL", "ERROR"})
        or c.details.get("operational_alarm")
        for c in checks
    ):
        return []
    by_id = {c.check_id: c for c in checks}
    controls = [by_id[name] for name in REQUIRED_CONTROLS if name in by_id]
    if len(controls) != len(REQUIRED_CONTROLS) or any(
        c.verdict != "PASS"
        or c.n <= 0
        or not c.evidence
        or not all(_finite(v) for v in [c.estimate, c.ci_low, c.ci_high])
        for c in controls
    ):
        return []
    fingerprints = [c for c in checks if _between(c)]
    if len(fingerprints) < 2 or not any(c.verdict == "FAIL" for c in fingerprints):
        return []
    identity = [
        (
            c.details["drift_context"],
            c.details["reference_pack_id"],
            c.reference_tier,
            c.details["qualification"],
        )
        for c in fingerprints
    ]
    if any(value != identity[0] for value in identity):
        return []
    accuracy = [c for c in checks if _small_loss(c)]
    if not accuracy or any(
        c.check_id.split(".")[0] in {"R01", "R02"} and c.verdict in {"FAIL", "ERROR"}
        for c in checks
    ):
        return []
    return [*fingerprints, *accuracy, *controls]
