"""Offline one-token evidence extraction. Missing observations are never imputed."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import Field

from servegauge.core.models import CheckResult, Tier, ValueModel


class FingerprintSample(ValueModel):
    session_id: str = Field(min_length=1)
    draw: int = Field(ge=0)
    response: dict[str, Any] | None = None
    cache_group: str | None = Field(default=None, min_length=1)


class FingerprintEvidence(ValueModel):
    version: Literal["robust-fingerprint-v3"] = "robust-fingerprint-v3"
    probe_id: str = Field(min_length=1)
    reference_tier: Tier
    anchor_tokens: list[str] = Field(default_factory=list)
    anchor_source: Literal["fixed-probe-configuration"] | None = None
    independent_sessions: bool = False
    matched_sampling_design: bool = False
    calibration_scope: str | None = None
    seed: int = 0
    target: list[FingerprintSample]
    reference: list[FingerprintSample]


@dataclass
class ResponseEvidence:
    text: str | None
    logprobs: dict[str, float] | None
    reason: str | None


def _logprob(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid_probability")
    number = float(value)
    if not math.isfinite(number) or number > 0:
        raise ValueError("invalid_probability")
    return number


def read_response(response: dict[str, Any] | None, *, tolerance: float) -> ResponseEvidence:
    if response is None:
        return ResponseEvidence(None, None, "missing_response")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return ResponseEvidence(None, None, "invalid_response")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        reason = (
            "censored_response"
            if choice.get("finish_reason") == "length"
            else "unfinished_response"
        )
        return ResponseEvidence(None, None, reason)
    message = choice.get("message")
    text = message.get("content") if isinstance(message, dict) else None
    text = text if isinstance(text, str) and text else None
    container = choice.get("logprobs")
    logs = container.get("content") if isinstance(container, dict) else None
    if not isinstance(logs, list) or not logs:
        return ResponseEvidence(text, None, "missing_logprobs")
    if len(logs) != 1:
        return ResponseEvidence(text, None, "one_output_token_required")
    try:
        row = logs[0]
        selected = _logprob(row["logprob"])
        if not isinstance(row["token"], str) or row["token"] != text:
            raise ValueError("token_content_mismatch")
        if "bytes" in row and row["bytes"] is not None:
            raw = row["bytes"]
            if not isinstance(raw, list) or any(type(v) is not int for v in raw):
                raise ValueError("invalid_token_bytes")
            if bytes(raw).decode("utf-8") != text:
                raise ValueError("invalid_token_bytes")
        top = row["top_logprobs"]
        if not isinstance(top, list) or not top:
            raise ValueError("missing_top_tokens")
        result: dict[str, float] = {}
        for token in top:
            name = token["token"]
            if not isinstance(name, str) or not name or name in result:
                raise ValueError("invalid_or_duplicate_token")
            result[name] = _logprob(token["logprob"])
        mass = sum(math.exp(v) for v in result.values())
        if not 0 < mass <= 1 + tolerance:
            raise ValueError("invalid_probability_mass")
        if text in result and not math.isclose(
            result[text], selected, rel_tol=0, abs_tol=tolerance
        ):
            raise ValueError("chosen_probability_mismatch")
        if text not in result and mass + math.exp(selected) > 1 + tolerance:
            raise ValueError("invalid_probability_mass")
    except (KeyError, TypeError, ValueError, OverflowError):
        return ResponseEvidence(text, None, "invalid_logprobs")
    return ResponseEvidence(text, result, None)


@dataclass
class PreparedEvidence:
    first: list[ResponseEvidence]
    second: list[ResponseEvidence]
    logprob_reason: str | None
    first_logs: list[dict[str, float]] = field(default_factory=list)
    second_logs: list[dict[str, float]] = field(default_factory=list)
    unavailable_reasons: dict[str, int] = field(default_factory=dict)


def prepare_evidence(
    evidence: FingerprintEvidence, *, minimum_anchor_tokens: int, tolerance: float
) -> PreparedEvidence:
    if minimum_anchor_tokens < 2 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("explicit_anchor_and_numeric_tolerance_policy_required")
    first, second = (
        [read_response(row.response, tolerance=tolerance) for row in rows]
        for rows in (evidence.target, evidence.reference)
    )
    counts = Counter(r.reason for r in first + second if r.reason is not None)
    anchors = evidence.anchor_tokens
    result = PreparedEvidence(first, second, None, unavailable_reasons=dict(counts))
    if (
        evidence.anchor_source != "fixed-probe-configuration"
        or len(anchors) < minimum_anchor_tokens
        or len(set(anchors)) != len(anchors)
        or any(not token for token in anchors)
    ):
        result.logprob_reason = "fixed_probe_anchors_required"
    elif not first or not second or any(r.logprobs is None for r in first + second):
        result.logprob_reason = "incomplete_logprob_evidence"
    elif any(not set(anchors) <= set(r.logprobs or {}) for r in first + second):
        result.logprob_reason = "missing_fixed_anchor"
    else:
        result.first_logs, result.second_logs = (
            [{token: (r.logprobs or {})[token] for token in anchors} for r in rows]
            for rows in (first, second)
        )
    return result


def unavailable_check(
    evidence: FingerprintEvidence, *, check_id: str, reason: str, groups: int
) -> CheckResult:
    """Unavailable evidence is not a negative probability or a successful comparison."""
    return CheckResult(
        check_id=check_id,
        check_class="report_only",
        suite_id="S6",
        endpoint_id="offline:target",
        run_id="offline-fingerprint",
        n=groups,
        reference_tier=evidence.reference_tier,
        verdict="INCONCLUSIVE",
        reason=reason,
        details={
            "target_requests": len(evidence.target),
            "reference_requests": len(evidence.reference),
            "independent_unit": "session/cache component",
        },
    )
