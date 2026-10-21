"""Protocol checks score sanitized wire captures, never backend fault flags."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

import numpy as np
from pydantic import Field

from servegauge.core.completion import completion_error
from servegauge.core.errors import SafeError
from servegauge.core.models import (
    CheckClass,
    CheckResult,
    EndpointSpec,
    Exchange,
    Item,
    ModelProfile,
    ValueModel,
)
from servegauge.stats.intervals import wilson
from servegauge.tokencount.counters import text_tokens
from servegauge.tokencount.model_text import uses_glm_profile
from servegauge.verdicts.config import DecisionPolicy


class Samples(ValueModel):
    check_id: str
    check_class: CheckClass
    binary: bool = True
    two_sided: bool = False
    metric: str = "mean"
    values: dict[str, float] = Field(default_factory=dict)
    repeats: dict[str, list[float]] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    replay_candidates: list[list[str]] = Field(default_factory=list)
    status: str | None = None
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)

    def mark(self, status: str, reason: str) -> None:
        if self.status != "ERROR":
            self.status, self.reason = status, reason


def choice(record: Exchange) -> dict[str, Any]:
    if completion_error(record.response):
        raise ValueError("provider_completion_error")
    choices = (record.response or {}).get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("invalid_choice_envelope")
    return dict(choices[0])


def message(record: Exchange) -> dict[str, Any]:
    value = choice(record).get("message")
    if not isinstance(value, dict):
        raise ValueError("invalid_message_envelope")
    return value


def calls(record: Exchange) -> list[dict[str, Any]]:
    value = message(record).get("tool_calls") or []
    if not isinstance(value, list) or any(not isinstance(call, dict) for call in value):
        raise ValueError("invalid_tool_call_envelope")
    return value


def content(record: Exchange) -> str:
    value = message(record).get("content")
    if value is not None and not isinstance(value, str):
        raise ValueError("invalid_text_content")
    return value or ""


def logprob_sanity(record: Exchange, tolerance: float) -> tuple[bool, float | None]:
    container = choice(record).get("logprobs")
    if not isinstance(container, dict):
        return True, None
    logs = container.get("content")
    if not isinstance(logs, list) or not logs:
        return True, None
    reconstructed, selected = bytearray(), []
    try:
        for row in logs:
            lp = float(row["logprob"])
            if not math.isfinite(lp) or lp > 0:
                return True, None
            selected.append(lp)
            if isinstance(row.get("bytes"), list):
                reconstructed.extend(bytes(row["bytes"]))
            else:
                reconstructed.extend(str(row["token"]).encode())
            top = row["top_logprobs"]
            probabilities = [float(token["logprob"]) for token in top]
            if any(not math.isfinite(v) or v > 0 for v in probabilities):
                return True, None
            if probabilities != sorted(probabilities, reverse=True):
                return True, None
            if sum(math.exp(v) for v in probabilities) > 1 + tolerance:
                return True, None
        invalid = reconstructed.decode("utf-8") != content(record)
    except (KeyError, ValueError, TypeError, OverflowError):
        return True, None
    return invalid, float(np.mean(selected))


def invariant_violation(
    item: Item, record: Exchange, policy: DecisionPolicy, *, profile: ModelProfile | None = None
) -> bool:
    check = item.expected["check_id"]
    if check == "P01":
        if uses_glm_profile(profile) or item.expected.get("text_counter_profile") == "glm-5.3":
            from servegauge.scoring.glm_protocol import cap_violation

            return cap_violation(item, record, profile)
        cap = record.request.get("max_completion_tokens", record.request.get("max_tokens"))
        if not isinstance(cap, int):
            raise ValueError("wire_output_cap_missing")
        output = content(record)
        reasoning = message(record).get("reasoning") or ""
        visible = text_tokens(output) + text_tokens(str(reasoning))
        reported = (record.usage or {}).get("completion_tokens")
        exceeds = visible > cap or (isinstance(reported, int) and reported > cap)
        requested = item.expected["full_text"]
        known_cut = bool(output) and len(output) < len(requested) and requested.startswith(output)
        return exceeds or (known_cut and choice(record).get("finish_reason") != "length")
    if check == "P02":
        return item.expected["stop"] in content(record)
    if check == "P09":
        actual, mode = calls(record), item.expected["mode"]
        return (
            bool(actual)
            if mode == "none"
            else not actual
            or (
                mode == "named"
                and any(c.get("function", {}).get("name") != "convert" for c in actual)
            )
        )
    if check == "P10":
        return len(calls(record)) > 1
    if check == "P12":
        return record.status == 200
    if check == "P06.structure":
        return logprob_sanity(record, float(policy.entries["P06"]["prob_sum_tolerance"]))[0]
    raise ValueError("unknown_protocol_invariant")


def stream_violation(record: Exchange) -> bool:
    meta = record.metadata
    return bool(
        not record.sse_events
        or not meta.get("terminal")
        or meta.get("incomplete")
        or meta.get("invalid_utf8")
        or meta.get("invalid_json_events")
        or meta.get("tool_delta_missing_index")
        or not record.usage
    )


def _structure(record: Exchange) -> list[tuple[str, str, tuple[str, ...]]]:
    result = []
    for call in calls(record):
        function = call["function"]
        arguments = json.loads(function["arguments"])
        if not isinstance(arguments, dict):
            raise ValueError("tool_arguments_not_object")
        result.append((str(call["type"]), str(function["name"]), tuple(sorted(arguments))))
    return sorted(result)


def score_capabilities(
    items: list[Item], records: list[Exchange], endpoint: EndpointSpec, run_id: str
) -> tuple[list[CheckResult], dict[str, bool | None]]:
    by_item = {record.item_id: record for record in records if record.endpoint_id == endpoint.id}
    results: list[CheckResult] = []
    observed: dict[str, bool | None] = {}
    for item in items:
        record = by_item.get(item.item_id)
        feature = item.expected.get("capability", "answers")
        result = CheckResult(
            check_id=item.expected["check_id"],
            suite_id="S0",
            check_class="report_only",
            endpoint_id=endpoint.id,
            run_id=run_id,
            n=0,
            verdict="INFO",
            reason="capability probe unavailable",
            reference_tier="MOCK" if endpoint.kind == "mock" else None,
            details={
                "capability": feature,
                "declared": endpoint.metadata.get("supported_parameters"),
            },
        )
        if record is None or record.error:
            observed[feature] = None
            result.verdict, result.reason = "ERROR", "capability transport or capture failure"
        elif record.metadata.get("request_not_sent"):
            observed[feature] = False if record.metadata.get("verdict") == "NOT_SUPPORTED" else None
            result.verdict = record.metadata.get("verdict", "INCONCLUSIVE")
            result.reason = record.metadata.get("skip_reason", "unsupported capability")
        else:
            accepted = record.status == 200
            observed[feature] = accepted
            result.n, result.estimate = 1, float(accepted)
            result.ci_low, result.ci_high = wilson(int(accepted), 1)
            result.reason = "request accepted" if accepted else "capability probe rejected"
            result.details["http_status"] = record.status
            result.evidence = [record.exchange_id]
            if accepted:
                try:
                    message(record)
                    if feature == "answers" and not content(record) and not calls(record):
                        observed[feature] = None
                        result.verdict, result.reason = (
                            "INCONCLUSIVE",
                            "basic probe returned no final answer",
                        )
                    if feature == "streaming":
                        observed[feature] = bool(
                            record.sse_events and record.metadata.get("terminal")
                        )
                    if feature == "reasoning":
                        result.details["reasoning_returned"] = bool(
                            message(record).get("reasoning")
                            or message(record).get("reasoning_details")
                        )
                except ValueError as error:
                    observed[feature] = None
                    result.verdict = "ERROR"
                    result.estimate = result.ci_low = result.ci_high = None
                    result.reason = (
                        "provider reported completion error"
                        if str(error) == "provider_completion_error"
                        else "invalid capability response envelope"
                    )
        results.append(result)
    return results, observed


def protocol_samples(
    items: list[Item],
    records: list[Exchange],
    endpoint_id: str,
    profile: ModelProfile,
    policy: DecisionPolicy,
) -> dict[str, Samples]:
    by_item: dict[str, list[Exchange]] = defaultdict(list)
    for record in records:
        if record.endpoint_id == endpoint_id:
            by_item[record.item_id].append(record)
    groups: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        if item.suite_id != "S1":
            continue
        check = item.expected["check_id"]
        if check == "P09":
            check += "." + item.expected["variant"]
        groups[check].append(item)
    output: dict[str, Samples] = {}
    for check, probes in groups.items():
        comparative = check in {"P03", "P04.top_p", "P04.top_k", "P07", "P13"}
        report_only = check == "P05"
        result = Samples(
            check_id=check,
            check_class="report_only"
            if report_only
            else "comparative"
            if comparative
            else "invariant",
            binary=not comparative or check == "P13",
            two_sided=check == "P13",
        )
        output[check] = result
        if uses_glm_profile(profile) and check in {"P01", "P07"}:
            result.details["counter_scope"] = (
                "claimed-profile local text projection or API-reported count; "
                "no generated-token, hidden-reasoning or billing certificate"
            )
        trials: dict[int, list[Item]] = defaultdict(list)
        for item in probes:
            trials[item.expected["trial"]].append(item)
        for trial, trial_items in sorted(trials.items()):
            captures = {
                item.expected["variant"]: sorted(by_item[item.item_id], key=lambda r: r.sample_idx)
                for item in trial_items
            }
            all_captures = [r for values in captures.values() for r in values]
            result.evidence.extend(r.exchange_id for r in all_captures)
            if any(
                len(by_item[item.item_id]) != int(item.expected.get("samples", 1))
                for item in trial_items
            ):
                result.mark("ERROR", "expected capture missing")
                continue
            if any(r.error for r in all_captures):
                result.mark("ERROR", "transport or excluded-routing capture")
                continue
            unsent = [
                r
                for variant, values in captures.items()
                for r in values
                if r.metadata.get("request_not_sent")
                and not (check == "P13" and variant == "required")
            ]
            if unsent:
                status = (
                    "INCONCLUSIVE"
                    if any(r.metadata.get("verdict") == "INCONCLUSIVE" for r in unsent)
                    else "NOT_SUPPORTED"
                )
                result.mark(status, "required capability unavailable; see capture metadata")
                continue
            try:
                if check in {
                    "P03",
                    "P04.top_p",
                    "P04.top_k",
                    "P09.none",
                    "P09.required",
                    "P09.named",
                    "P11",
                } and any(choice(r).get("finish_reason") == "length" for r in all_captures):
                    raise LookupError(
                        "output cap censored the requested behavior; "
                        "increase configured output headroom"
                    )
                if check in {"P03", "P04.top_p", "P04.top_k"}:
                    ratios = {
                        variant: len({content(r) for r in values}) / len(values)
                        for variant, values in captures.items()
                    }
                    value = ratios["high"] - ratios["low"]
                elif check == "P07":
                    counts = {}
                    count_modes = set()
                    for variant, values in captures.items():
                        r = values[0]
                        if uses_glm_profile(profile):
                            from servegauge.scoring.glm_protocol import reasoning_count

                            glm_count, count_mode = reasoning_count(r, profile)
                            counts[variant] = glm_count
                            count_modes.add(count_mode)
                            continue
                        details = (r.usage or {}).get("completion_tokens_details")
                        count = (
                            details.get("reasoning_tokens") if isinstance(details, dict) else None
                        )
                        if not isinstance(count, int) or count < 0:
                            reasoning = message(r).get("reasoning")
                            count = text_tokens(reasoning) if isinstance(reasoning, str) else None
                        if count is None or choice(r).get("finish_reason") == "length":
                            raise LookupError(
                                "reasoning counts unavailable or censored by output cap"
                            )
                        counts[variant] = count
                    if len(count_modes) > 1:
                        raise LookupError("GLM effort comparison requires identical count channels")
                    if count_modes:
                        result.details["reasoning_count_channel"] = next(iter(count_modes))
                    offset = float(policy.entries["P07"]["token_pseudocount"])
                    value = math.log((counts["high"] + offset) / (counts["low"] + offset))
                    result.details["metric"] = "log_high_over_low_reasoning_tokens"
                    result.details["decision_rule"] = "effort"
                elif check == "P05":
                    value = float(content(captures["first"][0]) == content(captures["repeat"][0]))
                elif check == "P11":
                    streamed, ordinary = captures["stream"][0], captures["json"][0]
                    value = float(stream_violation(streamed))
                    if not value and _structure(streamed) != _structure(ordinary):
                        result.details["sample_structure_mismatch"] = True
                        if not value:
                            result.mark(
                                "INCONCLUSIVE",
                                "sample structure differed; cannot isolate streaming from sampling",
                            )
                elif check == "P14":
                    raise LookupError("immutable enforcement requires a qualified maker contract")
                elif check == "P13":
                    value = float(captures["optional"][0].status != 200)
                else:
                    value = float(
                        any(
                            (
                                invariant_violation(item, record, policy, profile=profile)
                                if uses_glm_profile(profile)
                                else invariant_violation(item, record, policy)
                            )
                            for item in trial_items
                            for record in by_item[item.item_id]
                        )
                    )
                result.values[str(trial)] = value
                if result.check_class == "invariant" and value:
                    result.replay_candidates.append([item.item_id for item in trial_items])
            except LookupError as error:
                result.mark("INCONCLUSIVE", str(error))
            except SafeError:
                result.mark("ERROR", "verified local tokenizer cache unavailable")
            except (KeyError, TypeError, ValueError, IndexError):
                result.mark(
                    "ERROR",
                    "invalid response shape for deterministic scorer",
                )
        if check == "P06.structure":
            calibrated = Samples(
                check_id="P06.calibration",
                check_class="comparative",
                binary=False,
                two_sided=True,
                evidence=result.evidence,
                status=result.status,
                reason=result.reason,
            )
            for item in probes:
                available = by_item[item.item_id]
                if (
                    available
                    and not available[0].error
                    and not available[0].metadata.get("request_not_sent")
                ):
                    _, mean = logprob_sanity(
                        available[0], float(policy.entries["P06"]["prob_sum_tolerance"])
                    )
                    if mean is not None:
                        calibrated.values[str(item.expected["trial"])] = mean
            output[calibrated.check_id] = calibrated
    output.setdefault(
        "P12",
        Samples(
            check_id="P12",
            check_class="invariant",
            status="INCONCLUSIVE",
            reason="advertised context unavailable",
        ),
    )
    output.setdefault(
        "P13",
        Samples(
            check_id="P13",
            check_class="comparative",
            status="NOT_APPLICABLE",
            reason="router-only parameter mode check",
        ),
    )
    output.setdefault(
        "P14",
        Samples(
            check_id="P14",
            check_class="invariant",
            status="NOT_APPLICABLE" if not profile.immutable_params else "INCONCLUSIVE",
            reason="profile declares no immutable parameters"
            if not profile.immutable_params
            else "maker contract required",
        ),
    )
    return output
