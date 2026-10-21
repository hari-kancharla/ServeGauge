"""Independent P06/T06 raw-channel assertions; standard library only."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any


@dataclass(frozen=True)
class Observation:
    available: bool
    violated: bool | None
    signatures: tuple[str, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.available) is not bool or (
            type(self.violated) is not bool if self.available else self.violated is not None
        ):
            raise ValueError("independent_observation_state_invalid")


class Unavailable(ValueError):
    pass


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate_json_keys")
    return result


def _constant(value: str) -> None:
    raise ValueError("non_finite_json_number")


def _json(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_object, parse_constant=_constant)


def _choice(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict) or response.get("error"):
        raise Unavailable("response_unavailable")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise Unavailable("choice_unavailable")
    selected = choices[0]
    if not isinstance(selected.get("message"), dict):
        raise Unavailable("message_unavailable")
    return dict(selected)


def _stream(events: Any) -> dict[str, Any]:
    if not isinstance(events, list) or not events:
        raise Unavailable("stream_capture_unavailable")
    text: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    logs: list[Any] = []
    finish: str | None = None
    done = False
    observed = False
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("raw_lines"), list):
            raise Unavailable("stream_raw_lines_unavailable")
        pieces = []
        for line in event["raw_lines"]:
            if not isinstance(line, str):
                raise Unavailable("stream_raw_line_invalid")
            if line.startswith("data:"):
                value = line[5:]
                pieces.append(value[1:] if value.startswith(" ") else value)
        data = "\n".join(pieces)
        if not pieces:
            continue
        if event.get("is_comment") or event.get("data", data) != data:
            raise Unavailable("stream_projection_changed")
        if data == "[DONE]":
            if done:
                raise Unavailable("duplicate_stream_terminator")
            done = True
            continue
        if done:
            raise Unavailable("stream_data_after_terminator")
        chunk = _json(data)
        if not isinstance(chunk, dict) or not isinstance(chunk.get("choices"), list):
            raise Unavailable("stream_chunk_unavailable")
        for selected in chunk["choices"]:
            if not isinstance(selected, dict) or type(selected.get("index")) is not int:
                raise Unavailable("stream_choice_index_unavailable")
            if selected["index"] != 0:
                continue
            observed = True
            delta = selected.get("delta") or {}
            if not isinstance(delta, dict):
                raise Unavailable("stream_delta_invalid")
            content = delta.get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise Unavailable("stream_text_unavailable")
                text.append(content)
            if selected.get("finish_reason") is not None:
                if not isinstance(selected["finish_reason"], str):
                    raise Unavailable("stream_finish_marker_invalid")
                finish = selected["finish_reason"]
            probability = selected.get("logprobs")
            if isinstance(probability, dict) and probability.get("content") is not None:
                if not isinstance(probability["content"], list):
                    raise Unavailable("stream_logprobs_invalid")
                logs.extend(probability["content"])
            tools = delta.get("tool_calls")
            if tools is None:
                continue
            if not isinstance(tools, list):
                raise Unavailable("stream_tool_envelope_invalid")
            for part in tools:
                if not isinstance(part, dict):
                    raise Unavailable("stream_tool_delta_invalid")
                index = part.get("index")
                if type(index) is not int or index < 0:
                    # An unindexed delta cannot be assigned to a structured call.
                    continue
                target = calls.setdefault(index, {"function": {"name": "", "arguments": ""}})
                function = part.get("function") or {}
                if not isinstance(function, dict):
                    raise Unavailable("stream_function_invalid")
                for key in ["name", "arguments"]:
                    if key in function:
                        if not isinstance(function[key], str):
                            raise Unavailable("stream_function_fragment_invalid")
                        target["function"][key] += function[key]
    if not done or not observed:
        raise Unavailable("stream_tail_or_choice_unavailable")
    return {
        "message": {"content": "".join(text), "tool_calls": [calls[i] for i in sorted(calls)]},
        "finish_reason": finish,
        "logprobs": {"content": logs},
    }


def _captured(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if row.get("status") != 200 or row.get("error") or row.get("request_not_sent"):
        raise Unavailable("successful_capture_required")
    request = row.get("request")
    if not isinstance(request, dict):
        raise Unavailable("original_request_unavailable")
    if request.get("stream") is True or row.get("sse_events"):
        selected = _stream(row.get("sse_events"))
    elif row.get("response_text") is not None:
        if not isinstance(row["response_text"], str):
            raise Unavailable("response_text_invalid")
        selected = _choice(_json(row["response_text"]))
    else:
        selected = _choice(row.get("response"))
    message = selected["message"]
    if message.get("content") is not None and not isinstance(message["content"], str):
        raise Unavailable("visible_text_invalid")
    return request, selected


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid_probability_type")
    result = float(value)
    if not math.isfinite(result) or result > 0:
        raise ValueError("invalid_log_probability")
    return result


def _bytes(request: dict[str, Any], selected: dict[str, Any], tolerance: float) -> set[str]:
    if request.get("logprobs") is not True:
        raise Unavailable("logprobs_not_requested")
    container = selected.get("logprobs")
    rows = container.get("content") if isinstance(container, dict) else None
    if not isinstance(rows, list) or not rows:
        return {"requested_chosen_logprobs_absent"}
    reconstructed = bytearray()
    try:
        for row in rows:
            if not isinstance(row, dict):
                return {"chosen_logprob_invalid"}
            _number(row["logprob"])
            if isinstance(row.get("bytes"), list):
                if any(type(value) is not int or not 0 <= value <= 255 for value in row["bytes"]):
                    return {"token_bytes_invalid"}
                reconstructed.extend(row["bytes"])
            else:
                if not isinstance(row.get("token"), str):
                    return {"token_text_absent"}
                reconstructed.extend(row["token"].encode("utf-8"))
            top = row["top_logprobs"]
            if not isinstance(top, list):
                return {"top_logprobs_invalid"}
            probabilities = [_number(value["logprob"]) for value in top]
            if probabilities != sorted(probabilities, reverse=True):
                return {"top_logprobs_unsorted"}
            if math.fsum(math.exp(value) for value in probabilities) > 1 + tolerance:
                return {"top_probability_mass_exceeds_one"}
        if reconstructed.decode("utf-8") != (selected["message"].get("content") or ""):
            return {"token_bytes_disagree_with_visible_text"}
    except (KeyError, TypeError, ValueError, OverflowError):
        return {"chosen_logprob_or_bytes_invalid"}
    return set()


def _calls(selected: dict[str, Any]) -> list[Any]:
    value = selected["message"].get("tool_calls")
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(part, dict) for part in value):
        raise Unavailable("structured_tool_envelope_invalid")
    return value


def _tool_drop(request: dict[str, Any], selected: dict[str, Any]) -> set[str]:
    if _calls(selected):
        return set()
    if selected.get("finish_reason") == "tool_calls":
        return {"tool_finish_marker_without_structured_calls"}
    text = selected["message"].get("content") or ""
    try:
        parsed = _json(text)
    except (ValueError, RecursionError):
        return set()
    offered = set()
    for tool in request.get("tools") or []:
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
            name = tool["function"].get("name")
            if isinstance(name, str):
                offered.add(name)
    pending = list(parsed) if isinstance(parsed, list) else [parsed]
    while pending:
        candidate = pending.pop()
        if not isinstance(candidate, dict):
            continue
        if isinstance(candidate.get("tool_calls"), list):
            pending.extend(candidate["tool_calls"])
        function = candidate.get("function", candidate)
        if (
            not isinstance(function, dict)
            or not isinstance(function.get("name"), str)
            or function["name"] not in offered
        ):
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = _json(arguments)
            except (ValueError, RecursionError):
                continue
        if isinstance(arguments, dict):
            return {"offered_call_json_without_structured_calls"}
    return set()


def assess_unit(
    check_id: str,
    captures: list[dict[str, Any]],
    *,
    probability_tolerance: float,
    paired_structure: bool = False,
) -> Observation:
    if check_id not in {"P06.structure", "T06"}:
        raise ValueError("independent_check_unsupported")
    if not math.isfinite(probability_tolerance) or probability_tolerance < 0:
        raise ValueError("independent_probability_tolerance_invalid")
    if not captures:
        return Observation(False, None, reason="captures_missing")
    signatures: set[str] = set()
    try:
        counts = []
        for row in captures:
            request, selected = _captured(row)
            if check_id == "P06.structure":
                signatures.update(_bytes(request, selected, probability_tolerance))
            else:
                if selected.get("finish_reason") == "length":
                    raise Unavailable("tool_output_censored")
                counts.append(len(_calls(selected)))
                signatures.update(_tool_drop(request, selected))
        if paired_structure:
            if check_id != "T06" or len(captures) != 2:
                raise Unavailable("complete_structural_pair_required")
            if counts[0] != counts[1]:
                signatures.add("stream_and_nonstream_call_counts_disagree")
    except (Unavailable, KeyError, TypeError, ValueError, OverflowError, RecursionError):
        return Observation(False, None, reason="capture_channel_unavailable_or_invalid")
    return Observation(True, bool(signatures), tuple(sorted(signatures)))


def summarize(
    primary: dict[str, Observation],
    confirmation_batches: list[list[Observation]],
    *,
    alpha: float,
    tolerance: float,
    k_fail: int,
    reference_tier: str | None,
) -> dict[str, Any]:
    if not 0 < alpha < 1 or not 0 <= tolerance <= 1 or type(k_fail) is not int or k_fail < 1:
        raise ValueError("independent_policy_invalid")
    n = len(primary)
    if not n or any(not value.available for value in primary.values()):
        return {
            "n": n,
            "observed_n": sum(value.available for value in primary.values()),
            "violations": None,
            "ci_low": None,
            "ci_high": None,
            "reference_tier": reference_tier,
            "verdict": "ERROR",
            "reproduced": False,
            "replay_adds_independent_n": False,
        }
    count = sum(value.violated is True for value in primary.values())
    z = NormalDist().inv_cdf(1 - alpha / 2)
    squared = z * z
    denominator = 1 + squared / n
    center = (count / n + squared / (2 * n)) / denominator
    radius = z * math.sqrt(count / n * (1 - count / n) / n + squared / (4 * n * n)) / denominator
    low = 0.0 if count == 0 else max(0.0, center - radius)
    high = 1.0 if count == n else min(1.0, center + radius)
    reproduced = any(
        batch and all(value.available for value in batch) and any(value.violated for value in batch)
        for batch in confirmation_batches
    )
    verdict = "INCONCLUSIVE"
    if count >= k_fail and reproduced:
        verdict = "FAIL"
    elif count == 0 and math.nextafter(high, math.inf) <= tolerance:
        verdict = "PASS"
    return {
        "n": n,
        "observed_n": n,
        "violations": count,
        "estimate": count / n,
        "ci_low": low,
        "ci_high": high,
        "reference_tier": reference_tier,
        "reference_estimate": None,
        "verdict": verdict,
        "reproduced": reproduced,
        "replay_adds_independent_n": False,
    }


def get_assert(output: Any, context: dict[str, Any]) -> dict[str, Any]:
    """Promptfoo deterministic assertion with an explicit non-Boolean verdict.

    Promptfoo's Boolean field is true only for an adequately powered PASS. Its
    reason carries the full independent result, including INCONCLUSIVE and ERROR.
    The pinned launcher must retain that result rather than relabel a Boolean
    failure as an observed invariant violation.
    """
    try:
        value = _json(output) if isinstance(output, str) else output
        policy = context["vars"]["independent_policy"]
        keys = policy["unit_keys"]
        if (
            not isinstance(value, dict)
            or value.get("replay_adds_independent_n") is not False
            or not isinstance(keys, list)
            or not keys
            or any(not isinstance(key, str) or not key for key in keys)
            or len(set(keys)) != len(keys)
            or not isinstance(value.get("primary"), dict)
            or set(value["primary"]) != set(keys)
            or not isinstance(value.get("confirmations"), list)
        ):
            raise ValueError("independent_exact_population_required")

        def observe(unit: Any) -> Observation:
            if (
                not isinstance(unit, dict)
                or type(unit.get("paired_structure")) is not bool
                or not isinstance(unit.get("captures"), list)
                or any(not isinstance(row, dict) for row in unit["captures"])
            ):
                raise ValueError("independent_unit_descriptor_invalid")
            return assess_unit(
                policy["check_id"],
                unit["captures"],
                probability_tolerance=policy["probability_tolerance"],
                paired_structure=unit["paired_structure"],
            )

        batches = value["confirmations"]
        if any(not isinstance(batch, list) for batch in batches):
            raise ValueError("independent_confirmation_descriptor_invalid")
        result = summarize(
            {key: observe(value["primary"][key]) for key in keys},
            [[observe(unit) for unit in batch] for batch in batches],
            alpha=policy["alpha"],
            tolerance=policy["tolerance"],
            k_fail=policy["k_fail"],
            reference_tier=policy["reference_tier"],
        )
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError):
        result = {
            "verdict": "ERROR",
            "n": None,
            "violations": None,
            "ci_low": None,
            "ci_high": None,
            "reference_tier": None,
            "reason": "independent_scope_policy_or_channels_unavailable",
            "replay_adds_independent_n": False,
        }
    passed = result["verdict"] == "PASS"
    return {
        "pass": passed,
        "score": 1 if passed else 0,
        "reason": json.dumps(result, sort_keys=True, allow_nan=False, separators=(",", ":")),
    }
