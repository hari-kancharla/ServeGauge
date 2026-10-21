"""Pure S2 scoring from captured envelopes and declared generated-item rules."""

from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from typing import Any

from jsonschema import Draft202012Validator

from servegauge.core.hashing import canonical_json
from servegauge.core.json import strict_json
from servegauge.core.models import Exchange, Item, ModelProfile
from servegauge.profiles import field_verified
from servegauge.scoring.common import Samples, calls, choice, content, message, stream_violation
from servegauge.scoring.toolcall_scoring import arguments_match, parsed_call

GENERIC_MARKERS = ("<tool_call>", "</tool_call>", "<|call|>", "[TOOL_CALLS]", "<think>", "</think>")


@lru_cache(maxsize=512)
def _validator(schema: bytes) -> Draft202012Validator:
    return Draft202012Validator(strict_json(schema.decode()))


def schema_accuracy(item: Item, record: Exchange) -> bool:
    offered = {
        tool["function"]["name"]: tool["function"]["parameters"] for tool in item.payload["tools"]
    }
    actual = calls(record)
    if not actual:
        return False
    try:
        for call in actual:
            name, args = parsed_call(call)
            if name not in offered or not _validator(canonical_json(offered[name])).is_valid(args):
                return False
    except (ValueError, KeyError, TypeError):
        return False
    return True


def fidelity(item: Item, record: Exchange) -> bool:
    """Unordered multiset match via augmenting paths, including ambiguous tolerances."""
    expected = item.expected["calls"]
    try:
        actual = [parsed_call(call) for call in calls(record)]
    except (ValueError, KeyError, TypeError):
        return False
    if len(actual) != len(expected):
        return False
    edges = [
        [
            j
            for j, target in enumerate(expected)
            if name == target["name"]
            and arguments_match(
                args,
                target["arguments"],
                numeric_tolerance=float(item.expected.get("numeric_tolerance", 0)),
            )
        ]
        for name, args in actual
    ]
    matched: dict[int, int] = {}

    def augment(index: int, seen: set[int]) -> bool:
        for target in edges[index]:
            if target not in seen:
                seen.add(target)
                if target not in matched or augment(matched[target], seen):
                    matched[target] = index
                    return True
        return False

    return all(augment(index, set()) for index in range(len(actual)))


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [s for child in value for s in _strings(child)]
    if isinstance(value, dict):
        return [s for k, child in value.items() for s in [str(k), *_strings(child)]]
    return []


def leaked(
    record: Exchange, profile: ModelProfile, *, expected_calls: list[dict[str, Any]] | None = None
) -> bool:
    markers = [
        *GENERIC_MARKERS,
        *(profile.format_markers if field_verified(profile, "format_markers") else []),
    ]
    if any(marker in content(record) for marker in markers):
        return True
    for call in calls(record):
        function = call.get("function", {})
        name = function.get("name", "")
        if any(marker in str(name) for marker in markers):
            return True
        allowed = {
            text
            for expected in expected_calls or []
            if expected.get("name") == name
            for text in _strings(expected.get("arguments"))
        }
        arguments = function.get("arguments", "")
        try:
            values = (
                _strings(strict_json(arguments))
                if isinstance(arguments, str)
                else _strings(arguments)
            )
        except ValueError:
            values = [str(arguments)]
        if any(
            text not in allowed and any(marker in text for marker in markers) for text in values
        ):
            return True
    reasoning = message(record).get("reasoning") or ""
    details = message(record).get("reasoning_details") or []
    return any(
        marker in str(reasoning) + json.dumps(details, ensure_ascii=False) for marker in markers
    )


def dropped(record: Exchange) -> bool:
    actual = calls(record)
    if choice(record).get("finish_reason") == "tool_calls" and not actual:
        return True
    if not actual and content(record):
        offered = {
            tool.get("function", {}).get("name")
            for tool in record.request.get("tools", [])
            if isinstance(tool, dict)
        }

        def parseable(candidate: Any) -> bool:
            if not isinstance(candidate, dict):
                return False
            nested = candidate.get("tool_calls")
            if isinstance(nested, list):
                return any(parseable(child) for child in nested)
            function = candidate.get("function", candidate)
            if (
                not isinstance(function, dict)
                or not isinstance(function.get("name"), str)
                or function["name"] not in offered
            ):
                return False
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = strict_json(arguments)
                except ValueError:
                    return False
            return isinstance(arguments, dict)

        try:
            value = strict_json(content(record))
            candidates = value if isinstance(value, list) else [value]
            for candidate in candidates:
                if parseable(candidate):
                    return True
        except ValueError:
            pass
    return False


def structural_violation(item: Item, record: Exchange) -> bool:
    if item.payload.get("stream") and stream_violation(record):
        return True
    actual = calls(record)
    if len(actual) != len(item.expected["calls"]):
        return True
    return bool(actual) and not schema_accuracy(item, record)


def history_format_sample(profile: ModelProfile) -> Samples:
    return Samples(
        check_id="T08",
        check_class="comparative",
        status="NOT_APPLICABLE" if profile.tool_id_format is None else "INCONCLUSIVE",
        reason="profile declares no preferred tool-id format"
        if profile.tool_id_format is None
        else "preferred history-format contract not qualified",
    )


def tool_samples(
    items: list[Item], records: list[Exchange], endpoint_id: str, profile: ModelProfile
) -> dict[str, Samples]:
    by_item: dict[str, list[Exchange]] = defaultdict(list)
    for record in records:
        if record.endpoint_id == endpoint_id:
            by_item[record.item_id].append(record)
    output = {
        check: Samples(
            check_id=check, check_class="comparative", metric=("f1" if check == "T01" else "mean")
        )
        for check in ("T01", "T02", "T03")
    }
    output.update(
        {check: Samples(check_id=check, check_class="invariant") for check in ("T05", "T06")}
    )
    for item in items:
        if item.suite_id != "S2" or item.expected["check_id"].startswith("T07"):
            continue
        check = item.expected["check_id"]
        corpus = check == "tools-corpus"
        checks = ["T01", "T02", "T03", "T05", "T06"] if corpus else [check, "T05", "T06"]
        captures = sorted(by_item[item.item_id], key=lambda record: record.sample_idx)
        for key in checks:
            output.setdefault(key, Samples(check_id=key, check_class="invariant"))
            output[key].evidence.extend(record.exchange_id for record in captures)
        status, reason = None, ""
        if len(captures) != item.expected.get("samples", 1):
            status, reason = "ERROR", "expected tool capture missing"
        elif any(record.error for record in captures):
            status, reason = "ERROR", "transport or excluded-routing capture"
        elif any(record.metadata.get("request_not_sent") for record in captures):
            status = (
                "INCONCLUSIVE"
                if any(record.metadata.get("verdict") == "INCONCLUSIVE" for record in captures)
                else "NOT_SUPPORTED"
            )
            reason = "required tool capability unavailable; see capture metadata"
        elif any(record.status != 200 for record in captures):
            status, reason = "NOT_SUPPORTED", "tool request rejected"
        if status is None:
            try:
                if any(choice(record).get("finish_reason") == "length" for record in captures):
                    status, reason = "INCONCLUSIVE", "output cap censored tool behavior"
            except ValueError:
                status, reason = "ERROR", "invalid tool response envelope"
        if status:
            for key in checks:
                output[key].mark(status, reason)
            continue
        trial = item.item_id
        try:
            if corpus:
                flags = [
                    float(choice(record).get("finish_reason") == "tool_calls")
                    for record in captures
                ]
                output["T01"].values[trial] = flags[0]
                output["T01"].repeats[trial] = flags
                output["T01"].details.setdefault("history_by_item", {})[trial] = bool(
                    item.expected.get("history")
                )
                if flags[0]:
                    output["T02"].values[trial] = float(schema_accuracy(item, captures[0]))
                if item.expected["should_call"]:
                    output["T03"].values[trial] = float(fidelity(item, captures[0]))
            else:
                output[check].values[trial] = float(
                    any(structural_violation(item, record) for record in captures)
                )
                signatures = output[check].details.setdefault("failure_signatures", [])
                for record in captures:
                    for signature, present in (
                        (
                            "missing_stream_index",
                            bool(record.metadata.get("tool_delta_missing_index")),
                        ),
                        ("call_count", len(calls(record)) != len(item.expected["calls"])),
                        (
                            "argument_schema",
                            bool(calls(record)) and not schema_accuracy(item, record),
                        ),
                    ):
                        if present and signature not in signatures:
                            signatures.append(signature)
            invariant_trial = trial if corpus else "structure-" + str(item.expected["trial"])
            for key, failed in (
                (
                    "T05",
                    any(
                        leaked(record, profile, expected_calls=item.expected.get("calls"))
                        for record in captures
                    ),
                ),
                ("T06", any(dropped(record) for record in captures)),
            ):
                output[key].values[invariant_trial] = max(
                    output[key].values.get(invariant_trial, 0), float(failed)
                )
            for key in checks:
                outcome_key = invariant_trial if key in {"T05", "T06"} else trial
                if output[key].check_class == "invariant" and output[key].values.get(outcome_key):
                    output[key].replay_candidates.append([item.item_id])
        except (ValueError, KeyError, TypeError, IndexError):
            for key in checks:
                output[key].mark("ERROR", "invalid tool response envelope")
    # Paired structure probes provide a direct streamed/non-streamed drop observation.
    pairs: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        if item.suite_id == "S2" and item.expected["check_id"].startswith("T04"):
            pairs[str(item.expected["trial"])].append(item)
    for pair in pairs.values():
        if len(pair) == 2 and all(by_item[item.item_id] for item in pair):
            first, second = (by_item[item.item_id][0] for item in pair)
            if any(
                record.error or record.status != 200 or record.metadata.get("request_not_sent")
                for record in (first, second)
            ):
                continue
            try:
                if any(
                    choice(record).get("finish_reason") == "length" for record in (first, second)
                ):
                    continue
                if len(calls(first)) != len(calls(second)):
                    output["T06"].values["structure-" + str(pair[0].expected["trial"])] = 1.0
                    output["T06"].replay_candidates.append([item.item_id for item in pair])
            except ValueError:
                output["T06"].mark("ERROR", "invalid paired tool envelope")
    output["T08"] = history_format_sample(profile)
    return {
        key: sample
        for key, sample in output.items()
        if sample.values or sample.status or sample.evidence
    }
