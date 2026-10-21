"""B02 complete text projections and B03 generation field pairs, from captures only.

These are relative, matched-reference measurements. Retokenizing returned text is
not recovery of generated token IDs or proof of the provider's bill.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from servegauge.core.completion import completion_error
from servegauge.core.errors import SafeError
from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.models import Exchange, Item, ModelProfile
from servegauge.core.redaction import REDACTED
from servegauge.scoring.common import Samples
from servegauge.suites.s7_accounting import extended_contract, extended_groups
from servegauge.tokencount.glm_counter import glm_text_tokens
from servegauge.tokencount.hf_counter import hf_text_tokens
from servegauge.tokencount.profile import counter_kind


class AccountingUnavailable(ValueError):
    """A fixed reason code safe to include in a report."""


def _count(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AccountingUnavailable("nonnegative_integer_token_count_required")
    return value


def _complete(record: Exchange) -> None:
    if (
        record.error
        or record.status != 200
        or record.metadata.get("request_not_sent")
        or record.metadata.get("exclude_from_scoring")
    ):
        raise AccountingUnavailable("successful_sent_capture_required")
    if any(
        record.metadata.get(key)
        for key in (
            "incomplete",
            "invalid_utf8",
            "invalid_json_events",
            "capture_truncated",
            "stream_multiple_choices",
            "stream_identity_conflict",
        )
    ) or (record.request.get("stream") and not record.metadata.get("terminal")):
        raise AccountingUnavailable("complete_capture_required")
    if completion_error(record.response):
        raise AccountingUnavailable("completion_error")
    if REDACTED.encode() in canonical_json([record.request, record.response]):
        raise AccountingUnavailable("redacted_accounting_capture")


def completion_channels(record: Exchange, *, allow_length: bool = False) -> tuple[str, str, str]:
    """Return text once; explicit cap checks may count a terminal length-censored projection.

    B02 keeps the uncensored default. Allowing length does not recover generated
    token IDs or establish hidden reasoning completeness.
    """
    _complete(record)
    choices = (record.response or {}).get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise AccountingUnavailable("one_complete_choice_required")
    selected = choices[0]
    if (
        selected.get("finish_reason") not in ({"stop", "length"} if allow_length else {"stop"})
        or type(record.request.get("n", 1)) is not int
        or record.request.get("n", 1) != 1
        or type(selected.get("index", 0)) is not int
        or selected.get("index", 0) != 0
    ):
        raise AccountingUnavailable("uncensored_single_text_completion_required")
    message = selected.get("message")
    if not isinstance(message, dict) or any(
        message.get(key)
        for key in (
            "tool_calls",
            "function_call",
            "audio",
            "images",
            "refusal",
        )
    ):
        raise AccountingUnavailable("text_only_completion_required")
    visible = message.get("content")
    if visible is None:
        visible = ""
    if not isinstance(visible, str):
        raise AccountingUnavailable("text_content_required")
    request_reasoning = record.request.get("reasoning")
    if isinstance(request_reasoning, dict) and request_reasoning.get("exclude"):
        raise AccountingUnavailable("reasoning_was_excluded")
    reported = _count((record.usage or {}).get("completion_tokens"))
    details = (record.usage or {}).get("completion_tokens_details", {})
    if not isinstance(details, dict):
        raise AccountingUnavailable("invalid_completion_token_details")
    reason_count = None
    for key, value in details.items():
        count = _count(value)
        if key == "reasoning_tokens":
            reason_count = count
            if count > reported:
                raise AccountingUnavailable("reasoning_count_exceeds_completion")
        elif count and key != "accepted_prediction_tokens":
            raise AccountingUnavailable("unsupported_output_token_channel")
    aliases = []
    for key in ("reasoning", "reasoning_content"):
        if message.get(key) is not None:
            if not isinstance(message[key], str):
                raise AccountingUnavailable("text_reasoning_required")
            aliases.append(message[key])
    raw_details = message.get("reasoning_details")
    if raw_details is not None:
        if not isinstance(raw_details, list):
            raise AccountingUnavailable("reasoning_details_list_required")
        chunks = []
        for detail in raw_details:
            if (
                not isinstance(detail, dict)
                or detail.get("type") != "reasoning.text"
                or not isinstance(detail.get("text"), str)
            ):
                raise AccountingUnavailable("hidden_summarized_or_unsupported_reasoning")
            chunks.append(detail["text"])
        if chunks:
            aliases.append("".join(chunks))
    if aliases and any(alias != aliases[0] for alias in aliases):
        raise AccountingUnavailable("conflicting_reasoning_mirrors")
    reasoning = aliases[0] if aliases else ""
    if not reasoning and reason_count != 0:
        raise AccountingUnavailable("reasoning_visibility_unknown_or_hidden")
    if reasoning and not raw_details:
        raise AccountingUnavailable("raw_reasoning_type_not_established")
    if reasoning and reason_count == 0:
        raise AccountingUnavailable("reasoning_visibility_conflicts_with_zero_count")
    return visible, reasoning, "raw-text" if reasoning else "zero-reasoning"


def _generation(record: Exchange, channel: str) -> tuple[int, int]:
    _complete(record)
    generation = record.provider_meta or {}
    response = record.response or {}
    if (
        not isinstance(response.get("id"), str)
        or not response["id"]
        or generation.get("id") != response["id"]
        or not isinstance(response.get("model"), str)
        or not response["model"]
        or generation.get("model") != response["model"]
        or generation.get("cancelled") is not False
    ):
        raise AccountingUnavailable("matched_complete_generation_record_required")
    normalized = _count(generation.get("tokens_" + channel))
    native = _count(generation.get("native_tokens_" + channel))
    if native == 0:
        raise AccountingUnavailable("positive_native_denominator_required")
    return normalized, native


def _append(sample: Samples, unit: str, value: float, layout: str, counts: dict[str, Any]) -> None:
    if not math.isfinite(value):
        raise AccountingUnavailable("finite_accounting_ratio_required")
    sample.repeats.setdefault(unit, []).append(value)
    sample.values[unit] = math.fsum(sample.repeats[unit]) / len(sample.repeats[unit])
    hashes = sample.details.setdefault("input_hashes", {}).setdefault(unit, [])
    if layout not in hashes:
        hashes.append(layout)
        hashes.sort()
    sample.details.setdefault("counts", {}).setdefault(unit, []).append(counts)


def extended_accounting_samples(
    items: list[Item],
    records: list[Exchange],
    endpoint_id: str,
    profile: ModelProfile,
) -> dict[str, Samples]:
    contract = extended_contract(items)
    if contract is None:
        return {}
    by_item: dict[str, list[Exchange]] = defaultdict(list)
    for record in records:
        if record.endpoint_id == endpoint_id:
            by_item[record.item_id].append(record)
    output: dict[str, Samples] = {}
    kind = counter_kind(profile)
    for item in items:
        for group, unit in extended_groups(item):
            completion = group.startswith("B02")
            sample = output.setdefault(
                group,
                Samples(check_id=group, check_class="comparative", binary=False, two_sided=True),
            )
            sample.details.update(
                accounting_contract=contract,
                units="(reported completion - local visible/raw reasoning tokens) / local tokens"
                if completion
                else "(generation tokens - generation native tokens) / native tokens",
                calibration="identical request and channel layouts; matched-n reference required",
                scope="static captures; text projection is not generated-token or invoice proof",
                semantics="raw text assumed complete as returned; opaque channels abstain"
                if completion
                else "distinct captured field definitions; equality is not assumed",
            )
            available = sorted(by_item[item.item_id], key=lambda record: record.sample_idx)
            if [record.sample_idx for record in available] != list(
                range(int(item.expected.get("samples", 1)))
            ):
                sample.mark("ERROR", "accounting capture missing or duplicated")
                continue
            for record in available:
                sample.evidence.append(record.exchange_id)
                try:
                    # Route/model names are identities, not prompt or channel layouts.
                    request = {
                        k: v for k, v in record.request.items() if k not in {"model", "provider"}
                    }
                    layout: dict[str, Any] = {
                        "request": request,
                        "sample_idx": record.sample_idx,
                        "contract": contract,
                        "check": group,
                    }
                    if completion:
                        if kind is None:
                            raise AccountingUnavailable(
                                "verified_local_profile_counter_unavailable"
                            )
                        visible, reasoning, mode = completion_channels(record)
                        count_text = glm_text_tokens if kind == "glm-5.3" else hf_text_tokens
                        local_visible = count_text(visible)
                        local_reasoning = count_text(reasoning)
                        local = local_visible + local_reasoning
                        if local <= 0:
                            raise AccountingUnavailable("positive_local_denominator_required")
                        reported = _count((record.usage or {}).get("completion_tokens"))
                        value = (reported - local) / local
                        layout.update(mode=mode, tokenizer=profile.tokenizer)
                        counts = {
                            "exchange_id": record.exchange_id,
                            "reported": reported,
                            "local_visible": local_visible,
                            "local_reasoning": local_reasoning,
                            "local": local,
                            "relative_overhead": value,
                        }
                    else:
                        channel = "prompt" if group.startswith("B03.prompt") else "completion"
                        normalized, native = _generation(record, channel)
                        value = (normalized - native) / native
                        counts = {
                            "exchange_id": record.exchange_id,
                            "generation_tokens": normalized,
                            "generation_native_tokens": native,
                            "relative_difference": value,
                        }
                    _append(sample, unit, value, content_hash(layout), counts)
                except SafeError:
                    sample.mark("ERROR", "local tokenizer cache unavailable")
                except AccountingUnavailable as error:
                    sample.mark(
                        "ERROR" if record.error else "INCONCLUSIVE",
                        "accounting evidence unavailable: " + str(error),
                    )
                except (ValueError, KeyError, TypeError, AttributeError, OverflowError):
                    sample.mark(
                        "INCONCLUSIVE", "accounting evidence shape or local counter invalid"
                    )
            if not sample.values and sample.status is None:
                sample.mark("INCONCLUSIVE", "no qualified accounting observations")
    return output
