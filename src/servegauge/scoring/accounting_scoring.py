"""B01 relative template overhead, calibrated against identical reference inputs."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any

from jinja2 import TemplateError

from servegauge.core.errors import SafeError
from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.models import Exchange, Item, ModelProfile
from servegauge.core.redaction import REDACTED
from servegauge.scoring.common import Samples
from servegauge.suites.s7_accounting import billing_group
from servegauge.tokencount.glm_counter import (
    glm_request_controls,
    glm_text_tokens,
    render_glm_chat,
)
from servegauge.tokencount.hf_counter import hf_text_tokens, render_hf_chat
from servegauge.tokencount.profile import counter_kind


def accounting_samples(
    items: list[Item],
    records: list[Exchange],
    endpoint_id: str,
    profile: ModelProfile,
    cfg: dict[str, Any],
) -> dict[str, Samples]:
    by_item: dict[str, list[Exchange]] = defaultdict(list)
    for record in records:
        if record.endpoint_id == endpoint_id:
            by_item[record.item_id].append(record)
    output: dict[str, Samples] = {}
    cache: dict[str, tuple[int, str]] = {}
    kind = counter_kind(profile)
    for item in items:
        group, unit = billing_group(item)
        sample = output.setdefault(
            group, Samples(check_id=group, check_class="comparative", binary=False, two_sided=True)
        )
        sample.details.update(
            counter_date=cfg["counter_date"],
            counter_version=cfg["counter_version"],
            units="(reported prompt tokens - official local count) / local count",
            calibration="matched-n, identical-input reference overhead required before verdict",
            scope="static traffic; dynamic agent continuations need their own matched layouts",
        )
        if kind is None:
            sample.mark("INCONCLUSIVE", "verified local profile counter unavailable")
            continue
        if kind == "glm-5.3":
            sample.details.update(
                counter_version="glm-hf-api-adapter-v1",
                claimed_tokenizer=profile.tokenizer,
                provider_template_verified=False,
            )
        available = sorted(by_item[item.item_id], key=lambda record: record.sample_idx)
        if len(available) != item.expected.get("samples", 1):
            sample.mark("ERROR", "prompt accounting capture missing")
            continue
        for record in available:
            sample.evidence.append(record.exchange_id)
            if record.error:
                sample.mark("ERROR", "prompt accounting transport/excluded routing")
                continue
            if record.metadata.get("request_not_sent"):
                sample.mark(
                    "INCONCLUSIVE", "input group incomplete because a feature was unavailable"
                )
                continue
            if record.status != 200:
                sample.details.setdefault("excluded_rejections", []).append(record.exchange_id)
                continue
            reported = (record.usage or {}).get("prompt_tokens")
            if not isinstance(reported, int) or isinstance(reported, bool) or reported < 0:
                sample.mark("INCONCLUSIVE", "reported prompt token count unavailable or invalid")
                continue
            body = record.request
            if REDACTED.encode() in canonical_json(body):
                sample.mark("INCONCLUSIVE", "redacted input cannot supply an exact local count")
                continue
            reasoning = body.get("reasoning")
            default_effort = "max" if kind == "glm-5.3" else "medium"
            effort = body.get(
                "reasoning_effort",
                reasoning.get("effort", default_effort)
                if isinstance(reasoning, dict)
                else default_effort,
            )
            if not isinstance(effort, str):
                sample.mark("INCONCLUSIVE", "reasoning template setting unavailable")
                continue
            if kind == "glm-5.3":
                try:
                    glm_request_controls(body)
                except (ValueError, TypeError):
                    sample.mark("INCONCLUSIVE", "unknown or conflicting reasoning controls")
                    continue
            counter_input = {
                "messages": body["messages"],
                "tools": body.get("tools"),
                "effort": effort,
                "date": cfg["counter_date"],
                "prompt_controls": {
                    key: body[key]
                    for key in ("response_format", "tool_choice", "parallel_tool_calls")
                    if key in body
                },
            }
            if kind == "glm-5.3":
                counter_input["clear_thinking"] = body.get("clear_thinking", False)
                counter_input["prompt_controls"].update(
                    {key: body[key] for key in ("clear_thinking", "thinking") if key in body}
                )
            signature = content_hash(counter_input)
            try:
                if signature not in cache:
                    if kind == "glm-5.3":
                        rendered = render_glm_chat(
                            body["messages"],
                            tools=body.get("tools"),
                            reasoning_effort=effort,
                            clear_thinking=body.get("clear_thinking", False),
                        )
                        local_count = glm_text_tokens(rendered)
                    else:
                        rendered = render_hf_chat(
                            body["messages"],
                            tools=body.get("tools"),
                            reasoning_effort=effort,
                            conversation_date=cfg["counter_date"],
                        )
                        local_count = hf_text_tokens(rendered)
                    cache[signature] = (
                        local_count,
                        content_hash(
                            [
                                hashlib.sha256(rendered.encode()).hexdigest(),
                                counter_input["prompt_controls"],
                            ]
                        ),
                    )
                local, input_hash = cache[signature]
                if local <= 0:
                    raise ValueError("nonpositive_local_count")
                value = (reported - local) / local
                if not math.isfinite(value):
                    raise ValueError("nonfinite_count_ratio")
            except SafeError:
                sample.mark("ERROR", "local tokenizer cache unavailable")
                continue
            except (ValueError, TypeError, KeyError, AttributeError, TemplateError, OverflowError):
                sample.mark("INCONCLUSIVE", "input cannot be counted by the verified template")
                continue
            sample.repeats.setdefault(unit, []).append(value)
            sample.values[unit] = math.fsum(sample.repeats[unit]) / len(sample.repeats[unit])
            hashes = sample.details.setdefault("input_hashes", {}).setdefault(unit, [])
            if input_hash not in hashes:
                hashes.append(input_hash)
                hashes.sort()
            sample.details.setdefault("counts", {}).setdefault(unit, []).append(
                {
                    "exchange_id": record.exchange_id,
                    "reported": reported,
                    "local": local,
                    "overhead_tokens": reported - local,
                    "relative_overhead": value,
                }
            )
    for sample in output.values():
        if not sample.values and sample.status is None:
            sample.mark("INCONCLUSIVE", "no successful requests with qualified prompt counts")
    return output
