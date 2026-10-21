"""S3 JSON syntax, schema/semantic accuracy and sourced support-contract observations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from jsonschema import Draft202012Validator

from servegauge.core.capabilities import declared_structured, strict_schema_contract
from servegauge.core.errors import SafeError
from servegauge.core.json import strict_json
from servegauge.core.models import EndpointSpec, Exchange, Item, ModelProfile
from servegauge.scoring.common import Samples, choice, content
from servegauge.scoring.glm_protocol import GLMObservationUnavailable
from servegauge.scoring.toolcall_scoring import arguments_match
from servegauge.tokencount.counters import text_tokens
from servegauge.tokencount.model_text import uses_glm_profile


def parsed_object(record: Exchange) -> dict[str, Any] | None:
    try:
        value = strict_json(content(record))
        return value if isinstance(value, dict) else None
    except ValueError:
        return None


def json_samples(
    items: list[Item],
    records: list[Exchange],
    endpoint: EndpointSpec,
    *,
    profile: ModelProfile | None = None,
) -> dict[str, Samples]:
    by_item: dict[str, list[Exchange]] = defaultdict(list)
    for record in records:
        if record.endpoint_id == endpoint.id:
            by_item[record.item_id].append(record)
    output: dict[str, Samples] = {}
    for item in items:
        if item.suite_id != "S3":
            continue
        check = item.expected["check_id"]
        keys = ["J02.validity", "J02.semantic"] if check == "J02" else [check]
        if check == "J02" and item.expected["category"] == "flat":
            keys.append("J04")
        for key in keys:
            output.setdefault(
                key,
                Samples(
                    check_id=key,
                    check_class="invariant"
                    if key in {"J02.validity", "J04", "J05"}
                    else "comparative",
                ),
            )
        captures = by_item[item.item_id]
        for key in keys:
            output[key].evidence.extend(record.exchange_id for record in captures)
        if len(captures) != 1:
            for key in keys:
                output[key].mark("ERROR", "expected JSON capture missing")
            continue
        record = captures[0]
        if check == "J05" and (uses_glm_profile(profile) or endpoint.claimed_profile == "glm-5.3"):
            output[check].details["counter_scope"] = (
                "claimed-profile local text projection and API-reported count; "
                "not generated-token IDs or billing proof"
            )
        if record.error or record.metadata.get("request_not_sent"):
            for key in keys:
                output[key].mark(
                    "ERROR" if record.error else record.metadata.get("verdict", "INCONCLUSIVE"),
                    "JSON transport/excluded routing"
                    if record.error
                    else "required JSON capability unavailable; see capture metadata",
                )
            continue
        trial = str(item.expected["trial"])
        declared = declared_structured(endpoint)
        if "J04" in keys:
            sample = output["J04"]
            sample.details.update(
                declared=declared,
                observed_statuses={
                    **sample.details.get("observed_statuses", {}),
                    trial: record.status,
                },
            )
            if declared is None:
                sample.mark("INCONCLUSIVE", "sourced support declaration unavailable")
            elif not declared:
                sample.mark("NOT_APPLICABLE", "no positive support declaration to contradict")
            elif record.status == 200:
                sample.values[trial] = 0.0
            elif record.status in {400, 404, 422, 501}:
                error = (record.response or {}).get("error", {})
                code = error.get("code") if isinstance(error, dict) else None
                if code in {
                    "unsupported_response_format",
                    "json_schema_not_supported",
                    "unsupported_parameter",
                }:
                    sample.values[trial] = 1.0
                    sample.replay_candidates.append([item.item_id])
                else:
                    sample.mark("INCONCLUSIVE", "rejection does not isolate declared support")
        if record.status != 200:
            for key in keys:
                if key != "J04":
                    output[key].mark("NOT_SUPPORTED", "JSON mode or schema feature rejected")
            continue
        try:
            finish = choice(record).get("finish_reason")
            if check == "J05":
                text = content(record)
                cap = record.request.get("max_completion_tokens", record.request.get("max_tokens"))
                if not isinstance(cap, int):
                    raise ValueError("output_cap_unavailable")
                full = item.expected["full_text"]
                cut = bool(text) and len(text) < len(full) and full.startswith(text)
                if (
                    uses_glm_profile(profile)
                    or endpoint.claimed_profile == "glm-5.3"
                    or item.expected.get("text_counter_profile") == "glm-5.3"
                ):
                    from servegauge.scoring.glm_protocol import cap_violation

                    value = float(cap_violation(item, record, profile))
                else:
                    value = float(text_tokens(text) > cap or (cut and finish != "length"))
                if not text and finish != "length":
                    output[check].mark("INCONCLUSIVE", "no visible truncation observation")
                output[check].values[trial] = value
            elif finish == "length":
                for key in keys:
                    if key != "J04":
                        output[key].mark("INCONCLUSIVE", "output cap censored JSON behavior")
            else:
                parsed = parsed_object(record)
                valid = parsed is not None and Draft202012Validator(
                    item.expected["schema"]
                ).is_valid(parsed)
                semantic = valid and arguments_match(
                    parsed,
                    item.expected["value"],
                    numeric_tolerance=item.expected["numeric_tolerance"],
                )
                if check == "J02":
                    output["J02.validity"].values[trial] = float(not valid)
                    output["J02.semantic"].values[trial] = float(semantic)
                    if not strict_schema_contract(endpoint, item.expected["category"]):
                        output["J02.validity"].mark(
                            "INCONCLUSIVE",
                            "native strict guarantee for the schema feature is unverified",
                        )
                    if not valid:
                        output["J02.validity"].replay_candidates.append([item.item_id])
                else:
                    output[check].values[trial] = float(
                        parsed is not None if check == "J01" else semantic
                    )
            if check == "J05" and output[check].values.get(trial):
                output[check].replay_candidates.append([item.item_id])
        except GLMObservationUnavailable as error:
            for key in keys:
                output[key].mark("INCONCLUSIVE", str(error))
        except SafeError:
            for key in keys:
                output[key].mark("ERROR", "verified local tokenizer cache unavailable")
        except (ValueError, TypeError, KeyError, IndexError):
            for key in keys:
                output[key].mark("ERROR", "invalid JSON response envelope")
    return output
