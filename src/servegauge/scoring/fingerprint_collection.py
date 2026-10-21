"""Collect complete observed fingerprint evidence without imputing missing responses."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

from servegauge.core.hashing import content_hash
from servegauge.core.models import CheckResult, EndpointSpec, Exchange, Item
from servegauge.scoring.common import Samples, choice, content
from servegauge.scoring.fingerprint_evidence import read_response
from servegauge.verdicts.config import DecisionPolicy, load_policy


def sampling_request_hash(body: dict[str, Any]) -> str:
    """Normalize transport aliases; preserve every potentially behavioral option."""
    value = {k: v for k, v in body.items() if k not in {"model", "provider"}}
    if value.get("plugins") == [{"id": "context-compression", "enabled": False}]:
        value.pop("plugins")
    if "max_completion_tokens" in value and "max_tokens" not in value:
        value["max_tokens"] = value.pop("max_completion_tokens")
    if "reasoning_effort" in value and "reasoning" not in value:
        value["reasoning"] = {"effort": value.pop("reasoning_effort")}
    return content_hash(value)


def fingerprint_samples(
    items: list[Item],
    records: list[Exchange],
    endpoint: EndpointSpec,
    policy: DecisionPolicy | None = None,
) -> dict[str, Samples]:
    inventory: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        if item.suite_id == "S6":
            inventory[item.expected["check_id"]].append(item)
    if not inventory:
        return {}
    by_item: dict[str, list[Exchange]] = defaultdict(list)
    for record in records:
        if record.endpoint_id == endpoint.id and record.suite_id == "S6":
            by_item[record.item_id].append(record)
    output = {}
    tolerance = float((policy or load_policy()).entries["P06"]["prob_sum_tolerance"])
    for method, probes in inventory.items():
        sample = Samples(
            check_id=method,
            check_class="comparative",
            binary=False,
            two_sided=True,
            metric="fingerprint",
        )
        observations: list[dict[str, Any]] = []
        counts = {
            "requests": 0,
            "captures": 0,
            "missing_responses": 0,
            "duplicate_identities": 0,
            "missing_logprobs": 0,
            "censored_responses": 0,
            "invalid_responses": 0,
        }
        expected_requests = sum(i.expected["samples"] for i in probes)
        for item in probes:
            rows = by_item[item.item_id]
            identities: dict[int, list[Exchange]] = defaultdict(list)
            for row in rows:
                identities[row.sample_idx].append(row)
            counts["duplicate_identities"] += sum(max(0, len(v) - 1) for v in identities.values())
            if any(not 0 <= index < item.expected["samples"] for index in identities):
                sample.mark("INCONCLUSIVE", "unexpected fingerprint sample identity")
            for index in range(item.expected["samples"]):
                if index not in identities:
                    counts["missing_responses"] += 1
                    observations.append(
                        {
                            "cell": item.item_id,
                            "draw": index,
                            "response": None,
                            "reason": "missing_capture",
                        }
                    )
            for row in sorted(rows, key=lambda r: (r.sample_idx, r.exchange_id)):
                counts["captures"] += 1
                counts["requests"] += int(
                    not row.metadata.get("request_not_sent")
                    and row.error not in {"NOT_SUPPORTED", "INCONCLUSIVE"}
                )
                sample.evidence.append(row.exchange_id)
                observation: dict[str, Any] = {
                    "cell": item.item_id,
                    "draw": row.sample_idx,
                    "exchange_id": row.exchange_id,
                    "request_hash": content_hash(row.request),
                    "sampling_request_hash": sampling_request_hash(row.request),
                    "metadata": copy.deepcopy(row.metadata),
                    "response": copy.deepcopy(row.response),
                    "text": None,
                    "reason": None,
                }
                if row.metadata.get("request_not_sent") or row.error in {
                    "NOT_SUPPORTED",
                    "INCONCLUSIVE",
                }:
                    sample.mark(
                        row.metadata.get("verdict", row.error or "INCONCLUSIVE"),
                        "fingerprint capability unavailable; see capture",
                    )
                    observation["reason"] = "request_not_sent"
                elif row.error or row.status != 200:
                    sample.mark("ERROR", "fingerprint transport or capture failure")
                    observation["reason"] = "capture_failure"
                else:
                    try:
                        finish, text = choice(row).get("finish_reason"), content(row)
                        if not text or finish not in {"stop", "length"}:
                            raise ValueError("invalid_final_text")
                        censored = finish == "length" and (
                            method != "FP2" or len(text) < item.expected["fixed_length"]
                        )
                        if censored:
                            counts["censored_responses"] += 1
                            observation["reason"] = "censored_response"
                        else:
                            observation["text"] = text
                        if method == "FP3":
                            parsed = read_response(row.response, tolerance=tolerance)
                            if parsed.logprobs is None:
                                counts["missing_logprobs"] += 1
                                observation["logprob_reason"] = parsed.reason
                    except (ValueError, KeyError, TypeError):
                        counts["invalid_responses"] += 1
                        observation["reason"] = "invalid_response"
                observations.append(observation)
        if sample.status is None and any(
            counts[k]
            for k in (
                "missing_responses",
                "duplicate_identities",
                "censored_responses",
                "invalid_responses",
            )
        ):
            sample.mark("INCONCLUSIVE", "complete matched fingerprint evidence required")
        sample.details = {
            **counts,
            "expected_requests": expected_requests,
            "cells": len(probes),
            "observations": observations,
            "design": copy.deepcopy(probes[0].expected["design"]),
            "design_hash": probes[0].expected["design_hash"],
            "item_ids": [i.item_id for i in probes],
            "sampling_design": copy.deepcopy(endpoint.metadata.get("fingerprint_sampling", {})),
            "policy_hash": content_hash((policy or load_policy()).model_dump(mode="json")),
            "observation_hash": content_hash(observations),
        }
        output[method] = sample
    return output


def pending_checks(samples: dict[str, Samples], endpoint_id: str, run_id: str) -> list[CheckResult]:
    """Raw collection alone cannot support a calibrated comparative verdict."""
    return [
        CheckResult(
            check_id=method,
            check_class="comparative",
            suite_id="S6",
            endpoint_id=endpoint_id,
            run_id=run_id,
            n=sample.details["cells"],
            verdict="ERROR"
            if sample.status == "ERROR"
            else "NOT_SUPPORTED"
            if sample.status == "NOT_SUPPORTED"
            else "INCONCLUSIVE",
            reference_tier=None,
            reason=sample.reason or "matched fingerprint reference calibration unavailable",
            evidence=sorted(set(sample.evidence)),
            details={
                k: v
                for k, v in sample.details.items()
                if k not in {"observations", "sampling_design", "design"}
            },
        )
        for method, sample in samples.items()
        if sample.metric == "fingerprint"
    ]
