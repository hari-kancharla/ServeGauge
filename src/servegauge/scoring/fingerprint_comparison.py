"""Validate captured strata, sampling declarations and whole cache components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from servegauge.core.errors import SafeError
from servegauge.core.hashing import content_hash
from servegauge.scoring.common import Samples
from servegauge.scoring.fingerprint_analysis import group_evidence
from servegauge.scoring.fingerprint_evidence import (
    FingerprintEvidence,
    FingerprintSample,
    PreparedEvidence,
    ResponseEvidence,
    prepare_evidence,
)
from servegauge.stats.fingerprint_energy import energy_cell
from servegauge.stats.fingerprint_strata import Cell, text_cell
from servegauge.verdicts.config import DecisionPolicy

if TYPE_CHECKING:
    from servegauge.embeddings.local import LocalEncoder


def signature(sample: Samples) -> str:
    base = sample.check_id.lower()
    design = sample.details["design"]
    sizes = design["sizes"]
    value = {
        "method": sample.check_id,
        "version": design["version"],
        "prompts": design["prompt_hash"],
        "output_cap": design["output_cap"],
        "samples": sizes[base + "_samples"],
        "cells": sizes[base + "_cells"] if base != "fp3" else 1,
        "anchors": design["anchor_tokens"] if base == "fp3" else None,
        "minimum_anchor_tokens": design.get("minimum_anchor_tokens") if base == "fp3" else None,
        "top_logprobs": design["top_logprobs"] if base == "fp3" else None,
        "length": design["fp2_length"] if base == "fp2" else None,
        "minimum_independent_groups": design.get("minimum_independent_groups"),
        "policy": sample.details.get("policy_hash"),
    }
    if base == "fp4":
        value["embedding_model"] = design["embedding_model"]
        value["embedding_version"] = design["embedding_version"]
    return content_hash(value)


def _layout(sample: Samples) -> dict[str, list[dict[str, Any]]]:
    if sample.status is not None:
        raise ValueError("unavailable_fingerprint_capture")
    design = sample.details["design"]
    base = sample.check_id.lower()
    count = design["sizes"][base + "_samples"]
    ids = sample.details["item_ids"]
    cells = design["sizes"][base + "_cells"] if base != "fp3" else 1
    if not ids or len(ids) != cells or len(ids) != len(set(ids)):
        raise ValueError("unique_prompt_cells_required")
    rows = sample.details["observations"]
    mapping = {(r["cell"], r["draw"]): r for r in rows}
    wanted = {(cell, i) for cell in ids for i in range(count)}
    if len(mapping) != len(rows) or set(mapping) != wanted:
        raise ValueError("complete_matched_fingerprint_identities_required")
    if any(not isinstance(r.get("text"), str) or not r["text"] or r.get("reason") for r in rows):
        raise ValueError("complete_uncensored_fingerprint_text_required")
    return {cell: [mapping[(cell, i)] for i in range(count)] for cell in sorted(ids)}


def _sampling(sample: Samples, rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    declared: dict[str, Any] = sample.details["sampling_design"]
    if (
        declared.get("independent_sessions") is not True
        or declared.get("matched_sampling_design") is not True
        or declared.get("independent_reference_captures") is not True
    ):
        raise ValueError("independent_matched_sampling_declaration_required")
    if len(rows) > 1 and declared.get("independent_prompt_strata") is not True:
        raise ValueError("independent_prompt_strata_required")
    size = declared.get("session_size")
    if type(size) is not int or size < 1:
        raise ValueError("positive_session_size_required")
    cache = declared.get("cache_groups", {})
    if not isinstance(cache, dict) or any(not isinstance(v, str) or not v for v in cache.values()):
        raise ValueError("explicit_response_cache_lineages_required")
    known = {f"{cell}/{r['draw']}" for cell, values in rows.items() for r in values}
    if any(
        not isinstance(k, str)
        or not any(k.startswith(prefix + "-") for prefix in ("FP1", "FP2", "FP3", "FP4"))
        or (k.startswith(sample.check_id + "-") and k not in known)
        for k in cache
    ):
        raise ValueError("unknown_cache_sample_identity")
    seen: dict[str, set[str]] = {}
    for cell, values in rows.items():
        for row in values:
            group = cache.get(f"{cell}/{row['draw']}")
            if group is not None:
                seen.setdefault(group, set()).add(cell)
    if any(len(cells) > 1 for cells in seen.values()):
        raise ValueError("cross_prompt_cache_dependence_requires_new_design")
    return declared


def collection_cells(
    samples: list[Samples],
    *,
    method: str,
    policy: DecisionPolicy,
    encoder: LocalEncoder | None = None,
) -> list[Cell]:
    base = "FP3" if method.startswith("FP3-") else method
    if len(samples) != 4 or any(s.check_id != base for s in samples):
        raise ValueError("target_and_three_matched_reference_captures_required")
    if len({signature(s) for s in samples}) != 1:
        raise ValueError("fingerprint_configuration_mismatch")
    vectors = {}
    if method == "FP4" and (
        encoder is None
        or encoder.spec.model_dump(mode="json") != samples[0].details["design"]["embedding_model"]
    ):
        raise ValueError("matching_pinned_embedding_runtime_required")
    if any(
        s.details.get("policy_hash") != content_hash(policy.model_dump(mode="json"))
        for s in samples
    ):
        raise ValueError("fingerprint_policy_mismatch")
    layouts = [_layout(s) for s in samples]
    if method == "FP4":
        texts = sorted(
            {row["text"] for layout in layouts for rows in layout.values() for row in rows}
        )
        try:
            assert encoder is not None
            encoded = encoder.encode(texts)
        except SafeError as error:
            raise ValueError(error.code) from None
        vectors = dict(zip(texts, encoded, strict=True))
    identities = [
        row.get("exchange_id") for layout in layouts for rows in layout.values() for row in rows
    ]
    if any(not isinstance(v, str) or not v for v in identities) or len(set(identities)) != len(
        identities
    ):
        raise ValueError("independent_capture_evidence_required")
    if any(set(layout) != set(layouts[0]) for layout in layouts):
        raise ValueError("matched_prompt_cells_required")
    sampling = [_sampling(s, rows) for s, rows in zip(samples, layouts, strict=True)]
    if len({s["session_size"] for s in sampling}) != 1:
        raise ValueError("matched_session_sizes_required")
    result = []
    for cell in layouts[0]:
        sides = [rows[cell] for rows in layouts]
        if any(len(side) != len(sides[0]) for side in sides):
            raise ValueError("matched_request_counts_required")
        for i in range(len(sides[0])):
            hashes = {side[i].get("sampling_request_hash") for side in sides}
            if len(hashes) != 1 or None in hashes:
                raise ValueError("matched_actual_sampling_requests_required")
        wire_sides = [
            [
                FingerprintSample(
                    session_id=str(row["draw"] // declared["session_size"]),
                    draw=row["draw"],
                    response=row["response"],
                    cache_group=declared.get("cache_groups", {}).get(f"{cell}/{row['draw']}"),
                )
                for row in rows
            ]
            for rows, declared in zip(sides, sampling, strict=True)
        ]
        partitions = []
        parsed_sides: list[list[dict[str, float]]] = []
        for index in range(1, 4):
            evidence = FingerprintEvidence(
                probe_id=cell,
                reference_tier="MOCK",
                anchor_source="fixed-probe-configuration",
                anchor_tokens=samples[0].details["design"]["anchor_tokens"],
                independent_sessions=True,
                matched_sampling_design=True,
                target=wire_sides[0],
                reference=wire_sides[index],
            )
            if base == "FP3":
                prepared = prepare_evidence(
                    evidence,
                    minimum_anchor_tokens=samples[0].details["design"]["minimum_anchor_tokens"],
                    tolerance=float(policy.entries["P06"]["prob_sum_tolerance"]),
                )
                if prepared.logprob_reason:
                    raise ValueError("incomplete_fixed_anchor_logprob_evidence")
                if index == 1:
                    parsed_sides.append(prepared.first_logs)
                parsed_sides.append(prepared.second_logs)
            else:
                prepared = PreparedEvidence(
                    [ResponseEvidence(r["text"], None, None) for r in sides[0]],
                    [ResponseEvidence(r["text"], None, None) for r in sides[index]],
                    None,
                )
            grouping = group_evidence(evidence, prepared)
            if grouping.reason:
                raise ValueError(grouping.reason)
            partitions.append(sorted(tuple(a) for a, _ in grouping.pairs))
        if any(part != partitions[0] for part in partitions):
            raise ValueError("matched_cache_component_partitions_required")
        parts = partitions[0]
        if len(parts) < 2:
            raise ValueError("at_least_two_independent_components_per_prompt_required")
        if base != "FP3":
            groups = [[[side[i]["text"] for i in part] for part in parts] for side in sides]
            result.append(
                energy_cell(
                    [
                        [np.asarray([vectors[text] for text in row]) for row in side]
                        for side in groups
                    ]
                )
                if base == "FP4"
                else text_cell(
                    groups, method=base, length=samples[0].details["design"]["fp2_length"]
                )
            )
        else:
            anchors = samples[0].details["design"]["anchor_tokens"]
            arrays = []
            with np.errstate(over="ignore", invalid="ignore"):
                for values in parsed_sides:
                    raw = np.asarray([[row[k] for k in anchors] for row in values])
                    means = np.asarray([np.mean(raw[list(part)], axis=0) for part in parts])
                    if method == "FP3-dispersion":
                        moments = np.asarray(
                            [np.mean(raw[list(part)] ** 2, axis=0) for part in parts]
                        )
                        means = np.concatenate((means, moments), axis=1)
                    arrays.append(means)
            result.append(Cell(method, *arrays))
    return result
