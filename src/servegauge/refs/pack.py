"""Matched MOCK, trusted BASELINE and R3 packs with captured fingerprint evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.ids import safe_filename
from servegauge.core.models import ReferencePack, ValueModel
from servegauge.core.redaction import Redactor
from servegauge.refs.calibrate import calibrate
from servegauge.refs.fingerprint import build_fingerprints, captured_controls
from servegauge.scoring.common import Samples
from servegauge.scoring.toolcall_scoring import f1_reference_statistic
from servegauge.verdicts.config import DecisionPolicy


class ReferenceEndpoint(ValueModel):
    id: str
    profile: str
    kind: str
    seed: int | None = None
    clean: bool = False


class ReferenceRun(ValueModel):
    run_id: str
    profile: str
    configuration: dict[str, Any]
    item_set_hashes: dict[str, str]
    endpoints: list[ReferenceEndpoint]
    # check -> endpoint -> item -> repeats; item is the independent sampling unit.
    checks: dict[str, dict[str, dict[str, list[float]]]]
    inputs: dict[str, dict[str, dict[str, list[str]]]] = Field(default_factory=dict)
    baseline_endpoint_id: str | None = None
    manifest: dict[str, Any] = Field(default_factory=dict)
    fingerprints: dict[str, dict[str, Samples]] = Field(default_factory=dict)
    fingerprint_policy: DecisionPolicy | None = None


def build_pack(
    source: ReferenceRun,
    tier: Literal["MOCK", "BASELINE", "R3"],
    *,
    seed: int,
    resamples: int = 2000,
    controls: list[ReferenceRun] | None = None,
    reference_runs: list[ReferenceRun] | None = None,
    fingerprint_qualification: str | None = None,
) -> ReferencePack:
    if tier not in {"MOCK", "BASELINE", "R3"}:
        raise ValueError("tier_requires_qualified_owner_reference")
    endpoints = {endpoint.id: endpoint for endpoint in source.endpoints}
    if len(endpoints) != len(source.endpoints) or any(
        e.profile != source.profile for e in endpoints.values()
    ):
        raise ValueError("unique_same_profile_endpoints_required")
    if not source.checks and not source.fingerprints:
        raise ValueError("reference_requires_scored_outcomes")
    if fingerprint_qualification is not None:
        if tier != "MOCK" or not source.fingerprints:
            raise ValueError("fingerprint_qualification_requires_captured_MOCK")
        from servegauge.refs.fingerprint_qualification import load_qualification

        load_qualification(fingerprint_qualification)
    if tier == "MOCK":
        selected = [e.id for e in endpoints.values() if e.kind == "mock" and e.clean]
        seeds = [endpoints[e].seed for e in selected]
        if len(selected) < 3 or None in seeds or len(set(seeds)) != len(selected):
            raise ValueError("MOCK_requires_three_clean_instances_with_different_seeds")
        views = {"*": selected}
    elif tier == "BASELINE":
        if source.baseline_endpoint_id not in endpoints:
            raise ValueError("trusted_baseline_endpoint_required")
        views = {"*": [str(source.baseline_endpoint_id)]}
    else:
        if len(endpoints) < 4:
            raise ValueError("R3_requires_at_least_three_OTHER_endpoints")
        views = {target: [peer for peer in endpoints if peer != target] for target in endpoints}
    outcomes: dict[str, Any] = {}
    calibration: dict[str, Any] = {}
    unavailable: dict[str, dict[str, str]] = {}
    for target, peers in views.items():
        outcomes[target], calibration[target] = {}, {}
        unavailable[target] = {}
        for check, by_endpoint in source.checks.items():
            eligible = [peer for peer in peers if by_endpoint.get(peer)]
            if tier == "R3" and len(eligible) < 3:
                continue
            if tier != "R3" and len(eligible) != len(peers):
                raise ValueError("reference_peer_outcomes_missing")
            item_sets = [set(by_endpoint[peer]) for peer in eligible]
            if any(items != item_sets[0] for items in item_sets):
                if check == "T02":
                    unavailable[target][check] = (
                        "conditional observed-call supports differ; "
                        "matched-n recalibration required"
                    )
                    continue
                raise ValueError("reference_items_must_match_exactly")
            item_ids = sorted(item_sets[0])
            shared_inputs = None
            if check.startswith(("B", "S8.health.")):
                layouts = [source.inputs.get(check, {}).get(peer, {}) for peer in eligible]
                if any(set(layout) != set(item_ids) for layout in layouts) or any(
                    layout != layouts[0] for layout in layouts
                ):
                    continue
                shared_inputs = layouts[0]
            rows = []
            for peer in eligible:
                row = []
                for item in item_ids:
                    values = by_endpoint[peer][item]
                    if not values or not np.all(np.isfinite(values)):
                        raise ValueError("finite_nonempty_reference_outcomes_required")
                    row.append(float(np.mean(values)))
                rows.append(row)
            matrix = np.asarray(rows)
            pooled = np.median(matrix, axis=0) if tier == "R3" else np.mean(matrix, axis=0)
            outcomes[target][check] = {
                "item_ids": item_ids,
                "values": pooled.tolist(),
                "peer_ids": eligible,
                "peer_item_means": rows,
                "median_absolute_deviation": np.median(
                    np.abs(matrix - np.median(matrix, axis=0)), axis=0
                ).tolist(),
            }
            named_seed = int(content_hash([seed, target, check])[:16], 16)
            if shared_inputs is not None:
                outcomes[target][check]["input_hashes"] = shared_inputs
            if check == "T01":
                if any(len(by_endpoint[peer][item]) < 2 for peer in eligible for item in item_ids):
                    outcomes[target].pop(check)
                    continue
                draws = np.asarray(
                    [
                        [[by_endpoint[peer][item][index] for index in (0, 1)] for item in item_ids]
                        for peer in eligible
                    ]
                )
                pooled_draws = np.median(draws, axis=0)
                if not np.all(np.isin(pooled_draws, [0, 1])):
                    # An even consensus tie does not define a binary trigger label.
                    outcomes[target].pop(check)
                    continue
                anchor, repeat = pooled_draws[:, 0].tolist(), pooled_draws[:, 1].tolist()
                outcomes[target][check].update(
                    f1_anchor=anchor,
                    f1_repeat=repeat,
                    peer_item_repeats=draws.tolist(),
                    metric="tool_call_f1",
                )
                calibration[target][check] = calibrate(
                    pooled_draws.tolist(),
                    seed=named_seed,
                    resamples=resamples,
                    statistic=f1_reference_statistic,
                )
            else:
                calibration[target][check] = calibrate(
                    pooled.tolist(), seed=named_seed, resamples=resamples
                )
                if check == "R01":
                    repeats = [by_endpoint[peer][item] for peer in eligible for item in item_ids]
                    widths = {len(row) for row in repeats}
                    if len(widths) != 1:
                        outcomes[target].pop(check)
                        calibration[target].pop(check)
                        continue
                    draws = np.asarray(
                        [[by_endpoint[peer][item] for item in item_ids] for peer in eligible]
                    )
                    combined = np.median(draws, axis=0) if tier == "R3" else np.mean(draws, axis=0)
                    if not np.all(np.isin(combined, [0, 1])):
                        outcomes[target].pop(check)
                        calibration[target].pop(check)
                        continue
                    calibration[target][check]["binary_repeats"] = combined.tolist()
    if reference_runs and not source.fingerprints:
        raise ValueError("additional_reference_runs_require_fingerprint_captures")
    effective_controls = controls if controls is not None else captured_controls(source, tier)
    fingerprints = (
        build_fingerprints(
            source,
            views,
            effective_controls,
            tier=tier,
            seed=seed,
            resamples=resamples,
            reference_runs=reference_runs,
            qualification=fingerprint_qualification,
        )
        if source.fingerprints
        else {}
    )
    # Preserve existing minimal packs byte-for-byte, including their identity.
    source_value = source.model_dump(mode="json")
    if not source.fingerprints:
        source_value.pop("fingerprints")
        source_value.pop("fingerprint_policy")
    method_version = "matched-components-v4" if fingerprints else "matched-input-v3"
    identity: dict[str, Any] = {
        "source": source_value,
        "tier": tier,
        "seed": seed,
        "resamples": resamples,
        "method_version": method_version,
    }
    if fingerprints:
        if fingerprints.get("qualification"):
            identity["fingerprint_qualification"] = fingerprints["qualification"]
        embedding_runtimes = {
            target: view["metrics"]["FP4"]["embedding_identity"]
            for target, view in fingerprints["views"].items()
            if "FP4" in view["metrics"]
        }
        if embedding_runtimes:
            identity["embedding_runtimes"] = embedding_runtimes
        identity["fingerprint_controls"] = [c.model_dump(mode="json") for c in effective_controls]
        if reference_runs:
            identity["fingerprint_reference_runs"] = [
                r.model_dump(mode="json") for r in reference_runs
            ]
    pack = ReferencePack(
        pack_id="ref-" + content_hash(identity)[:20],
        profile=source.profile,
        tier=tier,
        manifest={
            "source_run": source.run_id,
            "source_manifest": source.manifest,
            "configuration": source.configuration,
            "configuration_hash": content_hash(source.configuration),
            "item_set_hashes": source.item_set_hashes,
            "endpoint_declarations": [e.model_dump() for e in source.endpoints],
            "n_per_check": {
                t: {c: v["n"] for c, v in checks.items()} for t, checks in calibration.items()
            },
            "method_version": method_version,
            "unavailable_views": unavailable,
            "fingerprints": fingerprints.get("version", "not implemented in fast track"),
            "qualification": "MOCK declarations, owner-trusted BASELINE, or consensus; "
            "no hardware attestation",
        },
        outcomes=outcomes,
        calibration=calibration,
        fingerprints=fingerprints,
    )
    pack.manifest["pack_content_hash"] = _pack_digest(pack)
    return pack


def _pack_digest(pack: ReferencePack) -> str:
    value = pack.model_dump(mode="json")
    value["manifest"].pop("pack_content_hash", None)
    return content_hash(value)


def save_pack(pack: ReferencePack, root: Path = Path("refs")) -> Path:
    path = root / safe_filename(pack.profile) / safe_filename(pack.pack_id) / "pack.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = Redactor.from_environment().redact(pack.model_dump(mode="json"))
    encoded = canonical_json(clean) + b"\n"
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError("reference_pack_is_immutable")
    path.write_bytes(encoded)
    return path


def load_pack(identifier: str, root: Path = Path("refs")) -> ReferencePack:
    path = Path(identifier)
    candidates = [path] if path.is_file() else list(root.rglob("pack.json"))
    matches = []
    for candidate in candidates:
        pack = ReferencePack.model_validate(json.loads(candidate.read_text(encoding="utf-8")))
        if pack.manifest.get("pack_content_hash") != _pack_digest(pack):
            raise ValueError("reference_pack_content_mismatch")
        if candidate == path or pack.pack_id == identifier:
            matches.append(pack)
    if len(matches) != 1:
        raise ValueError("unique_reference_pack_required")
    return matches[0]


def reference_for(
    pack: ReferencePack,
    *,
    target: str,
    check: str,
    item_ids: list[str],
    configuration: dict[str, Any],
    input_hashes: dict[str, list[str]] | None = None,
) -> tuple[list[float], dict[str, Any]]:
    def comparison_config(value: dict[str, Any]) -> dict[str, Any]:
        if "checks" not in value:
            return value
        if check not in value["checks"]:
            raise ValueError("reference_check_configuration_missing")
        return {
            **{k: v for k, v in value.items() if k not in {"checks", "item_set_hash"}},
            "check": value["checks"][check],
        }

    if content_hash(comparison_config(configuration)) != content_hash(
        comparison_config(pack.manifest["configuration"])
    ):
        raise ValueError("reference_configuration_mismatch")
    key = target if pack.tier == "R3" else "*"
    if key not in pack.outcomes or check not in pack.outcomes[key]:
        raise ValueError("reference_view_missing")
    view, band = pack.outcomes[key][check], pack.calibration[key][check]
    if view["item_ids"] != item_ids or band["n"] != len(item_ids):
        raise ValueError("reference_requires_matched_items_and_n; recalibrate")
    if pack.tier == "R3" and target in view["peer_ids"]:
        raise ValueError("R3_target_leakage")
    if check.startswith(("B", "S8.health.")) and (
        input_hashes is None or input_hashes != view.get("input_hashes")
    ):
        raise ValueError(
            "accounting_reference_requires_identical_input_layouts"
            if check.startswith("B")
            else "operational_reference_requires_identical_input_layouts"
        )
    details = dict(band)
    if check == "T01":
        details.update(f1_anchor=view["f1_anchor"], f1_repeat=view["f1_repeat"])
    return list(view["values"]), details
