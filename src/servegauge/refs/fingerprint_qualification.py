"""Pinned, opt-in text-equivalence qualification for one measured mock design.

This is an empirical scripted-process qualification, never a provider or model
identity certificate. Unsupported designs and stale implementation/assets refuse.
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from servegauge.core.hashing import content_hash
from servegauge.core.models import EndpointSpec
from servegauge.scoring.common import Samples
from servegauge.scoring.fingerprint_comparison import signature
from servegauge.verdicts.config import DecisionPolicy

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "configs/fingerprint_qualification.json"


def _path(name: str) -> Path:
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in name
        or any(p.casefold() == "starter files" for p in relative.parts)
    ):
        raise ValueError("qualification_canonical_asset_path_required")
    prefix = PurePosixPath("src/servegauge")
    if relative.is_relative_to(prefix):
        return Path(__file__).resolve().parents[1] / relative.relative_to(prefix)
    if not relative.parts or relative.parts[0] not in {"configs", "docs"}:
        raise ValueError("qualification_registered_asset_path_required")
    return ROOT / relative


def _pinned(name: str, digest: str) -> bytes:
    raw = _path(name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("qualification_asset_or_implementation_changed")
    return raw


def load_qualification(name: str) -> dict[str, Any]:
    try:
        registry = json.loads(REGISTRY.read_bytes())
        if registry["id"] != name:
            raise ValueError("qualification_unknown_profile")
        for path, digest in registry["implementation"].items():
            _pinned(path, digest)
        observed = {
            "python": platform.python_version(),
            "platform": platform.system(),
            "architecture": platform.machine(),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
        }
        if observed != registry["runtime"]:
            raise ValueError("qualification_numerical_runtime_changed")
        assets = {
            label: json.loads(_pinned(v["path"], v["sha256"]))
            for label, v in registry["assets"].items()
        }
        if any(
            assets[k]["accepted"] is not True
            for k in ["interval_acceptance", "null_acceptance", "wire_acceptance"]
        ):
            raise ValueError("qualification_accepted_studies_required")
        if any(v != 1 for v in assets["interval_acceptance"]["scales"].values()):
            raise ValueError("qualification_unapplied_interval_widening")
        envelope = assets["wire"]
        wire = envelope["value"]
        if (
            wire["accepted"] is not True
            or envelope["value_hash"] != content_hash(wire)
            or any(p["accepted"] is not True for p in wire["per_pack"])
        ):
            raise ValueError("qualification_intact_wire_study_required")
        for path, digest in assets["interval_acceptance"]["study_code_sha256"].items():
            _pinned(path, digest)
        for path, digest in wire["configuration"]["wire"]["null_source_files"].items():
            _pinned(path, digest)
        calibration = assets["calibration"]
        if (
            calibration["value_hash"] != content_hash(calibration["value"])
            or calibration["source_hash"] != wire["source_hash"]
            or calibration["configuration_hash"] != wire["configuration_hash"]
        ):
            raise ValueError("qualification_matched_calibration_required")
        sources = calibration["value"]["sources"]
        original = sources[0]
        sample_set = next(iter(original["fingerprints"].values()))
        policy = DecisionPolicy.model_validate(wire["configuration"]["policy"])
        requests = {
            method: {
                row["cell"]: row["sampling_request_hash"]
                for row in sample["details"]["observations"]
            }
            for method, sample in sample_set.items()
        }
        metrics = calibration["value"]["pack"]["fingerprints"]["views"]["*"]["metrics"]
        recipe = original["manifest"]["endpoint_metadata"][0]["metadata"]["mock_declaration"]
        return {
            "id": name,
            "binding": content_hash(registry),
            "family": sorted(sample_set),
            "n": {m: metrics[m if m != "FP3" else "FP3-mean"]["n"] for m in sample_set},
            "margins": {m: policy.margin(m) for m in ["FP1", "FP2"]},
            "policy_hash": content_hash(policy.model_dump(mode="json")),
            "profile": original["profile"],
            "model": original["manifest"]["endpoint_metadata"][0]["model"],
            "signatures": {m: signature(Samples.model_validate(s)) for m, s in sample_set.items()},
            "requests": requests,
            "sampling_version": recipe["sampling_version"],
            "healthy_logit_jitter": recipe["healthy_logit_jitter"],
            "reference_backend": recipe["backend"],
            "control_backends": [
                s["manifest"]["endpoint_metadata"][0]["metadata"]["mock_declaration"]["backend"]
                for s in sources[1:]
            ],
            "scope": "Text equivalence only for the pinned scripted mock process; "
            "no provider, model-weight or hardware certification",
        }
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        raise ValueError("qualification_registered_evidence_unavailable_or_invalid") from None


def _endpoint(
    endpoint: EndpointSpec | None, proof: dict[str, Any], *, backend: str | None = None
) -> None:
    if (
        endpoint is None
        or endpoint.kind != "mock"
        or endpoint.claimed_profile != proof["profile"]
        or endpoint.model != proof["model"]
        or endpoint.api_key_env is not None
    ):
        raise ValueError("qualification_mock_endpoint_required")
    url = urlsplit(endpoint.base_url)
    recipe = endpoint.metadata.get("mock_declaration", {})
    allowed = [proof["reference_backend"], *proof["control_backends"]]
    if (
        url.hostname not in {"localhost", "127.0.0.1", "::1"}
        or url.scheme != "http"
        or url.username
        or url.password
        or url.query
        or url.fragment
        or recipe.get("sampling_version") != proof["sampling_version"]
        or recipe.get("healthy_logit_jitter") != proof["healthy_logit_jitter"]
        or recipe.get("backend") not in allowed
        or type(recipe.get("seed")) is not int
        or (backend is not None and recipe.get("backend") != backend)
    ):
        raise ValueError("qualification_mock_recipe_mismatch")


def _sample(sample: Samples, proof: dict[str, Any]) -> None:
    method = sample.check_id
    details = sample.details
    sampling = details.get("sampling_design", {})
    if (
        sample.status is not None
        or method not in proof["family"]
        or signature(sample) != proof["signatures"][method]
        or details.get("policy_hash") != proof["policy_hash"]
    ):
        raise ValueError("qualification_capture_design_mismatch")
    if (
        type(sampling.get("session_size")) is not int
        or sampling["session_size"] != 1
        or sampling.get("cache_groups")
        or any(
            sampling.get(k) is not True
            for k in [
                "independent_sessions",
                "independent_reference_captures",
                "matched_sampling_design",
                "independent_prompt_strata",
            ]
        )
    ):
        raise ValueError("qualification_independent_iid_components_required")
    rows = details["observations"]
    size = details["design"]["sizes"][method.lower() + "_samples"]
    identities = {(row["cell"], row["draw"]) for row in rows}
    expected = {(cell, i) for cell in proof["requests"][method] for i in range(size)}
    if (
        len(identities) != len(rows)
        or identities != expected
        or len(set(sample.evidence)) != len(rows)
        or set(sample.evidence) != {row["exchange_id"] for row in rows}
    ):
        raise ValueError("qualification_complete_independent_identities_required")
    if len(rows) != proof["n"][method] or any(
        row.get("reason")
        or not isinstance(row.get("text"), str)
        or not row["text"]
        or row["sampling_request_hash"] != proof["requests"][method].get(row["cell"])
        for row in rows
    ):
        raise ValueError("qualification_complete_matched_requests_required")


def qualify_data(
    data: dict[str, Any],
    references: list[dict[str, Samples]],
    controls: list[dict[str, Samples]],
    declarations: list[EndpointSpec],
    *,
    tier: str,
    name: str,
) -> dict[str, Any]:
    proof = load_qualification(name)
    if (
        tier != "MOCK"
        or len(references) != 3
        or len(controls) != 2
        or len(declarations) != 5
        or set(data["views"]) != {"*"}
        or content_hash(data["policy"]) != proof["policy_hash"]
    ):
        raise ValueError("qualification_three_reference_two_control_mock_required")
    for i, endpoint in enumerate(declarations):
        _endpoint(
            endpoint,
            proof,
            backend=proof["reference_backend"] if i < 3 else proof["control_backends"][i - 3],
        )
    captures = {
        endpoint.id: endpoint.metadata["mock_declaration"]["seed"] for endpoint in declarations
    }
    if len(captures) != 5 or len(set(captures.values())) != 5:
        raise ValueError("qualification_independent_capture_seeds_required")
    for capture in [*references, *controls]:
        if sorted(capture) != proof["family"]:
            raise ValueError("qualification_exact_method_family_required")
        for sample in capture.values():
            _sample(sample, proof)
    original = data["views"]["*"]
    metrics = copy.deepcopy(original["metrics"])
    if set(metrics) != {"FP1", "FP2", "FP3-mean", "FP3-dispersion"}:
        raise ValueError("qualification_complete_method_metrics_required")
    for method in proof["margins"]:
        metric = metrics[method]
        if (
            metric["n"] != proof["n"][method]
            or not metric["bands_separated"]
            or metric["noise_band"] > proof["margins"][method]
        ):
            raise ValueError("qualification_eligible_reference_bands_required")
        metric.update(equivalence_qualified=True, limitations=proof["scope"])
    binding = {k: proof[k] for k in ["id", "binding", "family", "margins", "scope"]}
    binding["captures"] = captures
    return {**data, "qualification": binding, "views": {"*": {**original, "metrics": metrics}}}


def target_binding(
    data: dict[str, Any], sample: Samples, endpoint: EndpointSpec | None, policy: DecisionPolicy
) -> dict[str, Any]:
    binding = data["qualification"]
    proof = load_qualification(binding["id"])
    fields = ["id", "binding", "family", "margins", "scope"]
    if {k: binding[k] for k in fields} != {k: proof[k] for k in fields} or content_hash(
        policy.model_dump(mode="json")
    ) != proof["policy_hash"]:
        raise ValueError("qualification_pack_binding_mismatch")
    _endpoint(endpoint, proof)
    assert endpoint is not None
    captures = binding["captures"]
    expected = set(data["views"]["*"]["references"]) | {c["endpoint_id"] for c in data["controls"]}
    if (
        set(captures) != expected
        or len(set(captures.values())) != 5
        or endpoint.metadata["mock_declaration"]["seed"] in captures.values()
    ):
        raise ValueError("qualification_target_seed_must_be_independent")
    _sample(sample, proof)
    reference_ids = {
        identifier
        for source in data["views"]["*"]["references"].values()
        for identifier in source[sample.check_id]["evidence"]
    }
    if reference_ids.intersection(sample.evidence):
        raise ValueError("qualification_target_capture_must_be_independent")
    return dict(binding)
