"""Private immutable GPU study preparation; never launches or funds a campaign."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.redaction import Redactor
from servegauge.selfhost.study_design import StudyDesign, compile_matrix
from servegauge.selfhost.study_recipe import VARIANTS, study_template
from servegauge.storage.comparison import _guard, _private_json
from servegauge.storage.source_identity import capture_source


def _paths(*paths: Path) -> None:
    # Validate all caller paths before statting even the first path.
    for path in paths:
        if ".." in path.parts or any(p.casefold() in {"starter files", ".git"} for p in path.parts):
            raise ValueError("gpu_study_path_refused")
    for path in paths:
        _guard(path)


def _write(path: Path, value: Any) -> bytes:
    _paths(path)
    if Redactor.from_environment().redact(value) != value:
        raise ValueError("gpu_study_private_value_refused")
    blob = canonical_json(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    return blob


def prepare_study(
    root: Path, design: StudyDesign, *, out: Path, backend: str = "ssh"
) -> dict[str, Any]:
    _paths(root, out)
    private = root.absolute() / ".cache/gpu-studies"
    if not out.absolute().is_relative_to(private) or out.absolute() == private or out.exists():
        raise ValueError("gpu_study_fresh_private_directory_required")
    matrix = compile_matrix(design)
    source = capture_source()
    payloads: dict[str, Any] = {
        "design.json": design.model_dump(mode="json"),
        "matrix.json": matrix,
        "source.json": source,
        **{f"owner-{variant}.json": study_template(backend, variant) for variant in VARIANTS},
    }
    out.mkdir(parents=True, mode=0o700)
    inventory = {
        name: hashlib.sha256(_write(out / name, value)).hexdigest()
        for name, value in payloads.items()
    }
    manifest = {
        "version": "gpu-study-kit-v1",
        "state": "prepared_only",
        "study_id": design.study_id,
        "backend": backend,
        "design_hash": content_hash(design.model_dump(mode="json")),
        "source_tree_hash": source["source_tree_hash"],
        "files_sha256": inventory,
        "planned_sessions": len(matrix),
        "planned_units": len(matrix) // 4,
        "campaign_cost_usd": None,
        "funding_reviewed": False,
        "gpu_sessions_started": 0,
        "model_api_calls": 0,
        "scope": "configured_interventions_not_precision_attestation",
        "independence": "owner_declared" if design.independence_reviewed else "unreviewed",
        "seed_scope": "inventory_identity_not_server_rng_or_independence_proof",
    }
    _write(out / "kit.json", manifest)
    return manifest


def verify_study(kit: Path, *, kit_hash: str) -> dict[str, Any]:
    _paths(kit)
    for child in kit.iterdir():
        _paths(child)
        if not child.is_file():
            raise ValueError("gpu_study_flat_private_inventory_required")
    manifest = _private_json((kit / "kit.json").read_bytes())
    if not isinstance(manifest, dict) or content_hash(manifest) != kit_hash:
        raise ValueError("gpu_study_kit_hash_changed")
    if manifest.get("version") == "gpu-study-kit-v2":
        from servegauge.storage.gpu_collection import verify_collection_study

        return verify_collection_study(kit, kit_hash=kit_hash)
    design = StudyDesign.model_validate_json((kit / "design.json").read_bytes())
    matrix = compile_matrix(design)
    if set(p.name for p in kit.iterdir()) != {"kit.json", *manifest["files_sha256"]}:
        raise ValueError("gpu_study_kit_inventory_changed")
    for name, digest in manifest["files_sha256"].items():
        if Path(name).name != name or name == "kit.json":
            raise ValueError("gpu_study_flat_file_required")
        path = kit / name
        _paths(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("gpu_study_kit_file_changed")
    source = _private_json((kit / "source.json").read_bytes())
    if (
        manifest["version"] != "gpu-study-kit-v1"
        or manifest["state"] != "prepared_only"
        or manifest["design_hash"] != content_hash(design.model_dump(mode="json"))
        or manifest["study_id"] != design.study_id
        or manifest["source_tree_hash"] != source["source_tree_hash"]
        or json.loads((kit / "matrix.json").read_bytes()) != matrix
        or manifest["planned_sessions"] != len(matrix)
        or manifest["planned_units"] != len(matrix) // 4
        or manifest["campaign_cost_usd"] is not None
        or manifest["funding_reviewed"] is not False
        or manifest["gpu_sessions_started"] != 0
        or manifest["model_api_calls"] != 0
    ):
        raise ValueError("gpu_study_derived_contract_changed")
    for variant in VARIANTS:
        if _private_json((kit / f"owner-{variant}.json").read_bytes()) != study_template(
            manifest["backend"], variant
        ):
            raise ValueError("gpu_study_owner_template_changed")
    return dict(manifest)
