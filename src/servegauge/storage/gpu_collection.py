"""Versioned complete owner study kit, pinned workspace assets and private inventory."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from servegauge.config import load_budget
from servegauge.core.hashing import content_hash
from servegauge.embeddings.local import LocalEmbeddingSpec
from servegauge.profiles import load_profile
from servegauge.selfhost.remote_recipe import STUDY_COLLECTION, STUDY_PROFILE
from servegauge.selfhost.study_collection import study_fingerprint_items
from servegauge.selfhost.study_protocol import (
    CollectionSpec,
    collection_matrix,
    collection_owner_template,
)
from servegauge.selfhost.study_recipe import VARIANTS
from servegauge.storage.comparison import _private_json
from servegauge.storage.gpu_study import _paths, _write
from servegauge.storage.source_identity import capture_source
from servegauge.suites.config import suite_config
from servegauge.suites.s0_capabilities import capability_items
from servegauge.verdicts.config import load_policy


def collection_assets(root: Path, spec: CollectionSpec) -> dict[str, Any]:
    _paths(root)
    if root.absolute() != Path.cwd().absolute():
        raise ValueError("study_current_workspace_required")
    names = [
        "configs/budgets.yaml",
        "configs/margins.yaml",
        "configs/suites.yaml",
        "configs/fingerprints.yaml",
        "configs/fingerprints_embed.yaml",
        "profiles/qwen3-0.6b-study.yaml",
    ]
    _paths(*(root / name for name in names))
    embed = yaml.safe_load((root / "configs/fingerprints_embed.yaml").read_bytes())
    relative = Path(embed["model_manifest"])
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
        raise ValueError("study_workspace_embedding_manifest_required")
    path = root / relative
    _paths(path)
    if not path.absolute().is_relative_to(root.absolute() / "configs"):
        raise ValueError("study_workspace_embedding_manifest_required")
    names.append(relative.as_posix())
    embedding = (
        LocalEmbeddingSpec.model_validate(spec.fingerprints["embedding_model"])
        if "embedding_model" in spec.fingerprints
        else LocalEmbeddingSpec.model_validate_json(path.read_bytes())
    )
    overrides = {
        **spec.fingerprints,
        "collection_recipe": STUDY_COLLECTION,
        "embedding_model": embedding.model_dump(mode="json"),
    }
    items = study_fingerprint_items(
        spec.design.level, spec.design.seed, overrides, profile_id=STUDY_PROFILE
    )
    return {
        "version": "gpu-study-assets-v1",
        "profile": load_profile(STUDY_PROFILE, root / "profiles").model_dump(mode="json"),
        "policy": load_policy(root / "configs/margins.yaml").model_dump(mode="json"),
        "budget_policy": load_budget(root).model_dump(mode="json"),
        "suite_config": suite_config(),
        "overrides": overrides,
        "embedding_model": embedding.model_dump(mode="json"),
        "fingerprint_design": items[0].expected["design"],
        "fingerprint_requests_per_capture": sum(item.expected["samples"] for item in items),
        "capability_requests_per_capture": sum(
            item.expected.get("samples", 1) for item in capability_items()
        ),
        "source_files_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
        },
    }


def _manifest(
    spec: CollectionSpec,
    matrix: list[dict[str, Any]],
    assets: dict[str, Any],
    source: dict[str, Any],
    inventory: dict[str, str],
) -> dict[str, Any]:
    return {
        "version": "gpu-study-kit-v2",
        "state": "prepared_only",
        "study_id": spec.design.study_id,
        "backend": spec.backend,
        "scope": spec.scope,
        "design_hash": content_hash(spec.design.model_dump(mode="json")),
        "collection_hash": content_hash(spec.model_dump(mode="json")),
        "assets_hash": content_hash(assets),
        "source_tree_hash": source["source_tree_hash"],
        "files_sha256": inventory,
        "planned_sessions": len(matrix),
        "planned_units": len(matrix) // 4,
        "planned_fingerprint_requests": assets["fingerprint_requests_per_capture"] * len(matrix),
        "planned_capability_requests": assets["capability_requests_per_capture"] * len(matrix),
        "campaign_cost_usd": None,
        "funding_reviewed": False,
        "gpu_sessions_started": 0,
        "model_api_calls": 0,
        "physical_precision_verified": False,
        "seed_scope": "inventory_identity_not_server_rng_or_independence_proof",
    }


def prepare_collection_study(root: Path, spec: CollectionSpec, *, out: Path) -> dict[str, Any]:
    _paths(root, out)
    private = root.absolute() / ".cache/gpu-studies"
    if not out.absolute().is_relative_to(private) or out.absolute() == private or out.exists():
        raise ValueError("gpu_study_fresh_private_directory_required")
    matrix, assets, source = (
        collection_matrix(spec),
        collection_assets(root, spec),
        capture_source(),
    )
    payloads = {
        "design.json": spec.design.model_dump(mode="json"),
        "collection.json": spec.model_dump(mode="json"),
        "matrix.json": matrix,
        "source.json": source,
        "assets.json": assets,
        **{
            f"owner-{variant}.json": collection_owner_template(spec, variant)
            for variant in VARIANTS
        },
    }
    out.mkdir(parents=True, mode=0o700)
    inventory = {
        name: hashlib.sha256(_write(out / name, value)).hexdigest()
        for name, value in payloads.items()
    }
    value = _manifest(spec, matrix, assets, source, inventory)
    _write(out / "kit.json", value)
    return value


def verify_collection_study(
    kit: Path, *, kit_hash: str, root: Path | None = None
) -> dict[str, Any]:
    _paths(*([kit] if root is None else [root, kit]))
    expected = {
        "design.json",
        "collection.json",
        "matrix.json",
        "source.json",
        "assets.json",
        *(f"owner-{variant}.json" for variant in VARIANTS),
    }
    paths = list(kit.iterdir())
    _paths(*paths)
    if {path.name for path in paths} != expected | {"kit.json"} or any(
        not p.is_file() for p in paths
    ):
        raise ValueError("study_complete_flat_inventory_required")
    manifest = _private_json((kit / "kit.json").read_bytes())
    if not isinstance(manifest, dict) or content_hash(manifest) != kit_hash:
        raise ValueError("study_collection_kit_hash_changed")
    spec = CollectionSpec.model_validate(_private_json((kit / "collection.json").read_bytes()))
    assets = _private_json((kit / "assets.json").read_bytes())
    source = _private_json((kit / "source.json").read_bytes())
    matrix = collection_matrix(spec)
    inventory = {name: hashlib.sha256((kit / name).read_bytes()).hexdigest() for name in expected}
    if manifest.get("files_sha256") != inventory:
        raise ValueError("study_collection_file_changed")
    if not isinstance(assets, dict) or any(
        type(assets.get(name)) is not int or assets[name] < 1
        for name in ("fingerprint_requests_per_capture", "capability_requests_per_capture")
    ):
        raise ValueError("study_collection_asset_schema_changed")
    try:
        wanted = _manifest(spec, matrix, assets, source, inventory)
    except (TypeError, KeyError):
        raise ValueError("study_collection_asset_schema_changed") from None
    if (
        manifest != wanted
        or _private_json((kit / "matrix.json").read_bytes()) != matrix
        or _private_json((kit / "design.json").read_bytes()) != spec.design.model_dump(mode="json")
    ):
        raise ValueError("study_collection_derived_contract_changed")
    if source["source_identity"]["source_hash"] != source["source_tree_hash"]:
        raise ValueError("study_collection_source_identity_changed")
    for variant in VARIANTS:
        if _private_json((kit / f"owner-{variant}.json").read_bytes()) != collection_owner_template(
            spec, variant
        ):
            raise ValueError("study_collection_template_changed")
    if root is not None and (
        collection_assets(root, spec) != assets
        or capture_source()["source_tree_hash"] != manifest["source_tree_hash"]
    ):
        raise ValueError("study_collection_workspace_or_source_changed")
    return manifest


def collection_inputs(
    root: Path, kit: Path, kit_hash: str
) -> tuple[CollectionSpec, dict[str, Any], list[dict[str, Any]]]:
    verify_collection_study(kit, kit_hash=kit_hash, root=root)
    spec = CollectionSpec.model_validate(_private_json((kit / "collection.json").read_bytes()))
    return spec, _private_json((kit / "assets.json").read_bytes()), collection_matrix(spec)
