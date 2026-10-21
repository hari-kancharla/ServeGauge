"""The complete study cannot hide omitted methods, changed assets or reused slots."""

import json
from pathlib import Path

import pytest

from servegauge.core.hashing import content_hash
from servegauge.selfhost.study_protocol import CollectionSpec, collection_matrix
from servegauge.storage.gpu_collection import prepare_collection_study, verify_collection_study
from servegauge.storage.workspace import initialize


def specification(**changes):
    return CollectionSpec.model_validate(
        {
            "version": "gpu-study-collection-v1",
            "design": {"version": "gpu-study-v1", "study_id": "qa", "seed": 17, "replicates": 2},
            "requests_per_minute": 60000.0,
            "scope": "mock_software_qa",
            "backend": "ssh",
            "fingerprints": {
                "methods": ["FP1", "FP2", "FP3", "FP4"],
                "fp1_cells": 2,
                "fp1_samples": 4,
                "fp2_cells": 2,
                "fp2_samples": 4,
                "fp3_samples": 4,
                "fp4_cells": 2,
                "fp4_samples": 4,
            },
            **changes,
        }
    )


def test_complete_matrix_is_exclusive_and_namespaced_by_collection_design():
    first = specification()
    rows = collection_matrix(first)
    assert len(rows) == 80
    assert all(
        len({row[key] for row in rows}) == 80
        for key in ("slot_id", "run_id", "session_id", "capture_seed")
    )
    second = specification(scope="owner_gpu")
    assert not {r["run_id"] for r in rows} & {r["run_id"] for r in collection_matrix(second)}
    changed = first.model_copy(deep=True)
    changed.fingerprints["fp1_samples"] = 5
    assert not {r["session_id"] for r in rows} & {
        r["session_id"] for r in collection_matrix(changed)
    }
    assert first.sampling_declaration()["independent_sessions"] is False
    assert first.sampling_declaration()["independent_reference_captures"] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"scope": "gpu"},
        {"backend": "unreviewed"},
        {"session_size": True},
        {"session_size": 0},
        {"iid_condition_quartets_reviewed": True},
        {"within_capture_components_reviewed": 1},
        {"extra": "ignored"},
        {"fingerprints": {"methods": ["FP1", "FP2", "FP3"]}},
    ],
)
def test_incomplete_or_ambiguous_complete_design_refuses(changes):
    with pytest.raises(ValueError):
        specification(**changes)


def test_v2_preparation_pins_all_assets_and_still_never_launches(tmp_path, monkeypatch):
    initialize(tmp_path)
    monkeypatch.chdir(tmp_path)
    spec = specification()
    out = tmp_path / ".cache/gpu-studies/full"
    value = prepare_collection_study(tmp_path, spec, out=out)
    assert value["version"] == "gpu-study-kit-v2" and value["state"] == "prepared_only"
    assert value["model_api_calls"] == value["gpu_sessions_started"] == 0
    assert value["campaign_cost_usd"] is None and not value["funding_reviewed"]
    assert value["planned_sessions"] == 80 and value["planned_units"] == 20
    actual = verify_collection_study(out, kit_hash=content_hash(value))
    assert actual == value
    assets = json.loads((out / "assets.json").read_bytes())
    assert assets["profile"]["id"] == "qwen3-0.6b-study"
    assert assets["embedding_model"]["revision"]
    assert assets["fingerprint_design"]["sizes"]["methods"] == ["FP1", "FP2", "FP3", "FP4"]
    assert all(
        assets["fingerprint_design"]["sizes"][name] == 4
        for name in ("fp1_samples", "fp2_samples", "fp3_samples", "fp4_samples")
    )
    with pytest.raises(ValueError):
        prepare_collection_study(tmp_path, spec, out=out)
    (out / "assets.json").write_text("{}")
    with pytest.raises(ValueError):
        verify_collection_study(out, kit_hash=content_hash(value))


def test_current_assets_must_match_reviewed_kit_and_paths_refuse_before_stat(tmp_path, monkeypatch):
    initialize(tmp_path)
    monkeypatch.chdir(tmp_path)
    spec = specification()
    out = tmp_path / ".cache/gpu-studies/full"
    value = prepare_collection_study(tmp_path, spec, out=out)
    assert verify_collection_study(out, kit_hash=content_hash(value), root=tmp_path) == value
    path = tmp_path / "profiles/qwen3-0.6b-study.yaml"
    path.write_text(path.read_text().replace('"qwen3"', '"changed-family"'))
    with pytest.raises(ValueError):
        verify_collection_study(out, kit_hash=content_hash(value), root=tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("protected supplied paths must refuse lexically")

    monkeypatch.setattr(Path, "is_symlink", forbidden)
    with pytest.raises(ValueError):
        prepare_collection_study(tmp_path, spec, out=tmp_path / "Starter Files/x")
    with pytest.raises(ValueError):
        verify_collection_study(tmp_path / "Starter Files/x", kit_hash="0" * 64)


@pytest.mark.parametrize(
    "changes",
    [
        {"engine_images": {"0.32.0": None}},
        {"engine_images": {"0.31.0": "vllm/vllm-openai:latest", "0.32.0": None}},
        {
            "engine_images": {
                version: "vllm/vllm-openai@sha256:" + "a" * 64 for version in ("0.31.0", "0.32.0")
            }
        },
        {"modal_image_ids": {version: "im-same" for version in ("0.31.0", "0.32.0")}},
        {"requests_per_minute": True},
        {"concurrency_per_endpoint": True},
    ],
)
def test_exact_distinct_builds_and_sampling_traffic_are_part_of_the_design(changes):
    with pytest.raises(ValueError):
        specification(**changes)
