"""Private immutable kit inventory and lexical path refusal before filesystem reads."""

import json
from pathlib import Path

import pytest

from servegauge.core.hashing import content_hash
from servegauge.selfhost.study_design import StudyDesign
from servegauge.storage.gpu_study import prepare_study, verify_study


def test_prepare_complete_but_unfunded_matrix_and_bound_inventory(tmp_path):
    design = StudyDesign(version="gpu-study-v1", study_id="pilot", seed=3, replicates=2)
    out = tmp_path / ".cache/gpu-studies/pilot"
    value = prepare_study(tmp_path, design, out=out)
    assert value["state"] == "prepared_only"
    assert value["planned_sessions"] == 80 and value["planned_units"] == 20
    assert value["gpu_sessions_started"] == value["model_api_calls"] == 0
    assert value["campaign_cost_usd"] is None
    assert not value["funding_reviewed"]
    assert value == verify_study(out, kit_hash=content_hash(value))
    assert len(json.loads((out / "matrix.json").read_bytes())) == 80
    for name in ["bf16", "fp8", "mxfp4", "other-engine", "wrong-template"]:
        config = json.loads((out / f"owner-{name}.json").read_bytes())
        assert config["image"] is None and config["hourly_usd"] is None
        assert config["recipe_reviewed"] is False
    with pytest.raises(ValueError):
        prepare_study(tmp_path, design, out=out)
    (out / "matrix.json").write_text("[]")
    with pytest.raises(ValueError):
        verify_study(out, kit_hash=content_hash(value))


def test_source_pin_and_private_tree_refuse_symlinks_foreign_files_and_bad_hash(tmp_path):
    design = StudyDesign(version="gpu-study-v1", study_id="pilot", seed=3, replicates=2)
    out = tmp_path / ".cache/gpu-studies/pilot"
    value = prepare_study(tmp_path, design, out=out)
    with pytest.raises(ValueError):
        verify_study(out, kit_hash="0" * 64)
    (out / "extra").write_text("foreign")
    with pytest.raises(ValueError):
        verify_study(out, kit_hash=content_hash(value))
    (out / "extra").unlink()
    blob = (out / "matrix.json").read_bytes()
    (out / "matrix.json").unlink()
    external = tmp_path / "external"
    external.write_bytes(blob)
    (out / "matrix.json").symlink_to(external)
    with pytest.raises(ValueError):
        verify_study(out, kit_hash=content_hash(value))


def test_protected_path_refuses_before_any_stat(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("protected supplied paths must refuse lexically")

    monkeypatch.setattr(Path, "is_symlink", forbidden)
    design = StudyDesign(version="gpu-study-v1", study_id="pilot", seed=3, replicates=2)
    with pytest.raises(ValueError):
        prepare_study(tmp_path, design, out=tmp_path / "Starter Files/private")
    with pytest.raises(ValueError):
        verify_study(tmp_path / "Starter Files/private", kit_hash="0" * 64)
