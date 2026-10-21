"""Sealed owner kit keeps full source closure and separately reviewed controls."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from comparison_workflow import prepare_comparison_fixture

from servegauge.core.hashing import content_hash
from servegauge.storage.comparison_controls import ControlSelection
from servegauge.storage.comparison_kit import prepare_kit, verify_kit


@pytest.fixture
async def kit_inputs(tmp_path):
    base = tmp_path / "fixture"
    snapshot, failures, _ = await prepare_comparison_fixture(base)
    controls = ControlSelection.model_validate_json((base / "control-selection.json").read_text())
    return (
        base / "source",
        snapshot,
        failures.selection_hash,
        controls,
        base / "source/.cache/comparisons/kit",
    )


def prepare(inputs):
    root, snapshot, anchor, controls, out = inputs
    return prepare_kit(
        root,
        snapshot,
        selection_hash=anchor,
        controls=controls,
        controls_hash=content_hash(controls.model_dump(mode="json")),
        out=out,
    )


async def test_preparation_and_independent_anchor_verification_never_execute_either_procedure(
    kit_inputs, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("preparation cannot execute")

    monkeypatch.setattr("servegauge.comparison.case_provider.run_raw_case", forbidden)
    monkeypatch.setattr("servegauge.runner.check.run_check", forbidden)
    manifest = prepare(kit_inputs)
    out = kit_inputs[-1]
    assert manifest["state"] == "prepared_only" and manifest["owner_comparison_executed"] is False
    assert manifest["setup_seconds"] is None
    assert manifest["synthetic_only"] is True and manifest["m15_eligible"] is False
    assert {case["original_check"]["verdict"] for case in manifest["cases"]} == {
        "FAIL",
        "INCONCLUSIVE",
    }
    assert {case["population_n"] for case in manifest["cases"]} == {4}
    assert len(manifest["cases"]) == 4
    assert verify_kit(out, kit_hash=content_hash(manifest)) == manifest
    assert all(name.startswith(("snapshot/", "procedures/")) for name in manifest["files_sha256"])
    if os.name != "nt":
        assert out.stat().st_mode & 0o077 == 0
        assert all((out / name).stat().st_mode & 0o077 == 0 for name in manifest["files_sha256"])


@pytest.mark.parametrize(
    "mutation", ["wrong_anchor", "payload", "manifest", "extra", "symlink", "control_selection"]
)
async def test_modified_kit_or_self_resealed_manifest_cannot_override_the_external_anchor(
    kit_inputs, mutation
):
    manifest = prepare(kit_inputs)
    anchor = content_hash(manifest)
    out = kit_inputs[-1]
    if mutation == "wrong_anchor":
        anchor = "0" * 64
    elif mutation == "manifest":
        value = json.loads((out / "manifest.json").read_text())
        value["setup_seconds"] = 0
        (out / "manifest.json").write_text(json.dumps(value))
    elif mutation == "extra":
        (out / "unexpected").write_text("refuse")
    elif mutation == "symlink":
        (out / "escape").symlink_to(out.parent / "missing")
    else:
        name = (
            "procedures/program.json"
            if mutation == "payload"
            else "procedures/controls-selection.json"
        )
        path = out / name
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        verify_kit(out, kit_hash=anchor)


async def test_foreign_destination_and_outside_private_namespace_refuse(kit_inputs, tmp_path):
    root, snapshot, anchor, controls, out = kit_inputs
    for invalid in [tmp_path / "public-output", out]:
        if invalid == out:
            out.mkdir()
            (out / "foreign").write_text("keep")
        with pytest.raises(ValueError):
            prepare_kit(
                root,
                snapshot,
                selection_hash=anchor,
                controls=controls,
                controls_hash=content_hash(controls.model_dump(mode="json")),
                out=invalid,
            )
    assert (out / "foreign").read_text() == "keep"


def test_protected_path_refused_before_any_filesystem_read(tmp_path, monkeypatch):
    accesses = []

    def forbidden(*args, **kwargs):
        accesses.append(args)
        raise AssertionError("no access")

    monkeypatch.setattr(Path, "is_symlink", forbidden)
    with pytest.raises(ValueError, match="path_refused"):
        verify_kit(tmp_path / "Starter Files" / "kit", kit_hash="0" * 64)
    assert not accesses
