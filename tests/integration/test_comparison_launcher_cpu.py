"""Optional installed CPU workflow; no provider, account or GPU is used."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from comparison_workflow import prepare_comparison_fixture

from servegauge.core.hashing import content_hash
from servegauge.runner.comparison_launcher import launch
from servegauge.storage.comparison import _guard
from servegauge.storage.comparison_controls import ControlSelection
from servegauge.storage.comparison_kit import prepare_kit


@pytest.mark.slow
@pytest.mark.enable_socket
@pytest.mark.asyncio
async def test_actual_installed_cpu_parent_and_both_procedures(tmp_path):
    image = os.environ.get("SERVEGAUGE_COMPARISON_IMAGE")
    docker = os.environ.get("SERVEGAUGE_COMPARISON_DOCKER")
    config = os.environ.get("SERVEGAUGE_COMPARISON_DOCKER_CONFIG")
    if not image or not docker or not config:
        pytest.skip("explicit pinned local image/executable/anonymous client config required")
    for path in [Path(docker).parent, Path(config)]:
        _guard(path)
    trials = int(os.environ.get("SERVEGAUGE_COMPARISON_TRIALS", "4"))
    fixture = tmp_path / "fixture"
    snapshot, failures, _ = await prepare_comparison_fixture(fixture, trials=trials)
    controls = ControlSelection.model_validate_json(
        (fixture / "control-selection.json").read_bytes()
    )
    root = fixture / "source"
    kit = root / ".cache/comparisons/kit"
    manifest = prepare_kit(
        root,
        snapshot,
        selection_hash=failures.selection_hash,
        controls=controls,
        controls_hash=content_hash(controls.model_dump(mode="json")),
        out=kit,
    )
    result = launch(
        root,
        kit,
        kit_hash=content_hash(manifest),
        mode="synthetic",
        review=None,
        docker=Path(docker),
        config=Path(config),
        host=os.environ.get(
            "SERVEGAUGE_COMPARISON_DOCKER_HOST",
            "npipe:////./pipe/docker_engine" if os.name == "nt" else "unix:///var/run/docker.sock",
        ),
        image=image,
        out=root / ".cache/comparisons/results",
    )
    assert result["owner_comparison_executed"] is False and result["total_setup_seconds"] is None
    assert all(item["harness_error"] is None for item in result["procedures"])
    for agreement in result["agreements"]:
        assert all(
            agreement[key] is True
            for key in [
                "same_n",
                "same_violations",
                "same_reference_tier",
                "same_verdict",
                "intervals_agree_within_float_roundoff",
            ]
        )
