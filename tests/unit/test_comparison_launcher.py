"""Parent authorization and private output scope precede any container launch."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from unit.test_comparison_authorization import manifest

from servegauge.core.hashing import content_hash
from servegauge.runner.comparison_launcher import launch


def test_unapproved_owner_comparison_does_not_create_output_or_start_a_container(
    tmp_path, monkeypatch
):
    value = manifest()
    root = tmp_path / "workspace"
    root.mkdir()
    kit = root / ".cache/comparisons/kit"
    output = root / ".cache/comparisons/result"
    monkeypatch.setattr("servegauge.runner.comparison_launcher.verify_kit", lambda *a, **k: value)

    def forbidden(*args, **kwargs):
        raise AssertionError("no container may start")

    monkeypatch.setattr("servegauge.runner.comparison_launcher._run_procedure", forbidden)
    with pytest.raises(ValueError, match="owner_environment"):
        launch(
            root,
            kit,
            kit_hash=content_hash(value),
            mode="owner",
            review=None,
            docker=Path("/missing/docker"),
            config=root / "anonymous",
            host="unix:///var/run/docker.sock",
            image="sha256:" + "a" * 64,
            out=output,
            environment={},
        )
    assert not output.exists()


def test_recorded_sources_cannot_run_through_synthetic_mode(tmp_path, monkeypatch):
    value = manifest()
    root = tmp_path / "workspace"
    root.mkdir()
    output = root / ".cache/comparisons/result"
    monkeypatch.setattr("servegauge.runner.comparison_launcher.verify_kit", lambda *a, **k: value)

    def forbidden(*args, **kwargs):
        raise AssertionError("no execution")

    monkeypatch.setattr("servegauge.runner.comparison_launcher._run_procedure", forbidden)
    with pytest.raises(ValueError, match="synthetic_source"):
        launch(
            root,
            root / "kit",
            kit_hash=content_hash(value),
            mode="synthetic",
            review=None,
            docker=Path("/missing/docker"),
            config=root / "anonymous",
            host="unix:///var/run/docker.sock",
            image="sha256:" + "a" * 64,
            out=output,
            environment={},
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "module,arguments",
    [
        ("servegauge.runner.comparison_launcher", ["synthetic-private-cli-value"]),
        (
            "servegauge.runner.comparison",
            [
                "--kit",
                "unused",
                "--kit-hash",
                "0" * 64,
                "--procedure",
                "synthetic-private-cli-value",
            ],
        ),
    ],
)
def test_invalid_cli_values_are_not_echoed_in_diagnostics(module, arguments):
    result = subprocess.run(
        [sys.executable, "-m", module, *arguments], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 3
    assert "synthetic-private-cli-value" not in result.stdout + result.stderr
