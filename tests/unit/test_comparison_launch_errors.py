"""A failed comparison child remains inspectable and cannot be reported as success."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from unit.test_comparison_authorization import manifest

from servegauge.core.hashing import content_hash
from servegauge.runner import comparison_launcher as launcher


@pytest.mark.parametrize(
    "errors", [(None, None), ("failed", None), (None, "failed"), ("failed", "failed")]
)
def test_parent_failure_state_keeps_error_rows_and_private_result_inventory(
    tmp_path, monkeypatch, errors
):
    value = manifest(synthetic=True, eligible=False)
    value["root_cause_limit"] = "Cause unverified."
    monkeypatch.setattr(launcher, "verify_kit", lambda *a, **k: value)

    def procedure(**kwargs):
        index = 0 if kwargs["procedure"] == "ordinary" else 1
        return {
            "procedure": kwargs["procedure"],
            "harness_error": errors[index],
            "first_valid_result_utc": None,
            "cases": [
                {
                    "case_id": "case",
                    "verdict": "ERROR",
                    "n": None,
                    "violations": None,
                    "ci_low": None,
                    "ci_high": None,
                    "reference_tier": None,
                }
            ],
        }

    monkeypatch.setattr(launcher, "_run_procedure", procedure)
    output = tmp_path / ".cache/comparisons/results"
    result = launcher.launch(
        tmp_path,
        tmp_path / "kit",
        kit_hash=content_hash(value),
        mode="synthetic",
        review=None,
        docker=Path("/missing/docker"),
        config=tmp_path / "anonymous",
        host="unix:///var/run/docker.sock",
        image="sha256:" + "a" * 64,
        out=output,
        environment={},
    )
    expected = "comparison_failed" if any(errors) else "captured_procedures_executed"
    assert result["state"] == expected
    assert json.loads((output / "comparison-results.json").read_text()) == result
    assert all(row["verdict"] == "ERROR" for item in result["procedures"] for row in item["cases"])
    assert all(item["same_n"] is None for item in result["agreements"])


@pytest.mark.parametrize(
    "errors", [(None, None), ("failed", None), (None, "failed"), ("failed", "failed")]
)
def test_cli_exits_five_for_failed_children_and_retains_external_result_anchor(
    tmp_path, monkeypatch, capsys, errors
):
    result = {
        "state": "comparison_failed" if any(errors) else "captured_procedures_executed",
        "kit_hash": "a" * 64,
        "owner_comparison_executed": False,
        "procedures": [{"harness_error": error} for error in errors],
    }
    monkeypatch.setattr(launcher, "launch", lambda *a, **k: result)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "comparison_launcher",
            "run-synthetic",
            "--root",
            str(tmp_path),
            "--kit",
            str(tmp_path / "kit"),
            "--kit-hash",
            "a" * 64,
            "--out",
            str(tmp_path / ".cache/comparisons/results"),
            "--docker",
            "/missing/docker",
            "--docker-config",
            str(tmp_path / "anonymous"),
            "--image",
            "sha256:" + "b" * 64,
        ],
    )
    if any(errors):
        with pytest.raises(SystemExit) as caught:
            launcher.main()
        assert caught.value.code == 5
    else:
        launcher.main()
    summary = json.loads(capsys.readouterr().out)
    assert summary["result_hash"] == content_hash(result)
    assert summary["state"] == result["state"]
    assert summary["harness_failures"] == sum(error is not None for error in errors)
