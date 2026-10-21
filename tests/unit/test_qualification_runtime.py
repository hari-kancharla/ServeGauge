"""An unsupported numerical runtime must refuse empirical qualification."""

import json
from pathlib import Path

import pytest

import servegauge.refs.fingerprint_qualification as qualification


@pytest.mark.parametrize("changed", ["python", "platform", "architecture", "numpy", "scipy"])
def test_each_runtime_mismatch_refuses_before_evidence_admission(monkeypatch, changed):
    required = json.loads(Path("configs/fingerprint_qualification.json").read_text())["runtime"]
    observed = {**required, changed: "unsupported-runtime"}
    monkeypatch.setattr(qualification.platform, "python_version", lambda: observed["python"])
    monkeypatch.setattr(qualification.platform, "system", lambda: observed["platform"])
    monkeypatch.setattr(qualification.platform, "machine", lambda: observed["architecture"])
    monkeypatch.setattr(qualification, "version", lambda name: observed[name])
    with pytest.raises(ValueError, match="qualification_numerical_runtime_changed"):
        qualification.load_qualification("captured-scripted-text-v1")
