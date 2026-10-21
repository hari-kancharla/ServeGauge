"""First-valid timing is observed once; missing evidence never receives a time."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from unit.test_independent_comparison import assertion_input

from servegauge.storage.comparison_timing import get_assert


def test_first_valid_result_is_recorded_once_without_changing_the_independent_verdict(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    value, context = assertion_input(10, bad=True)
    result = get_assert(value, context)
    assert json.loads(result["reason"])["verdict"] == "FAIL"
    path = tmp_path / "first-valid-result.json"
    first = path.read_bytes()
    marker = json.loads(first)
    assert (
        marker["procedure"] == "external" and marker["scope"] == "first_valid_independent_assertion"
    )
    assert marker["utc"].endswith("+00:00")
    if os.name != "nt":
        assert path.stat().st_mode & 0o077 == 0
    get_assert(*assertion_input(400))
    assert path.read_bytes() == first


def test_error_has_no_fabricated_first_valid_time(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = get_assert({}, {})
    assert json.loads(result["reason"])["verdict"] == "ERROR"
    assert not (tmp_path / "first-valid-result.json").exists()


def test_malformed_existing_timing_data_is_refused_without_echoing_its_value(tmp_path, monkeypatch):
    from servegauge.storage.comparison_timing import record_first_valid

    marker = {
        "procedure": "external",
        "utc": "synthetic-private-value",
        "scope": "first_valid_independent_assertion",
    }
    path = tmp_path / "first-valid-result.json"
    path.write_text(json.dumps(marker))
    before = path.read_bytes()
    with pytest.raises(ValueError) as failure:
        record_first_valid(tmp_path, procedure="external")
    assert "synthetic-private-value" not in str(failure.value)
    assert path.read_bytes() == before


def test_protected_or_symlink_timing_destination_refuses_without_writing(tmp_path, monkeypatch):
    protected = tmp_path / "Starter Files" / "first-valid-result.json"
    accesses = []
    from servegauge.storage.comparison_timing import record_first_valid

    def forbidden(*args, **kwargs):
        accesses.append(args)
        raise AssertionError("no access")

    monkeypatch.setattr(Path, "is_symlink", forbidden)
    with pytest.raises(ValueError):
        record_first_valid(protected.parent, procedure="external")
    assert not accesses
