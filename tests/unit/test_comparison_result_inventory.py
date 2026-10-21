"""Private comparison summaries retain uncertainty and a separate result anchor."""

from __future__ import annotations

import hashlib

import pytest

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.storage.comparison_results import render_comparison, verify_results


def fixture_result(root):
    root.mkdir()
    (root / "evidence.txt").write_text("synthetic private evidence")
    case = {
        "case_id": "control",
        "n": 10,
        "original_population_n": 10,
        "violations": 0,
        "ci_low": 0.0,
        "ci_high": 0.2775327998628892,
        "reference_tier": None,
        "verdict": "INCONCLUSIVE",
    }
    value = {
        "version": "comparison-results-v1",
        "owner_comparison_executed": False,
        "mode": "synthetic",
        "kit_hash": "a" * 64,
        "total_setup_seconds": None,
        "root_cause_limit": "Cause remains unverified.",
        "procedures": [
            {
                "procedure": name,
                "cases": [case],
                "execution_seconds": 1.0,
                "first_valid_result_utc": None,
                "harness_error": None,
            }
            for name in ["ordinary", "external"]
        ],
        "files_sha256": {
            "evidence.txt": hashlib.sha256((root / "evidence.txt").read_bytes()).hexdigest()
        },
    }
    (root / "comparison-results.json").write_bytes(canonical_json(value) + b"\n")
    return value


def test_result_anchor_binds_every_evidence_file_and_cannot_accept_later_edits(tmp_path):
    root = tmp_path / "results"
    value = fixture_result(root)
    anchor = content_hash(value)
    assert verify_results(root, result_hash=anchor) == value
    (root / "evidence.txt").write_text("changed")
    with pytest.raises(ValueError):
        verify_results(root, result_hash=anchor)
    with pytest.raises(ValueError):
        verify_results(root, result_hash="0" * 64)


def test_private_report_preserves_inconclusive_interval_and_unmeasured_setup(tmp_path):
    value = fixture_result(tmp_path / "results")
    text = render_comparison(value)
    assert "synthetic" in text.lower()
    assert "INCONCLUSIVE" in text and "0.2775328" in text
    assert "unmeasured" in text.lower() and "no reference" in text.lower()
    assert "Cause remains unverified." in text
    assert "winner" not in text.lower()
