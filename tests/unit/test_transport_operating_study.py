"""A small diagnostic study never becomes an operating qualification."""

import json

import pytest

from servegauge.validation.transport_study import transport_study


def test_diagnostic_retains_full_family_shapes_seeds_missingness_and_all_results(tmp_path):
    out = tmp_path / "study"
    result = transport_study(out, replicates=2)
    assert result["operating_qualification_accepted"] is False
    assert result["eligible_for_operating_review"] is False
    assert len(result["family_cases"]) == 4 and len(result["readiness_cases"]) == 3
    assert all(
        r["families"] == 2 and r["streams"] == 100 and r["evaluation_updates"] == 120
        for r in result["family_cases"]
    )
    assert all(r["replicates"] == 2 for r in result["readiness_cases"])
    assert result["held_out_public_replay"]["trajectories"] == 2
    assert result["held_out_public_replay"]["full_family_paths"] == 0
    assert json.loads((out / "study.json").read_text()) == result
    assert result["model_api_calls"] == 0 and result["new_spend_usd"] == 0
    with pytest.raises(ValueError, match="fresh"):
        transport_study(out, replicates=2)


def test_invalid_replication_count_refuses_before_writes(tmp_path):
    for count in [0, True, -1]:
        with pytest.raises(ValueError):
            transport_study(tmp_path / "absent", replicates=count)
        assert not (tmp_path / "absent").exists()
