"""Fixed-seed statistical acceptance. Never search seeds or change alpha to make it pass."""

import json
from pathlib import Path

import pytest

from servegauge.stats.simulate import null_simulations


@pytest.mark.slow
def test_type1_and_multiple_testing_2000_replicates() -> None:
    report = null_simulations(replicates=2000, seed=20261018)
    destination = Path(".cache/evidence/m5-null-simulations.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    for name, result in report["tests"].items():
        if result["class"] == "conservative":
            assert result["estimate"] <= 0.065, (name, result)
        elif result["class"] == "asymptotic":
            assert 0.035 <= result["estimate"] <= 0.065, (name, result)
    assert report["holm"]["estimate"] <= 0.05
    assert report["bh"]["estimate"] <= 0.05


def test_null_simulation_output_has_counts_intervals_and_assumptions() -> None:
    report = null_simulations(replicates=20, seed=20261018)
    for result in [*report["tests"].values(), report["holm"], report["bh"]]:
        assert result["n"] == 20
        assert len(result["interval"]) == 2
        assert result["reference_tier"] == "MOCK"
        assert result["assumptions"]
