import pytest
from tests_support import exchange

from servegauge.scoring.fingerprints import distances, fingerprint_checks
from servegauge.scoring.operational import cache_samples, operational_samples
from servegauge.suites.s1_protocol import probe
from servegauge.verdicts.config import load_policy


def test_controlled_fingerprint_displacement_uses_matched_cells_and_noise():
    cells = [["a"] * 20 + ["b"] * 10 for _ in range(40)]
    other = [["c"] * 20 + ["d"] * 10 for _ in range(40)]
    controls = [other, [["z"] * 30 for _ in range(40)]]
    clean = fingerprint_checks(
        cells, [cells, cells, cells], controls, seed=19, length=16, policy=load_policy()
    )
    bad = fingerprint_checks(
        other, [cells, cells, cells], controls, seed=19, length=16, policy=load_policy()
    )
    assert all(r.verdict == "PASS" for r in clean)
    assert all(
        r.verdict == "FAIL" and r.details["fingerprint_position"] == "different-model-band"
        for r in bad
    )
    with pytest.raises(ValueError, match="three_clean"):
        fingerprint_checks(cells, [cells], controls, seed=19, length=16, policy=load_policy())
    with pytest.raises(ValueError, match="matched_prompt"):
        distances(cells, [row[:-1] for row in cells], method="FP1", length=16)
    with pytest.raises(ValueError, match="unsupported_fingerprint"):
        distances(cells, cells, method="unknown", length=16)


def test_cache_transitions_are_clustered_and_errors_are_operational_observations():
    item = probe("P03.cache", 1, "repeat", "color", {"type": "sample"}, samples=4)
    records = [
        exchange(
            i,
            endpoint_id="e",
            item_id=item.item_id,
            response={"choices": [{"message": {"content": text}}]},
        )
        for i, text in enumerate(("red", "blue", "red", "blue"))
    ]
    assert cache_samples([item], records, "e").values[item.item_id] == 1
    records[1].status = 503
    assert cache_samples([item], records, "e").status == "ERROR"
    observed = operational_samples(records, "e")
    assert sum(observed.values.values()) == 3
    records[0].metadata["request_not_sent"] = True
    assert len(operational_samples(records, "e").values) == 3
