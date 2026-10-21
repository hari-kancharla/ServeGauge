import copy

import pytest

from servegauge.refs.fingerprint import band_position, control_positions


@pytest.mark.parametrize(
    "delta,expected",
    [
        (-0.2, "below-same-model-band"),
        (-0.01, "same-model-band"),
        (0.0, "same-model-band"),
        (0.01, "same-model-band"),
        (0.05, "between-bands"),
        (0.1, "different-model-band"),
        (0.2, "different-model-band"),
        (0.25, "beyond-different-model-band"),
    ],
)
def test_signed_control_position_respects_both_band_boundaries(delta, expected):
    assert band_position(delta, 0.01, [0.1, 0.2], separated=True, split_beyond=True) == expected


def test_overlap_abstains_and_combined_band_retains_its_existing_positive_label():
    assert band_position(0.5, 0.1, [0.05, 0.2], separated=False) == "calibration-overlap"
    assert band_position(0.5, 0.01, [0.1, 0.2], separated=True) == "different-model-band-or-beyond"


@pytest.mark.parametrize(
    "delta,noise,band",
    [(float("nan"), 0.1, [0.2, 0.3]), (0.1, -0.1, [0.2, 0.3]), (0.1, 0.01, [0.3, 0.2])],
)
def test_invalid_band_evidence_cannot_receive_a_position(delta, noise, band):
    with pytest.raises(ValueError):
        band_position(delta, noise, band, separated=True)


def controls():
    return [
        {
            "run_id": f"run-{i}",
            "endpoint_id": f"control-{i}",
            "model_identity": f"model-{i}",
            "capture_hash": f"hash-{i}",
        }
        for i in range(2)
    ]


def test_mixture_can_exceed_nearer_control_and_remain_below_farther_one():
    metric = {
        "noise_band": 0.02,
        "control_bands": [{"interval": [0.1, 0.2]}, {"interval": [0.4, 0.6]}],
    }
    declarations = controls()
    original = copy.deepcopy((metric, declarations))
    result = control_positions(metric, declarations, 0.3)
    assert [row["position"] for row in result] == ["beyond-different-model-band", "between-bands"]
    assert [row["model_identity"] for row in result] == ["model-0", "model-1"]
    assert [row["capture_hash"] for row in result] == ["hash-0", "hash-1"]
    assert (metric, declarations) == original


@pytest.mark.parametrize("case", ["count", "identity", "interval"])
def test_control_positions_require_matched_band_provenance(case):
    metric = {
        "noise_band": 0.02,
        "control_bands": [{"interval": [0.1, 0.2]}, {"interval": [0.4, 0.6]}],
    }
    declarations = controls()
    if case == "count":
        declarations.pop()
    elif case == "identity":
        declarations[0].pop("capture_hash")
    else:
        metric["control_bands"][0]["interval"] = [0.2, 0.1]
    with pytest.raises(ValueError):
        control_positions(metric, declarations, 0.3)
