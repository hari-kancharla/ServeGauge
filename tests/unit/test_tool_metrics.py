import numpy as np
import pytest

from servegauge.scoring.toolcall_scoring import (
    arguments_match,
    f1_influence,
    f1_projection,
    paired_f1,
    parsed_call,
    trigger_counts,
)


def test_confusion_f1_known_answers_and_noncollapsed_boundary_intervals() -> None:
    counts = trigger_counts([1, 1, 0, 0], [1, 0, 1, 0])
    assert counts == {
        "n": 4,
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "precision": 0.5,
        "recall": 0.5,
        "tool_call_f1": 0.5,
    }
    np.testing.assert_allclose(f1_influence([1, 1, 0, 0], [1, 0, 1, 0]), [1, -0.5, -0.5, 0])
    assert trigger_counts([0, 0], [0, 0])["tool_call_f1"] == 0
    assert f1_projection([0, 0], [0, 0], alpha=0.05) == (0, 1)
    low, high = f1_projection([1] * 400 + [0] * 100, [1] * 400 + [0] * 100, alpha=0.05)
    assert 0.97 < low < 1 and high == 1
    with pytest.raises(ValueError):
        trigger_counts([2], [1])


def test_paired_F1_is_paired_and_small_n_retains_uncertainty() -> None:
    small = paired_f1([1, 1], [0, 0], [1, 1], seed=8, alpha=0.05)
    assert small.p_value == 0.25 and small.estimate == 0 and small.reference_estimate == 1
    labels = [1] * 400 + [0] * 100
    perfect = paired_f1(labels, labels, labels, seed=8, alpha=0.05)
    assert perfect.estimate == perfect.reference_estimate == 1
    assert perfect.delta_interval[0] < 0 < perfect.delta_interval[1]
    assert perfect.one_sided_bounds[0] > -0.03
    assert perfect.p_value == 1
    short = paired_f1([1] * 10, [1] * 10, [1] * 10, seed=8, alpha=0.05)
    assert short.one_sided_bounds[0] < -0.03


def test_argument_types_unicode_and_explicit_numeric_tolerance() -> None:
    assert arguments_match(
        {"city": "Zu\u0308rich", "n": [1.0, True]},
        {"city": "Zürich", "n": [1, True]},
        numeric_tolerance=1e-9,
    )
    assert not arguments_match(True, 1, numeric_tolerance=1e-9)
    assert not arguments_match({"x": 1, "extra": 2}, {"x": 1}, numeric_tolerance=1e-9)
    assert not arguments_match([1], [1, 2], numeric_tolerance=1e-9)
    assert not arguments_match("1", 1, numeric_tolerance=1e-9)
    assert not arguments_match(float("inf"), 1, numeric_tolerance=1e-9)
    assert parsed_call({"function": {"name": "a", "arguments": '{"x":1}'}}) == ("a", {"x": 1})
    for invalid in [
        {},
        {"function": {"name": "a", "arguments": {}}},
        {"function": {"name": "a", "arguments": "[1]"}},
        {"function": {"name": "a", "arguments": "{"}},
    ]:
        with pytest.raises(ValueError):
            parsed_call(invalid)
