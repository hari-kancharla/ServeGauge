"""Finite disjoint study matrix; preparation never implies funding or execution."""

import pytest

from servegauge.selfhost.study_design import StudyDesign, compile_matrix


def test_complete_matrix_has_no_reused_run_session_or_seed():
    design = StudyDesign(version="gpu-study-v1", study_id="pilot", seed=91, replicates=2)
    rows = compile_matrix(design)
    assert len(rows) == 2 * 2 * 5 * 4
    assert len({r["run_id"] for r in rows}) == len(rows)
    assert len({r["session_id"] for r in rows}) == len(rows)
    assert len({r["capture_seed"] for r in rows}) == len(rows)
    assert {r["partition"] for r in rows} == {"calibration", "evaluation"}
    assert {r["condition"] for r in rows} == {
        "bf16",
        "fp8",
        "mxfp4",
        "other-engine",
        "wrong-template",
    }
    for unit in {r["unit_id"] for r in rows}:
        group = [r for r in rows if r["unit_id"] == unit]
        assert len(group) == 4
        assert {r["role"] for r in group} == {"target", "reference-b", "reference-c", "reference-d"}
        assert {r["recipe"] for r in group if r["role"] != "target"} == {"bf16"}
    assert rows == compile_matrix(design)
    assert not design.independence_reviewed


@pytest.mark.parametrize(
    "changes",
    [
        {"study_id": "../unsafe"},
        {"study_id": "Starter Files"},
        {"replicates": True},
        {"replicates": 1},
        {"replicates": 1001},
        {"seed": -1},
        {"version": "old"},
        {"extra_args": ["--trust-remote-code"]},
    ],
)
def test_matrix_requires_bounded_explicit_design(changes):
    value = dict(version="gpu-study-v1", study_id="pilot", seed=91, replicates=2)
    with pytest.raises(ValueError):
        StudyDesign.model_validate({**value, **changes})
