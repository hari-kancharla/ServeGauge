"""Controls retain full matched source units and independent aggregate uncertainty."""

from __future__ import annotations

import pytest
from unit.test_comparison_sources import hashes
from unit.test_comparison_sources import source as source

from servegauge.core.hashing import content_hash
from servegauge.storage.comparison import prepare_sources
from servegauge.storage.comparison_controls import (
    ControlRequest,
    ControlSelection,
    compile_controls,
    propose_controls,
)
from servegauge.storage.comparison_program import compile_program


def prepared(source):
    root, selection, snapshot = source
    anchor = content_hash(selection.model_dump(mode="json"))
    prepare_sources(root, selection, snapshot)
    requests = [
        ControlRequest(
            case_id="control-" + c.case_id, failure_case_id=c.case_id, endpoint_id="base"
        )
        for c in selection.cases
    ]
    return snapshot, anchor, requests


def test_controls_keep_full_units_original_counts_and_underpowered_verdict(source):
    snapshot, anchor, requests = prepared(source)
    before = hashes(snapshot)
    failure = compile_program(snapshot, selection_hash=anchor)
    selection = propose_controls(snapshot, selection_hash=anchor, requests=requests)
    program = compile_controls(
        snapshot,
        selection,
        selection_hash=anchor,
        controls_hash=content_hash(selection.model_dump(mode="json")),
    )
    assert len(program.cases) == 2
    assert hashes(snapshot) == before
    assert compile_program(snapshot, selection_hash=anchor).program_hash() == failure.program_hash()
    for control, original in zip(program.cases, failure.cases, strict=True):
        assert control.population_n == original.population_n == 4
        assert control.logical_unit_keys == original.logical_unit_keys
        assert control.original_check.violations == 0
        assert control.original_check.verdict == "INCONCLUSIVE"
        assert control.original_check.ci_high > 0.01
        assert control.original_check.details.get("replay_batch_ids", []) == []
        assert {
            row.source.record.endpoint_id for phase in control.phases for row in phase.exchanges
        } == {"base"}
        assert {row.source.role for phase in control.phases for row in phase.exchanges} == {
            "primary",
            "preflight",
        }
        assert (
            control.original_check.details["comparison_aggregate_origin"]
            == "raw_reconstructed_control"
        )


@pytest.mark.parametrize(
    "mutation", ["failure_endpoint", "unknown_endpoint", "unknown_case", "duplicate_case"]
)
def test_control_scope_cannot_borrow_failed_or_unknown_endpoint_or_case(source, mutation):
    snapshot, anchor, requests = prepared(source)
    if mutation == "failure_endpoint":
        requests[0].endpoint_id = "a"
    elif mutation == "unknown_endpoint":
        requests[0].endpoint_id = "missing"
    elif mutation == "unknown_case":
        requests[0].failure_case_id = "missing"
    else:
        requests[1].case_id = requests[0].case_id
    with pytest.raises(ValueError):
        propose_controls(snapshot, selection_hash=anchor, requests=requests)


@pytest.mark.parametrize(
    "mutation", ["failure_anchor", "control_anchor", "check_hash", "endpoint", "source_bytes"]
)
def test_controls_require_separate_reviewed_anchors_and_unchanged_raw(source, mutation):
    snapshot, anchor, requests = prepared(source)
    selection = propose_controls(snapshot, selection_hash=anchor, requests=requests)
    controls_hash = content_hash(selection.model_dump(mode="json"))
    if mutation == "failure_anchor":
        anchor = "0" * 64
    elif mutation == "control_anchor":
        controls_hash = "0" * 64
    elif mutation == "check_hash":
        selection.controls[0].check_hash = "0" * 64
    elif mutation == "endpoint":
        selection.controls[0].endpoint_id = "a"
    else:
        path = snapshot / "selection.json"
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        compile_controls(snapshot, selection, selection_hash=anchor, controls_hash=controls_hash)


def test_control_selection_requires_exactly_one_distinct_control_for_each_failure(source):
    snapshot, anchor, requests = prepared(source)
    for invalid in [requests[:1], requests + requests[:1], [requests[0], requests[0]]]:
        with pytest.raises(ValueError):
            propose_controls(snapshot, selection_hash=anchor, requests=invalid)


def test_control_validation_rejects_invalid_control_selection():
    with pytest.raises(ValueError):
        ControlSelection(failure_selection_hash="0" * 64, controls=[])
