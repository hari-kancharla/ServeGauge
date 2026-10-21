"""Uncertain/error external rows and changed finite audits cannot become successes."""

from __future__ import annotations

import copy
import json

import pytest
from unit.test_comparison_sources import source as source

from servegauge.comparison.case_provider import run_raw_case
from servegauge.core.hashing import content_hash
from servegauge.storage.comparison import prepare_sources
from servegauge.storage.comparison_program import compile_program
from servegauge.storage.comparison_results import normalize_external, validate_audit


def cases():
    return [
        {"case_id": "failure", "population_n": 10, "original_check": {"reference_tier": None}},
        {"case_id": "control", "population_n": 10, "original_check": {"reference_tier": None}},
    ]


def external_row(case, verdict="INCONCLUSIVE"):
    summary = {
        "n": 10,
        "observed_n": 10,
        "violations": 0,
        "ci_low": 0.0,
        "ci_high": 0.2775327998628892,
        "reference_tier": None,
        "verdict": verdict,
        "replay_adds_independent_n": False,
    }
    return {
        "testCase": {"description": case, "vars": {"case_id": case}},
        "gradingResult": {"componentResults": [{"reason": json.dumps(summary)}]},
    }


def test_inconclusive_and_disagreement_are_preserved_without_inventing_violations():
    raw = {"results": {"results": [external_row("failure"), external_row("control")]}}
    actual = normalize_external(raw, cases())
    assert {row["verdict"] for row in actual} == {"INCONCLUSIVE"}
    assert all(
        row["violations"] == 0 and row["n"] == 10 and row["original_population_n"] == 10
        for row in actual
    )


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "duplicate",
        "not_json",
        "wrong_n",
        "unknown_verdict",
        "invalid_interval",
        "inflated_n",
    ],
)
def test_invalid_external_rows_are_errors_with_unknown_counts_not_zero(problem):
    rows = [external_row("failure"), external_row("control")]
    if problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows.append(copy.deepcopy(rows[1]))
    elif problem == "not_json":
        rows[1]["gradingResult"]["componentResults"][0]["reason"] = "harness error"
    else:
        target = rows[1]["gradingResult"]["componentResults"][0]
        value = json.loads(target["reason"])
        if problem == "wrong_n":
            value["n"] = 400
        elif problem == "unknown_verdict":
            value["verdict"] = "GREAT"
        elif problem == "invalid_interval":
            value["ci_low"] = 0.9
        else:
            value["replay_adds_independent_n"] = True
        target["reason"] = json.dumps(value)
    actual = normalize_external({"results": {"results": rows}}, cases())
    row = next(row for row in actual if row["case_id"] == "control")
    assert row["verdict"] == "ERROR" and row["violations"] is None
    assert row["ci_low"] is row["ci_high"] is None
    assert row["original_population_n"] == 10


@pytest.mark.asyncio
async def test_finite_audit_matches_every_address_attempt_representation_and_source_hash(
    source, tmp_path
):
    root, selection, snapshot = source
    prepare_sources(root, selection, snapshot)
    program = compile_program(
        snapshot, selection_hash=content_hash(selection.model_dump(mode="json"))
    )
    case = program.cases[0]
    output = tmp_path / "raw-output"
    await run_raw_case(program, case_id=case.case_id, root=output)
    audit = json.loads((output / "finite-audit.json").read_text())
    before = copy.deepcopy(audit)
    validate_audit(case, audit)
    assert audit == before
    for problem in ["request_hash", "attempt_hash", "endpoint", "extra", "missing", "refused"]:
        changed = copy.deepcopy(audit)
        if problem in {"request_hash", "attempt_hash"}:
            changed["observations"][0][problem] = "0" * 64
        elif problem == "endpoint":
            changed["observations"][0]["address"]["endpoint_id"] = "other"
        elif problem == "extra":
            changed["observations"].append(copy.deepcopy(changed["observations"][0]))
        elif problem == "missing":
            changed["observations"].pop()
        else:
            changed["refusals"] = ["refused"]
        with pytest.raises(ValueError):
            validate_audit(case, changed)
