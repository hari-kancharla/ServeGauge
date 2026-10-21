"""Independent raw P06/T06 assertions versus ordinary captured procedures."""

from __future__ import annotations

import json

import pytest
from comparison_workflow import prepare_comparison_fixture

from servegauge.comparison.case_provider import run_raw_case
from servegauge.comparison.independent import get_assert
from servegauge.runner.captured import CapturedCheck
from servegauge.runner.check import run_check
from servegauge.storage.comparison import _private_json
from servegauge.storage.workspace import initialize


@pytest.mark.asyncio
async def test_two_failure_and_full_matched_control_procedures_agree_without_new_n(tmp_path):
    snapshot, failures, controls = await prepare_comparison_fixture(tmp_path / "fixture")
    sources = {}
    for path in (snapshot / "sources").iterdir():
        for seal in (path / "data/raw").glob("*/analysis.json"):
            value = _private_json(seal.read_bytes())["value"]
            sources[value["run_id"]] = value
    for program in [failures, controls]:
        for case in program.cases:
            policy = sources[case.original_check.run_id]["metadata"]["policy"]
            context = {
                "vars": {
                    "independent_policy": {
                        "check_id": case.original_check.check_id,
                        "unit_keys": case.logical_unit_keys,
                        "probability_tolerance": policy["entries"]["P06"]["prob_sum_tolerance"],
                        "alpha": policy["alpha"],
                        "tolerance": policy["tolerance"],
                        "k_fail": policy["k_fail"],
                        "reference_tier": case.original_check.reference_tier,
                    }
                }
            }
            raw = await run_raw_case(
                program, case_id=case.case_id, root=tmp_path / ("independent-" + case.case_id)
            )
            independent = json.loads(get_assert(raw, context)["reason"])
            assert independent["verdict"] == case.original_check.verdict
            assert independent["violations"] == case.original_check.violations
            assert independent["n"] == case.population_n == 4
            assert independent["ci_low"] == pytest.approx(case.original_check.ci_low)
            assert independent["ci_high"] == pytest.approx(case.original_check.ci_high)
            replay = tmp_path / ("ordinary-" + case.case_id)
            initialize(replay)
            captured = CapturedCheck(program, case_id=case.case_id)
            result = await run_check(
                captured.local_plan(),
                root=replay,
                approved=True,
                tests=case.original_check.check_id,
                baseline_id=captured.baseline_id,
                inventory=captured.primary_items(),
                capability_inventory=captured.capability_items(),
                transports=captured.transports(),
                captured=captured,
            )
            actual = next(
                row for row in result.checks if row.check_id == case.original_check.check_id
            )
            assert actual.n == independent["n"] and actual.verdict == independent["verdict"]
            assert actual.violations == independent["violations"]
