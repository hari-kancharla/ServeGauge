"""A diagnosis replay must reproduce its full scoped witness, including controls."""

import copy

import pytest

from servegauge.core.models import CheckResult
from servegauge.diagnose.confidence import confidence
from servegauge.diagnose.replay_expectations import (
    expected_context,
    mark_reproduced,
    replay_matches,
)
from servegauge.diagnose.rules import diagnoses


def result(name, verdict="FAIL", *, replay=False):
    return CheckResult(
        check_id=name,
        suite_id="S2",
        check_class="comparative",
        endpoint_id="e",
        run_id="source",
        n=400,
        estimate=0.2 if verdict == "FAIL" else 1,
        ci_low=0.1,
        ci_high=1,
        reference_tier="BASELINE",
        reference_estimate=1,
        verdict=verdict,
        reason="captured evidence",
        evidence=[name + "-capture"],
        details={"replay_confirmed": replay},
    )


@pytest.mark.parametrize(
    "failed,control,code",
    [
        ("T02", "T01", "GUIDED_DECODING_ABSENT"),
        ("T01", "T02", "TEMPLATE_MISMATCH"),
        ("R03.8000", "R03.4000", "LONG_OUTPUT_FAULT"),
    ],
)
def test_passing_prerequisite_is_part_of_complete_witness(failed, control, code):
    checks = [result(failed, replay=True), result(control, "PASS")]
    diagnosis = next(d for d in diagnoses(checks) if d.code == code)
    assert set(diagnosis.check_ids) == {failed, control}
    assert set(diagnosis.evidence) == {failed + "-capture", control + "-capture"}
    assert diagnosis.confidence == "low"
    context = expected_context(diagnosis, checks)
    assert context["expected"] == {failed: "FAIL", control: "PASS"}
    replay = copy.deepcopy(checks)
    for check in replay:
        check.run_id = "replay"
    assert replay_matches(replay, context, endpoint_id="e")
    assert mark_reproduced(replay, context)
    assert all(c.details.get("bundle_replay_reproduced") == diagnosis.diagnosis_id for c in replay)
    assert next(d for d in diagnoses(replay) if d.code == code).confidence == "medium"
    replay[1].verdict = "INCONCLUSIVE"
    assert not replay_matches(replay, context, endpoint_id="e")


def test_passing_controls_and_other_scopes_cannot_inflate_confidence():
    positive = result("T02", replay=True)
    control = result("T01", "PASS", replay=True)
    assert confidence([positive, control]) == "medium"
    unrelated = result("P01", replay=True)
    unrelated.endpoint_id = "other"
    assert confidence([positive, unrelated]) == "medium"
    unrelated.endpoint_id = "e"
    unrelated.run_id = "other-run"
    assert confidence([positive, unrelated]) == "medium"
    positive.details = {}
    assert confidence([positive, control]) == "low"


def test_same_fail_verdict_is_insufficient_if_diagnosis_signature_changed():
    checks = [result("T02"), result("T01", "PASS")]
    diagnosis = diagnoses(checks)[0]
    context = expected_context(diagnosis, checks)
    parser = result("T06")
    assert not replay_matches([*checks, parser], context, endpoint_id="e")
    assert not mark_reproduced([*checks, parser], context)
    assert not any(c.details.get("bundle_replay_reproduced") for c in checks)


def test_report_only_alarm_must_reproduce_as_an_alarm():
    check = result("S8.success", "INFO")
    check.check_class = "report_only"
    check.details["operational_alarm"] = True
    diagnosis = diagnoses([check])[0]
    context = expected_context(diagnosis, [check])
    assert context["expected"] == {"S8.success": "OPERATIONAL_ALARM"}
    assert replay_matches([check], context, endpoint_id="e")
    check.details["operational_alarm"] = False
    assert not replay_matches([check], context, endpoint_id="e")


def test_ambiguous_scope_unsupported_and_forged_witnesses_fail_closed():
    checks = [result("P01")]
    diagnosis = diagnoses(checks)[0]
    context = expected_context(diagnosis, checks)
    repeated = checks[0].model_copy(deep=True)
    repeated.run_id = "other-run"
    assert not replay_matches([*checks, repeated], context, endpoint_id="e")
    assert not replay_matches(checks, context, endpoint_id="other")
    forged = diagnosis.model_copy(update={"confidence": "high"})
    with pytest.raises(ValueError, match="unsupported_diagnosis"):
        expected_context(forged, checks)
    with pytest.raises(ValueError, match="unsupported_diagnosis"):
        expected_context(diagnosis, [])
    altered = copy.deepcopy(context)
    altered["expected"]["P01"] = "PASS"
    assert not replay_matches(checks, altered, endpoint_id="e")


def test_legacy_replay_metadata_keeps_its_recorded_interpretation():
    checks = [result("P01"), result("T01", "PASS")]
    old = {"diagnosis_id": "old", "check_ids": ["P01", "T01"]}
    mark_reproduced(checks, old)
    assert checks[0].details["bundle_replay_reproduced"] == "old"
    assert "bundle_replay_reproduced" not in checks[1].details
