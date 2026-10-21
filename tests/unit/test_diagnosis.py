import pytest

from servegauge.core.models import CheckResult
from servegauge.diagnose.confidence import confidence
from servegauge.diagnose.rules import diagnoses


def result(check, *, verdict="FAIL", replay=True, evidence=None):
    return CheckResult(
        check_id=check,
        suite_id="S1",
        check_class="invariant",
        endpoint_id="e",
        run_id="r",
        n=400,
        estimate=0.1,
        ci_low=0.07,
        ci_high=0.14,
        reference_estimate=0,
        violations=40,
        verdict=verdict,
        reason="observed behavior",
        evidence=evidence or [check],
        details={
            "replay_confirmed": replay,
            **(
                {"failure_signatures": ["missing_stream_index"]} if check.startswith("T04.") else {}
            ),
        },
    )


@pytest.mark.parametrize(
    "check,code",
    [
        ("P01", "PARAM_IGNORED"),
        ("P09.none", "PARAM_IGNORED"),
        ("T05", "PARSER_LEAK"),
        ("T06", "PARSER_DROP"),
        ("T04.parallel.stream", "PARSER_DROP"),
        ("P11", "PARSER_DROP"),
        ("J02.validity", "GUIDED_DECODING_ABSENT"),
        ("L04", "CONTEXT_TRUNCATION"),
        ("B01.context.32000", "ACCOUNTING_MISMATCH"),
        ("T07.passback", "TEMPLATE_MISMATCH"),
    ],
)
def test_scoped_codes_and_qualified_language(check, code):
    diagnosis = diagnoses([result(check)])[0]
    assert diagnosis.code == code and diagnosis.confidence == "medium"
    assert diagnosis.alternatives and diagnosis.falsifiers and "CI [" in diagnosis.summary
    assert diagnosis.evidence and check in diagnosis.check_ids


def test_argument_schema_failure_does_not_imply_parser_drop():
    structural = result("T04.parallel.stream")
    structural.details["failure_signatures"] = ["argument_schema"]
    found = diagnoses([structural, result("T02"), result("T01", verdict="PASS")])
    assert {d.code for d in found} == {"GUIDED_DECODING_ABSENT"}


def test_missing_full_diagnosis_paths_require_their_supporting_observations():
    assert diagnoses([result("P03.cache")])[0].code == "CACHE_ANOMALY"
    assert (
        diagnoses([result("R03.8000"), result("R03.4000", verdict="PASS")])[0].code
        == "LONG_OUTPUT_FAULT"
    )
    operational = result("S8.success", verdict="INFO", replay=False)
    operational.check_class = "report_only"
    assert not diagnoses([operational])
    operational.details["operational_alarm"] = True
    assert diagnoses([operational])[0].code == "INFRA_UNRELIABLE"
    fingerprints = [result("FP1"), result("FP2")]
    for r in fingerprints:
        r.details["fingerprint_position"] = "different-model-band"
    assert not diagnoses(fingerprints)
    assert diagnoses([*fingerprints, result("R01")])[0].code == "DIFFERENT_MODEL_LIKE"


def test_independent_support_and_replay_are_both_required():
    first, second = result("P01"), result("P02")
    assert confidence([first, second]) == "high"
    second.evidence = first.evidence
    assert confidence([first, second]) == "medium"
    first.details = second.details = {}
    assert confidence([first, second]) == "low"
    assert confidence([result("B01.one"), result("B01.two")]) == "medium"
    assert not diagnoses([result("R03.8000"), result("L05", verdict="INCONCLUSIVE")])
    assert diagnoses([result("T01"), result("T02", verdict="PASS")])[0].code == "TEMPLATE_MISMATCH"
    assert (
        diagnoses([result("T02"), result("T01", verdict="PASS")])[0].code
        == "GUIDED_DECODING_ABSENT"
    )
    assert [
        diagnosis.code
        for diagnosis in diagnoses([result("T02"), result("T01", verdict="PASS"), result("T06")])
    ] == ["PARSER_DROP"]
