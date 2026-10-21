"""Local drafts retain the complete numeric witness without interpreting payloads."""

import pytest

from servegauge.core.models import CheckResult, Diagnosis
from servegauge.diagnose.issue import issue_markdown


def fixture():
    check = CheckResult(
        check_id="P01",
        check_class="invariant",
        suite_id="S1",
        endpoint_id="endpoint",
        run_id="run",
        n=400,
        estimate=0.5,
        ci_low=0.45,
        ci_high=0.55,
        margin=0.01,
        violations=200,
        verdict="FAIL",
        reason="observed cap violation",
        evidence=["exchange-1"],
    )
    diagnosis = Diagnosis(
        diagnosis_id="diagnosis-fixture",
        endpoint_id="endpoint",
        run_id="run",
        failure_class="API_INCOMPATIBILITY",
        code="PARAM_IGNORED",
        confidence="medium",
        summary="P01 cap violations with an invariant expectation.",
        evidence=check.evidence,
        check_ids=[check.check_id],
        alternatives=["request transformation", "unsupported parameter"],
        falsifiers=["A matched clean rerun clears the cap."],
    )
    return diagnosis, [check]


def test_local_issue_draft_contains_counts_intervals_reference_and_competing_explanations():
    diagnosis, checks = fixture()
    text = issue_markdown(diagnosis, checks)
    rendered_text = text.replace("\\", "")
    for expected in [
        "PARAM_IGNORED",
        "API_INCOMPATIBILITY",
        "P01",
        "FAIL",
        "400",
        "200",
        "0.45",
        "0.55",
        "invariant expectation",
        "request transformation",
        "matched clean rerun",
        "[bundle.json](bundle.json)",
        "[replay.py](replay.py)",
    ]:
        assert expected in rendered_text
    assert "Draft" in text and "no hardware" in text


def test_passing_and_inconclusive_witness_states_are_preserved_with_unavailable_values():
    diagnosis, checks = fixture()
    for name, verdict in [("R01", "PASS"), ("FP2", "INCONCLUSIVE")]:
        c = checks[0].model_copy(deep=True)
        c.check_id, c.verdict = name, verdict
        c.reference_tier = "MOCK"
        c.reference_estimate = None
        c.delta = c.delta_ci_low = c.delta_ci_high = None
        checks.append(c)
        diagnosis.check_ids.append(name)
    text = issue_markdown(diagnosis, checks, redact_content=True)
    assert "PASS" in text and "INCONCLUSIVE" in text and "MOCK" in text
    assert "unavailable" in text and "private raw store" in text


def test_untrusted_metadata_is_escaped_and_cannot_add_html_or_external_links():
    diagnosis, checks = fixture()
    diagnosis.endpoint_id = "<img src=x> [click](https://invalid.example)\n| forged |"
    checks[0].endpoint_id = diagnosis.endpoint_id
    diagnosis.alternatives = ["<script>private</script>"]
    text = issue_markdown(diagnosis, checks)
    assert "<img" not in text and "<script>" not in text
    assert "[click](https://invalid.example)" not in text
    assert "\n| forged |" not in text


def test_known_sensitive_values_are_sanitized_before_markdown_escaping():
    diagnosis, checks = fixture()
    marker = "test_only_sensitive_material_with_underscores"
    diagnosis.summary = "Captured label: " + marker
    text = issue_markdown(
        diagnosis, checks, sanitize=lambda value: value.replace(marker, "[REDACTED]")
    )
    assert marker not in text and marker.replace("_", "\\_") not in text
    assert "REDACTED" in text


@pytest.mark.parametrize("fault", ["missing", "cross-run", "cross-endpoint", "duplicate"])
def test_incomplete_or_ambiguous_witness_cannot_be_drafted(fault):
    diagnosis, checks = fixture()
    if fault == "missing":
        checks = []
    elif fault == "cross-run":
        checks[0].run_id = "other"
    elif fault == "cross-endpoint":
        checks[0].endpoint_id = "other"
    else:
        checks.append(checks[0].model_copy(deep=True))
    with pytest.raises(ValueError, match="issue_witness"):
        issue_markdown(diagnosis, checks)
