"""Pure, versioned expectations for reproducing a complete diagnosis witness."""

from typing import Any

from servegauge.core.models import CheckResult, Diagnosis
from servegauge.diagnose.rules import diagnoses

VERSION = "diagnosis-replay-v2"


def observation(check: CheckResult) -> str:
    if check.check_class == "report_only" and check.details.get("operational_alarm"):
        return "OPERATIONAL_ALARM"
    return check.verdict


def expected_context(diagnosis: Diagnosis, checks: list[CheckResult]) -> dict[str, Any]:
    if diagnosis not in diagnoses(checks):
        raise ValueError("unsupported_diagnosis_witness")
    wanted = set(diagnosis.check_ids)
    support = [
        c
        for c in checks
        if c.run_id == diagnosis.run_id
        and c.endpoint_id == diagnosis.endpoint_id
        and c.check_id in wanted
    ]
    expected = {c.check_id: observation(c) for c in support}
    if not wanted or set(expected) != wanted or len(support) != len(wanted):
        raise ValueError("unsupported_diagnosis_witness")
    allowed = {"FAIL", "PASS", "OPERATIONAL_ALARM"}
    if diagnosis.code == "QUANTIZATION_LIKE_DRIFT":
        allowed.add("INCONCLUSIVE")
    if any(v not in allowed for v in expected.values()):
        raise ValueError("unsupported_diagnosis_witness")
    return {
        "version": VERSION,
        "diagnosis_id": diagnosis.diagnosis_id,
        "code": diagnosis.code,
        "endpoint_id": diagnosis.endpoint_id,
        "check_ids": sorted(expected),
        "expected": dict(sorted(expected.items())),
    }


def replay_matches(checks: list[CheckResult], context: dict[str, Any], *, endpoint_id: str) -> bool:
    if context.get("version") != VERSION:
        return False
    expected = context.get("expected")
    allowed = {"FAIL", "PASS", "OPERATIONAL_ALARM"}
    if context.get("code") == "QUANTIZATION_LIKE_DRIFT":
        allowed.add("INCONCLUSIVE")
    if (
        not isinstance(expected, dict)
        or not expected
        or sorted(expected) != context.get("check_ids")
        or any(v not in allowed for v in expected.values())
    ):
        return False
    support = [c for c in checks if c.endpoint_id == endpoint_id and c.check_id in expected]
    if len(support) != len(expected) or len({c.run_id for c in support}) != 1:
        return False
    if any(observation(c) != expected[c.check_id] for c in support):
        return False
    return any(
        d.code == context.get("code")
        and d.endpoint_id == endpoint_id
        and d.run_id == support[0].run_id
        and set(expected) <= set(d.check_ids)
        for d in diagnoses(checks)
    )


def mark_reproduced(checks: list[CheckResult], context: dict[str, Any] | None) -> bool:
    if not context:
        return False
    if "version" not in context:
        # Old sealed captures retain their original FAIL-only interpretation.
        found = False
        for check in checks:
            if check.verdict == "FAIL" and check.check_id in context["check_ids"]:
                check.details["bundle_replay_reproduced"] = context["diagnosis_id"]
                found = True
        return found
    endpoint = context.get("endpoint_id")
    if not isinstance(endpoint, str) or not replay_matches(checks, context, endpoint_id=endpoint):
        return False
    for check in checks:
        if check.endpoint_id == endpoint and check.check_id in context["expected"]:
            check.details["bundle_replay_reproduced"] = context["diagnosis_id"]
    return True
