"""Independent support requires different checks and disjoint captured evidence."""

from typing import Literal

from servegauge.core.models import CheckResult


def confidence(checks: list[CheckResult]) -> Literal["low", "medium", "high"]:
    reproduced = [
        check
        for check in checks
        if (check.verdict == "FAIL" or check.details.get("operational_alarm"))
        and (check.details.get("replay_confirmed") or check.details.get("bundle_replay_reproduced"))
    ]
    if not reproduced:
        return "low"
    for first in reproduced:
        for second in reproduced:
            if (
                first.run_id == second.run_id
                and first.endpoint_id == second.endpoint_id
                and first.check_id.split(".")[0] != second.check_id.split(".")[0]
                and first.evidence
                and second.evidence
                and set(first.evidence).isdisjoint(second.evidence)
            ):
                return "high"
    return "medium"
