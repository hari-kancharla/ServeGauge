"""Diagnose observed failures without consulting mock fault labels or asserting intent."""

from collections import defaultdict

from servegauge.core.hashing import content_hash
from servegauge.core.models import CheckResult, Diagnosis
from servegauge.diagnose.cache import cache_witness
from servegauge.diagnose.confidence import confidence
from servegauge.diagnose.drift import drift_witness
from servegauge.diagnose.taxonomy import ALTERNATIVES, CLASSES, FALSIFIERS
from servegauge.scoring.routing import ROUTING_CHECK


def diagnoses(checks: list[CheckResult]) -> list[Diagnosis]:
    groups: dict[tuple[str, str, str], list[CheckResult]] = defaultdict(list)
    passed = {
        (check.run_id, check.endpoint_id, check.check_id)
        for check in checks
        if check.verdict == "PASS"
    }
    controls = {
        (check.run_id, check.endpoint_id, check.check_id): check
        for check in checks
        if check.verdict == "PASS"
    }

    def parser_signature(check: CheckResult) -> bool:
        return check.check_id in {"T06", "P11"} or (
            check.check_id.startswith("T04.")
            and bool(
                set(check.details.get("failure_signatures", []))
                & {"call_count", "missing_stream_index"}
            )
        )

    parser_drop = {
        (check.run_id, check.endpoint_id)
        for check in checks
        if check.verdict == "FAIL" and parser_signature(check)
    }
    for check in checks:
        if check.verdict != "FAIL" and not (
            check.check_class == "report_only" and check.details.get("operational_alarm")
        ):
            continue
        name, code = check.check_id, None
        if name == "P03.cache":
            if "cache_contract" not in check.details:
                code = "CACHE_ANOMALY"
        elif name == ROUTING_CHECK and check.details.get("operational_alarm"):
            code = "ROUTING_INSTABILITY"
        elif (name == "S8.success" or name.startswith("S8.health.")) and check.details.get(
            "operational_alarm"
        ):
            code = "INFRA_UNRELIABLE"
        elif name.startswith("R03.") and any(
            run == check.run_id
            and e == check.endpoint_id
            and n.startswith("R03.")
            and float(n.split(".")[1]) < float(name.split(".")[1])
            for run, e, n in passed
        ):
            code = "LONG_OUTPUT_FAULT"
        elif name in {
            "P01",
            "P02",
            "P03",
            "P04.top_p",
            "P04.top_k",
            "P07",
            "P10",
            "P13",
        } or name.startswith("P09."):
            code = "PARAM_IGNORED"
        elif name == "T05":
            code = "PARSER_LEAK"
        elif parser_signature(check):
            code = "PARSER_DROP"
        elif name.startswith("L04"):
            code = "CONTEXT_TRUNCATION"
        elif name.startswith(("B01", "B02", "B03", "B04")):
            code = "ACCOUNTING_MISMATCH"
        elif (
            name.startswith("J02.validity")
            or name == "T02"
            and (check.run_id, check.endpoint_id, "T01") in passed
            and (check.run_id, check.endpoint_id) not in parser_drop
        ):
            code = "GUIDED_DECODING_ABSENT"
        elif (
            name.startswith("T07")
            or name == "T01"
            and (check.run_id, check.endpoint_id, "T02") in passed
        ):
            code = "TEMPLATE_MISMATCH"
        if code:
            groups[(check.run_id, check.endpoint_id, code)].append(check)
            prerequisite = None
            if code == "GUIDED_DECODING_ABSENT" and name == "T02":
                prerequisite = controls[(check.run_id, check.endpoint_id, "T01")]
            elif code == "TEMPLATE_MISMATCH" and name == "T01":
                prerequisite = controls[(check.run_id, check.endpoint_id, "T02")]
            elif code == "LONG_OUTPUT_FAULT":
                shorter = [
                    c
                    for (run, e, n), c in controls.items()
                    if run == check.run_id
                    and e == check.endpoint_id
                    and n.startswith("R03.")
                    and float(n.split(".")[1]) < float(name.split(".")[1])
                ]
                prerequisite = max(shorter, key=lambda c: float(c.check_id.split(".")[1]))
            if prerequisite is not None:
                groups[(check.run_id, check.endpoint_id, code)].append(prerequisite)
    scopes = {(check.run_id, check.endpoint_id) for check in checks}
    different_positions = {
        "different-model-band",
        "different-model-band-or-beyond",
        "beyond-different-model-band",
    }
    for run_id, endpoint in scopes:
        cached = cache_witness(
            [c for c in checks if c.run_id == run_id and c.endpoint_id == endpoint]
        )
        if cached:
            groups[(run_id, endpoint, "CACHE_ANOMALY")] = cached
        candidates = [
            c
            for c in checks
            if c.run_id == run_id and c.endpoint_id == endpoint and c.verdict == "FAIL"
        ]
        fingerprints = {
            c.check_id: c
            for c in candidates
            if c.check_id in {"FP1", "FP2", "FP3", "FP4"}
            and c.details.get("fingerprint_position") in different_positions
        }
        corroboration = [c for c in candidates if c.check_id.split(".")[0] in {"B05", "R01", "R02"}]
        if len(fingerprints) >= 2 and corroboration:
            groups[(run_id, endpoint, "DIFFERENT_MODEL_LIKE")] = [
                *fingerprints.values(),
                *corroboration,
            ]
        drift = drift_witness(
            [c for c in checks if c.run_id == run_id and c.endpoint_id == endpoint]
        )
        if drift:
            groups[(run_id, endpoint, "QUANTIZATION_LIKE_DRIFT")] = drift
    output = []
    for (run_id, endpoint, code), support in sorted(groups.items()):
        support = list({check.check_id: check for check in support}.values())
        identifiers = sorted({check.check_id for check in support})
        identity = [run_id, endpoint, code, identifiers]
        statements = []
        for check in support:
            effect = (
                f"{check.violations}/{check.n} violations, CI [{check.ci_low},{check.ci_high}]"
                if check.violations is not None
                else f"n={check.n}, delta={check.delta}, "
                f"CI [{check.delta_ci_low},{check.delta_ci_high}]"
            )
            statements.append(
                f"{check.check_id} ({check.verdict}): {effect}; reference "
                f"{check.reference_tier or 'invariant expectation'} / {check.reference_estimate}"
            )
        alternatives = ALTERNATIVES[code]
        if code == "CACHE_ANOMALY" and any("cache_contract" in c.details for c in support):
            alternatives = [
                *alternatives,
                "mixed routing or reference misspecification; no physical-cache attestation",
            ]
        level = confidence(support)
        if any(
            c.verdict == "PASS"
            and not (c.details.get("replay_confirmed") or c.details.get("bundle_replay_reproduced"))
            for c in support
        ):
            level = "low"
        if code == "DIFFERENT_MODEL_LIKE":
            reproduced = [
                c
                for c in support
                if c.details.get("replay_confirmed") or c.details.get("bundle_replay_reproduced")
            ]
            if len({c.check_id for c in reproduced if c.check_id.startswith("FP")}) < 2 or not any(
                c.check_id.split(".")[0] in {"B05", "R01", "R02"} for c in reproduced
            ):
                level = "low"
        if code == "QUANTIZATION_LIKE_DRIFT" and any(
            not (c.details.get("replay_confirmed") or c.details.get("bundle_replay_reproduced"))
            for c in support
        ):
            level = "low"
        disclosure = ""
        if code == "QUANTIZATION_LIKE_DRIFT":
            context = next(
                c.details["drift_context"] for c in support if c.check_id.startswith("FP")
            )
            label = context["disclosure"]
            disclosure = (
                f" Captured quantization declaration: {label['status']} / {label['label']}; "
                "not hardware verification."
            )
        output.append(
            Diagnosis(
                diagnosis_id="diagnosis-" + content_hash(identity)[:20],
                failure_class=CLASSES[code],
                code=code,
                endpoint_id=endpoint,
                run_id=run_id,
                confidence=level,
                summary="; ".join(statements)
                + f". {CLASSES[code]}; most likely code {code}. "
                + "Alternatives: "
                + "; ".join(alternatives)
                + "."
                + disclosure,
                evidence=sorted({identifier for check in support for identifier in check.evidence}),
                check_ids=identifiers,
                alternatives=alternatives,
                falsifiers=FALSIFIERS[code],
            )
        )
    return output
