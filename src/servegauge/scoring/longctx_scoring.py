"""S4 retrieval/canary observations and an explicitly discrete effective-context summary."""

from __future__ import annotations

from collections import defaultdict

from servegauge.core.models import CheckResult, Exchange, Item
from servegauge.scoring.common import Samples, choice
from servegauge.scoring.json_scoring import parsed_object
from servegauge.stats.intervals import paired_difference_interval
from servegauge.verdicts.config import DecisionPolicy


def context_samples(
    items: list[Item], records: list[Exchange], endpoint_id: str, policy: DecisionPolicy
) -> dict[str, Samples]:
    by_item = {record.item_id: record for record in records if record.endpoint_id == endpoint_id}
    output: dict[str, Samples] = {}
    for item in items:
        if item.suite_id != "S4" or item.expected["check_id"] != "context":
            continue
        length = item.expected["length"]
        keys = [f"L{index:02}.{length}" for index in (1, 2, 3, 5)]
        keys += ["L04", f"L04.{length}"]
        for key in keys:
            output.setdefault(
                key,
                Samples(
                    check_id=key,
                    check_class="invariant" if key.startswith("L04") else "comparative",
                ),
            )
        record = by_item.get(item.item_id)
        if record is None or record.error:
            for key in keys:
                output[key].mark("ERROR", "context capture missing or transport/excluded routing")
            continue
        for key in keys:
            output[key].evidence.append(record.exchange_id)
        if record.metadata.get("request_not_sent") or record.status != 200:
            status = record.metadata.get("verdict", "INCONCLUSIVE")
            reason = record.metadata.get(
                "skip_reason", "context request rejected; template overhead may differ"
            )
            for key in keys:
                if key == "L04":
                    output[key].details.setdefault("unmeasured_items", {})[item.item_id] = {
                        "status": status,
                        "reason": reason,
                    }
                else:
                    output[key].mark(status, reason)
            continue
        try:
            censored = choice(record).get("finish_reason") == "length"
            actual = parsed_object(record)
        except (TypeError, ValueError):
            for key in keys:
                output[key].mark("ERROR", "invalid context response envelope")
            continue
        if censored:
            for key in keys:
                if key.startswith("L04"):
                    output[key].details["incomplete_trials"] = (
                        output[key].details.get("incomplete_trials", 0) + 1
                    )
                else:
                    output[key].mark("INCONCLUSIVE", "output cap censored context behavior")
            continue
        expected = item.expected["answers"]
        parsed = actual or {}
        outcomes = {
            f"L01.{length}": parsed.get("single") == expected["single"],
            f"L02.{length}": all(
                parsed.get(field) == expected[field] for field in ("multi1", "multi2", "track")
            ),
            f"L03.{length}": isinstance(parsed.get("total"), int)
            and not isinstance(parsed.get("total"), bool)
            and parsed.get("total") == expected["total"],
        }
        canary_failure = (
            parsed.get("head") != expected["head"] or parsed.get("tail") != expected["tail"]
        )
        outcomes[f"L05.{length}"] = all(outcomes.values()) and not canary_failure
        trial = item.item_id
        for key, value in outcomes.items():
            output[key].values[trial] = float(value)
            output[key].details.update(length=length, units="task success probability")
        for key in ("L04", f"L04.{length}"):
            if actual is None:
                output[key].details["incomplete_trials"] = (
                    output[key].details.get("incomplete_trials", 0) + 1
                )
            else:
                output[key].values[trial] = float(canary_failure)
                if canary_failure:
                    output[key].replay_candidates.append([item.item_id])
            output[key].details.setdefault("canary_observations", {})[trial] = {
                "length": length,
                "head_returned": parsed.get("head") == expected["head"],
                "tail_returned": parsed.get("tail") == expected["tail"],
                "valid_JSON_object": actual is not None,
            }
    if "L04" in output and not output["L04"].values and output["L04"].status is None:
        output["L04"].mark("INCONCLUSIVE", "no complete supported canary observations")
    output.update(prefix_samples(items, records, endpoint_id, policy))
    return output


def prefix_samples(
    items: list[Item], records: list[Exchange], endpoint_id: str, policy: DecisionPolicy
) -> dict[str, Samples]:
    groups: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        if item.suite_id == "S4" and item.expected["check_id"] == "L06":
            groups[item.expected["trial"]].append(item)
    if not groups:
        return {}
    output = {
        key: Samples(check_id=key, check_class="comparative")
        for key in ("L06", "L06.fresh", "L06.warm")
    }
    by_item = {record.item_id: record for record in records if record.endpoint_id == endpoint_id}
    fresh, warm = [], []
    for trial, probes in groups.items():
        captured = {item.expected["phase"]: by_item.get(item.item_id) for item in probes}
        if set(captured) != {"prime", "fresh", "warm"} or any(
            record is None for record in captured.values()
        ):
            for sample in output.values():
                sample.mark("ERROR", "prefix phase capture missing")
            continue
        available = [record for record in captured.values() if record is not None]
        for sample in output.values():
            sample.evidence.extend(record.exchange_id for record in available)
        prime = captured["prime"]
        warmed = captured["warm"]
        assert prime is not None and warmed is not None
        if any(record.error for record in available):
            for sample in output.values():
                sample.mark("ERROR", "prefix phase transport/excluded routing")
            continue
        fresh_record = captured["fresh"]
        assert fresh_record is not None
        first_messages = prime.request.get("messages", [])
        warm_messages = warmed.request.get("messages", [])
        fresh_messages = fresh_record.request.get("messages", [])
        if (
            prime.exchange_id not in warmed.metadata.get("completed_dependencies", [])
            or min(len(first_messages), len(warm_messages), len(fresh_messages)) < 2
            or first_messages[:-1] != warm_messages[:-1]
            or first_messages[-1] == warm_messages[-1]
            or fresh_messages[-1] != warm_messages[-1]
            or fresh_messages[:-1] == warm_messages[:-1]
        ):
            for sample in output.values():
                sample.mark("INCONCLUSIVE", "completed shared-prefix exposure is unverified")
            continue
        if any(
            record.metadata.get("request_not_sent") or record.status != 200 for record in available
        ):
            for sample in output.values():
                sample.mark("INCONCLUSIVE", "prefix exposure was not completely observed")
            continue
        try:
            if any(choice(record).get("finish_reason") == "length" for record in available):
                for sample in output.values():
                    sample.mark("INCONCLUSIVE", "output cap censored prefix phases")
                continue
            scores = {}
            for item in probes:
                record = by_item[item.item_id]
                value = parsed_object(record) or {}
                scores[item.expected["phase"]] = float(
                    value.get(item.expected["key"]) == item.expected["value"]
                )
            first, second = scores["fresh"], scores["warm"]
            fresh.append(first)
            warm.append(second)
            output["L06"].values[trial] = float(first == second == 1)
            output["L06.fresh"].values[trial] = first
            output["L06.warm"].values[trial] = second
        except (ValueError, TypeError, KeyError, IndexError):
            for sample in output.values():
                sample.mark("ERROR", "invalid prefix response envelope")
    for sample in output.values():
        sample.details.update(
            metric="both final questions correct"
            if sample.check_id == "L06"
            else "phase correctness",
            exposure="shared prefix completes before the warm request with a different question",
            native_cache_state="unverified; fresh prefix is unique in this plan",
        )
    if fresh:
        output["L06"].details["within_endpoint_warm_minus_fresh"] = {
            "n": len(fresh),
            "estimate": (sum(warm) - sum(fresh)) / len(fresh),
            "interval": list(paired_difference_interval(warm, fresh, alpha=policy.alpha)),
            "reference_tier": None,
            "reference_estimate": None,
            "verdict": "INFO",
        }
    return output


def effective_context(results: list[CheckResult]) -> list[CheckResult]:
    grouped: dict[tuple[str, str], list[CheckResult]] = defaultdict(list)
    for result in results:
        if result.check_id.startswith("L05."):
            grouped[(result.endpoint_id, result.run_id)].append(result)
    summaries = []
    for (endpoint, run_id), checks in grouped.items():
        passing = [int(check.check_id.split(".")[1]) for check in checks if check.verdict == "PASS"]
        largest = max(passing) if passing else None
        failed_above = [
            int(check.check_id.split(".")[1])
            for check in checks
            if check.verdict == "FAIL"
            and largest is not None
            and int(check.check_id.split(".")[1]) > largest
        ]
        references = [
            int(check.check_id.split(".")[1])
            for check in checks
            if check.reference_estimate is not None
        ]
        summaries.append(
            CheckResult(
                check_id="L05",
                check_class="report_only",
                suite_id="S4",
                endpoint_id=endpoint,
                run_id=run_id,
                n=sum(check.n for check in checks),
                verdict="INFO" if largest is not None else "INCONCLUSIVE",
                estimate=largest,
                ci_low=None,
                ci_high=None,
                reference_tier=next(
                    (check.reference_tier for check in checks if check.reference_tier), None
                ),
                reference_estimate=max(references) if references else None,
                reason="largest tested input length non-inferior to stated reference; grid bracket"
                if largest is not None
                else "no input length qualified at this n/reference",
                evidence=sorted({evidence for check in checks for evidence in check.evidence}),
                details={
                    "interval_kind": "tested grid bracket; no parameter confidence interval",
                    "grid_interval": [largest, min(failed_above) if failed_above else None],
                    "right_censored": not failed_above,
                    "units": "visible input tokens (nominal grid)",
                    "per_length": [
                        {
                            "check_id": check.check_id,
                            "n": check.n,
                            "estimate": check.estimate,
                            "interval": [check.ci_low, check.ci_high],
                            "reference_tier": check.reference_tier,
                            "reference_estimate": check.reference_estimate,
                            "verdict": check.verdict,
                        }
                        for check in checks
                    ],
                },
            )
        )
    return summaries
