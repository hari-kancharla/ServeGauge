"""Pure S5 reconstruction; repeats remain clustered by independent problem."""

import math
from collections import defaultdict

from servegauge.core.models import Exchange, Item
from servegauge.items.longout import LongOutput
from servegauge.scoring.common import Samples, choice, content
from servegauge.scoring.degeneration import degeneration
from servegauge.scoring.math_scoring import exact_math_answer
from servegauge.suites.config import suite_config


def reasoning_samples(
    items: list[Item], records: list[Exchange], endpoint_id: str
) -> dict[str, Samples]:
    by_item: dict[str, list[Exchange]] = defaultdict(list)
    for record in records:
        if record.endpoint_id == endpoint_id:
            by_item[record.item_id].append(record)
    output: dict[str, Samples] = {}
    for item in items:
        if item.suite_id != "S5":
            continue
        check = item.expected["check_id"]
        groups = (
            [check, "R05"]
            if check == "R01"
            else [
                check,
                *(
                    f"R04.{metric}.{check.split('.')[-1]}"
                    for metric in ("repeated_ngram", "compression_ratio", "longest_repeat")
                ),
            ]
        )
        for group in groups:
            output.setdefault(
                group,
                Samples(
                    check_id=group,
                    check_class="report_only" if group == "R05" else "comparative",
                    binary=group.startswith(("R01", "R03")),
                    two_sided=group.startswith("R04"),
                ),
            )
        captures = sorted(by_item[item.item_id], key=lambda r: r.sample_idx)
        for group in groups:
            sample = output[group]
            sample.evidence.extend(record.exchange_id for record in captures)
            if len(captures) != item.expected["samples"]:
                sample.mark("ERROR", "missing S5 captures")
        for record in captures:
            status = (
                "ERROR"
                if record.error
                else record.metadata.get("verdict")
                if record.metadata.get("request_not_sent")
                else None
            )
            if status:
                for group in groups:
                    output[group].mark(
                        status, "S5 transport or required output capability unavailable"
                    )
                continue
            try:
                if record.status != 200:
                    raise ValueError("unexpected_status")
                if choice(record).get("finish_reason") == "length":
                    for group in groups:
                        output[group].mark(
                            "INCONCLUSIVE", "configured output cap censored generation"
                        )
                    continue
                text = content(record)
                if check == "R01":
                    values = {check: float(exact_math_answer(text, item.expected["answer"]))}
                    counts = (record.usage or {}).get("completion_tokens_details")
                    count = counts.get("reasoning_tokens") if isinstance(counts, dict) else None
                    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                        values["R05"] = float(count)
                    else:
                        output["R05"].mark("INCONCLUSIVE", "reasoning token count unavailable")
                else:
                    case = LongOutput(
                        item.expected["start"],
                        item.expected["count"],
                        item.expected["kind"],
                        item.expected["full_text"],
                    )
                    values = {
                        check: float(case.matches(text)),
                        **{
                            f"R04.{metric}.{check.split('.')[-1]}": value
                            for metric, value in degeneration(
                                text, ngram=suite_config()["reasoning"]["degeneration_ngram"]
                            ).items()
                        },
                    }
                for group, value in values.items():
                    sample = output[group]
                    sample.repeats.setdefault(item.item_id, []).append(value)
                    sample.values[item.item_id] = math.fsum(sample.repeats[item.item_id]) / len(
                        sample.repeats[item.item_id]
                    )
                    sample.details["independent_unit"] = (
                        "problem" if check == "R01" else "generated long-output rule"
                    )
                    if check == "R01" and group == check:
                        sample.binary = False
                        sample.details["binary_repeats"] = sample.repeats
                        sample.details["metric"] = "average exact-match correctness over repeats"
            except (ValueError, KeyError, TypeError):
                for group in groups:
                    output[group].mark("ERROR", "invalid S5 wire envelope")
    return output
