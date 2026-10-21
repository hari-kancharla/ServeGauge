"""S5 independent code-verified math and bounded long structured outputs."""

from typing import Any

from servegauge.core.models import Item, ModelProfile
from servegauge.items.longout import long_output
from servegauge.items.mathgen import math_problem
from servegauge.suites.config import suite_config
from servegauge.tokencount.counters import text_tokens
from servegauge.tokencount.model_text import profile_codec


def reasoning_items(
    level: str,
    seed: int,
    overrides: dict[str, Any] | None = None,
    *,
    profile: ModelProfile | None = None,
) -> list[Item]:
    codec = profile_codec(profile)
    cfg = suite_config()["reasoning"]
    design = {**cfg[level], **(overrides or {})}
    if (
        set(design) != set(cfg[level])
        or any(
            not isinstance(design[k], int) or isinstance(design[k], bool) or design[k] < 1
            for k in ("items", "samples", "long_items")
        )
        or not isinstance(design["lengths"], list)
        or not design["lengths"]
        or any(
            not isinstance(n, int) or isinstance(n, bool) or n not in cfg["allowed_lengths"]
            for n in design["lengths"]
        )
    ):
        raise ValueError("known_positive_reasoning_design_required")
    items = []
    for index in range(design["items"]):
        problem = math_problem(seed + index, tier=index % 3 + 1)
        items.append(
            Item(
                item_id=f"math-{seed}-{index}",
                suite_id="S5",
                version="math-v1",
                payload={
                    "messages": [{"role": "user", "content": problem.prompt}],
                    "params": {"max_tokens": cfg["math_output_cap"], "reasoning_effort": "high"},
                },
                expected={
                    "check_id": "R01",
                    "samples": design["samples"],
                    "trial": str(index),
                    "answer": problem.answer,
                    "tier": index % 3 + 1,
                    "mock_task": {"type": "math", "operation": problem.kind, **problem.parameters},
                },
            )
        )
    for length in sorted(set(design["lengths"])):
        for index in range(design["long_items"]):
            kind = "json" if index % 2 == 0 else "sequence"
            case = long_output(
                seed + index,
                length,
                kind,
                **({"text_counter": codec.count} if codec is not None else {}),
            )
            prompt = (
                f"Produce exactly {case.count} records, starting with integer {case.start}. "
                + (
                    "Output only a JSON array of objects with integer i and square=i*i, "
                    "in ascending i order."
                    if kind == "json"
                    else "Output one line per integer value in ascending order. "
                    "Line index starts at zero. "
                    "Each line is 'INDEX_PLUS_ONE: VALUE; checksum=CHECK', "
                    "with CHECK=(VALUE*17+INDEX)%997. "
                    "No extra text."
                )
            )
            count_text = codec.count if codec is not None else text_tokens
            needed = count_text(case.text) + cfg["output_headroom"]
            items.append(
                Item(
                    item_id=f"long-{seed}-{length}-{index}",
                    suite_id="S5",
                    version="long-glm-v1" if codec is not None else "long-v1",
                    payload={
                        "messages": [{"role": "user", "content": prompt}],
                        "params": {"max_tokens": needed},
                        "stream": False,
                    },
                    expected={
                        "check_id": f"R03.{length}",
                        "samples": 1,
                        "trial": str(index),
                        "minimum_output_tokens": needed,
                        "full_text": case.text,
                        "kind": kind,
                        "start": case.start,
                        "count": case.count,
                        "mock_task": {
                            "type": "long_output",
                            "kind": kind,
                            "start": case.start,
                            "count": case.count,
                        },
                        **(
                            {
                                "text_counter_profile": codec.profile,
                                "text_counter_revision": codec.revision,
                                "visible_output_tokens": count_text(case.text),
                            }
                            if codec is not None
                            else {}
                        ),
                    },
                )
            )
    return items
