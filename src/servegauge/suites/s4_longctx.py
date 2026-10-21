"""S4 independent generated contexts with a declared length/depth grid."""

from __future__ import annotations

from typing import Any

from servegauge.core.hashing import content_hash
from servegauge.core.models import EndpointSpec, Item, ModelProfile
from servegauge.items.haystacks import context_case
from servegauge.suites.config import suite_config
from servegauge.tokencount.model_text import profile_codec


def context_items(
    level: str,
    seed: int,
    endpoints: list[EndpointSpec],
    overrides: dict[str, Any] | None = None,
    *,
    profile: ModelProfile | None = None,
) -> list[Item]:
    if profile is None and any(endpoint.claimed_profile == "glm-5.3" for endpoint in endpoints):
        raise ValueError("glm_context_requires_verified_profile")
    codec = profile_codec(profile)
    cfg = suite_config()["long_context"]
    design = {**cfg[level], **(overrides or {})}
    if set(design) != set(cfg[level]) or any(
        not isinstance(design[key], int) or isinstance(design[key], bool) or design[key] < 1
        for key in ("task_types", "repeats")
    ):
        raise ValueError("known_positive_context_design_required")
    lengths, depths = design["lengths"], design["depths"]
    if (
        not isinstance(lengths, list)
        or not lengths
        or any(
            not isinstance(length, int)
            or isinstance(length, bool)
            or not 1000 <= length <= cfg["maximum_probe_context"]
            for length in lengths
        )
    ):
        raise ValueError("bounded_context_lengths_required")
    if (
        not isinstance(depths, list)
        or not depths
        or any(
            not isinstance(depth, int) or isinstance(depth, bool) or not 0 <= depth <= 100
            for depth in depths
        )
    ):
        raise ValueError("bounded_context_depths_required")
    declared = [
        int(endpoint.metadata["context_length"])
        for endpoint in endpoints
        if isinstance(endpoint.metadata.get("context_length"), int)
    ]
    maximum = max(declared, default=cfg["maximum_probe_context"])
    lengths = sorted(set(lengths))
    if level == "full":
        final = min(maximum, cfg["maximum_probe_context"]) - cfg["output_headroom"]
        if final >= 1000:
            lengths = sorted({*lengths, final})
    items = []
    for length in lengths:
        # Retain an unsent placeholder for a known impossible length, without allocating its text.
        possible = length + cfg["output_cap"] <= maximum
        for repeat in range(design["repeats"] if possible else 1):
            for depth in depths if possible else [0]:
                for task_type in range(design["task_types"] if possible else 1):
                    named_seed = int(
                        content_hash([seed, length, depth, task_type, repeat])[:16], 16
                    )
                    case = (
                        context_case(
                            named_seed,
                            length=length,
                            depth=depth,
                            task_type=task_type,
                            aggregate_records=cfg["aggregate_records"],
                            padding_tokens=cfg["padding_tokens"],
                            maximum_excess=cfg["maximum_length_excess"],
                            iterations=cfg["construction_iterations"],
                            **({"codec": codec} if codec is not None else {}),
                        )
                        if possible
                        else None
                    )
                    items.append(
                        Item(
                            item_id=f"context-{length}-{named_seed}",
                            suite_id="S4",
                            version="context-glm-v1" if codec is not None else "context-v1",
                            payload={
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": case.text
                                        if case
                                        else "Requested input length exceeds declared context.",
                                    }
                                ],
                                "params": {"max_tokens": cfg["output_cap"]},
                                "stream": False,
                            },
                            expected={
                                "check_id": "context",
                                "length": length,
                                "depth": depth,
                                "trial": str(named_seed),
                                "samples": 1,
                                "minimum_context_tokens": (case.visible_tokens if case else length)
                                + cfg["output_cap"],
                                "visible_prompt_tokens": case.visible_tokens if case else None,
                                "needle_position_tokens": case.needle_position_tokens
                                if case
                                else None,
                                "answers": case.expected if case else {},
                                "mock_task": case.mock_task if case else None,
                                "allowed_statuses": [200, 400, 413, 422],
                                **(
                                    {
                                        "text_counter_profile": codec.profile,
                                        "text_counter_revision": codec.revision,
                                    }
                                    if codec is not None
                                    else {}
                                ),
                            },
                            tags=[
                                "generated",
                                "synthetic-prose" if task_type else "synthetic-records",
                            ],
                        )
                    )
    for index in range(cfg["prefix_cache_items"][level]):
        named_seed = int(content_hash([seed, "prefix", index])[:16], 16)
        length = cfg["prefix_cache_lengths"][level]
        possible = length + cfg["output_cap"] <= maximum
        case = (
            context_case(
                named_seed,
                length=length,
                depth=50,
                task_type=index % 2,
                aggregate_records=cfg["aggregate_records"],
                padding_tokens=cfg["padding_tokens"],
                maximum_excess=cfg["maximum_length_excess"],
                iterations=cfg["construction_iterations"],
                **({"codec": codec} if codec is not None else {}),
            )
            if possible
            else None
        )
        prefix = (
            case.text.partition("\nReturn one JSON object:")[0] if case else "Unavailable prefix."
        )
        for phase in ("prime", "fresh", "warm"):
            field = "single" if phase == "prime" else "multi2"
            key = case.mock_task["keys"][field] if case else "unavailable"
            exposure = "fresh" if phase == "fresh" else "shared"
            system = f"Synthetic prefix nonce {named_seed}-{exposure}.\n" + prefix
            question = f"Return only a JSON object mapping the record key {key} to its value."
            from servegauge.tokencount.counters import text_tokens

            count_text = codec.count if codec is not None else text_tokens
            visible = count_text(system) + count_text(question)
            items.append(
                Item(
                    item_id=f"prefix-{named_seed}-{phase}",
                    suite_id="S4",
                    version="prefix-exposure-glm-v1" if codec is not None else "prefix-exposure-v1",
                    payload={
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": question},
                        ],
                        "params": {"max_tokens": cfg["output_cap"]},
                        "stream": False,
                    },
                    expected={
                        "check_id": "L06",
                        "trial": str(named_seed),
                        "phase": phase,
                        "samples": 1,
                        "depends_on": [f"prefix-{named_seed}-prime"] if phase == "warm" else [],
                        "key": key,
                        "value": case.expected[field] if case else None,
                        "minimum_context_tokens": (visible if case else length) + cfg["output_cap"],
                        "visible_prompt_tokens": visible,
                        "allowed_statuses": [200, 400, 413, 422],
                        "mock_task": {"type": "retrieve", "keys": [key]},
                        "cache_cold_or_hit_verified": False,
                        **(
                            {
                                "text_counter_profile": codec.profile,
                                "text_counter_revision": codec.revision,
                            }
                            if codec is not None
                            else {}
                        ),
                    },
                    tags=["generated", "prefix-exposure", "native-cache-state-unverified"],
                )
            )
    return items
