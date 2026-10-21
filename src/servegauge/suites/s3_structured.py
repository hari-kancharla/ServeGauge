"""S3 JSON modes, own schema ladder, and deliberate output-cap interaction."""

import json
from typing import Any

from servegauge.core.models import Item, ModelProfile
from servegauge.items.schemas import LADDER, schema_case
from servegauge.suites.config import suite_config
from servegauge.tokencount.model_text import profile_codec


def structured_items(
    level: str,
    seed: int,
    overrides: dict[str, Any] | None = None,
    *,
    profile: ModelProfile | None = None,
) -> list[Item]:
    codec = profile_codec(profile)
    cfg = suite_config()["structured"]
    sizes = {**cfg[level], **(overrides or {})}
    if (
        set(sizes) != {"items"}
        or not isinstance(sizes["items"], int)
        or isinstance(sizes["items"], bool)
        or sizes["items"] < 1
    ):
        raise ValueError("positive_S3_item_count_required")
    items = []
    for index in range(sizes["items"]):
        case = schema_case(seed + index, LADDER[index % len(LADDER)])
        for check in ("J01", "J02", "J03"):
            payload: dict[str, Any] = {
                "messages": [{"role": "user", "content": case.prompt}],
                "params": {"max_tokens": cfg["output_cap"]},
                "stream": False,
            }
            if check == "J01":
                payload["response_format"] = {"type": "json_object"}
            if check == "J02":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "tw_generated", "strict": True, "schema": case.schema},
                }
            items.append(
                Item(
                    item_id=f"json-{seed}-{index}-{check}",
                    suite_id="S3",
                    version="json-ladder-v1",
                    payload=payload,
                    expected={
                        "check_id": check,
                        "trial": str(seed + index),
                        "samples": 1,
                        "category": case.category,
                        "value": case.value,
                        "schema": case.schema,
                        "numeric_tolerance": float(cfg["numeric_tolerance"]),
                        "requires": ["json_object"]
                        if check == "J01"
                        else ["json_schema"]
                        if check == "J02"
                        else [],
                        "allowed_statuses": [200, 400, 404, 422, 501],
                        "mock_task": {"type": "json", "value": case.value},
                    },
                    tags=["generated", case.category],
                )
            )
        value = {"records": list(range(seed + index, seed + index + cfg["truncation_records"]))}
        text = json.dumps(value, separators=(",", ":"))
        items.append(
            Item(
                item_id=f"json-{seed}-{index}-J05",
                suite_id="S3",
                version="json-truncation-glm-v1" if codec is not None else "json-truncation-v1",
                payload={
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Case {seed + index}. "
                            "Copy this JSON exactly, without extra text: " + text,
                        }
                    ],
                    "params": {"max_tokens": cfg["truncation_cap"]},
                    "stream": False,
                },
                expected={
                    "check_id": "J05",
                    "trial": str(seed + index),
                    "samples": 1,
                    "full_text": text,
                    "mock_task": {"type": "echo", "text": text},
                    **(
                        {
                            "text_counter_profile": codec.profile,
                            "text_counter_revision": codec.revision,
                        }
                        if codec is not None
                        else {}
                    ),
                },
                tags=["generated", "deliberate-truncation"],
            )
        )
    return items
