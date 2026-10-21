"""Own token-budgeted synthetic records and prose, with no external text or oracle leakage."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from servegauge.core.hashing import content_hash
from servegauge.tokencount.counters import decode, encode, text_tokens
from servegauge.tokencount.model_text import TextCodec


@dataclass(frozen=True)
class ContextCase:
    seed: int
    length: int
    depth: int
    text: str
    expected: dict[str, Any]
    mock_task: dict[str, Any]
    visible_tokens: int
    needle_position_tokens: int


def context_case(
    seed: int,
    *,
    length: int,
    depth: int,
    task_type: int,
    aggregate_records: int,
    padding_tokens: int,
    maximum_excess: int,
    iterations: int,
    codec: TextCodec | None = None,
) -> ContextCase:
    if (
        length < 1000
        or not 0 <= depth <= 100
        or task_type not in {0, 1}
        or aggregate_records < 1
        or padding_tokens < 0
        or maximum_excess < padding_tokens
    ):
        raise ValueError("invalid_context_design")
    encode_text = codec.encode if codec is not None else encode
    decode_tokens = codec.decode if codec is not None else decode
    count_text = codec.count if codec is not None else text_tokens
    rng = random.Random(seed)
    tag = content_hash(seed)[:8]
    names = {
        name: prefix + "_" + tag
        for name, prefix in (
            ("single", "S"),
            ("multi1", "M1"),
            ("multi2", "M2"),
            ("track", "V"),
            ("head", "H"),
            ("tail", "T"),
        )
    }
    values = {name: f"v{rng.getrandbits(32):08x}" for name in names}
    old_value = f"v{rng.getrandbits(32):08x}"
    numbers = [rng.randint(1, 9) for _ in range(aggregate_records)]
    aggregate_prefix = "A_" + tag + "_"
    instruction = "Read these generated records. Each KEY=VALUE line is literal data.\n"
    question = (
        "\nReturn one JSON object: single is "
        + names["single"]
        + ", multi1 is "
        + names["multi1"]
        + ", multi2 is "
        + names["multi2"]
        + ", track is the final assignment of "
        + names["track"]
        + ", total is the sum of all integer records starting "
        + aggregate_prefix
        + ", head is "
        + names["head"]
        + ", tail is "
        + names["tail"]
        + ". Use null only for a record you cannot find. Return no other text.\n"
    )
    markers = [
        (depth / 100, f"{names['single']}={values['single']}"),
        (0.1, f"{names['multi1']}={values['multi1']}"),
        (0.9, f"{names['multi2']}={values['multi2']}"),
        (0.25, f"{names['track']}={old_value}"),
        (0.75, f"{names['track']}={values['track']}"),
    ]
    markers.extend(
        ((index + 1) / (aggregate_records + 1), f"{aggregate_prefix}{index}={number}")
        for index, number in enumerate(numbers)
    )
    markers.sort()
    head = instruction + names["head"] + "=" + values["head"] + "\n"
    tail = question + names["tail"] + "=" + values["tail"] + "\n"

    def compose(tokens: list[int]) -> str:
        parts = [head]
        previous = 0
        for fraction, record in markers:
            stop = round(fraction * len(tokens))
            parts.extend((decode_tokens(tokens[previous:stop]), "\n", record, "\n"))
            previous = stop
        parts.extend((decode_tokens(tokens[previous:]), tail))
        return "".join(parts)

    # Generated background contains no KEY=VALUE records and never includes answers.
    source = []
    for index in range(length // 8 + 256):
        a, b = rng.randrange(10000), rng.randrange(10000)
        source.append(
            f"Log {index}: unit {a} recorded metric {b}.\n"
            if task_type == 0
            else f"The synthetic lantern {index} crossed sector {a} beside depot {b}.\n"
        )
    pool = encode_text("".join(source))
    overhead = count_text(compose([]))
    count = length + padding_tokens - overhead
    text, visible = "", 0
    for _ in range(iterations):
        if not 0 <= count <= len(pool):
            raise ValueError("context_filler_pool_outside_design")
        text = compose(pool[:count])
        visible = count_text(text)
        if length <= visible <= length + maximum_excess:
            break
        count += length + padding_tokens - visible
    else:
        raise ValueError("context_length_construction_did_not_converge")
    marker = names["single"] + "=" + values["single"]
    expected: dict[str, Any] = {**values, "total": sum(numbers)}
    task = {"type": "context", "keys": names, "aggregate_prefix": aggregate_prefix}
    return ContextCase(
        seed, length, depth, text, expected, task, visible, count_text(text[: text.index(marker)])
    )
