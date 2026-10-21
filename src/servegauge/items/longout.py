"""Own deterministic rules and exact checkers for long generation."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from servegauge.tokencount.counters import text_tokens


@lru_cache(maxsize=512)
def render_output(start: int, count: int, kind: str) -> str:
    if kind == "json":
        return json.dumps(
            [{"i": i, "square": i * i} for i in range(start, start + count)], separators=(",", ":")
        )
    if kind != "sequence":
        raise ValueError("unknown_long_output_rule")
    return "\n".join(
        f"{index + 1}: {value}; checksum={(value * 17 + index) % 997}"
        for index, value in enumerate(range(start, start + count))
    )


@dataclass(frozen=True)
class LongOutput:
    start: int
    count: int
    kind: str
    text: str

    def matches(self, observed: str) -> bool:
        if self.kind == "json":
            try:
                value = json.loads(observed)
            except (ValueError, TypeError):
                return False
            # Canonical text also distinguishes booleans and floats from exact integers.
            return json.dumps(value, separators=(",", ":")) == self.text
        return observed.strip() == self.text


def long_output(
    seed: int, target: int, kind: str, *, text_counter: Callable[[str], int] | None = None
) -> LongOutput:
    count_text = text_counter if text_counter is not None else text_tokens
    start = 1000 + seed % 9000
    low, high = 1, target
    while low < high:
        middle = (low + high) // 2
        if count_text(render_output(start, middle, kind)) < target:
            low = middle + 1
        else:
            high = middle
    return LongOutput(start, low, kind, render_output(start, low, kind))
