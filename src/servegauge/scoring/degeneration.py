"""Deterministic descriptive metrics; reference calibration supplies decision bands."""

import zlib
from functools import lru_cache

import numpy as np


def _longest_repeat(text: str) -> int:
    # Suffix ranks, doubled each pass. No hash collisions or quadratic string copies.
    n = len(text)
    if n < 2:
        return 0
    ranks = np.asarray([ord(char) for char in text], dtype=np.int64)
    width, longest = 1, 0
    while width < n:
        second = np.full(n, -1, dtype=np.int64)
        second[:-width] = ranks[width:]
        order = np.lexsort((second, ranks))
        same = (ranks[order[1:]] == ranks[order[:-1]]) & (second[order[1:]] == second[order[:-1]])
        if not np.any(same):
            break
        new = np.empty(n, dtype=np.int64)
        new[order] = np.r_[0, np.cumsum(~same)]
        ranks = new
        width *= 2
    # Kasai LCP from the final suffix ordering, including overlapping substrings.
    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = np.arange(n)
    height = 0
    for index in range(n):
        position = int(inverse[index])
        if position == 0:
            height = 0
            continue
        other = int(order[position - 1])
        while (
            index + height < n
            and other + height < n
            and text[index + height] == text[other + height]
        ):
            height += 1
        longest = max(longest, height)
        height = max(0, height - 1)
    return longest


def degeneration(text: str, *, ngram: int) -> dict[str, float]:
    if ngram < 1:
        raise ValueError("positive_ngram_required")
    return dict(
        zip(
            ("repeated_ngram", "compression_ratio", "longest_repeat"),
            _metrics(text, ngram),
            strict=True,
        )
    )


@lru_cache(maxsize=1024)
def _metrics(text: str, ngram: int) -> tuple[float, float, float]:
    # Pure deterministic work. A fresh dictionary prevents caller mutation.
    grams = [text[i : i + ngram] for i in range(max(0, len(text) - ngram + 1))]
    raw = text.encode("utf-8")
    return (
        1 - len(set(grams)) / len(grams) if grams else 0.0,
        len(zlib.compress(raw)) / len(raw) if raw else 0.0,
        _longest_repeat(text) / len(text) if text else 0.0,
    )
