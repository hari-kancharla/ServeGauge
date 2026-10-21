"""Holm for confirmatory families; BH for explicitly exploratory, independent/PRDS tables."""

import math
from collections.abc import Sequence


def _validate(values: Sequence[float]) -> None:
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("finite_p_values_in_unit_interval_required")


def holm(values: Sequence[float]) -> list[float]:
    _validate(values)
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [1.0] * len(values)
    maximum = 0.0
    for rank, index in enumerate(order):
        maximum = max(maximum, (len(values) - rank) * values[index])
        result[index] = min(1.0, maximum)
    return result


def benjamini_hochberg(values: Sequence[float]) -> list[float]:
    _validate(values)
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [1.0] * len(values)
    minimum = 1.0
    for rank in range(len(order), 0, -1):
        index = order[rank - 1]
        minimum = min(minimum, values[index] * len(values) / rank)
        result[index] = minimum
    return result
