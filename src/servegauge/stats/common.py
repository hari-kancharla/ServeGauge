"""Finite numeric inputs and explicit probability-domain checks."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

type FloatArray = NDArray[np.float64]


def finite_array(values: ArrayLike, *, min_rows: int = 1) -> FloatArray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError):
        raise ValueError("finite_numeric_data_required") from None
    if result.ndim == 0 or len(result) < min_rows or not np.all(np.isfinite(result)):
        raise ValueError("nonempty_finite_numeric_data_required")
    return result


def probability(value: float, *, closed: bool = False) -> None:
    valid = 0 <= value <= 1 if closed else 0 < value < 1
    if not math.isfinite(value) or not valid:
        raise ValueError("probability_outside_valid_domain")


def paired_arrays(x: ArrayLike, y: ArrayLike) -> tuple[FloatArray, FloatArray]:
    a, b = finite_array(x), finite_array(y)
    if a.shape != b.shape:
        raise ValueError("paired_samples_must_have_identical_shape")
    return a, b
