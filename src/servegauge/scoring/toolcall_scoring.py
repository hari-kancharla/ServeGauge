"""Deterministic tool-call metrics and paired trigger-F1 inference."""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from servegauge.core.json import strict_json
from servegauge.stats.bootstrap import clustered_bootstrap
from servegauge.stats.common import FloatArray, paired_arrays
from servegauge.stats.intervals import wilson
from servegauge.stats.tests import two_sample_permutation


def _flags(labels: ArrayLike, predictions: ArrayLike) -> tuple[FloatArray, FloatArray]:
    a, b = paired_arrays(labels, predictions)
    if a.ndim != 1 or not np.all(np.isin(a, [0, 1])) or not np.all(np.isin(b, [0, 1])):
        raise ValueError("paired_binary_classifier_flags_required")
    return a, b


def _f1(labels: FloatArray, predictions: FloatArray) -> float:
    denominator = float(np.sum(labels) + np.sum(predictions))
    return 2 * float(np.sum(labels * predictions)) / denominator if denominator else 0.0


def trigger_counts(labels: ArrayLike, predictions: ArrayLike) -> dict[str, int | float]:
    a, b = _flags(labels, predictions)
    tp = int(np.sum((a == 1) & (b == 1)))
    fp = int(np.sum((a == 0) & (b == 1)))
    fn = int(np.sum((a == 1) & (b == 0)))
    return {
        "n": len(a),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "tool_call_f1": _f1(a, b),
    }


def f1_projection(
    labels: ArrayLike, predictions: ArrayLike, *, alpha: float
) -> tuple[float, float]:
    """Monotone F1 projection of simultaneous Wilson precision/recall intervals.

    Two intervals each spend alpha/2. A zero denominator remains [0,1], rather
    than creating certainty about a capability that was never observed.
    """
    counts = trigger_counts(labels, predictions)
    tp, fp, fn = (int(counts[k]) for k in ("tp", "fp", "fn"))
    precision = wilson(tp, tp + fp, alpha=alpha / 2) if tp + fp else (0.0, 1.0)
    recall = wilson(tp, tp + fn, alpha=alpha / 2) if tp + fn else (0.0, 1.0)

    def harmonic(p: float, r: float) -> float:
        return 2 * p * r / (p + r) if p + r else 0.0

    return harmonic(precision[0], recall[0]), harmonic(precision[1], recall[1])


def f1_reference_statistic(rows: FloatArray) -> float:
    return _f1(rows[:, 0], rows[:, 1])


def f1_influence(labels: ArrayLike, predictions: ArrayLike) -> FloatArray:
    a, b = _flags(labels, predictions)
    denominator = float(np.mean(a + b))
    if not denominator:
        return np.zeros_like(a)
    return (2 * a * b - _f1(a, b) * (a + b)) / denominator


@dataclass(frozen=True)
class F1Contrast:
    estimate: float
    reference_estimate: float
    interval: tuple[float, float]
    reference_interval: tuple[float, float]
    delta_interval: tuple[float, float]
    one_sided_bounds: tuple[float, float]
    p_value: float
    paired_variance: float
    n: int
    resamples: int


def paired_f1(
    labels: ArrayLike,
    endpoint: ArrayLike,
    reference: ArrayLike,
    *,
    seed: int,
    alpha: float,
    resamples: int = 2000,
    permutations: int = 2000,
) -> F1Contrast:
    a, b = _flags(labels, endpoint)
    _, c = _flags(labels, reference)
    matrix = np.column_stack((a, b, c))

    def endpoint_metric(rows: FloatArray) -> float:
        return _f1(rows[:, 0], rows[:, 1])

    def reference_metric(rows: FloatArray) -> float:
        return _f1(rows[:, 0], rows[:, 2])

    def difference(rows: FloatArray) -> float:
        return endpoint_metric(rows) - reference_metric(rows)

    def metric_interval(column: int) -> tuple[float, float]:
        result = clustered_bootstrap(
            matrix,
            seed=seed,
            resamples=resamples,
            alpha=alpha,
            statistic=endpoint_metric if column == 1 else reference_metric,
        )
        low, high = f1_projection(a, matrix[:, column], alpha=alpha)
        return min(low, result.low), max(high, result.high)

    def difference_interval(level: float) -> tuple[float, float]:
        result = clustered_bootstrap(
            matrix, seed=seed, resamples=resamples, alpha=level, statistic=difference
        )
        endpoint_bounds = f1_projection(a, b, alpha=level / 2)
        reference_bounds = f1_projection(a, c, alpha=level / 2)
        return min(result.low, endpoint_bounds[0] - reference_bounds[1]), max(
            result.high, endpoint_bounds[1] - reference_bounds[0]
        )

    def permutation_statistic(first: FloatArray, second: FloatArray) -> float:
        return f1_reference_statistic(first) - f1_reference_statistic(second)

    permutation = two_sample_permutation(
        np.column_stack((a, b)),
        np.column_stack((a, c)),
        permutation_statistic,
        seed=seed,
        resamples=permutations,
        paired=True,
        alternative="less",
    )
    influence = f1_influence(a, b) - f1_influence(a, c)
    variance = float(np.var(influence, ddof=1)) if len(a) > 1 else 1.0
    return F1Contrast(
        _f1(a, b),
        _f1(a, c),
        metric_interval(1),
        metric_interval(2),
        difference_interval(alpha),
        difference_interval(2 * alpha),
        permutation.p_value,
        variance,
        len(a),
        resamples,
    )


def arguments_match(actual: Any, expected: Any, *, numeric_tolerance: float) -> bool:
    """Types, Unicode NFC and numerical tolerance are declared by the generated item."""
    if isinstance(expected, bool) or expected is None:
        return actual is expected
    if isinstance(expected, (int, float)):
        try:
            return (
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and math.isfinite(actual)
                and math.isclose(
                    actual, expected, rel_tol=numeric_tolerance, abs_tol=numeric_tolerance
                )
            )
        except OverflowError:
            return False
    if isinstance(expected, str):
        return isinstance(actual, str) and unicodedata.normalize(
            "NFC", actual
        ) == unicodedata.normalize("NFC", expected)
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                arguments_match(x, y, numeric_tolerance=numeric_tolerance)
                for x, y in zip(actual, expected, strict=True)
            )
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(
                arguments_match(actual[k], v, numeric_tolerance=numeric_tolerance)
                for k, v in expected.items()
            )
        )
    return False


def parsed_call(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    function = call.get("function")
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        raise ValueError("invalid_function_call")
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise ValueError("arguments_must_be_a_JSON_string")
    try:
        value = strict_json(arguments)
    except ValueError:
        raise ValueError("invalid_argument_JSON") from None
    if not isinstance(value, dict):
        raise ValueError("arguments_must_be_object")
    return function["name"], value
