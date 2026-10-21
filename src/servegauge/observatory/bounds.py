"""Directed numerical enclosures for inactive temporal diagnostics."""

from __future__ import annotations

import math
from decimal import ROUND_CEILING, Context, Decimal, localcontext
from fractions import Fraction
from functools import lru_cache


def _directed_float(value: Fraction, *, upward: bool) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("finite_directed_conversion_required")
    represented = Fraction.from_float(result)
    if (upward and represented < value) or (not upward and represented > value):
        result = math.nextafter(result, math.inf if upward else -math.inf)
    return result


def floor_fraction(value: Fraction) -> float:
    return _directed_float(value, upward=False)


def _counts(left: int, right: int) -> None:
    if type(left) is not int or type(right) is not int or min(left, right) <= 0:
        raise ValueError("positive_integer_cut_counts_required")


@lru_cache(maxsize=32768)
def _parameters(left: int, right: int, delta: float) -> tuple[float, int]:
    # Decimal ln/sqrt use correctly rounded results. Step upward after each
    # transcendental operation; directed basic operations enclose their inputs.
    with localcontext(Context(prec=80, rounding=ROUND_CEILING)):
        ratio = Fraction(4 * (left + right)) / Fraction.from_float(delta)
        logarithm = (Decimal(ratio.numerator) / Decimal(ratio.denominator)).ln().next_plus()
        factor = Fraction(1, 2) * (Fraction(1, left) + Fraction(1, right))
        upper = (
            ((Decimal(factor.numerator) / Decimal(factor.denominator)) * logarithm)
            .sqrt()
            .next_plus()
        )
        radius = _directed_float(Fraction(upper), upward=True)
    cross_product = Fraction.from_float(radius) * left * right
    return radius, cross_product.numerator // cross_product.denominator


def _checked(left: int, right: int, delta: float) -> tuple[float, int]:
    _counts(left, right)
    if not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("finite_open_probability_required")
    # Validate before the cache: bool/int equality cannot bypass count checks.
    return _parameters(left, right, float(delta))


def upper_cut_bound(left: int, right: int, delta: float) -> float:
    return _checked(left, right, delta)[0]


def cut_limit(left: int, right: int, delta: float) -> int:
    """Integer cross-product boundary, avoiding rounded sample-mean decisions."""
    return _checked(left, right, delta)[1]


def effect_interval(
    left_successes: int, left: int, right_successes: int, right: int, radius: float
) -> tuple[float, float, float]:
    _counts(left, right)
    if (
        type(left_successes) is not int
        or type(right_successes) is not int
        or not 0 <= left_successes <= left
        or not 0 <= right_successes <= right
        or not math.isfinite(radius)
        or radius < 0
    ):
        raise ValueError("bounded_integer_successes_and_finite_radius_required")
    difference = Fraction(right_successes, right) - Fraction(left_successes, left)
    bound = Fraction.from_float(radius)
    low = max(Fraction(-1), difference - bound)
    high = min(Fraction(1), difference + bound)
    return float(difference), floor_fraction(low), _directed_float(high, upward=True)
