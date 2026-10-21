"""Conservative machine arithmetic before any temporal admission is possible."""

import math
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from servegauge.observatory.bounds import cut_limit, effect_interval, upper_cut_bound
from servegauge.observatory.rolling import load_rolling_policy


def test_all_supported_allocations_are_downward_rounded_and_tight():
    base = load_rolling_policy()
    for streams in range(1, 101):
        for updates in range(1, 121):
            policy = base.model_copy(update={"max_streams": streams, "max_updates": updates})
            exact = min(
                Fraction(str(policy.max_false_events_30_days)),
                Fraction(str(policy.family_alpha)),
            ) / (streams * updates)
            actual = Fraction.from_float(policy.update_budget)
            assert 0 < actual <= exact
            assert Fraction.from_float(math.nextafter(policy.update_budget, math.inf)) > exact


def test_exact_binary_allocation_is_not_needlessly_reduced():
    policy = load_rolling_policy().model_copy(
        update={"family_alpha": 0.03125, "max_streams": 1, "max_updates": 1}
    )
    assert policy.update_budget == 1 / 32


def test_bound_is_upward_rounded_against_high_precision_formula():
    delta = load_rolling_policy().delta_for(2)
    with localcontext() as context:
        context.prec = 100
        for left, right in [(400, 400), (400, 800), (800, 400), (12000, 36000), (800, 95200)]:
            d = Decimal.from_float(delta)
            exact = (
                Decimal("0.5")
                * (Decimal(1) / left + Decimal(1) / right)
                * (Decimal(4 * (left + right)) / d).ln()
            ).sqrt()
            actual = upper_cut_bound(left, right, delta)
            assert Decimal.from_float(actual) >= exact
            assert Decimal.from_float(math.nextafter(actual, -math.inf)) <= exact
            bound = Fraction.from_float(actual) * left * right
            assert cut_limit(left, right, delta) == bound.numerator // bound.denominator


def test_effect_interval_encloses_exact_rational_endpoints():
    radius = 0.05
    difference, low, high = effect_interval(1, 3, 2, 7, radius)
    effect = Fraction(2, 7) - Fraction(1, 3)
    assert difference == float(effect)
    assert Fraction.from_float(low) <= effect - Fraction.from_float(radius)
    assert Fraction.from_float(high) >= effect + Fraction.from_float(radius)
    assert effect_interval(0, 400, 400, 400, 0.5) == (1.0, 0.5, 1.0)


@pytest.mark.parametrize(
    "left,right,delta",
    [(0, 400, 0.01), (400, 0, 0.01), (True, 400, 0.01), (400, 400, 0), (400, 400, float("nan"))],
)
def test_invalid_cut_inputs_refuse(left, right, delta):
    with pytest.raises(ValueError):
        upper_cut_bound(left, right, delta)


def test_prefix_weights_telescope_and_allocations_never_round_up():
    policy = load_rolling_policy()
    exact_b = Fraction(str(policy.family_alpha)) / (policy.max_streams * policy.max_updates)
    allocated = Fraction()
    for length in range(2, 122):
        exact = exact_b * Fraction(2, length * (length + 1))
        actual = Fraction.from_float(policy.delta_for(length))
        assert 0 < actual <= exact
        assert Fraction.from_float(math.nextafter(policy.delta_for(length), math.inf)) > exact
        allocated += actual
    assert allocated < exact_b
    assert sum(
        (Fraction(2, length * (length + 1)) for length in range(2, 122)), Fraction()
    ) == 1 - Fraction(2, 122)
    for invalid in [0, 1, True, 122]:
        with pytest.raises(ValueError):
            policy.delta_for(invalid)
