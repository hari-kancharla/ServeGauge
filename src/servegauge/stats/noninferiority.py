"""Decision boundaries are strict and always receive declared practical margins."""

import math


def non_inferior(lower_one_sided_bound: float, margin: float) -> bool:
    if not math.isfinite(lower_one_sided_bound) or not math.isfinite(margin) or margin < 0:
        raise ValueError("finite_bound_and_nonnegative_margin_required")
    return lower_one_sided_bound > -margin


def equivalent(lower_tost_bound: float, upper_tost_bound: float, margin: float) -> bool:
    if lower_tost_bound > upper_tost_bound:
        raise ValueError("ordered_interval_required")
    return non_inferior(lower_tost_bound, margin) and non_inferior(-upper_tost_bound, margin)
