"""Declared SPEC check classes, shared by planning and inference."""

from servegauge.core.models import CheckClass


def check_class(check: str) -> CheckClass:
    if check.startswith(("S0", "S8.")) or check in {"P05", "L05"} or check.startswith("R05"):
        return "report_only"
    if check in {
        "P03",
        "P03.cache",
        "P03.cache.fresh",
        "P04.top_p",
        "P04.top_k",
        "P06.calibration",
        "P07",
        "P13",
        "T01",
        "T02",
        "T03",
        "T08",
        "J01",
        "J02.semantic",
        "J03",
        "L06",
    } or check.startswith(
        ("T07.", "L01.", "L02.", "L03.", "L05.", "L06.", "R01", "R02", "R03", "R04", "B", "FP")
    ):
        return "comparative"
    if check in {
        "P01",
        "P02",
        "P06.structure",
        "P10",
        "P11",
        "P12",
        "P14",
        "T05",
        "T06",
        "J02.validity",
        "J04",
        "J05",
        "L04",
    } or check.startswith(("P09.", "T04.", "L04.")):
        return "invariant"
    raise ValueError("check_class_not_declared")
