"""Strict JSON decoding: non-finite numbers are wire data, never model values."""

import json
import math
from typing import Any


def _reject(value: str) -> float:
    raise ValueError("nonfinite_JSON_number")


def _number(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite_JSON_number")
    return result


def strict_json(text: str) -> Any:
    return json.loads(text, parse_constant=_reject, parse_float=_number)
