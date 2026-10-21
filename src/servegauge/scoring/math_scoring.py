"""Exact generated-integer answers; no benchmark text or approximate grading."""


def exact_math_answer(text: str, expected: int) -> bool:
    if not isinstance(expected, int) or isinstance(expected, bool):
        raise ValueError("integer_math_gold_required")
    return text.strip() == str(expected)
