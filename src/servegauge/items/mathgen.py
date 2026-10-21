"""Own deterministic integer problems; independent brute force verifies every family."""

from __future__ import annotations

import random
from functools import cache
from typing import Literal

from servegauge.core.models import ValueModel

type MathKind = Literal["mod_power", "linear_system", "constrained_count", "digit_progression"]


class MathProblem(ValueModel):
    seed: int
    tier: int
    kind: MathKind
    prompt: str
    parameters: dict[str, int]
    answer: int
    license: str = "CC-BY-4.0"


def _digit_count(limit: int, modulus: int, residue: int, digit_mod: int, digit_residue: int) -> int:
    digits = [int(digit) for digit in str(limit)]

    @cache
    def count(position: int, remainder: int, digit_remainder: int, tight: bool) -> int:
        if position == len(digits):
            return int(remainder == residue and digit_remainder == digit_residue)
        upper = digits[position] if tight else 9
        return sum(
            count(
                position + 1,
                (remainder * 10 + digit) % modulus,
                (digit_remainder + digit) % digit_mod,
                tight and digit == upper,
            )
            for digit in range(upper + 1)
        )

    return count(0, 0, 0, True)


def math_problem(seed: int, *, tier: int) -> MathProblem:
    if tier not in {1, 2, 3}:
        raise ValueError("math_tier_must_be_1_2_or_3")
    rng = random.Random(seed)
    kinds: tuple[MathKind, ...] = (
        "mod_power",
        "linear_system",
        "constrained_count",
        "digit_progression",
    )
    kind = kinds[seed % 4]
    parameters: dict[str, int]
    if kind == "mod_power":
        a, exponent = rng.randint(2, 30 * tier), rng.randint(2, (5, 17, 53)[tier - 1])
        modulus, offset = rng.randint(2, 97 * tier), rng.randint(-100 * tier, 100 * tier)
        parameters = {"a": a, "exponent": exponent, "modulus": modulus, "offset": offset}
        answer = (pow(a, exponent, modulus) + offset) % modulus
        question = (
            f"What is the least nonnegative residue of ({a}^{exponent} + ({offset})) "
            f"modulo {modulus}?"
        )
    elif kind == "linear_system":
        bound = (3, 8, 16)[tier - 1]
        while True:
            a, b, c, d = (rng.randint(-2 * tier, 2 * tier) for _ in range(4))
            determinant = a * d - b * c
            if determinant:
                break
        x, y = rng.randint(-bound, bound), rng.randint(-bound, bound)
        u, v = a * x + b * y, c * x + d * y
        parameters = {"a": a, "b": b, "c": c, "d": d, "u": u, "v": v, "solution_bound": bound}
        answer = (u * d - b * v) // determinant
        question = f"Solve {a}x + ({b})y = {u} and {c}x + ({d})y = {v}. What is x?"
    elif kind == "constrained_count":
        limit = rng.randint(40, (150, 700, 2500)[tier - 1])
        modulus, digit_modulus = rng.randint(2, 7 * tier), rng.randint(2, 7)
        residue, digit_residue = rng.randrange(modulus), rng.randrange(digit_modulus)
        parameters = {
            "limit": limit,
            "modulus": modulus,
            "residue": residue,
            "digit_modulus": digit_modulus,
            "digit_residue": digit_residue,
        }
        answer = _digit_count(limit, modulus, residue, digit_modulus, digit_residue)
        question = (
            f"How many integers n from 0 through {limit} inclusive have n mod {modulus} "
            f"equal to {residue}, and decimal digit sum mod {digit_modulus} "
            f"equal to {digit_residue}?"
        )
    else:
        digits = tier + 1
        possibilities = [
            (first, step)
            for first in range(1, 10)
            for step in (-2, -1, 1, 2)
            if all(0 <= first + index * step <= 9 for index in range(digits))
        ]
        first, step = rng.choice(possibilities)
        sequence = [first + index * step for index in range(digits)]
        parameters = {
            "digits": digits,
            "last": sequence[-1],
            "digit_sum": sum(sequence),
            "step": step,
        }
        answer = int("".join(str(digit) for digit in sequence))
        question = (
            f"An integer has exactly {digits} decimal digits, with no leading zero. "
            f"Each digit after the first is the previous digit plus ({step}). "
            f"Its final digit is {sequence[-1]} and its digit sum is {sum(sequence)}. "
            "What is the integer?"
        )
    return MathProblem(
        seed=seed,
        tier=tier,
        kind=kind,
        prompt=f"Generated problem {seed}. {question} Return only the integer answer.",
        parameters=parameters,
        answer=answer,
    )
