from servegauge.items.mathgen import math_problem
from servegauge.mock.math_tasks import math_answer


def brute(problem):
    p = problem.parameters
    if problem.kind == "mod_power":
        value = 1
        for _ in range(p["exponent"]):
            value = value * p["a"] % p["modulus"]
        return (value + p["offset"]) % p["modulus"]
    if problem.kind == "linear_system":
        bound = p["solution_bound"]
        solutions = [
            (x, y)
            for x in range(-bound, bound + 1)
            for y in range(-bound, bound + 1)
            if p["a"] * x + p["b"] * y == p["u"] and p["c"] * x + p["d"] * y == p["v"]
        ]
        assert len(solutions) == 1
        return solutions[0][0]
    if problem.kind == "constrained_count":
        return sum(
            number % p["modulus"] == p["residue"]
            and sum(int(digit) for digit in str(number)) % p["digit_modulus"] == p["digit_residue"]
            for number in range(p["limit"] + 1)
        )
    solutions = []
    for number in range(10 ** (p["digits"] - 1), 10 ** p["digits"]):
        digits = [int(digit) for digit in str(number)]
        if (
            digits[-1] == p["last"]
            and sum(digits) == p["digit_sum"]
            and all(
                right - left == p["step"]
                for left, right in zip(digits[:-1], digits[1:], strict=True)
            )
        ):
            solutions.append(number)
    assert len(solutions) == 1
    return solutions[0]


def test_5000_generated_math_problems_against_independent_brute_force():
    seen = set()
    for seed in range(5000):
        tier = 1 + seed % 3
        problem = math_problem(seed, tier=tier)
        assert problem.answer == brute(problem), (seed, tier, problem)
        assert math_answer({"operation": problem.kind, **problem.parameters}) == problem.answer
        assert str(seed) in problem.prompt and problem.license == "CC-BY-4.0"
        seen.add((problem.kind, problem.tier))
        if seed < 20:
            assert math_problem(seed, tier=tier) == problem
    assert len(seen) == 12
