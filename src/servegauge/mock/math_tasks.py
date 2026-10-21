"""Independent small synthetic-task solver for the declared mock interpreter."""

from typing import Any


def math_answer(task: dict[str, Any]) -> int:
    operation = task.get("operation", "mod")
    a, b = int(task.get("a", 1)), int(task.get("b", 2))
    if operation == "mod":
        return a % max(b, 1)
    if operation == "linear":
        return (int(task["c"]) - b) // a
    if operation == "count":
        return sum(i % b == 0 for i in range(a + 1))
    if operation == "mod_power":
        value = 1
        modulus = int(task["modulus"])
        for _ in range(int(task["exponent"])):
            value = value * a % modulus
        return (value + int(task["offset"])) % modulus
    if operation == "linear_system":
        c, d, u, v = (int(task[key]) for key in ("c", "d", "u", "v"))
        return (u * d - b * v) // (a * d - b * c)
    if operation == "constrained_count":
        return sum(
            number % int(task["modulus"]) == int(task["residue"])
            and sum(int(digit) for digit in str(number)) % int(task["digit_modulus"])
            == int(task["digit_residue"])
            for number in range(int(task["limit"]) + 1)
        )
    if operation == "digit_progression":
        size, last, step = (int(task[key]) for key in ("digits", "last", "step"))
        return int("".join(str(last - index * step) for index in reversed(range(size))))
    return sum(int(ch) for ch in str(abs(a)))
