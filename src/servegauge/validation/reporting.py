"""Markdown companions to machine-readable validation receipts."""

from pathlib import Path
from typing import Any


def rate(value: dict[str, Any]) -> str:
    return (
        f"{value['successes']}/{value['n']} ({value['estimate']:.3f}; "
        f"95% [{value['ci_low']:.3f}, {value['ci_high']:.3f}])"
    )


def fault_report(result: dict[str, Any], out: Path) -> None:
    lines = [
        "# Controlled offline fault validation",
        "",
        f"Acceptance: **{'met' if result['accepted'] else 'not met'}**. "
        "Synthetic MOCK references only; zero model API calls and $0 spent.",
        "",
        "At least 60 clean and 30 fault seeds per case, three clean reference "
        "instances, original fault strengths and frozen verdict rules. "
        "Intervals describe Monte Carlo uncertainty, not provider reliability.",
        "",
        "| Fault | Expected check / code | Detection | Correct diagnosis | Accepted |",
        "|---|---|---|---|---|",
    ]
    for name, case in result["cases"].items():
        lines.append(
            f"| {name} | {case['expected_check']} / {case['expected_code']} | "
            f"{rate(case['power'])} | {rate(case['correct_diagnosis_power'])} | "
            f"{case['accepted']} |"
        )
    lines.extend(
        [
            "",
            "## Clean controls",
            "",
            "| Case / check | False-positive rate | Comparative PASS rate |",
            "|---|---|---|",
        ]
    )
    for name, case in result["cases"].items():
        for check, row in case["clean"].items():
            passing = (
                rate(row["comparative_pass_rate"])
                if row["comparative_pass_rate"]
                else "not comparative"
            )
            lines.append(f"| {name} / {check} | {rate(row['false_positive_rate'])} | {passing} |")
    lines.extend(
        [
            "",
            "## Scope and limits",
            "",
            "- Per-check power, diagnosis sets, seeds, hashes, intervals, p-values and "
            "configuration: [faults.json](faults.json).",
            "- Core sizes come from suite configuration. Context/output cases use production "
            "grids and rules; unused suites are not dispatched.",
            "- Responses remain clustered by item. Confirmation uses fresh scripted generation "
            "and the original bounded replay policy.",
            "- Clean mock outputs are unusually accurate. These false-positive rates "
            "do not describe arbitrary healthy language models.",
            "- FP1/FP2 use matched cells, two different-model controls and code-verified math. "
            "Full S6 CLI/reference-pack integration remains separate.",
            "- The power report studies FP3/FP4, mixtures and logit perturbations. "
            "Simulated FP4 vectors do not validate a text embedding model.",
            "- S8 remains report-only. An operational alarm supports an infrastructure "
            "hypothesis without a model-quality FAIL.",
            "- Acceptance uses frozen point thresholds. Wilson bounds remain visible "
            "even when too wide to certify those thresholds.",
            "",
        ]
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "faults.md").write_text("\n".join(lines), encoding="utf-8")
