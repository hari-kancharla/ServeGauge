"""Container-side execution of the sealed, explicitly authorized comparison kit."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Literal, NoReturn

from servegauge.comparison.authorization import (
    OWNER_VARIABLES,
    OwnerReview,
    authorize,
    check_review,
)
from servegauge.core.hashing import canonical_json
from servegauge.profiles import load_profile
from servegauge.runner.captured import CapturedCheck
from servegauge.runner.check import run_check
from servegauge.runner.ledger import Ledger
from servegauge.storage.comparison import _guard, _private_json, _relative
from servegauge.storage.comparison_kit import _write, verify_kit
from servegauge.storage.comparison_program import ComparisonProgram
from servegauge.storage.comparison_results import normalize_external, validate_audit
from servegauge.storage.comparison_timing import record_first_valid
from servegauge.storage.source_identity import capture_source
from servegauge.storage.workspace import initialize
from servegauge.suites.config import suite_config
from servegauge.verdicts.config import load_policy

Procedure = Literal["ordinary", "external"]


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ValueError("comparison_command_usage")


def _runtime_identity(procedure: Procedure) -> dict[str, Any]:
    identity = {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "source": capture_source()["source_identity"],
        "promptfoo_version": None,
        "node_version": None,
    }
    if procedure == "external":
        package = _private_json(
            Path("/opt/promptfoo/node_modules/promptfoo/package.json").read_bytes()
        )
        version = subprocess.run(
            ["/opt/node/bin/node", "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout.strip()
        if package.get("version") != "0.124.1" or version != "v26.7.0":
            raise ValueError("comparison_pinned_external_runtime_required")
        identity.update(promptfoo_version=package["version"], node_version=version)
    return identity


def require_container_boundary() -> None:
    """Check actual Linux kernel state, not an environment opt-out or fake flag."""
    try:
        mounts = {
            line.split()[1]: line.split()[3].split(",")
            for line in Path("/proc/mounts").read_text().splitlines()
        }
        state = {
            line.split(":", 1)[0]: line.split(":", 1)[1].strip()
            for line in Path("/proc/self/status").read_text().splitlines()
            if ":" in line
        }
        interfaces = [
            line.split(":")[0].strip()
            for line in Path("/proc/net/dev").read_text().splitlines()[2:]
        ]
        active = [
            name
            for name in interfaces
            if int(Path("/sys/class/net", name, "flags").read_text(), 16) & 1
        ]
        if (
            "ro" not in mounts["/"]
            or "ro" not in mounts["/inputs"]
            or "rw" not in mounts["/work"]
            or "noexec" not in mounts["/tmp"]
            or state["NoNewPrivs"] != "1"
            or int(state["CapEff"], 16) != 0
            or int(state["Uid"].split()[1]) == 0
            or active != ["lo"]
            or Path("/proc/net/route").read_text().splitlines()[1:]
            or any(
                line.split()[-1] != "lo"
                for line in Path("/proc/net/ipv6_route").read_text().splitlines()
            )
            or any(os.environ.get(name) for name in ["HOME", *OWNER_VARIABLES])
        ):
            raise ValueError
    except (OSError, ValueError, KeyError, IndexError):
        raise ValueError("comparison_verified_cpu_container_boundary_required") from None


def external_config(kit: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """No raw prompt templating, model grader, reported result or fault-label input."""
    program_blob = _relative(kit, "procedures/program.json").read_bytes()
    contexts = _private_json(_relative(kit, "procedures/contexts.json").read_bytes())
    tests = []
    for case in manifest["cases"]:
        original = case["original_check"]
        policy = contexts[case["case_id"]]["policy"]
        tests.append(
            {
                "description": case["case_id"],
                "vars": {
                    "case_id": case["case_id"],
                    "independent_policy": {
                        "check_id": original["check_id"],
                        "unit_keys": case["logical_unit_keys"],
                        "probability_tolerance": policy["entries"]["P06"]["prob_sum_tolerance"],
                        "alpha": policy["alpha"],
                        "tolerance": policy["tolerance"],
                        "k_fail": policy["k_fail"],
                        "reference_tier": original["reference_tier"],
                    },
                },
                "assert": [{"type": "python", "value": "file://assertion.py"}],
            }
        )
    return {
        "providers": [
            {
                "id": "file://provider.py",
                "config": {
                    "program": "program.json",
                    "program_sha256": hashlib.sha256(program_blob).hexdigest(),
                },
            }
        ],
        "prompts": ["{{case_id}}"],
        "tests": tests,
        "sharing": False,
        "evaluateOptions": {"maxConcurrency": 1, "cache": False, "showProgressBar": False},
    }


def _ticket(work: Path, kit_hash: str) -> dict[str, Any]:
    try:
        value = _private_json(_relative(work, "execution-ticket.json").read_bytes())
        if (
            not isinstance(value, dict)
            or set(value) != {"kit_hash", "mode", "owner_review"}
            or value["kit_hash"] != kit_hash
            or value["mode"] not in {"synthetic", "owner"}
        ):
            raise ValueError
        return dict(value)
    except (OSError, ValueError, KeyError, TypeError):
        raise ValueError("comparison_execution_ticket_unavailable_or_changed") from None


def _check_environment(kit: Path, program: ComparisonProgram) -> None:
    contexts = _private_json(_relative(kit, "procedures/contexts.json").read_bytes())
    if set(contexts) != {case.case_id for case in program.cases}:
        raise ValueError("comparison_exact_context_scope_required")
    for case in program.cases:
        primary = next(
            phase for phase in case.phases if phase.source_run_id == case.original_check.run_id
        )
        actual = {
            "profile": load_profile(primary.source_plan.profile).model_dump(mode="json"),
            "policy": load_policy().model_dump(mode="json"),
            "suite_config": suite_config(),
        }
        if canonical_json(actual) != canonical_json(contexts[case.case_id]):
            raise ValueError("comparison_original_profile_policy_and_suite_required")


async def execute_procedure(
    kit: Path,
    *,
    kit_hash: str,
    procedure: Procedure,
    work: Path,
) -> dict[str, Any]:
    """Execute only after parent approval and actual kernel boundary verification."""
    for path in [kit, work]:
        if ".." in path.parts or any(part.casefold() == "starter files" for part in path.parts):
            raise ValueError("comparison_path_refused")
    for path in [kit, work]:
        _guard(path)
    if work.absolute() != Path.cwd() or procedure not in {"ordinary", "external"}:
        raise ValueError("comparison_explicit_execution_scope_required")
    ticket = _ticket(work, kit_hash)
    require_container_boundary()
    initialize(work)  # Private defaults only; no captured procedure has started.
    manifest = verify_kit(kit, kit_hash=kit_hash)
    review = (
        OwnerReview.model_validate(ticket["owner_review"])
        if ticket["owner_review"] is not None
        else None
    )
    if ticket["mode"] == "owner":
        check_review(manifest, kit_hash=kit_hash, review=review)
    else:
        authorize(manifest, kit_hash=kit_hash, mode="synthetic", review=review)
    program = ComparisonProgram.model_validate(
        _private_json(_relative(kit, "procedures/program.json").read_bytes())
    )
    _check_environment(kit, program)
    runtime = _runtime_identity(procedure)
    rows = []
    if procedure == "ordinary":
        for case in program.cases:
            captured = CapturedCheck(program, case_id=case.case_id)
            output = work / ("case-" + case.case_id)
            _guard(output)
            output.mkdir(mode=0o700)
            initialize(output)
            result = await run_check(
                captured.local_plan(),
                root=output,
                approved=True,
                tests=case.original_check.check_id,
                baseline_id=captured.baseline_id,
                inventory=captured.primary_items(),
                capability_inventory=captured.capability_items(),
                transports=captured.transports(),
                captured=captured,
            )
            actual = next(
                row for row in result.checks if row.check_id == case.original_check.check_id
            )
            audit = captured.tape.audit()
            validate_audit(case, audit)
            _write(output / "finite-audit.json", canonical_json(audit) + b"\n")
            ledger = Ledger(output)
            ledger.refresh()
            if ledger.pending or ledger.development_spend() != 0:
                raise ValueError("comparison_zero_cost_reserved_ledger_required")
            rows.append(
                {
                    "case_id": case.case_id,
                    "original_population_n": case.population_n,
                    **{
                        key: getattr(actual, key)
                        for key in [
                            "n",
                            "violations",
                            "ci_low",
                            "ci_high",
                            "reference_tier",
                            "verdict",
                        ]
                    },
                    "replay_adds_independent_n": False,
                }
            )
            if actual.n > 0 and actual.ci_low is not None and actual.ci_high is not None:
                record_first_valid(work, procedure="ordinary")
    else:
        for name, blob in {
            "program.json": _relative(kit, "procedures/program.json").read_bytes(),
            "provider.py": b"from servegauge.comparison.case_provider import call_api\n",
            "assertion.py": b"from servegauge.storage.comparison_timing import get_assert\n",
            "promptfoo.json": canonical_json(external_config(kit, manifest)) + b"\n",
        }.items():
            _write(work / name, blob)
        command = [
            "/opt/node/bin/node",
            "/opt/promptfoo/node_modules/promptfoo/dist/src/entrypoint.js",
            "eval",
            "--config",
            "promptfoo.json",
            "--no-cache",
            "--no-write",
            "--no-share",
            "--no-table",
            "--no-progress-bar",
            "--max-concurrency",
            "1",
            "--repeat",
            "1",
            "--output",
            "promptfoo-results.json",
        ]
        completed = subprocess.run(
            command, cwd=work, env=dict(os.environ), capture_output=True, timeout=600
        )
        _write(work / "external.stdout", completed.stdout)
        _write(work / "external.stderr", completed.stderr)
        if completed.returncode not in {0, 100}:
            raise ValueError("comparison_external_tool_execution_failed")
        value = _private_json(_relative(work, "promptfoo-results.json").read_bytes())
        rows = normalize_external(value, manifest["cases"])
        for case in program.cases:
            validate_audit(
                case,
                _private_json(
                    _relative(work, f"case-{case.case_id}/finite-audit.json").read_bytes()
                ),
            )
    marker = work / "first-valid-result.json"
    _guard(marker)
    first = _private_json(marker.read_bytes())["utc"] if marker.exists() else None
    value = {
        "procedure": procedure,
        "kit_hash": kit_hash,
        "cases": rows,
        "owner_comparison_executed": ticket["mode"] == "owner",
        "total_setup_seconds": None,
        "first_valid_result_utc": first,
        "new_model_api_calls": 0,
        "new_spend_usd": 0,
        "prebuilt_custom_work_reused": True,
        "replay_adds_independent_n": False,
        "runtime": runtime,
    }
    verify_kit(kit, kit_hash=kit_hash)
    _write(work / "procedure-results.json", canonical_json(value) + b"\n")
    return value


def main() -> None:
    parser = Parser(description=__doc__)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--kit-hash", required=True)
    parser.add_argument("--procedure", choices=["ordinary", "external"], required=True)
    try:
        args = parser.parse_args()
        if args.procedure == "ordinary":
            procedure: Procedure = "ordinary"
        else:
            procedure = "external"
        asyncio.run(
            execute_procedure(
                args.kit, kit_hash=args.kit_hash, procedure=procedure, work=Path.cwd()
            )
        )
    except Exception as error:
        if Path.cwd() == Path("/work"):
            try:
                frames = [
                    {
                        "module": Path(frame.filename).name,
                        "line": frame.lineno,
                        "function": frame.name,
                    }
                    for frame in traceback.extract_tb(error.__traceback__)
                ]
                _write(
                    Path("/work/procedure-error.json"),
                    canonical_json(
                        {
                            "error_type": type(error).__name__,
                            "frames": frames,
                            "exception_values_and_locals_omitted": True,
                        }
                    )
                    + b"\n",
                )
            except (OSError, ValueError):
                pass
        parser.exit(
            3, "Comparison procedure refused or failed; inspect private sanitized evidence.\n"
        )


if __name__ == "__main__":
    main()
