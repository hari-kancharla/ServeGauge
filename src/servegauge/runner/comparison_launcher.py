"""Owner-gated parent launcher; both procedures use the same private CPU boundary."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

from servegauge.comparison.authorization import (
    Mode,
    OwnerReview,
    SetupRecord,
    authorize,
    setup_measurement,
)
from servegauge.comparison.container import prepare_container
from servegauge.comparison.isolation import _lexical
from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.redaction import Redactor
from servegauge.storage.comparison import _files, _guard, _private_json, _require_sanitized
from servegauge.storage.comparison_kit import _write, verify_kit
from servegauge.storage.comparison_program import ComparisonProgram
from servegauge.storage.comparison_results import (
    normalize_external,
    render_comparison,
    validate_audit,
    verify_results,
)
from servegauge.storage.source_identity import capture_source


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ValueError("comparison_command_usage")


def _failure_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case["case_id"],
            "original_population_n": case["population_n"],
            "n": None,
            "violations": None,
            "ci_low": None,
            "ci_high": None,
            "verdict": "ERROR",
            "reference_tier": case["original_check"]["reference_tier"],
            "replay_adds_independent_n": False,
        }
        for case in manifest["cases"]
    ]


def _run_procedure(
    *,
    kit: Path,
    manifest: dict[str, Any],
    kit_hash: str,
    mode: Mode,
    review: OwnerReview | None,
    procedure: str,
    docker: Path,
    config: Path,
    host: str,
    image: str,
    output: Path,
    user: str,
) -> dict[str, Any]:
    command = [
        "/opt/runtime/.venv/bin/python",
        "-m",
        "servegauge.runner.comparison",
        "--kit",
        "/inputs",
        "--kit-hash",
        kit_hash,
        "--procedure",
        procedure,
    ]
    process = prepare_container(
        docker=docker,
        config=config,
        host=host,
        image=image,
        inputs=kit,
        output=output,
        user=user,
        arguments=command,
    )
    ticket = {
        "kit_hash": kit_hash,
        "mode": mode,
        "owner_review": review.model_dump(mode="json") if review is not None else None,
    }
    _write(output / "execution-ticket.json", canonical_json(ticket) + b"\n")
    _write(output / "container-command.json", canonical_json(process.arguments) + b"\n")
    started = datetime.now(UTC).isoformat()
    clock = time.perf_counter()
    try:
        completed = subprocess.run(
            process.arguments, env=process.environment, capture_output=True, timeout=1200
        )
        elapsed = time.perf_counter() - clock
        _write(output / "container.stdout", completed.stdout)
        _write(output / "container.stderr", completed.stderr)
        if completed.returncode != 0:
            raise ValueError("comparison_child_failed")
        result = _private_json((output / "procedure-results.json").read_bytes())
        if (
            not isinstance(result, dict)
            or result.get("kit_hash") != kit_hash
            or result.get("procedure") != procedure
            or result.get("owner_comparison_executed") is not (mode == "owner")
            or result.get("total_setup_seconds") is not None
            or result.get("new_model_api_calls") != 0
            or result.get("new_spend_usd") != 0
            or result.get("replay_adds_independent_n") is not False
            or not isinstance(result.get("cases"), list)
            or sorted(row["case_id"] for row in result["cases"])
            != sorted(case["case_id"] for case in manifest["cases"])
        ):
            raise ValueError("comparison_child_result_scope_changed")
        program = ComparisonProgram.model_validate(
            _private_json((kit / "procedures/program.json").read_bytes())
        )
        runtime = result.get("runtime")
        if (
            not isinstance(runtime, dict)
            or not isinstance(runtime.get("source"), dict)
            or runtime["source"].get("source_hash") != capture_source()["source_tree_hash"]
        ):
            raise ValueError("comparison_current_installed_source_identity_required")
        if procedure == "ordinary":
            originals = {case.case_id: case for case in program.cases}
            for row in result["cases"]:
                original = originals[row["case_id"]].original_check
                keys = ["n", "violations", "ci_low", "ci_high", "reference_tier", "verdict"]
                if canonical_json({key: row.get(key) for key in keys}) != canonical_json(
                    {key: getattr(original, key) for key in keys}
                ):
                    raise ValueError("comparison_ordinary_result_changed")
        else:
            raw = _private_json((output / "promptfoo-results.json").read_bytes())
            if canonical_json(normalize_external(raw, manifest["cases"])) != canonical_json(
                result["cases"]
            ):
                raise ValueError("comparison_external_result_changed")
        for case in program.cases:
            path = output / ("case-" + case.case_id) / "finite-audit.json"
            _guard(path)
            validate_audit(case, _private_json(path.read_bytes()))
        _require_sanitized(result)
        result.update(execution_started_utc=started, execution_seconds=elapsed, harness_error=None)
        return dict(result)
    except (ValueError, TypeError, KeyError, OSError, subprocess.TimeoutExpired):
        elapsed = time.perf_counter() - clock
        return {
            "procedure": procedure,
            "kit_hash": kit_hash,
            "cases": _failure_rows(manifest),
            "owner_comparison_executed": mode == "owner",
            "total_setup_seconds": None,
            "first_valid_result_utc": None,
            "execution_started_utc": started,
            "execution_seconds": elapsed,
            "harness_error": "comparison_child_or_evidence_failed",
            "new_model_api_calls": 0,
            "new_spend_usd": 0,
            "replay_adds_independent_n": False,
        }


def _agreement(ordinary: dict[str, Any], external: dict[str, Any]) -> dict[str, Any]:
    available = ordinary["verdict"] != "ERROR" and external["verdict"] != "ERROR"
    intervals = all(
        type(row.get(key)) in {int, float} and math.isfinite(row[key])
        for row in [ordinary, external]
        for key in ["ci_low", "ci_high"]
    )
    difference = (
        max(abs(ordinary[key] - external[key]) for key in ["ci_low", "ci_high"])
        if available and intervals
        else None
    )
    return {
        "case_id": ordinary["case_id"],
        **{
            "same_" + key: ordinary.get(key) == external.get(key) if available else None
            for key in ["n", "violations", "reference_tier", "verdict"]
        },
        "interval_max_abs_difference": difference,
        "intervals_agree_within_float_roundoff": difference <= 32 * math.ulp(1.0)
        if difference is not None
        else None,
    }


def launch(
    root: Path,
    kit: Path,
    *,
    kit_hash: str,
    mode: Mode,
    review: OwnerReview | None,
    docker: Path,
    config: Path,
    host: str,
    image: str,
    out: Path,
    user: str | None = None,
    environment: Mapping[str, str] | None = None,
    setup_records: Mapping[str, SetupRecord] | None = None,
) -> dict[str, Any]:
    """Execute a reviewed owner kit or a strictly synthetic kit; never set owner mode."""
    _lexical([root, kit, docker, config, out])
    for path in [root, kit, config, out]:
        _guard(path)
    private = root.absolute() / ".cache/comparisons"
    if (
        not out.absolute().is_relative_to(private)
        or out.absolute() == private
        or out.exists()
        or out.absolute().is_relative_to(kit.absolute())
        or kit.absolute().is_relative_to(out.absolute())
    ):
        raise ValueError("comparison_new_private_result_destination_required")
    manifest = verify_kit(kit, kit_hash=kit_hash)
    authorize(manifest, kit_hash=kit_hash, mode=mode, review=review, environment=environment)
    notes = setup_records or {}
    if not set(notes) <= {"ordinary", "external"}:
        raise ValueError("comparison_declared_setup_procedure_required")
    for procedure, note in notes.items():
        _require_sanitized(note.model_dump(mode="json"))
        setup_measurement(note, kit_hash=kit_hash, procedure=procedure, first_valid_utc=None)
    if user is None:
        if sys.platform == "win32":
            user = "65534:65534"
        else:
            user = f"{os.getuid()}:{os.getgid()}"
    out.mkdir(mode=0o700, parents=True)
    procedures = []
    for procedure in ["ordinary", "external"]:
        verify_kit(kit, kit_hash=kit_hash)
        procedures.append(
            _run_procedure(
                kit=kit,
                manifest=manifest,
                kit_hash=kit_hash,
                mode=mode,
                review=review,
                procedure=procedure,
                docker=docker,
                config=config,
                host=host,
                image=image,
                output=out / procedure,
                user=user,
            )
        )
        procedures[-1]["setup"] = setup_measurement(
            notes.get(procedure),
            kit_hash=kit_hash,
            procedure=procedure,
            first_valid_utc=procedures[-1]["first_valid_result_utc"],
        )
    verify_kit(kit, kit_hash=kit_hash)
    left = {row["case_id"]: row for row in procedures[0]["cases"]}
    right = {row["case_id"]: row for row in procedures[1]["cases"]}
    result = {
        "version": "comparison-results-v1",
        "kit_hash": kit_hash,
        "image": image,
        "state": "comparison_failed"
        if any(procedure["harness_error"] is not None for procedure in procedures)
        else "captured_procedures_executed",
        "owner_comparison_executed": mode == "owner",
        "mode": mode,
        "procedures": procedures,
        "agreements": [_agreement(left[key], right[key]) for key in sorted(left)],
        "total_setup_seconds": None,
        "manual_setup_steps": {procedure: note.manual_steps for procedure, note in notes.items()}
        if notes
        else None,
        "setup_scope": (
            "Measured execution uses prebuilt custom providers/assertions; "
            "total setup remains unmeasured."
        ),
        "custom_work": [
            "Pinned CPU image and finite source compiler",
            "Custom Python raw provider and independent P06/T06 assertions",
            "Ordinary reserved replay and matched control reconstruction",
        ],
        "root_cause_limit": manifest["root_cause_limit"],
        "new_model_api_calls": 0,
        "new_spend_usd": 0,
        "replay_adds_independent_n": False,
        "report_status": (
            "Owner interpretation and actual setup record required; no comparative winner inferred."
        ),
    }
    result["files_sha256"] = {
        path.relative_to(out).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in _files(out)
    }
    _require_sanitized(result)
    _write(out / "comparison-results.json", canonical_json(result) + b"\n")
    return result


def main() -> None:
    parser = Parser(description=__doc__)
    parser.add_argument(
        "command", choices=["run-owner", "run-synthetic", "review-owner", "setup-start", "report"]
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--kit-hash", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--confirm-completed-m15-report-site", action="store_true")
    parser.add_argument("--docker", type=Path)
    parser.add_argument("--docker-config", type=Path)
    parser.add_argument(
        "--docker-host",
        default="npipe:////./pipe/docker_engine"
        if os.name == "nt"
        else "unix:///var/run/docker.sock",
    )
    parser.add_argument("--image")
    parser.add_argument("--container-user")
    parser.add_argument("--procedure", choices=["ordinary", "external"])
    parser.add_argument("--manual-step", action="append", default=[])
    parser.add_argument("--ordinary-setup", type=Path)
    parser.add_argument("--external-setup", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--result-hash")
    failed_procedures = 0
    try:
        args = parser.parse_args()
        _lexical(
            [
                path
                for path in [
                    args.root,
                    args.kit,
                    args.out,
                    args.review,
                    args.docker,
                    args.docker_config,
                    args.ordinary_setup,
                    args.external_setup,
                    args.results,
                ]
                if path is not None
            ]
        )
        if args.command == "report":
            if args.results is None or args.result_hash is None:
                raise ValueError("comparison_external_result_anchor_required")
            _guard(args.results)
            result = verify_results(args.results, result_hash=args.result_hash)
            verify_kit(args.kit, kit_hash=args.kit_hash)
            if result["kit_hash"] != args.kit_hash:
                raise ValueError("comparison_matching_result_source_kit_required")
            _guard(args.out)
            private = args.root.absolute() / ".cache/comparisons"
            if (
                not args.out.absolute().is_relative_to(private)
                or args.out.absolute().is_relative_to(args.results.absolute())
                or args.out.absolute().is_relative_to(args.kit.absolute())
            ):
                raise ValueError("comparison_private_separate_report_destination_required")
            _write(args.out, render_comparison(result).encode())
            value = {
                "state": "private_report_draft_written",
                "result_hash": args.result_hash,
                "published": False,
                "owner_interpretation_required": True,
            }
        elif args.command == "setup-start":
            started = datetime.now(UTC).isoformat()
            if args.review is None or args.procedure is None:
                raise ValueError("comparison_review_and_setup_procedure_required")
            _guard(args.review)
            review = OwnerReview.model_validate(_private_json(args.review.read_bytes()))
            manifest = verify_kit(args.kit, kit_hash=args.kit_hash)
            authorize(manifest, kit_hash=args.kit_hash, mode="owner", review=review)
            note = SetupRecord(
                kit_hash=args.kit_hash,
                procedure=args.procedure,
                started_utc=started,
                manual_steps=args.manual_step,
            )
            _require_sanitized(note.model_dump(mode="json"))
            _guard(args.out)
            private = args.root.absolute() / ".cache/comparisons"
            if not args.out.absolute().is_relative_to(
                private
            ) or args.out.absolute().is_relative_to(args.kit.absolute()):
                raise ValueError("comparison_private_separate_setup_record_required")
            _write(args.out, canonical_json(note.model_dump(mode="json")) + b"\n")
            value = {
                "state": "owner_setup_start_recorded",
                "kit_hash": args.kit_hash,
                "procedure": args.procedure,
                "total_setup_seconds": None,
            }
        elif args.command == "review-owner":
            if not args.confirm_completed_m15_report_site or args.review is not None:
                raise ValueError("comparison_explicit_m15_completion_confirmation_required")
            manifest = verify_kit(args.kit, kit_hash=args.kit_hash)
            review = OwnerReview(
                kit_hash=args.kit_hash,
                selection_hash=manifest["selection_hash"],
                controls_hash=manifest["controls_hash"],
                source_run_ids=sorted(manifest["source_eligibility"]),
                approved_utc=datetime.now(UTC).isoformat(),
                m15_report_site_completed=True,
            )
            authorize(manifest, kit_hash=args.kit_hash, mode="owner", review=review)
            _guard(args.out)
            private = args.root.absolute() / ".cache/comparisons"
            if not args.out.absolute().is_relative_to(
                private
            ) or args.out.absolute().is_relative_to(args.kit.absolute()):
                raise ValueError("comparison_private_separate_review_destination_required")
            _write(args.out, canonical_json(review.model_dump(mode="json")) + b"\n")
            value = {"state": "owner_review_recorded", "kit_hash": args.kit_hash}
        else:
            if args.docker is None or args.docker_config is None or args.image is None:
                raise ValueError("comparison_explicit_pinned_local_runtime_required")
            review = None
            if args.review is not None:
                _guard(args.review)
                review = OwnerReview.model_validate(_private_json(args.review.read_bytes()))
            mode: Mode = "owner" if args.command == "run-owner" else "synthetic"
            notes = {}
            for procedure, path in [
                ("ordinary", args.ordinary_setup),
                ("external", args.external_setup),
            ]:
                if path is not None:
                    _guard(path)
                    data = _private_json(path.read_bytes())
                    _require_sanitized(data)
                    notes[procedure] = SetupRecord.model_validate(data)
            value = launch(
                args.root,
                args.kit,
                kit_hash=args.kit_hash,
                mode=mode,
                review=review,
                docker=args.docker,
                config=args.docker_config,
                host=args.docker_host,
                image=args.image,
                out=args.out,
                user=args.container_user,
                setup_records=notes,
            )
            failed_procedures = sum(
                procedure["harness_error"] is not None for procedure in value["procedures"]
            )
            value = {
                "state": value["state"],
                "kit_hash": value["kit_hash"],
                "result_hash": content_hash(value),
                "owner_comparison_executed": value["owner_comparison_executed"],
                "total_setup_seconds": None,
                "harness_failures": failed_procedures,
            }
        print(Redactor.from_environment().text(canonical_json(value).decode()))
        if failed_procedures:
            parser.exit(5, "Comparison child failed; private results and result hash retained.\n")
    except Exception:
        parser.exit(3, "Comparison launch refused or failed; inspect private sanitized evidence.\n")


if __name__ == "__main__":
    main()
