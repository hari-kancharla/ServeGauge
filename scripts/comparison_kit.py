"""Propose matched controls and prepare/verify a private kit; never run a procedure."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NoReturn

from servegauge.comparison.isolation import _lexical
from servegauge.core.hashing import canonical_json, content_hash
from servegauge.storage.comparison import _guard, _private_json
from servegauge.storage.comparison_controls import (
    ControlRequest,
    ControlSelection,
    propose_controls,
)
from servegauge.storage.comparison_kit import _write, prepare_kit, verify_kit


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ValueError("comparison_command_usage")


def main() -> None:
    parser = Parser(description=__doc__)
    parser.add_argument("command", choices=["controls", "prepare", "verify"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--selection-hash")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--controls-hash")
    parser.add_argument("--kit", type=Path)
    parser.add_argument("--kit-hash")
    parser.add_argument("--out", type=Path)
    try:
        args = parser.parse_args()
        paths = [
            path
            for path in [args.root, args.snapshot, args.request, args.controls, args.kit, args.out]
            if path is not None
        ]
        _lexical(paths)
        for path in paths:
            _guard(path)
        if args.command == "verify":
            if args.kit is None or args.kit_hash is None:
                raise ValueError("comparison_external_kit_anchor_required")
            manifest = verify_kit(args.kit, kit_hash=args.kit_hash)
            value = {
                "kit_hash": args.kit_hash,
                "state": manifest["state"],
                "m15_eligible": manifest["m15_eligible"],
                "synthetic_only": manifest["synthetic_only"],
                "owner_comparison_executed": False,
                "setup_seconds": None,
            }
        else:
            if args.snapshot is None or args.selection_hash is None or args.out is None:
                raise ValueError("comparison_snapshot_anchor_and_private_destination_required")
            private = args.root.absolute() / ".cache/comparisons"
            if (
                not args.out.absolute().is_relative_to(private)
                or args.out.absolute() == private
                or args.out.absolute().is_relative_to(args.snapshot.absolute())
            ):
                raise ValueError("comparison_new_separate_private_destination_required")
            if args.command == "controls":
                if args.request is None:
                    raise ValueError("comparison_control_request_required")
                request = _private_json(args.request.read_bytes())
                if (
                    not isinstance(request, dict)
                    or set(request) != {"controls"}
                    or not isinstance(request["controls"], list)
                ):
                    raise ValueError("comparison_control_request_invalid")
                controls = propose_controls(
                    args.snapshot,
                    selection_hash=args.selection_hash,
                    requests=[ControlRequest.model_validate(row) for row in request["controls"]],
                )
                _write(args.out, canonical_json(controls.model_dump(mode="json")) + b"\n")
                value = {
                    "state": "controls_proposed_only",
                    "owner_review": "required",
                    "controls_hash": content_hash(controls.model_dump(mode="json")),
                    "owner_comparison_executed": False,
                }
            else:
                if args.controls is None or args.controls_hash is None:
                    raise ValueError("comparison_separate_control_selection_and_anchor_required")
                controls = ControlSelection.model_validate(
                    _private_json(args.controls.read_bytes())
                )
                manifest = prepare_kit(
                    args.root,
                    args.snapshot,
                    selection_hash=args.selection_hash,
                    controls=controls,
                    controls_hash=args.controls_hash,
                    out=args.out,
                )
                value = {
                    "state": manifest["state"],
                    "kit_hash": content_hash(manifest),
                    "m15_eligible": manifest["m15_eligible"],
                    "synthetic_only": manifest["synthetic_only"],
                    "owner_comparison_executed": False,
                    "setup_seconds": None,
                }
        print(canonical_json(value).decode())
    except (ValueError, TypeError, KeyError, OSError):
        parser.exit(3, "Invalid comparison inputs or changed evidence; no procedure executed.\n")
    except Exception:
        parser.exit(5, "Kit preparation failed; inspect private sanitized evidence.\n")


if __name__ == "__main__":
    main()
