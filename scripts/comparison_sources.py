"""Prepare private comparison evidence; this command cannot run either comparison tool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NoReturn

from servegauge.core.hashing import content_hash
from servegauge.core.redaction import Redactor
from servegauge.storage.comparison import (
    ComparisonRequest,
    ComparisonSelection,
    _guard,
    _private_json,
    prepare_sources,
    propose_selection,
    save_selection,
    verify_sources,
)
from servegauge.storage.workspace import configure_workspace


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ValueError("comparison_command_usage")


def main() -> None:
    parser = Parser(description=__doc__)
    parser.add_argument("command", choices=["select", "prepare", "verify"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--request", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--out", type=Path)
    try:
        args = parser.parse_args()
        for path in [args.root, args.request, args.selection, args.out]:
            if path is not None:
                _guard(path)
        configure_workspace(args.root)
        if args.command == "select":
            if args.request is None or args.selection is not None:
                raise ValueError("comparison_select_requires_request_only")
            request = ComparisonRequest.model_validate(_private_json(args.request.read_bytes()))
            selection = propose_selection(args.root, request)
            if args.out is None:
                value = selection.model_dump(mode="json")
            else:
                save_selection(args.root, selection, args.out)
                value = {
                    "state": "selection_proposed_only",
                    "owner_review": "required",
                    "selection_hash": content_hash(selection.model_dump(mode="json")),
                    "comparison_executed": False,
                }
        else:
            if args.selection is None or args.out is None or args.request is not None:
                raise ValueError("comparison_selection_and_private_destination_required")
            selection = ComparisonSelection.model_validate(
                _private_json(args.selection.read_bytes())
            )
            if args.command == "prepare":
                value = prepare_sources(args.root, selection, args.out)
            else:
                value = verify_sources(
                    args.out, selection_hash=content_hash(selection.model_dump(mode="json"))
                )
        print(Redactor.from_environment().text(json.dumps(value, indent=2, sort_keys=True)))
    except (ValueError, OSError, TypeError, KeyError):
        parser.exit(3, "Invalid comparison inputs or changed evidence; no comparison executed.\n")
    except Exception:
        parser.exit(5, "Comparison preparation failed; inspect sanitized evidence.\n")


if __name__ == "__main__":
    main()
