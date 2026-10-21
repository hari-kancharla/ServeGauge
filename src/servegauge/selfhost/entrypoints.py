"""Backend-specific Python launch scripts share the guarded CLI and ledger path."""

from __future__ import annotations

import sys
from typing import NoReturn


def _refused() -> NoReturn:
    sys.stderr.write("Launcher backend differs from the reviewed arguments.\n")
    raise SystemExit(3)


def launcher(backend: str) -> None:
    from servegauge.cli import main

    if backend not in {"ssh", "modal"}:
        raise ValueError("selfhost_known_launcher_required")
    arguments = list(sys.argv[1:])
    if any(
        a.startswith("--backend=") and a.removeprefix("--backend=") != backend for a in arguments
    ):
        _refused()
    if arguments and arguments[0] in {"template", "up", "status", "down"}:
        if "--backend" in arguments:
            position = arguments.index("--backend")
            if position + 1 >= len(arguments) or arguments[position + 1] != backend:
                _refused()
        else:
            arguments.extend(["--backend", backend])
    sys.argv[1:] = ["selfhost", *arguments]
    main()
