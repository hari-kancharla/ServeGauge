"""Prepare or explicitly download the pinned optional tool; never run a comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import NoReturn

from servegauge.core.redaction import Redactor
from servegauge.storage.comparison import _guard, _private_json


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ValueError("comparison_runtime_usage")


def main() -> None:
    parser = Parser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "install"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    try:
        args = parser.parse_args()
        root = Path(__file__).absolute().parents[1]
        target = args.out.absolute()
        _guard(target)
        private = root / ".cache/tools"
        if target == private or not target.is_relative_to(private):
            raise ValueError("comparison_private_tool_directory_required")
        inputs = root / "scripts/comparison_runtime_inputs"
        _guard(inputs)
        for name in ["promptfoo-pin.json", "promptfoo-package.json", "promptfoo-package-lock.json"]:
            _guard(inputs / name)
        pin = _private_json((inputs / "promptfoo-pin.json").read_bytes())
        for source, digest in [
            ("promptfoo-package.json", pin["package_sha256"]),
            ("promptfoo-package-lock.json", pin["lock_sha256"]),
        ]:
            if hashlib.sha256((inputs / source).read_bytes()).hexdigest() != digest:
                raise ValueError("comparison_tool_input_changed")
        if args.command == "prepare":
            if args.offline:
                raise ValueError("comparison_offline_option_is_for_install")
            if target.exists():
                raise ValueError("comparison_fresh_tool_directory_required")
            target.mkdir(parents=True, mode=0o700)
            for source, destination in [
                ("promptfoo-package.json", "package.json"),
                ("promptfoo-package-lock.json", "package-lock.json"),
            ]:
                with (target / destination).open("xb") as stream:
                    stream.write((inputs / source).read_bytes())
            for name in ["user.npmrc", "global.npmrc"]:
                with (target / name).open("xb"):
                    pass
        for name, digest in [
            ("package.json", pin["package_sha256"]),
            ("package-lock.json", pin["lock_sha256"]),
        ]:
            path = target / name
            _guard(path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("comparison_prepared_tool_input_changed")
        for name in ["user.npmrc", "global.npmrc"]:
            path = target / name
            _guard(path)
            if path.read_bytes():
                raise ValueError("comparison_empty_npm_config_required")
        if args.command == "install":
            allowed = {"package.json", "package-lock.json", "user.npmrc", "global.npmrc"}
            for path in target.iterdir():
                if path.name not in allowed:
                    raise ValueError("comparison_only_prepared_tool_files_allowed")
                _guard(path)
            npm = shutil.which("npm")
            if npm is None:
                raise ValueError("comparison_npm_and_fresh_install_required")
            environment = {
                "PATH": os.pathsep.join([str(Path(npm).absolute().parent), "/usr/bin", "/bin"]),
                "NPM_CONFIG_USERCONFIG": str(target / "user.npmrc"),
                "NPM_CONFIG_GLOBALCONFIG": str(target / "global.npmrc"),
                "NPM_CONFIG_CACHE": str(root / ".cache/npm"),
                "NPM_CONFIG_UPDATE_NOTIFIER": "false",
            }
            if os.name == "nt":
                for name in ["SYSTEMROOT", "WINDIR"]:
                    if name in os.environ:
                        environment[name] = os.environ[name]
            with (target / "install.log").open("x") as log:
                result = subprocess.run(
                    [
                        npm,
                        "ci",
                        "--ignore-scripts",
                        "--no-audit",
                        "--no-fund",
                        "--omit=dev",
                        *(["--offline"] if args.offline else []),
                    ],
                    cwd=target,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=600,
                )
            if result.returncode:
                raise ValueError("comparison_tool_download_failed")
            for name, digest in [
                ("package.json", pin["package_sha256"]),
                ("package-lock.json", pin["lock_sha256"]),
            ]:
                if hashlib.sha256((target / name).read_bytes()).hexdigest() != digest:
                    raise ValueError("comparison_installed_tool_input_changed")
        value = {
            "state": "prepared_only" if args.command == "prepare" else "tool_installed_only",
            "tool": "promptfoo",
            "version": pin["tool_version"],
            "lock_sha256": pin["lock_sha256"],
            "lifecycle_scripts_enabled": False,
            "comparison_executed": False,
            "model_api_calls": 0,
            "new_model_spend_usd": 0,
        }
        print(Redactor.from_environment().text(json.dumps(value, indent=2, sort_keys=True)))
    except (ValueError, OSError, TypeError, KeyError, subprocess.SubprocessError):
        parser.exit(3, "Invalid runtime inputs or failed download; no comparison executed.\n")


if __name__ == "__main__":
    main()
