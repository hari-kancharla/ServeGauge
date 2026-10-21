"""Linux receiver body, assembled with its fixed recipe helper for authenticated SSH.

The assembled receiver uses only the standard library. Public launchers must use
the reserved controller; this private receiver never performs budget admission.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from servegauge.selfhost.remote_recipe import study_engine_args


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def guard(path: Path) -> None:
    if any(p.casefold() in {"starter files", ".git"} or p == ".." for p in path.parts):
        raise ValueError("remote_canonical_private_namespace_required")
    if any(p.is_symlink() for p in [*reversed(path.absolute().parents), path]):
        raise ValueError("remote_private_namespace_symlink_refused")


def command(
    args: list[str], *, environment: Mapping[str, str] | None = None, timeout: int = 60
) -> str:
    env = dict(os.environ)
    if environment:
        env.update(environment)
    value = subprocess.run(
        args,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
        text=True,
        timeout=timeout,
    ).stdout
    if len(value) > 65536:
        raise ValueError("remote_command_response_too_large")
    return value


def _expected_args(config: dict[str, Any]) -> list[str]:
    if config.get("version") == "selfhost-study-config-v1":
        return study_engine_args(config)
    return [
        "vllm",
        "serve",
        "openai/" + config["profile"],
        "--revision",
        config["model_revision"],
        "--tokenizer-revision",
        config["model_revision"],
        "--served-model-name",
        config["profile"],
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--tensor-parallel-size",
        "1",
        "--tool-call-parser",
        "openai",
        "--enable-auto-tool-choice",
    ]


def _request(payload: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    required = {"action", "session_id", "plan", "endpoint_key"}
    if not required <= payload.keys() or not payload.keys() <= required | {"expected_remote_id"}:
        raise ValueError("remote_typed_request_required")
    action, identifier, plan = payload["action"], payload["session_id"], payload["plan"]
    if action not in {"up", "status", "down"} or not re.fullmatch(
        r"selfhost-[a-f0-9]{32}", identifier
    ):
        raise ValueError("remote_session_identity_invalid")
    if not isinstance(plan, dict) or plan.get("plan_hash") != digest(
        {k: v for k, v in plan.items() if k != "plan_hash"}
    ):
        raise ValueError("remote_plan_hash_mismatch")
    config = plan["configuration"]
    study = config.get("version") == "selfhost-study-config-v1"
    if (
        plan["version"] != ("selfhost-study-plan-v1" if study else "selfhost-plan-v1")
        or config["backend"] != "ssh"
        or config.get("dedicated_host_reviewed") is not True
        or (not study and config["profile"] not in {"gpt-oss-120b", "gpt-oss-20b"})
        or not re.fullmatch(r"vllm/vllm-openai@sha256:[a-f0-9]{64}", config["image"])
        or not re.fullmatch(r"[a-f0-9]{40}", config["model_revision"])
        or not re.fullmatch(r"\d+\.\d+\.\d+", config["engine_version"])
        or config["gpu"] not in {"H100", "H200", "B200", "A100-80GB"}
        or type(plan["shutdown_minutes"]) is not int
        or not 1 <= plan["shutdown_minutes"] <= 1440
        or plan["engine_args"] != _expected_args(config)
    ):
        raise ValueError("remote_pinned_recipe_required")
    if action == "up" and (
        not isinstance(payload["endpoint_key"], str)
        or not 8 <= len(payload["endpoint_key"]) <= 4096
        or "\0" in payload["endpoint_key"]
    ):
        raise ValueError("remote_endpoint_authentication_required")
    if action != "up" and payload["endpoint_key"] is not None:
        raise ValueError("remote_cleanup_does_not_accept_endpoint_credentials")
    return action, identifier, plan


def _write(path: Path, value: dict[str, Any]) -> None:
    guard(path)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise ValueError("remote_existing_session_is_immutable") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())


def container_args(identifier: str, plan: dict[str, Any]) -> list[str]:
    return [
        "docker",
        "run",
        "--detach",
        "--platform",
        "linux/amd64",
        "--name",
        "servegauge-" + identifier.removeprefix("selfhost-"),
        "--label",
        "servegauge.session_id=" + identifier,
        "--label",
        "servegauge.plan_hash=" + plan["plan_hash"],
        "--publish",
        "127.0.0.1:8000:8000",
        "--gpus",
        "device=0",
        "--env",
        "VLLM_API_KEY",
        "--entrypoint",
        "vllm",
        plan["configuration"]["image"],
        *plan["engine_args"][1:],
    ]


def _snapshot(
    directory: Path, identifier: str, plan: dict[str, Any], execute: Callable[..., str]
) -> dict[str, Any]:
    name = "servegauge-" + identifier.removeprefix("selfhost-")
    record = json.loads(execute(["docker", "inspect", "--format", "{{json .}}", name], timeout=30))
    labels = record["Config"]["Labels"]
    if (
        labels.get("servegauge.session_id") != identifier
        or labels.get("servegauge.plan_hash") != plan["plan_hash"]
    ):
        raise ValueError("remote_container_binding_mismatch")
    if not re.fullmatch(r"[a-f0-9]{64}", record["Id"]):
        raise ValueError("remote_container_identity_invalid")
    state = record["State"]
    if type(state["Running"]) is not bool or type(state["ExitCode"]) is not int:
        raise ValueError("remote_container_state_invalid")
    receipt_path = directory / "runtime.json"
    guard(receipt_path)
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
    return {
        "state": "running" if state["Running"] else "terminated",
        "remote_id": record["Id"],
        "endpoint_url": "http://127.0.0.1:8000/v1",
        "remote_exit_code": None if state["Running"] else state["ExitCode"],
        "runtime_receipt": receipt,
    }


def handle(
    payload: dict[str, Any], *, home: Path | None = None, execute: Callable[..., str] = command
) -> dict[str, Any]:
    action, identifier, plan = _request(payload)
    home = Path.home() if home is None else home
    guard(home)
    directory = home / ".servegauge-selfhost" / identifier
    guard(directory)
    path = directory / "spec.json"
    guard(path)
    specification = {
        "version": "selfhost-remote-v1",
        "session_id": identifier,
        "plan_hash": plan["plan_hash"],
    }
    if action == "up":
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Exclusive intent precedes every system operation. Never replay startup
        # after a lost response, a shutdown failure or an interrupted image pull.
        _write(path, specification)
        deadline = time.monotonic() + plan["shutdown_minutes"] * 60
        execute(["sudo", "-n", "shutdown", "-h", "+" + str(plan["shutdown_minutes"])], timeout=30)
        config = plan["configuration"]

        def remaining() -> int:
            value = int(deadline - time.monotonic())
            if value <= 0:
                raise ValueError("remote_lifetime_exhausted")
            return value

        version = execute(
            [
                "docker",
                "run",
                "--rm",
                "--platform",
                "linux/amd64",
                "--entrypoint",
                "python3",
                config["image"],
                "-c",
                "import importlib.metadata;print(importlib.metadata.version('vllm'))",
            ],
            timeout=remaining(),
        ).strip()
        if version != config["engine_version"]:
            raise ValueError("remote_engine_version_mismatch")
        images = json.loads(
            execute(
                [
                    "docker",
                    "image",
                    "inspect",
                    "--format",
                    "{{json .RepoDigests}}",
                    config["image"],
                ],
                timeout=remaining(),
            )
        )
        if not isinstance(images, list) or config["image"] not in [
            s.removeprefix("docker.io/") for s in images
        ]:
            raise ValueError("remote_image_digest_mismatch")
        gpu = execute(
            [
                "nvidia-smi",
                "-i",
                "0",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            timeout=remaining(),
        ).strip()
        if "\n" in gpu or not re.fullmatch(r"[A-Za-z0-9 .,_()-]{1,256}", gpu):
            raise ValueError("remote_single_gpu_observation_required")
        observed, driver = [s.strip() for s in gpu.split(",", 1)]
        expected = config["gpu"].split("-")[0]
        if expected not in observed or (expected == "A100" and "80" not in observed):
            raise ValueError("remote_gpu_recipe_mismatch")
        argv = container_args(identifier, plan)
        container = execute(
            argv, environment={"VLLM_API_KEY": payload["endpoint_key"]}, timeout=remaining()
        ).strip()
        if not re.fullmatch(r"[a-f0-9]{64}", container):
            raise ValueError("remote_container_launch_unverified")
        _write(
            directory / "runtime.json",
            {
                "version": "selfhost-runtime-v1",
                "session_id": identifier,
                "plan_hash": plan["plan_hash"],
                "backend": "ssh",
                "recorded_at": datetime.now(UTC).isoformat(),
                "engine_version": version,
                "image": config["image"],
                "model_revision": config["model_revision"],
                "gpu_requested": config["gpu"],
                "gpu_observed": observed,
                "driver_version": driver,
                "engine_args": plan["engine_args"],
                "shutdown_armed": True,
                "shutdown_minutes": plan["shutdown_minutes"],
                "container_command": argv,
            },
        )
    elif not path.is_file() or json.loads(path.read_text()) != specification:
        raise ValueError("remote_saved_session_required")
    result = _snapshot(directory, identifier, plan, execute)
    expected_remote_id = payload.get("expected_remote_id")
    if expected_remote_id is not None and result["remote_id"] != expected_remote_id:
        raise ValueError("remote_original_container_identity_changed")
    if action == "down" and result["state"] == "running":
        execute(
            [
                "docker",
                "stop",
                "--time",
                "10",
                "servegauge-" + identifier.removeprefix("selfhost-"),
            ],
            timeout=30,
        )
        result = _snapshot(directory, identifier, plan, execute)
    return result


def main() -> None:
    try:
        if os.environ.get("SERVEGAUGE_REMOTE_OWNER_DISPATCH") != "1":
            raise ValueError("remote_reserved_owner_dispatch_required")
        value = sys.stdin.buffer.read(1048577)
        if len(value) > 1048576:
            raise ValueError("remote_request_too_large")
        result = handle(json.loads(value))
        print(json.dumps(result, sort_keys=True))
    except Exception:
        print('{"error":"remote_session_outcome_unverified"}')
        raise SystemExit(3) from None


if __name__ == "__main__":
    main()
