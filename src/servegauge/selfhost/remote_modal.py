"""Fixed Modal entrypoint, assembled with its recipe helper; the Sandbox limits lifetime."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from servegauge.selfhost.remote_recipe import study_engine_args


def runtime_observation(
    identifier: str, plan: dict[str, Any], *, engine_version: str, gpu: str
) -> dict[str, Any]:
    config = plan["configuration"]
    if config.get("version") == "selfhost-study-config-v1" and (
        plan.get("version") != "selfhost-study-plan-v1"
        or plan["engine_args"] != study_engine_args(config)
    ):
        raise ValueError("modal_remote_finite_study_recipe_required")
    if (
        not re.fullmatch(r"selfhost-[a-f0-9]{32}", identifier)
        or config["backend"] != "modal"
        or config["engine_version"] != engine_version
        or "\n" in gpu
        or not re.fullmatch(r"[A-Za-z0-9 .,_()-]{1,256}", gpu)
    ):
        raise ValueError("modal_remote_runtime_unverified")
    observed, driver = [v.strip() for v in gpu.split(",", 1)]
    expected = config["gpu"].split("-")[0]
    if expected not in observed or (expected == "A100" and "80" not in observed):
        raise ValueError("modal_remote_gpu_recipe_mismatch")
    return {
        "version": "selfhost-runtime-v1",
        "session_id": identifier,
        "plan_hash": plan["plan_hash"],
        "backend": "modal",
        "recorded_at": datetime.now(UTC).isoformat(),
        "engine_version": engine_version,
        "image": config["image"],
        "model_revision": config["model_revision"],
        "gpu_requested": config["gpu"],
        "gpu_observed": observed,
        "driver_version": driver,
        "engine_args": plan["engine_args"],
        "shutdown_armed": True,
        "shutdown_minutes": plan["shutdown_minutes"],
    }


def main() -> None:
    value = json.loads(os.environ["SERVEGAUGE_RUNTIME_PLAN"])
    identifier, plan = value["session_id"], value["plan"]
    blob = json.dumps(
        {k: v for k, v in plan.items() if k != "plan_hash"},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    if hashlib.sha256(blob).hexdigest() != plan["plan_hash"]:
        raise ValueError("modal_remote_plan_changed")
    if plan["engine_args"][:2] != ["vllm", "serve"] or not os.environ.get("VLLM_API_KEY"):
        raise ValueError("modal_remote_authenticated_engine_required")
    version = importlib.metadata.version("vllm")
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()
    receipt = runtime_observation(identifier, plan, engine_version=version, gpu=gpu)
    path = Path("/tmp/servegauge-runtime-v1.json")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(receipt, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.execvpe("vllm", plan["engine_args"], dict(os.environ))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit(3) from None
