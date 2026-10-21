"""Strict local binding of remote runtime observations; no reference qualification."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from servegauge.core.models import ValueModel
from servegauge.selfhost.remote_ssh import container_args


class RuntimeReceipt(ValueModel):
    version: Literal["selfhost-runtime-v1"]
    session_id: str = Field(pattern=r"^selfhost-[a-f0-9]{32}$")
    plan_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    backend: Literal["ssh", "modal"]
    recorded_at: str = Field(max_length=64)
    engine_version: str = Field(pattern=r"^\d+\.\d+\.\d+$", max_length=32)
    image: str = Field(pattern=r"^vllm/vllm-openai@sha256:[a-f0-9]{64}$")
    model_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    gpu_requested: str = Field(max_length=32)
    gpu_observed: str = Field(pattern=r"^[A-Za-z0-9 ._()-]{1,128}$")
    driver_version: str = Field(pattern=r"^[A-Za-z0-9 ._-]{1,128}$")
    engine_args: list[str] = Field(max_length=64)
    shutdown_armed: bool = Field(strict=True)
    shutdown_minutes: int = Field(ge=1, le=1440, strict=True)
    container_command: list[str] | None = Field(default=None, max_length=128)
    modal_sandbox_id: str | None = Field(default=None, pattern=r"^sb-[A-Za-z0-9_-]{1,128}$")
    modal_image_id: str | None = Field(default=None, pattern=r"^im-[A-Za-z0-9_-]{1,128}$")


def validate_runtime(
    identifier: str, plan: dict[str, Any], value: dict[str, Any]
) -> dict[str, Any]:
    receipt = RuntimeReceipt.model_validate(value)
    config = plan["configuration"]
    timestamp = datetime.fromisoformat(receipt.recorded_at)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("selfhost_aware_remote_observation_required")
    if (
        receipt.session_id != identifier
        or receipt.plan_hash != plan["plan_hash"]
        or receipt.backend != config["backend"]
        or receipt.engine_version != config["engine_version"]
        or receipt.image != config["image"]
        or receipt.model_revision != config["model_revision"]
        or receipt.gpu_requested != config["gpu"]
        or receipt.engine_args != plan["engine_args"]
        or receipt.shutdown_armed is not True
        or receipt.shutdown_minutes != plan["shutdown_minutes"]
    ):
        raise ValueError("selfhost_remote_runtime_binding_changed")
    gpu = config["gpu"].split("-")[0]
    if gpu not in receipt.gpu_observed or (gpu == "A100" and "80" not in receipt.gpu_observed):
        raise ValueError("selfhost_remote_gpu_recipe_mismatch")
    if config["backend"] == "ssh":
        if (
            receipt.container_command != container_args(identifier, plan)
            or receipt.modal_sandbox_id is not None
            or receipt.modal_image_id is not None
        ):
            raise ValueError("selfhost_remote_container_flags_changed")
    elif (
        receipt.container_command is not None
        or receipt.modal_sandbox_id is None
        or receipt.modal_image_id != config["modal_image_id"]
    ):
        raise ValueError("selfhost_remote_modal_identity_required")
    return receipt.model_dump(mode="json")
