"""Separate finite small-model experiment recipes; no reference or precision certificate."""

from __future__ import annotations

import re
from decimal import Decimal
from ipaddress import ip_network
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from servegauge.core.models import ValueModel
from servegauge.selfhost.remote_recipe import (
    STUDY_ENGINES,
    STUDY_MODEL,
    STUDY_REVISION,
    STUDY_WRONG_TEMPLATE,
    study_engine_args,
)
from servegauge.selfhost.remote_recipe import (
    STUDY_PROFILE as STUDY_PROFILE,
)

MODEL, REVISION = STUDY_MODEL, STUDY_REVISION
ENGINE_REVISIONS = dict(STUDY_ENGINES)
WRONG_TEMPLATE = STUDY_WRONG_TEMPLATE
VARIANTS = ("bf16", "fp8", "mxfp4", "other-engine", "wrong-template")


class StudySessionConfig(ValueModel):
    version: Literal["selfhost-study-config-v1"]
    purpose: Literal["perturbation-study"]
    backend: Literal["ssh", "modal"]
    profile: Literal["qwen3-0.6b-study"]
    model_repository: Literal["Qwen/Qwen3-0.6B"]
    model_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    engine_version: Literal["0.31.0", "0.32.0"]
    engine_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    precision: Literal["bf16", "fp8", "mxfp4-w4a16"]
    template: Literal["official", "wrong-v1"]
    image: str = Field(pattern=r"^vllm/vllm-openai@sha256:[a-f0-9]{64}$")
    gpu: Literal["H100"]
    hourly_usd: Decimal = Field(gt=0, max_digits=24, decimal_places=12)
    setup_max_usd: Decimal = Field(ge=0, max_digits=24, decimal_places=12)
    shutdown_grace_seconds: int = Field(ge=0, le=3600, strict=True)
    billing_basis_reviewed: bool = Field(strict=True)
    recipe_reviewed: bool = Field(strict=True)
    dedicated_host_reviewed: bool = Field(default=False, strict=True)
    api_key_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,127}$")
    ssh_host: str | None = None
    ssh_key_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]{0,127}$")
    ssh_port: int = Field(default=22, ge=1, le=65535, strict=True)
    modal_app: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,62}$")
    modal_image_id: str | None = Field(default=None, pattern=r"^im-[A-Za-z0-9_-]{1,128}$")
    inbound_cidrs: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def reviewed(self) -> Self:
        if not self.billing_basis_reviewed or not self.recipe_reviewed:
            raise ValueError("selfhost_reviewed_recipe_and_billing_basis_required")
        if self.api_key_env in {
            "OPENROUTER_API_KEY",
            "MODAL_TOKEN_ID",
            "MODAL_TOKEN_SECRET",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "SERVEGAUGE_OWNER_RUN",
            "TRUEWEIGHTS_OWNER_RUN",
        }:
            raise ValueError("selfhost_separate_endpoint_credential_required")
        if self.backend == "ssh":
            if (
                self.ssh_host is None
                or not re.fullmatch(
                    r"(?:[A-Za-z_][A-Za-z0-9_-]*@)?[A-Za-z0-9][A-Za-z0-9.-]{0,252}",
                    self.ssh_host,
                )
                or self.ssh_key_env is None
                or self.ssh_key_env == self.api_key_env
                or self.modal_app is not None
                or self.modal_image_id is not None
                or self.inbound_cidrs
                or not self.dedicated_host_reviewed
            ):
                raise ValueError("selfhost_canonical_ssh_destination_required")
        elif (
            self.modal_app is None
            or self.modal_image_id is None
            or self.ssh_host is not None
            or self.ssh_key_env is not None
            or not self.inbound_cidrs
        ):
            raise ValueError("selfhost_dedicated_modal_destination_required")
        for value in self.inbound_cidrs:
            network = ip_network(value, strict=True)
            if network.prefixlen == 0 or str(network) != value or network.is_multicast:
                raise ValueError("selfhost_canonical_restricted_inbound_network_required")
        study_engine_args(self.model_dump(mode="json"))
        return self


def study_template(backend: str, variant: str = "bf16") -> dict[str, Any]:
    """Incomplete owner input, with only source-verified recipe facts supplied."""
    if backend not in {"ssh", "modal"} or variant not in VARIANTS:
        raise ValueError("selfhost_known_study_variant_required")
    engine = "0.31.0" if variant == "other-engine" else "0.32.0"
    return {
        "version": "selfhost-study-config-v1",
        "purpose": "perturbation-study",
        "backend": backend,
        "profile": STUDY_PROFILE,
        "model_repository": MODEL,
        "model_revision": REVISION,
        "engine_version": engine,
        "engine_revision": ENGINE_REVISIONS[engine],
        "precision": "mxfp4-w4a16" if variant == "mxfp4" else "fp8" if variant == "fp8" else "bf16",
        "template": "wrong-v1" if variant == "wrong-template" else "official",
        "image": None,
        "gpu": "H100",
        "hourly_usd": None,
        "setup_max_usd": None,
        "shutdown_grace_seconds": 60,
        "billing_basis_reviewed": False,
        "recipe_reviewed": False,
        "dedicated_host_reviewed": False,
        "api_key_env": "SELFHOST_API_KEY",
        "ssh_host": None,
        "ssh_key_env": "SELFHOST_SSH_KEY_PATH" if backend == "ssh" else None,
        "ssh_port": 22,
        "modal_app": None,
        "modal_image_id": None,
        "inbound_cidrs": [],
    }
