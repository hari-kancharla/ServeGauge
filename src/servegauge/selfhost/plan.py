"""Pure pinned self-host plans and conservative declared session admission bounds."""

from __future__ import annotations

import re
from decimal import ROUND_CEILING, Decimal, localcontext
from ipaddress import ip_network
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from servegauge.core.hashing import content_hash
from servegauge.core.models import ValueModel
from servegauge.runner.budget import BudgetPolicy
from servegauge.selfhost.remote_recipe import study_engine_args
from servegauge.selfhost.study_recipe import STUDY_PROFILE, StudySessionConfig, study_template


class SessionConfig(ValueModel):
    version: Literal["selfhost-config-v1"]
    backend: Literal["ssh", "modal"]
    profile: Literal["gpt-oss-120b", "gpt-oss-20b"]
    engine_version: str = Field(pattern=r"^\d+\.\d+\.\d+$", max_length=32)
    image: str = Field(pattern=r"^vllm/vllm-openai@sha256:[a-f0-9]{64}$")
    model_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    gpu: Literal["H100", "H200", "B200", "A100-80GB"]
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
        if tuple(int(n) for n in self.engine_version.split(".")) < (0, 10, 1):
            raise ValueError("selfhost_official_recipe_engine_too_old")
        if not self.billing_basis_reviewed or not self.recipe_reviewed:
            raise ValueError("selfhost_reviewed_recipe_and_billing_basis_required")
        if self.api_key_env in {
            "OPENROUTER_API_KEY",
            "MODAL_TOKEN_ID",
            "MODAL_TOKEN_SECRET",
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
        ):
            raise ValueError("selfhost_dedicated_modal_destination_required")
        elif not self.inbound_cidrs:
            raise ValueError("selfhost_reviewed_modal_inbound_allowlist_required")
        for value in self.inbound_cidrs:
            network = ip_network(value, strict=True)
            if network.prefixlen == 0 or str(network) != value or network.is_multicast:
                raise ValueError("selfhost_canonical_restricted_inbound_network_required")
        return self


def parse_session_config(value: Any) -> SessionConfig | StudySessionConfig:
    if not isinstance(value, dict):
        raise ValueError("selfhost_configuration_object_required")
    if value.get("version") == "selfhost-config-v1":
        return SessionConfig.model_validate(value)
    if value.get("version") == "selfhost-study-config-v1":
        return StudySessionConfig.model_validate(value)
    raise ValueError("selfhost_known_configuration_version_required")


def engine_args(config: SessionConfig | StudySessionConfig) -> list[str]:
    """No shell fragments or credential values enter the pinned engine command."""
    if isinstance(config, StudySessionConfig):
        return study_engine_args(config.model_dump(mode="json"))
    return [
        "vllm",
        "serve",
        "openai/" + config.profile,
        "--revision",
        config.model_revision,
        "--tokenizer-revision",
        config.model_revision,
        "--served-model-name",
        config.profile,
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


def build_plan(
    config: SessionConfig | StudySessionConfig,
    policy: BudgetPolicy,
    *,
    max_usd: Decimal,
    shutdown_minutes: int,
) -> dict[str, Any]:
    """Admission estimate only; the controller must reserve it before owner dispatch."""
    if type(shutdown_minutes) is not int or not 1 <= shutdown_minutes <= 1440:
        raise ValueError("selfhost_bounded_shutdown_minutes_required")
    if not max_usd.is_finite() or max_usd <= 0:
        raise ValueError("selfhost_positive_finite_cap_required")
    gross_cap = min(policy.selfhost_usd_per_session, policy.per_run_usd, policy.monthly_usd)
    if max_usd > gross_cap:
        raise ValueError("selfhost_requested_cap_exceeds_policy")
    seconds = shutdown_minutes * 60 + config.shutdown_grace_seconds
    # Round admission upward at a declared micro-dollar accounting resolution.
    # This never rounds observed charges down or turns a declared rate into an invoice.
    with localcontext() as context:
        context.prec = 80
        compute = (config.hourly_usd * seconds / 3600).quantize(
            Decimal("0.000001"), rounding=ROUND_CEILING
        )
        reserved = compute + config.setup_max_usd
        if reserved > max_usd * policy.stop_at_fraction:
            raise ValueError("selfhost_declared_cost_exceeds_stopping_cap")
    value: dict[str, Any] = {
        "version": "selfhost-study-plan-v1"
        if isinstance(config, StudySessionConfig)
        else "selfhost-plan-v1",
        "profile": config.profile,
        "configuration": config.model_dump(mode="json"),
        "budget_policy": policy.model_dump(mode="json"),
        "engine_args": engine_args(config),
        "max_usd": str(max_usd),
        "shutdown_minutes": shutdown_minutes,
        "maximum_duration_seconds": seconds,
        "gpu_compute_upper_usd": str(compute.normalize()),
        "reserved_upper_usd": str(reserved.normalize()),
        "billing_status": "declared_bound_not_invoice",
        "reference_qualification": "not_a_reference_session"
        if isinstance(config, StudySessionConfig)
        else "pending_owner_capture",
        "model_api_calls": 0,
        "network_writes": 0,
    }
    return {**value, "plan_hash": content_hash(value)}


def validate_plan(value: dict[str, Any]) -> None:
    """Reconstruct every derived field; a rehashed edited plan is not sufficient."""
    try:
        expected = build_plan(
            parse_session_config(value["configuration"]),
            BudgetPolicy.model_validate(value["budget_policy"]),
            max_usd=Decimal(value["max_usd"]),
            shutdown_minutes=value["shutdown_minutes"],
        )
    except (ValueError, TypeError, KeyError, ArithmeticError):
        raise ValueError("selfhost_reviewed_plan_invalid") from None
    if value != expected:
        raise ValueError("selfhost_reviewed_plan_changed")


def config_template(backend: str, profile: str, *, variant: str | None = None) -> dict[str, Any]:
    """An intentionally incomplete owner template; no guessed rate or mutable pin."""
    if profile == STUDY_PROFILE:
        return study_template(backend, variant or "bf16")
    if variant is not None:
        raise ValueError("selfhost_variant_requires_experiment_profile")
    if backend not in {"ssh", "modal"} or profile not in {"gpt-oss-120b", "gpt-oss-20b"}:
        raise ValueError("selfhost_known_template_required")
    return {
        "version": "selfhost-config-v1",
        "backend": backend,
        "profile": profile,
        "engine_version": None,
        "image": None,
        "model_revision": None,
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
