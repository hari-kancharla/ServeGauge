"""Strict plan and requirements validation. Errors never echo source values."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
from pydantic import Field, model_validator

from servegauge.core.models import EndpointSpec, Level, Tier, ValueModel
from servegauge.selfhost.remote_recipe import STUDY_COLLECTION, STUDY_PROFILE


class ReferenceSpec(ValueModel):
    tier: Tier
    pack: str | None = None


class RunPlan(ValueModel):
    plan_id: str
    profile: str
    level: Level
    seed: int
    reference: ReferenceSpec
    concurrency_per_endpoint: int = Field(default=4, ge=1)
    requests_per_minute: float = Field(default=60, gt=0)
    tokens_per_minute: float | None = Field(default=None, gt=0)
    retries: int = Field(default=2, ge=0, le=10)
    retry_base_seconds: float = Field(default=0.5, ge=0)
    retry_max_seconds: float = Field(default=30, ge=0)
    timeout_seconds: float = Field(default=120, gt=0)
    max_usd: float = Field(ge=0)
    endpoints: list[EndpointSpec] = Field(min_length=1)
    suites: list[str] = Field(min_length=1)
    overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def endpoints_unique_and_safe(self) -> RunPlan:
        if self.profile == STUDY_PROFILE and (
            set(self.suites) - {"S0", "S6"}
            or any(endpoint.claimed_profile != STUDY_PROFILE for endpoint in self.endpoints)
            or (
                "S6" in self.suites
                and self.overrides.get("S6", {}).get("collection_recipe") != STUDY_COLLECTION
            )
        ):
            raise ValueError("experiment_profile_requires_supported_fingerprint_collection")
        ids = [endpoint.id for endpoint in self.endpoints]
        if len(ids) != len(set(ids)):
            raise ValueError("endpoint ids must be unique")
        for endpoint in self.endpoints:
            parts = urlsplit(endpoint.base_url)
            if (
                parts.scheme not in {"http", "https"}
                or not parts.hostname
                or parts.username
                or parts.query
                or parts.fragment
            ):
                raise ValueError("base_url must be HTTP(S) without credentials, query or fragment")
            if endpoint.kind == "mock" and parts.hostname not in {"localhost", "127.0.0.1", "::1"}:
                raise ValueError("mock endpoints must be loopback addresses")
        return self


class Requirement(ValueModel):
    id: str
    tests: list[str] = Field(min_length=1)
    margin: float | None = Field(default=None, gt=0)
    params: dict[str, Any] = Field(default_factory=dict)


class Requirements(ValueModel):
    app: str
    requires: list[Requirement] = Field(min_length=1)


def load_plan(path: Path) -> RunPlan:
    try:
        return RunPlan.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, yaml.YAMLError):
        raise ValueError(
            "Invalid plan: check required fields, endpoint URLs and nonnegative limits"
        ) from None


def load_requirements(path: Path) -> Requirements:
    try:
        return Requirements.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, yaml.YAMLError):
        raise ValueError(
            "Invalid requirements: app, requires, id and nonempty tests are required"
        ) from None
