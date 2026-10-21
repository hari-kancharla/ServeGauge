"""Typed observatory selections and UTC calendar windows, without dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from servegauge.core.hashing import content_hash
from servegauge.core.models import EndpointSpec, Level
from servegauge.mock.faults import FaultConfig
from servegauge.runner.check import normalize_tests
from servegauge.runner.plan import ReferenceSpec, RunPlan

Cadence = Literal["probe", "daily", "weekly", "monthly"]
Money = Annotated[float, Field(ge=0, allow_inf_nan=False)]
CADENCES = ("probe", "daily", "weekly", "monthly")


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, allow_inf_nan=False)


class CadenceSettings(Settings):
    level: Level
    tests: list[str] = Field(min_length=1)
    fingerprint_level: Level | None = None
    overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator("tests")
    @classmethod
    def known_tests(cls, value: list[str]) -> list[str]:
        normalize_tests(",".join(value))
        if len(value) != len(set(value)):
            raise ValueError("duplicate_cadence_tests")
        return value


def default_cadences() -> dict[str, CadenceSettings]:
    return {
        "probe": CadenceSettings(level="smoke", tests=["FP1", "FP3"]),
        "daily": CadenceSettings(
            level="core", tests=["S0", "S1", "T04", "T05", "FP1"], fingerprint_level="full"
        ),
        "weekly": CadenceSettings(level="core", tests=["S2", "S3", "S7", "S8"]),
        "monthly": CadenceSettings(
            level="full", tests=["all"], overrides={"S6": {"methods": ["FP1", "FP2", "FP3", "FP4"]}}
        ),
    }


class PublishingSettings(Settings):
    enabled: StrictBool = False
    repo_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    credential_env: str = Field(default="HF_TOKEN", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    license: Literal["cc-by-4.0"] = "cc-by-4.0"


class MonitorModel(Settings):
    id: str = Field(min_length=1, max_length=100)
    profile: str
    endpoints: list[EndpointSpec] = Field(min_length=1)
    reference: ReferenceSpec = Field(default_factory=lambda: ReferenceSpec(tier="MOCK"))
    faults: FaultConfig | None = None
    sampling_version: Literal["v1", "v2"] = "v2"
    mock_context_length: int | None = Field(default=None, strict=True, ge=256, le=131072)

    @model_validator(mode="after")
    def unique_endpoints(self) -> MonitorModel:
        if len({e.id for e in self.endpoints}) != len(self.endpoints):
            raise ValueError("duplicate_monitor_endpoints")
        if any(e.claimed_profile != self.profile for e in self.endpoints):
            raise ValueError("monitor_endpoint_profile_mismatch")
        RunPlan(
            plan_id="validate-selection",
            profile=self.profile,
            level="smoke",
            seed=0,
            reference=self.reference,
            max_usd=0,
            endpoints=self.endpoints,
            suites=["S0"],
        )
        return self


class ObservatoryConfig(Settings):
    version: Literal["observatory-v1"] = "observatory-v1"
    id: str = Field(min_length=1, max_length=100)
    enabled: StrictBool = False
    mode: Literal["mock", "owner"] = "mock"
    endpoint_selection: Literal["explicit"] = "explicit"
    seed: int = Field(default=20261020, ge=0, strict=True)
    concurrency_per_endpoint: int = Field(default=4, ge=1, strict=True)
    requests_per_minute: float = Field(default=60, gt=0, allow_inf_nan=False)
    models: list[MonitorModel] = Field(default_factory=list)
    cadences: dict[str, CadenceSettings] = Field(default_factory=default_cadences)
    budgets: dict[str, Money] = Field(default_factory=lambda: dict.fromkeys(CADENCES, 0.0))
    publishing: PublishingSettings = Field(default_factory=PublishingSettings)

    @model_validator(mode="after")
    def safe_selection(self) -> ObservatoryConfig:
        if self.enabled and not self.models:
            raise ValueError("enabled_observatory_needs_explicit_models")
        if len({m.id for m in self.models}) != len(self.models):
            raise ValueError("duplicate_monitor_models")
        if set(self.cadences) != set(CADENCES) or set(self.budgets) != set(CADENCES):
            raise ValueError("four_explicit_cadences_and_budgets_required")
        if self.mode == "mock" and (
            any(
                e.kind != "mock" or e.api_key_env is not None
                for m in self.models
                for e in m.endpoints
            )
            or any(self.budgets.values())
        ):
            raise ValueError("in_process_mock_requires_zero_budget_and_no_credentials")
        if self.mode == "owner" and any(
            m.faults is not None or m.mock_context_length is not None for m in self.models
        ):
            raise ValueError("scripted_faults_and_context_are_mock_only")
        if self.mode == "owner" and any(e.kind == "mock" for m in self.models for e in m.endpoints):
            raise ValueError("owner_mode_requires_real_explicit_endpoints")
        if any((m.reference.tier == "MOCK") != (self.mode == "mock") for m in self.models):
            raise ValueError("reference_tier_must_match_execution_mode")
        return self


def load_observatory(path: Path) -> ObservatoryConfig:
    try:
        return ObservatoryConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, yaml.YAMLError):
        raise ValueError("invalid_or_unavailable_observatory_configuration") from None


@dataclass(frozen=True)
class CadenceWindow:
    cadence: Cadence
    start: datetime
    end: datetime


def cadence_window(cadence: str, now: datetime) -> CadenceWindow:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("timezone_aware_window_time_required")
    moment = now.astimezone(UTC)
    start = moment.replace(hour=0, minute=0, second=0, microsecond=0)
    if cadence == "probe":
        start = start.replace(hour=(moment.hour // 6) * 6)
        return CadenceWindow("probe", start, start + timedelta(hours=6))
    if cadence == "daily":
        return CadenceWindow("daily", start, start + timedelta(days=1))
    if cadence == "weekly":
        start -= timedelta(days=start.weekday())
        return CadenceWindow("weekly", start, start + timedelta(days=7))
    if cadence == "monthly":
        start = start.replace(day=1)
        end = (
            start.replace(year=start.year + 1, month=1)
            if start.month == 12
            else start.replace(month=start.month + 1)
        )
        return CadenceWindow("monthly", start, end)
    raise ValueError("unknown_observatory_cadence")


def window_id(observatory_id: str, window: CadenceWindow) -> str:
    return "obs-" + content_hash(
        ["observatory-window-v1", observatory_id, window.cadence, window.start.isoformat()]
    )
