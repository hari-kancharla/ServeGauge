"""Explicit owner authorization, separate from preparatory and synthetic work."""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import Field, model_validator

from servegauge.core.hashing import content_hash
from servegauge.core.models import ValueModel
from servegauge.storage.comparison import Digest

Mode = Literal["owner", "synthetic"]
OWNER_VARIABLES = ("SERVEGAUGE_OWNER_RUN", "TRUEWEIGHTS_OWNER_RUN")


class OwnerReview(ValueModel):
    version: Literal["comparison-owner-review-v1"] = "comparison-owner-review-v1"
    kit_hash: Digest
    selection_hash: Digest
    controls_hash: Digest
    source_run_ids: list[str] = Field(min_length=1)
    approved_utc: str
    m15_report_site_completed: Literal[True]

    @model_validator(mode="after")
    def explicit_review(self) -> OwnerReview:
        aware_utc(self.approved_utc)
        if self.source_run_ids != sorted(set(self.source_run_ids)):
            raise ValueError("comparison_explicit_dated_owner_review_required")
        return self


def aware_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.utcoffset() != timedelta(0):
            raise ValueError
        return parsed
    except (TypeError, ValueError):
        raise ValueError("comparison_explicit_utc_timestamp_required") from None


class SetupRecord(ValueModel):
    version: Literal["comparison-setup-record-v1"] = "comparison-setup-record-v1"
    kit_hash: Digest
    procedure: Literal["ordinary", "external"]
    started_utc: str
    manual_steps: list[str] = Field(default_factory=list)
    prebuilt_custom_work_reused: Literal[True] = True

    @model_validator(mode="after")
    def explicit_start(self) -> SetupRecord:
        aware_utc(self.started_utc)
        return self


def setup_measurement(
    record: SetupRecord | None,
    *,
    kit_hash: str,
    procedure: str,
    first_valid_utc: str | None,
) -> dict[str, Any]:
    if record is None:
        return {
            "owner_started_utc": None,
            "owner_started_to_first_valid_seconds": None,
            "manual_steps": None,
            "total_setup_seconds": None,
            "prebuilt_custom_work_reused": True,
            "setup_scope": "unmeasured",
        }
    record = SetupRecord.model_validate(record.model_dump(mode="json"))
    if record.kit_hash != kit_hash or record.procedure != procedure:
        raise ValueError("comparison_matching_setup_record_required")
    elapsed = None
    if first_valid_utc is not None:
        difference = (aware_utc(first_valid_utc) - aware_utc(record.started_utc)).total_seconds()
        if difference >= 0:
            elapsed = difference
    return {
        "owner_started_utc": record.started_utc,
        "owner_started_to_first_valid_seconds": elapsed,
        "manual_steps": record.manual_steps,
        "total_setup_seconds": None,
        "prebuilt_custom_work_reused": True,
        "setup_scope": (
            "Owner-declared start to observed first valid result; prebuilt engineering excluded."
        ),
    }


def check_review(manifest: dict[str, Any], *, kit_hash: str, review: OwnerReview | None) -> None:
    """Verify the separately recorded human review; never manufacture owner mode."""
    if content_hash(manifest) != kit_hash or review is None:
        raise ValueError("comparison_explicit_owner_review_required")
    review = OwnerReview.model_validate(review.model_dump(mode="json"))
    scopes = manifest["source_eligibility"]
    if (
        manifest.get("synthetic_only") is not False
        or manifest.get("m15_eligible") is not True
        or not scopes
        or any(value.get("eligible_source_scope") is not True for value in scopes.values())
        or review.kit_hash != kit_hash
        or review.selection_hash != manifest["selection_hash"]
        or review.controls_hash != manifest["controls_hash"]
        or review.source_run_ids != sorted(scopes)
    ):
        raise ValueError("comparison_reviewed_m15_source_and_controls_required")


def authorize(
    manifest: dict[str, Any],
    *,
    kit_hash: str,
    mode: Mode,
    review: OwnerReview | None,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Check the actual parent environment; this function never sets any variable."""
    current = os.environ if environment is None else environment
    if content_hash(manifest) != kit_hash:
        raise ValueError("comparison_external_kit_anchor_changed")
    if mode == "synthetic":
        if (
            manifest.get("synthetic_only") is not True
            or review is not None
            or any(current.get(name) for name in OWNER_VARIABLES)
        ):
            raise ValueError("comparison_synthetic_source_and_non_owner_context_required")
    elif mode == "owner":
        if not any(current.get(name) == "1" for name in OWNER_VARIABLES):
            raise ValueError("comparison_owner_environment_required")
        check_review(manifest, kit_hash=kit_hash, review=review)
    else:
        raise ValueError("comparison_explicit_execution_mode_required")
