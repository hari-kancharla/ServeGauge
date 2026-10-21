"""Version-bound local right-of-reply assertions; never delivery verification."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, model_validator

from servegauge.core.hashing import content_hash
from servegauge.core.models import ValueModel

if TYPE_CHECKING:
    from servegauge.storage.analysis import Analysis

Digest = str


def validate_bundle_reference(value: str) -> str:
    if not isinstance(value, str) or "\\" in value or "\x00" in value:
        raise ValueError("notice_bundle_reference_refused")
    path = PurePosixPath(value)
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *[f"COM{i}" for i in range(1, 10)],
        *[f"LPT{i}" for i in range(1, 10)],
    }
    if any(ord(c) < 32 or c in '<>:"|?*' for c in value) or any(
        part.rstrip(" .") != part or part.split(".")[0].upper() in reserved for part in path.parts
    ):
        raise ValueError("notice_bundle_reference_refused")
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.parts[0] != "bundles"
        or path.name != "bundle.json"
        or path.as_posix() != value
        or any(p.lower() == "starter files" for p in path.parts)
    ):
        raise ValueError("notice_bundle_reference_refused")
    return value


class FailureSubject(ValueModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    version: Literal["notice-failure-v1"] = "notice-failure-v1"
    run_id: str = Field(min_length=1)
    endpoint_id: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    check_id: str = Field(min_length=1)
    synthetic_mock: bool = Field(strict=True)
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def subject_id(self) -> str:
        return "failure-" + content_hash(self.model_dump(mode="json"))


def aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("notice_aware_utc_timestamp_required")
    return value.astimezone(UTC)


def channel_origin(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.port not in {None, 443}
    ):
        raise ValueError("notice_public_https_channel_required")
    return "https://" + parsed.hostname.lower()


class NoticeAssertion(ValueModel):
    type: Literal["sent", "reply", "already_public", "withdrawn"]
    at: str
    owner_asserted: bool = Field(strict=True)
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    channel_origin: str | None = None

    @model_validator(mode="after")
    def explicit_assertion(self) -> NoticeAssertion:
        if self.owner_asserted is not True:
            raise ValueError("notice_explicit_owner_assertion_required")
        return self


def validate_history(spec: dict[str, Any], events: list[dict[str, Any]]) -> None:
    validate_bundle_reference(spec["bundle_reference"])
    subject = FailureSubject.model_validate(spec["subject"])
    if (
        spec.get("version") != "notice-spec-v1"
        or spec.get("subject_id") != subject.subject_id
        or spec.get("notice_id")
        != "notice-" + content_hash([subject.subject_id, spec["bundle_sha256"]])
        or len(spec["bundle_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in spec["bundle_sha256"])
    ):
        raise ValueError("notice_spec_binding_invalid")
    prepared = aware(datetime.fromisoformat(spec["prepared_at"]))
    sent_at = None
    seen = set()
    withdrawn = False
    for data in events:
        row = NoticeAssertion.model_validate(data)
        stamp = aware(datetime.fromisoformat(row.at))
        if withdrawn:
            raise ValueError("notice_assertion_history_invalid")
        if row.type in {"sent", "already_public", "withdrawn"} and row.type in seen:
            raise ValueError("notice_assertion_conflict")
        if row.type == "sent":
            if stamp < prepared or row.channel_origin is None:
                raise ValueError("notice_send_precedes_preparation_or_channel_missing")
            sent_at = stamp
        if row.type == "already_public" and row.channel_origin is None:
            raise ValueError("notice_public_source_required")
        if row.channel_origin != channel_origin(row.channel_origin):
            raise ValueError("notice_channel_not_canonical")
        if row.type == "reply" and (sent_at is None or stamp < sent_at):
            raise ValueError("notice_reply_without_prior_send")
        withdrawn = row.type == "withdrawn"
        seen.add(row.type)


def failure_subject(analysis: Analysis, endpoint_id: str, check_id: str) -> FailureSubject:
    endpoint = next(e for e in analysis.plan.endpoints if e.id == endpoint_id)
    checks = [c for c in analysis.checks if c.endpoint_id == endpoint_id and c.check_id == check_id]
    if len(checks) != 1 or checks[0].verdict != "FAIL":
        raise ValueError("notice_exact_failed_check_required")
    check = checks[0]
    return FailureSubject(
        run_id=check.run_id,
        endpoint_id=endpoint_id,
        provider_id=endpoint.openrouter_pin.tag if endpoint.openrouter_pin else endpoint.id,
        check_id=check_id,
        synthetic_mock=endpoint.kind == "mock",
        evidence_sha256=content_hash(
            {
                "check": check.model_dump(mode="json"),
                "plan": analysis.plan.model_dump(mode="json"),
                "manifest": analysis.manifest,
                "records": {
                    r.exchange_id: content_hash(r.model_dump(mode="json"))
                    for r in analysis.records
                    if r.endpoint_id == endpoint_id
                },
            }
        ),
    )


def notice_state(
    subject: FailureSubject, snapshots: list[dict[str, Any]], *, now: datetime
) -> dict[str, Any]:
    now = aware(now)
    eligible = []
    for snapshot in snapshots:
        spec, events = snapshot["spec"], snapshot["events"]
        validate_history(spec, events)
        if spec["subject"] != subject.model_dump(mode="json"):
            continue
        rows = [NoticeAssertion.model_validate(e) for e in events]
        if any(aware(datetime.fromisoformat(r.at)) > now for r in rows):
            continue
        if rows and rows[-1].type == "withdrawn":
            continue
        public = next((r for r in rows if r.type == "already_public"), None)
        sent = next((r for r in rows if r.type == "sent"), None)
        if public or (sent and now >= aware(datetime.fromisoformat(sent.at)) + timedelta(days=14)):
            eligible.append(
                {
                    "eligible": True,
                    "notice_id": spec["notice_id"],
                    "basis": "owner_asserted_already_public"
                    if public
                    else "owner_asserted_send_plus_14_days",
                    "owner_asserted": True,
                    "reply_count": sum(r.type == "reply" for r in rows),
                    "sent_at": sent.at if sent else None,
                }
            )
    return (
        eligible[0]
        if eligible
        else {
            "eligible": False,
            "basis": "missing_withdrawn_or_waiting_notice",
            "owner_asserted": False,
            "reply_count": 0,
            "sent_at": None,
        }
    )
