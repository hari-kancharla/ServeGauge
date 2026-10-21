"""Owner-reviewed origin tiers, separate from numerical or hardware qualification."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.models import EndpointSpec, Exchange, ModelProfile, ReferencePack, ValueModel
from servegauge.refs.pack import ReferenceRun, build_pack
from servegauge.selfhost.plan import validate_plan
from servegauge.selfhost.runtime import validate_runtime

BINDING = "qualified_reference"
RECIPE = "https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html"
GLM_CODING_SOURCE = "https://docs.z.ai/guides/llm/glm-5.3"
GLM_CODING_BASE = "https://api.z.ai/api/coding/paas/v4"
CREATORS = {
    "deepseek": {
        "weights": "deepseek-ai/",
        "slug": "deepseek/",
        "provider": "DeepSeek",
        "url": "https://api.deepseek.com",
        "source": "https://api-docs.deepseek.com/",
    },
    "moonshotai": {
        "weights": "moonshotai/",
        "slug": "moonshotai/",
        "provider": "Moonshot AI",
        "url": "https://api.moonshot.ai/v1",
        "source": "https://platform.kimi.ai/docs/overview",
    },
    "zai": {
        "weights": "zai-org/",
        "slug": "z-ai/",
        "provider": "Z.AI",
        "url": "https://api.z.ai/api/paas/v4",
        "source": "https://docs.z.ai/guides/overview/quick-start",
    },
}


def _aware(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("reference_aware_review_and_capture_dates_required")
    return result


class CreatorOrigin(ValueModel):
    kind: Literal["creator"]
    maker: Literal["deepseek", "moonshotai", "zai"]


class SelfhostOrigin(ValueModel):
    kind: Literal["selfhost"]
    session_id: str = Field(pattern=r"^selfhost-[a-f0-9]{32}$")
    plan_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    runtime_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    ssh_tunnel_reviewed: bool = Field(default=False, strict=True)


class Qualification(ValueModel):
    version: Literal["owner-reference-v1"]
    tier: Literal["R1", "R2"]
    profile: str = Field(min_length=1, max_length=128)
    profile_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    endpoint_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    reviewed_at: str = Field(max_length=64)
    source_url: str = Field(max_length=256)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_mapping_reviewed: bool = Field(strict=True)
    origin: Annotated[CreatorOrigin | SelfhostOrigin, Field(discriminator="kind")]

    @model_validator(mode="after")
    def reviewed(self) -> Self:
        _aware(self.reviewed_at)
        if not self.model_mapping_reviewed:
            raise ValueError("reference_owner_model_mapping_review_required")
        expected = {RECIPE}
        if isinstance(self.origin, CreatorOrigin):
            expected = {CREATORS[self.origin.maker]["source"]}
            if self.origin.maker == "zai" and self.profile == "glm-5.3":
                expected.add(GLM_CODING_SOURCE)
            if self.tier != "R1":
                raise ValueError("reference_origin_tier_mismatch")
        elif self.tier != "R2":
            raise ValueError("reference_origin_tier_mismatch")
        if self.source_url not in expected:
            raise ValueError("reference_researched_official_source_required")
        return self


def _unbound(endpoint: EndpointSpec) -> dict[str, Any]:
    value = endpoint.model_dump(mode="json")
    value["metadata"].pop(BINDING, None)
    # The runner appends observed preflight results before its primary capture.
    # Full captured metadata still enters the source proof; this cannot change origin.
    value["metadata"].pop("observed_capabilities", None)
    return value


def qualification_template(
    tier: str, profile: ModelProfile, endpoint: EndpointSpec, *, maker: str | None = None
) -> dict[str, Any]:
    """Incomplete local review form: a template never qualifies a capture."""
    if tier not in {"R1", "R2"} or (tier == "R1" and maker not in CREATORS):
        raise ValueError("reference_known_owner_tier_and_creator_required")
    source_url = CREATORS[str(maker)]["source"] if tier == "R1" else RECIPE
    if (
        tier == "R1"
        and maker == "zai"
        and profile.id == "glm-5.3"
        and endpoint.kind == "openai_compat"
        and endpoint.base_url == GLM_CODING_BASE
    ):
        source_url = GLM_CODING_SOURCE
    return {
        "version": "owner-reference-v1",
        "tier": tier,
        "profile": profile.id,
        "profile_hash": content_hash(profile.model_dump(mode="json")),
        "endpoint_hash": content_hash(_unbound(endpoint)),
        "reviewed_at": None,
        "source_url": source_url,
        "source_sha256": None,
        "model_mapping_reviewed": False,
        "origin": {"kind": "creator", "maker": maker}
        if tier == "R1"
        else {
            "kind": "selfhost",
            "session_id": None,
            "plan_hash": None,
            "runtime_sha256": None,
            "ssh_tunnel_reviewed": False,
        },
    }


def bind_endpoint(
    request: Qualification, profile: ModelProfile, endpoint: EndpointSpec
) -> EndpointSpec:
    if (
        request.profile != profile.id
        or endpoint.claimed_profile != profile.id
        or request.profile_hash != content_hash(profile.model_dump(mode="json"))
        or request.endpoint_hash != content_hash(_unbound(endpoint))
        or any(v in {"id", "weights"} for v in profile.unverified)
    ):
        raise ValueError("reference_reviewed_profile_or_endpoint_changed")
    parts = urlsplit(endpoint.base_url)
    if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
        raise ValueError("reference_canonical_credential_free_endpoint_required")
    if isinstance(request.origin, CreatorOrigin):
        creator = CREATORS[request.origin.maker]
        if not profile.weights.casefold().startswith(creator["weights"]):
            raise ValueError("reference_creator_model_namespace_mismatch")
        if endpoint.kind == "openrouter":
            pin = endpoint.openrouter_pin
            if (
                endpoint.base_url != "https://openrouter.ai/api/v1"
                or profile.openrouter_slug != endpoint.model
                or not endpoint.model.startswith(creator["slug"])
                or "openrouter_slug" in profile.unverified
                or endpoint.labels.get("provider_name") != creator["provider"]
                or pin is None
                or not pin.tag
                or not pin.require_parameters
                or pin.allow_fallbacks
            ):
                raise ValueError("reference_exact_maker_route_required")
        elif endpoint.kind != "openai_compat" or endpoint.base_url != (
            GLM_CODING_BASE if request.source_url == GLM_CODING_SOURCE else creator["url"]
        ):
            raise ValueError("reference_official_creator_endpoint_required")
    elif (
        endpoint.kind != "selfhosted"
        or profile.weights != "openai/" + profile.id
        or endpoint.model != profile.id
    ):
        raise ValueError("reference_official_selfhost_model_required")
    result = endpoint.model_copy(deep=True)
    binding = {
        "version": "qualified-reference-binding-v1",
        "qualification_hash": content_hash(request.model_dump(mode="json")),
    }
    if BINDING in result.metadata and result.metadata[BINDING] != binding:
        raise ValueError("reference_existing_pre_capture_binding_changed")
    result.metadata[BINDING] = binding
    return result


def _runtime(
    request: Qualification, endpoint: EndpointSpec, session: dict[str, Any] | None
) -> dict[str, Any]:
    origin = request.origin
    if not isinstance(origin, SelfhostOrigin) or session is None:
        raise ValueError("reference_selfhost_session_evidence_required")
    plan = session["plan"]
    validate_plan(plan)
    if plan.get("reference_qualification") != "pending_owner_capture":
        raise ValueError("reference_experiment_session_cannot_grant_r2")
    runtime = validate_runtime(origin.session_id, plan, session["runtime"])
    config = plan["configuration"]
    row = session["reservation"]
    running = session["running"]
    if (
        session["session_id"] != origin.session_id
        or plan["plan_hash"] != origin.plan_hash
        or hashlib.sha256(canonical_json(runtime) + b"\n").hexdigest() != origin.runtime_sha256
        or running.get("runtime_receipt_sha256") != origin.runtime_sha256
        or not running.get("remote_id")
        or plan["profile"] != request.profile
        or endpoint.api_key_env != config["api_key_env"]
        or row.get("owner") is not True
        or row.get("reservation_id") != session["reservation_id"]
        or row.get("dispatched") is not True
        or row.get("run_id") != origin.session_id
        or row.get("exchange_id") != origin.session_id
        or row.get("endpoint_id") != "selfhost:" + config["backend"]
        or row.get("request_hash")
        != content_hash({"session_id": origin.session_id, "plan_hash": origin.plan_hash})
        or Decimal(row["amount"]) != Decimal(plan["reserved_upper_usd"])
    ):
        raise ValueError("reference_selfhost_session_or_reservation_changed")
    remote_id = running["remote_id"]
    if (config["backend"] == "modal" and remote_id != runtime.get("modal_sandbox_id")) or (
        config["backend"] == "ssh" and not re.fullmatch(r"[a-f0-9]{64}", remote_id)
    ):
        raise ValueError("reference_runtime_remote_identity_changed")
    remote_url = running["endpoint_url"]
    if endpoint.base_url != remote_url:
        parts = urlsplit(endpoint.base_url)
        remote = urlsplit(remote_url)
        if (
            config["backend"] != "ssh"
            or not origin.ssh_tunnel_reviewed
            or parts.scheme != "http"
            or parts.hostname not in {"127.0.0.1", "::1"}
            or parts.path != "/v1"
            or parts.port is None
            or remote.scheme != "http"
            or remote.hostname not in {"127.0.0.1", "::1"}
            or remote.port != 8000
            or remote.path != "/v1"
        ):
            raise ValueError("reference_reviewed_client_to_remote_binding_required")
    return runtime


def qualify_capture(
    source: ReferenceRun,
    profile: ModelProfile,
    request: Qualification,
    records: list[Exchange],
    *,
    session: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pure origin checks after sealed-raw reconstruction; assertions are not attestation."""
    endpoints = [EndpointSpec.model_validate(v) for v in source.manifest["endpoint_metadata"]]
    matched = [e for e in endpoints if content_hash(_unbound(e)) == request.endpoint_hash]
    if len(matched) != 1 or source.profile != profile.id:
        raise ValueError("reference_unique_reviewed_capture_endpoint_required")
    endpoint = matched[0]
    expected = bind_endpoint(request, profile, endpoint)
    if endpoint != expected:
        raise ValueError("reference_pre_capture_binding_required")
    declared = [e for e in source.endpoints if e.id == endpoint.id]
    if len(declared) != 1 or declared[0].profile != profile.id or declared[0].kind != endpoint.kind:
        raise ValueError("reference_captured_endpoint_declaration_changed")
    if _aware(source.manifest["created_at"]) < _aware(request.reviewed_at):
        raise ValueError("reference_review_must_precede_capture")
    if source.manifest.get("run_id") != source.run_id:
        raise ValueError("reference_original_capture_identity_changed")
    primary = [r for r in records if r.run_id == source.run_id and r.endpoint_id == endpoint.id]
    if not primary or len({r.exchange_id for r in primary}) != len(primary):
        raise ValueError("reference_unique_raw_capture_required")
    runtime = _runtime(request, endpoint, session) if request.tier == "R2" else None
    generations: set[str] = set()
    for record in primary:
        if (
            record.request_url
            not in {endpoint.base_url + "/chat/completions", endpoint.base_url + "/completions"}
            or record.request.get("model") != endpoint.model
            or record.timings is None
            or record.timings.total_ms < 0
            or datetime.fromtimestamp(record.timings.start_unix, _aware(request.reviewed_at).tzinfo)
            < _aware(request.reviewed_at)
            or record.error is not None
            or record.status != 200
            or not record.attempts
            or record.attempts[-1].status != 200
            or record.attempts[-1].error is not None
            or any(
                record.metadata.get(k)
                for k in ["request_not_sent", "incomplete", "sensitive_response_redacted"]
            )
        ):
            raise ValueError("reference_complete_bound_raw_transport_required")
        if runtime and record.timings.start_unix < _aware(runtime["recorded_at"]).timestamp():
            raise ValueError("reference_runtime_observation_must_precede_capture")
        if runtime and record.timings.start_unix + record.timings.total_ms / 1000 > (
            _aware(runtime["recorded_at"]).timestamp() + runtime["shutdown_minutes"] * 60
        ):
            raise ValueError("reference_capture_outside_bounded_session_lease")
        if (
            runtime
            and session is not None
            and (
                record.timings.start_unix < _aware(session["running_at"]).timestamp()
                or (
                    session["terminated_at"] is not None
                    and record.timings.start_unix + record.timings.total_ms / 1000
                    > _aware(session["terminated_at"]).timestamp()
                )
            )
        ):
            raise ValueError("reference_capture_outside_observed_session_lifetime")
        if endpoint.kind == "openrouter":
            from servegauge.scoring.routing import _identity

            if _identity(record, endpoint) is not False:
                raise ValueError("reference_linked_maker_generation_evidence_required")
            generation_id = str(record.metadata["generation_record"]["id"])
            if generation_id in generations or len(record.attempts) != 1:
                raise ValueError("reference_unique_single_attempt_maker_capture_required")
            generations.add(generation_id)
            responses = record.metadata["generation_record"].get("provider_responses", [])
            if not isinstance(responses, list) or any(
                not isinstance(row, dict)
                or row.get("provider_name") != endpoint.labels["provider_name"]
                for row in responses
            ):
                raise ValueError("reference_maker_attempt_origin_unverified")
    proof: dict[str, Any] = {
        "version": "qualified-reference-capture-v1",
        "tier": request.tier,
        "source_run": source.run_id,
        "endpoint_id": endpoint.id,
        "qualification_hash": content_hash(request.model_dump(mode="json")),
        "source_binding": content_hash(
            {
                "source": source.model_dump(mode="json"),
                "records": [r.model_dump(mode="json") for r in primary],
            }
        ),
        "raw_requests": len(primary),
        "hardware_attested": False,
        "independence_qualified": False,
        "tag_verified": False,
        "trust_basis": (
            "owner-reviewed origin and model mapping; local integrity, not remote attestation"
        ),
    }
    if runtime:
        proof["engine"] = {
            k: runtime[k]
            for k in [
                "engine_version",
                "image",
                "model_revision",
                "gpu_observed",
                "driver_version",
                "engine_args",
                "recorded_at",
            ]
        }
        proof["session_id"] = (
            request.origin.session_id if isinstance(request.origin, SelfhostOrigin) else None
        )
    return proof


@dataclass
class CapturedReference:
    source: ReferenceRun
    records: list[Exchange]
    session: dict[str, Any] | None = None


def origin_review(request: Qualification, proofs: list[dict[str, Any]]) -> dict[str, Any]:
    """Public provenance excludes session/client identities while retaining the recipe."""
    return {
        "version": "owner-reference-origin-v1",
        "tier": request.tier,
        "reviewed_at": request.reviewed_at,
        "source_url": request.source_url,
        "source_sha256": request.source_sha256,
        "profile_hash": request.profile_hash,
        "model_mapping_reviewed": True,
        "creator": request.origin.maker if isinstance(request.origin, CreatorOrigin) else None,
        "captures": [
            {k: v for k, v in proof.items() if k not in {"session_id", "endpoint_id"}}
            for proof in proofs
        ],
        "trust_basis": "owner assertions and intact local evidence; no remote attestation",
    }


def build_owner_pack(
    source: ReferenceRun,
    profile: ModelProfile,
    request: Qualification,
    records: list[Exchange],
    *,
    seed: int,
    environment: Mapping[str, str],
    session: dict[str, Any] | None = None,
    controls: list[ReferenceRun] | None = None,
    references: list[CapturedReference] | None = None,
    resamples: int = 2000,
) -> ReferencePack:
    if environment.get("SERVEGAUGE_OWNER_RUN", environment.get("TRUEWEIGHTS_OWNER_RUN")) != "1":
        raise ValueError("reference_owner_authorization_required")
    if references and not source.fingerprints:
        raise ValueError("reference_additional_runs_require_fingerprint_capture")
    captures = [CapturedReference(source, records, session), *(references or [])]
    if len({c.source.run_id for c in captures}) != len(captures):
        raise ValueError("reference_distinct_capture_runs_required")
    proofs = [
        qualify_capture(c.source, profile, request, c.records, session=c.session) for c in captures
    ]
    selected = source.model_copy(
        update={"baseline_endpoint_id": proofs[0]["endpoint_id"]}, deep=True
    )
    baseline = build_pack(
        selected,
        "BASELINE",
        seed=seed,
        resamples=resamples,
        controls=controls,
        reference_runs=[c.source for c in references or []],
    )
    pack = baseline.model_copy(deep=True)
    pack.tier = request.tier
    pack.pack_id = (
        "ref-"
        + content_hash(
            {
                "baseline": baseline.manifest["pack_content_hash"],
                "request": request.model_dump(mode="json"),
                "captures": proofs,
            }
        )[:20]
    )
    pack.manifest["owner_qualification"] = {
        "request": request.model_dump(mode="json"),
        "captures": proofs,
    }
    pack.manifest["origin_review"] = origin_review(request, proofs)
    pack.manifest["qualification"] = (
        "Owner-reviewed origin tier; no provider equivalence, independence or hardware attestation"
    )
    pack.manifest.pop("pack_content_hash")
    pack.manifest["pack_content_hash"] = content_hash(pack.model_dump(mode="json"))
    return pack
