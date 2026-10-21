"""Owner-origin tiers need pre-capture bindings; numbers retain the frozen algorithm."""

import copy
import hashlib
import os
from datetime import datetime

import pytest
from unit.test_refs import reference_run
from unit.test_selfhost_plan import config
from unit.test_selfhost_runtime import prepared, receipt
from unit.test_selfhost_storage import SID

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.models import (
    Attempt,
    EndpointSpec,
    Exchange,
    ModelProfile,
    OpenRouterPin,
    Timings,
)
from servegauge.refs.owner import (
    Qualification,
    bind_endpoint,
    build_owner_pack,
    qualification_template,
    qualify_capture,
)
from servegauge.refs.pack import build_pack, load_pack, save_pack
from servegauge.selfhost.runtime import validate_runtime


def inputs(tier="R1", maker="deepseek", router=False):
    weights, slug, url = {
        "deepseek": ("deepseek-ai/DeepSeek-V4", "deepseek/deepseek-v4", "https://api.deepseek.com"),
        "moonshotai": ("moonshotai/Kimi-K3", "moonshotai/kimi-k3", "https://api.moonshot.ai/v1"),
        "zai": ("zai-org/GLM-5.3", "z-ai/glm-5.3", "https://api.z.ai/api/paas/v4"),
    }[maker]
    profile = ModelProfile(id="test-model", weights=weights, openrouter_slug=slug)
    endpoint = EndpointSpec(
        id="reference",
        kind="openai_compat",
        model=slug.split("/")[1],
        base_url=url,
        claimed_profile=profile.id,
    )
    if router:
        endpoint.kind = "openrouter"
        endpoint.base_url = "https://openrouter.ai/api/v1"
        endpoint.model = slug
        endpoint.openrouter_pin = OpenRouterPin(tag=maker)
        endpoint.labels["provider_name"] = {
            "deepseek": "DeepSeek",
            "moonshotai": "Moonshot AI",
            "zai": "Z.AI",
        }[maker]
    if tier == "R2":
        profile = ModelProfile(id="gpt-oss-120b", weights="openai/gpt-oss-120b")
        endpoint = EndpointSpec(
            id="reference",
            kind="selfhosted",
            base_url="http://127.0.0.1:8000/v1",
            model=profile.id,
            claimed_profile=profile.id,
            api_key_env="SELFHOST_API_KEY",
        )
    value = qualification_template(tier, profile, endpoint, maker=maker)
    value.update(
        reviewed_at="2026-10-20T12:00:00+00:00", source_sha256="d" * 64, model_mapping_reviewed=True
    )
    if tier == "R2":
        value["origin"].update(
            session_id=SID,
            plan_hash=prepared()["plan_hash"],
            runtime_sha256=hashlib.sha256(
                canonical_json(validate_runtime(SID, prepared(), receipt())) + b"\n"
            ).hexdigest(),
        )
    request = Qualification.model_validate(value)
    endpoint = bind_endpoint(request, profile, endpoint)
    source = reference_run()
    source.profile = profile.id
    source.endpoints = [
        source.endpoints[0].model_copy(
            update={
                "id": endpoint.id,
                "profile": profile.id,
                "kind": endpoint.kind,
                "seed": None,
                "clean": False,
            }
        )
    ]
    source.baseline_endpoint_id = endpoint.id
    source.checks = {"T02": {endpoint.id: source.checks["T02"]["a"]}}
    source.manifest = {
        "run_id": source.run_id,
        "created_at": "2026-10-20T13:00:01+00:00",
        "endpoint_metadata": [endpoint.model_dump(mode="json")],
    }
    return request, profile, endpoint, source


def records(source, endpoint):
    timing = Timings(
        start_unix=datetime.fromisoformat("2026-10-20T13:00:02+00:00").timestamp(), total_ms=100
    )
    response = {"id": "gen-fixture-1", "model": endpoint.model, "choices": []}
    request = {"model": endpoint.model, "messages": []}
    metadata = {}
    if endpoint.kind == "openrouter":
        request["provider"] = endpoint.openrouter_pin.model_dump(exclude={"tag"}) | {
            "only": [endpoint.openrouter_pin.tag]
        }
        metadata = {
            "generation_record_status": "available",
            "generation_record": {
                "id": response["id"],
                "provider_name": endpoint.labels["provider_name"],
            },
        }
    return [
        Exchange(
            exchange_id="fixture",
            run_id=source.run_id,
            endpoint_id=endpoint.id,
            suite_id="S2",
            item_id="item00",
            sample_idx=0,
            request=request,
            request_url=endpoint.base_url + "/chat/completions",
            status=200,
            response=response,
            timings=timing,
            metadata=metadata,
            attempts=[
                Attempt(
                    n=1,
                    status=200,
                    timings=timing,
                    response=response,
                    metadata=copy.deepcopy(metadata),
                )
            ],
        )
    ]


def session_proof(request, endpoint):
    return {
        "session_id": SID,
        "running_at": "2026-10-20T13:00:00+00:00",
        "terminated_at": None,
        "reservation_id": "e" * 32,
        "plan": prepared(),
        "runtime": receipt(),
        "running": {
            "remote_id": "f" * 64,
            "endpoint_url": endpoint.base_url,
            "runtime_receipt_sha256": request.origin.runtime_sha256,
        },
        "reservation": {
            "reservation_id": "e" * 32,
            "owner": True,
            "dispatched": True,
            "run_id": SID,
            "amount": prepared()["reserved_upper_usd"],
            "endpoint_id": "selfhost:ssh",
            "exchange_id": SID,
            "request_hash": content_hash({"session_id": SID, "plan_hash": prepared()["plan_hash"]}),
        },
    }


@pytest.mark.parametrize("maker", ["deepseek", "moonshotai", "zai"])
@pytest.mark.parametrize("router", [False, True])
def test_firstparty_origin_and_prebinding_are_required(maker, router):
    request, profile, endpoint, source = inputs(maker=maker, router=router)
    proof = qualify_capture(source, profile, request, records(source, endpoint))
    assert proof["tier"] == "R1" and proof["source_run"] == source.run_id
    assert proof["hardware_attested"] is False and proof["independence_qualified"] is False
    assert proof["tag_verified"] is False
    source.manifest["endpoint_metadata"][0]["metadata"].clear()
    with pytest.raises(ValueError, match="pre_capture"):
        qualify_capture(source, profile, request, records(source, endpoint))


@pytest.mark.parametrize(
    "change",
    [
        {"base_url": "https://api.deepseek.com.evil.test"},
        {"base_url": "https://api.deepseek.com?key=fixture"},
        {"kind": "mock"},
        {"claimed_profile": "other"},
        {"model": "other"},
    ],
)
def test_endpoint_replacement_and_spoofed_creator_refuse(change):
    request, profile, endpoint, _ = inputs()
    replacement = endpoint.model_copy(update=change, deep=True)
    with pytest.raises(ValueError):
        bind_endpoint(request, profile, replacement)


def test_review_is_strict_and_templates_are_intentionally_incomplete():
    request, profile, endpoint, _ = inputs()
    template = qualification_template("R1", profile, endpoint, maker="deepseek")
    with pytest.raises(ValueError):
        Qualification.model_validate(template)
    for changed in [
        {"model_mapping_reviewed": 1},
        {"model_mapping_reviewed": False},
        {"reviewed_at": "2026-10-20T12:00:00"},
        {"source_url": "https://example.test"},
        {"source_sha256": "latest"},
        {"extra": "fixture"},
    ]:
        with pytest.raises(ValueError):
            Qualification.model_validate(request.model_dump(mode="json") | changed)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "provider",
        "generation",
        "attempt",
        "fallback",
        "model",
        "url",
        "timing",
        "duplicate",
    ],
)
def test_unlinked_or_contradictory_router_capture_refuses(mutation):
    request, profile, endpoint, source = inputs(router=True)
    raw = records(source, endpoint)
    record = raw[0]
    if mutation == "missing":
        record.metadata.clear()
    elif mutation == "provider":
        record.metadata["generation_record"]["provider_name"] = "Other"
    elif mutation == "generation":
        record.response["id"] = "gen-other"
    elif mutation == "attempt":
        record.attempts[0].metadata.clear()
    elif mutation == "fallback":
        record.request["provider"]["allow_fallbacks"] = True
    elif mutation == "model":
        record.request["model"] = "other"
    elif mutation == "url":
        record.request_url += "?fixture=1"
    elif mutation == "timing":
        record.timings = None
    elif mutation == "duplicate":
        raw *= 2
    with pytest.raises(ValueError):
        qualify_capture(source, profile, request, raw)


def test_selfhost_needs_exact_runtime_running_session_and_dispatched_owner_reservation():
    request, profile, endpoint, source = inputs("R2")
    raw = records(source, endpoint)
    proof = session_proof(request, endpoint)
    result = qualify_capture(source, profile, request, raw, session=proof)
    assert result["tier"] == "R2" and result["engine"]["image"] == config().image
    for section, key, value in [
        ("runtime", "engine_args", []),
        ("runtime", "gpu_observed", "other"),
        ("runtime", "model_revision", "c" * 40),
        ("running", "runtime_receipt_sha256", "c" * 64),
        ("running", "endpoint_url", "http://127.0.0.1:8001/v1"),
        ("reservation", "owner", False),
        ("reservation", "dispatched", False),
        ("reservation", "request_hash", "c" * 64),
        ("reservation", "amount", "0"),
    ]:
        changed = copy.deepcopy(proof)
        changed[section][key] = value
        with pytest.raises(ValueError):
            qualify_capture(source, profile, request, raw, session=changed)
    with pytest.raises(ValueError):
        qualify_capture(source, profile, request, raw)


def test_qualified_pack_preserves_numeric_calibration_and_does_not_mutate_inputs(tmp_path):
    request, profile, endpoint, source = inputs()
    raw = records(source, endpoint)
    original = source.model_dump(mode="json")
    baseline = build_pack(source, "BASELINE", seed=8)
    pack = build_owner_pack(
        source, profile, request, raw, seed=8, environment={"SERVEGAUGE_OWNER_RUN": "1"}
    )
    assert pack.tier == "R1" and pack.pack_id != baseline.pack_id
    assert pack.outcomes == baseline.outcomes and pack.calibration == baseline.calibration
    assert pack.fingerprints == baseline.fingerprints
    assert source.model_dump(mode="json") == original
    assert load_pack(str(save_pack(pack, tmp_path))) == pack
    for environment in [{}, {"SERVEGAUGE_OWNER_RUN": "0"}, {"TRUEWEIGHTS_OWNER_RUN": "0"}]:
        with pytest.raises(ValueError, match="owner_authorization"):
            build_owner_pack(source, profile, request, raw, seed=8, environment=environment)
    assert os.environ.get("SERVEGAUGE_OWNER_RUN") is None
    assert os.environ.get("TRUEWEIGHTS_OWNER_RUN") is None


def test_preflight_observations_do_not_replace_the_reviewed_origin():
    request, profile, endpoint, source = inputs()
    source.manifest["endpoint_metadata"][0]["metadata"]["observed_capabilities"] = {"answers": True}
    assert qualify_capture(source, profile, request, records(source, endpoint))["tier"] == "R1"
    source.manifest["endpoint_metadata"][0]["defaults"]["temperature"] = 0.2
    with pytest.raises(ValueError):
        qualify_capture(source, profile, request, records(source, endpoint))


@pytest.mark.parametrize(
    "field,value",
    [("running_at", "2026-10-20T13:00:03+00:00"), ("terminated_at", "2026-10-20T13:00:02+00:00")],
)
def test_selfhost_capture_must_fit_observed_session_lifetime(field, value):
    request, profile, endpoint, source = inputs("R2")
    session = session_proof(request, endpoint) | {field: value}
    with pytest.raises(ValueError, match="session_lifetime"):
        qualify_capture(source, profile, request, records(source, endpoint), session=session)


def test_router_reused_generation_is_not_a_new_reference_observation():
    request, profile, endpoint, source = inputs(router=True)
    raw = records(source, endpoint)
    raw.append(raw[0].model_copy(update={"exchange_id": "second"}, deep=True))
    with pytest.raises(ValueError, match="unique_single_attempt"):
        qualify_capture(source, profile, request, raw)


def test_still_recorded_running_session_cannot_qualify_capture_after_its_remote_lease():
    request, profile, endpoint, source = inputs("R2")
    raw = records(source, endpoint)
    raw[0].timings.start_unix += prepared()["maximum_duration_seconds"] + 60
    with pytest.raises(ValueError, match="bounded_session_lease"):
        qualify_capture(source, profile, request, raw, session=session_proof(request, endpoint))


def test_ssh_client_port_mapping_requires_explicit_owner_review():
    request, profile, endpoint, source = inputs("R2")
    session = session_proof(request, endpoint)
    endpoint.base_url = "http://127.0.0.1:8008/v1"
    endpoint.metadata.clear()
    request.endpoint_hash = content_hash(endpoint.model_dump(mode="json"))
    endpoint = bind_endpoint(request, profile, endpoint)
    source.manifest["endpoint_metadata"] = [endpoint.model_dump(mode="json")]
    with pytest.raises(ValueError, match="client_to_remote"):
        qualify_capture(source, profile, request, records(source, endpoint), session=session)
    request.origin.ssh_tunnel_reviewed = True
    endpoint.metadata.clear()
    endpoint = bind_endpoint(request, profile, endpoint)
    source.manifest["endpoint_metadata"] = [endpoint.model_dump(mode="json")]
    assert (
        qualify_capture(source, profile, request, records(source, endpoint), session=session)[
            "tier"
        ]
        == "R2"
    )


def test_modal_reference_binds_observed_sandbox_and_exact_endpoint():
    from decimal import Decimal

    from unit.test_selfhost_plan import policy

    from servegauge.selfhost.plan import build_plan

    request, profile, endpoint, source = inputs("R2")
    plan = build_plan(
        config(
            backend="modal",
            ssh_host=None,
            ssh_key_env=None,
            modal_app="servegauge-owner",
            modal_image_id="im-fixture",
            inbound_cidrs=["198.51.100.7/32"],
        ),
        policy(),
        max_usd=Decimal("2"),
        shutdown_minutes=10,
    )
    value = receipt() | {
        "plan_hash": plan["plan_hash"],
        "backend": "modal",
        "container_command": None,
        "modal_sandbox_id": "sb-fixture",
        "modal_image_id": "im-fixture",
    }
    request.origin.plan_hash = plan["plan_hash"]
    request.origin.runtime_sha256 = hashlib.sha256(
        canonical_json(validate_runtime(SID, plan, value)) + b"\n"
    ).hexdigest()
    endpoint.base_url = "https://sandbox.example.test/v1"
    endpoint.metadata.clear()
    request.endpoint_hash = content_hash(endpoint.model_dump(mode="json"))
    endpoint = bind_endpoint(request, profile, endpoint)
    source.manifest["endpoint_metadata"] = [endpoint.model_dump(mode="json")]
    session = session_proof(request, endpoint)
    session.update(plan=plan, runtime=value)
    session["running"]["remote_id"] = "sb-fixture"
    session["reservation"]["endpoint_id"] = "selfhost:modal"
    session["reservation"]["request_hash"] = content_hash(
        {"session_id": SID, "plan_hash": plan["plan_hash"]}
    )
    assert (
        qualify_capture(source, profile, request, records(source, endpoint), session=session)[
            "tier"
        ]
        == "R2"
    )
    session["running"]["remote_id"] = "sb-other"
    with pytest.raises(ValueError, match="remote_identity"):
        qualify_capture(source, profile, request, records(source, endpoint), session=session)
