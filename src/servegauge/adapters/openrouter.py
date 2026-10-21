"""OpenRouter contracts: exact declared pins, safe parameters and provider-level verification."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import httpx

from servegauge.adapters.base import UnsupportedFeature
from servegauge.core.errors import SafeError
from servegauge.core.hashing import content_hash
from servegauge.core.models import ChatRequest, EndpointSpec, OpenRouterPin
from servegauge.core.redaction import Redactor, sensitive_key
from servegauge.runner.ratelimit import Sleep

API_BASE = "https://openrouter.ai/api/v1"
TRANSFORMS = [{"id": "context-compression", "enabled": False}]
TAG_LIMITATION = (
    "Generation records identify providers, not exact endpoint tags, variants or regions."
)


def price_provenance(
    source: str,
    model: str,
    tag: str,
    pricing: dict[str, Any],
    *,
    captured_at: str | None = None,
) -> dict[str, Any]:
    """Bind the public endpoint price snapshot, not an invoice or future price guarantee."""
    return {
        "version": "openrouter-token-prices-v1",
        "verified": True,
        "currency": "USD",
        "source": source,
        "model": model,
        "endpoint_tag": tag,
        "pricing_hash": content_hash(pricing),
        "captured_at": captured_at or datetime.now(UTC).isoformat(),
    }


async def fetch_catalog(
    client: httpx.AsyncClient, headers: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    value = await _metadata_get(client, API_BASE + "/models", headers)
    data = value.get("data")
    if not isinstance(data, list) or not all(isinstance(m, dict) for m in data):
        raise SafeError("model_catalog_schema_unverified")
    return data


class OpenRouterAdapter:
    def build_request(self, request: ChatRequest, endpoint: EndpointSpec) -> dict[str, Any]:
        pin, meta = endpoint.openrouter_pin, endpoint.metadata
        if pin is None or pin.allow_fallbacks:
            raise ValueError("exact_pin_with_fallbacks_disabled_required")
        supported = meta.get("supported_parameters")
        if not isinstance(supported, list) or not all(isinstance(p, str) for p in supported):
            raise UnsupportedFeature(
                "metadata_unverified", ["supported_parameters"], uncertain=True
            )
        if meta.get("tag") != pin.tag:
            raise ValueError("pin_differs_from_endpoint_snapshot")
        cap = meta.get("max_completion_tokens")
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
            raise UnsupportedFeature(
                "metadata_unverified", ["max_completion_tokens"], uncertain=True
            )
        if request.minimum_output_tokens and request.minimum_output_tokens > cap:
            raise UnsupportedFeature(
                "required_output_exceeds_endpoint_cap", ["max_completion_tokens"]
            )
        parameters = dict(request.params)
        if any(sensitive_key(key) for key in parameters):
            raise ValueError("credentials_must_come_from_environment_headers")
        requested = parameters.pop(
            "max_completion_tokens",
            parameters.get(
                "max_tokens",
                endpoint.defaults.get(
                    "max_completion_tokens", endpoint.defaults.get("max_tokens", 256)
                ),
            ),
        )
        if not isinstance(requested, int) or isinstance(requested, bool) or requested < 1:
            raise ValueError("explicit_positive_output_cap_required")
        output_key = "max_tokens" if "max_tokens" in supported else "max_completion_tokens"
        parameters.pop("max_tokens", None)
        parameters[output_key] = min(requested, cap)
        for key in ("tools", "tool_choice", "response_format"):
            value = getattr(request, key)
            if value is not None:
                parameters[key] = value
        required = set(parameters) - {
            "stream_options",
            "provider",
            "plugins",
            "model",
            "messages",
            "stream",
        }
        unsupported = sorted(required - set(supported))
        if (
            request.response_format
            and request.response_format.get("type") == "json_schema"
            and "structured_outputs" not in supported
        ):
            unsupported.append("structured_outputs")
        if request.tool_choice is not None:
            mode = "function" if isinstance(request.tool_choice, dict) else request.tool_choice
            choices = meta.get("supports_tool_choice", {})
            if not isinstance(choices, dict) or choices.get(mode) is not True:
                unsupported.append("tool_choice:" + mode)
        if unsupported and request.unsupported_probe != "P13":
            raise UnsupportedFeature("unsupported_parameter", unsupported)
        body = {
            key: value
            for key, value in parameters.items()
            if key not in {"provider", "plugins", "model", "messages", "stream"}
        }
        body.update(
            {
                "model": endpoint.model,
                "messages": request.messages,
                "stream": request.stream,
                "provider": {
                    "only": [pin.tag],
                    "allow_fallbacks": False,
                    "require_parameters": pin.require_parameters
                    and request.unsupported_probe != "P13",
                },
                "plugins": [dict(flag) for flag in TRANSFORMS],
            }
        )
        if request.stream:
            body["stream_options"] = {**body.get("stream_options", {}), "include_usage": True}
        return body


async def _metadata_get(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        response = await client.get(url, headers=headers, params=params)
        if response.status_code != 200:
            raise SafeError("metadata_unavailable")
        value = response.json()
        if not isinstance(value, dict):
            raise SafeError("metadata_schema_unverified")
        credential = (headers or {}).get("authorization", "").removeprefix("Bearer ")
        return dict(Redactor.from_environment().with_values(credential).redact(value))
    except (httpx.HTTPError, ValueError, TypeError):
        raise SafeError("metadata_unavailable") from None


async def discover_model(
    client: httpx.AsyncClient,
    model: str,
    *,
    headers: dict[str, str] | None = None,
    base_url: str = API_BASE,
) -> dict[str, Any]:
    parts = model.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("model must be author/slug")
    url = (
        base_url.rstrip("/")
        + "/models/"
        + "/".join(quote(p, safe="") for p in parts)
        + "/endpoints"
    )
    value = await _metadata_get(client, url, headers)
    data = value.get("data", {})
    if not isinstance(data, dict) or not isinstance(data.get("endpoints"), list):
        raise SafeError("endpoint_schema_unverified")
    endpoints = []
    for raw in data["endpoints"]:
        if (
            not isinstance(raw, dict)
            or not isinstance(raw.get("tag"), str)
            or not isinstance(raw.get("provider_name"), str)
        ):
            raise SafeError("endpoint_identity_unverified")
        record = EndpointSpec(
            id=f"or:{model}@{raw['tag']}",
            kind="openrouter",
            base_url=base_url,
            model=model,
            api_key_env="OPENROUTER_API_KEY",
            claimed_profile=parts[1],
            openrouter_pin=OpenRouterPin(tag=raw["tag"]),
            labels={
                key: str(raw.get(key) or "unknown")
                for key in ("provider_name", "quantization", "service_tier")
            },
            metadata={
                **raw,
                **(
                    {"price_provenance": price_provenance(url, model, raw["tag"], raw["pricing"])}
                    if isinstance(raw.get("pricing"), dict)
                    else {}
                ),
                "declared_json_schema": {
                    "value": "structured_outputs" in raw.get("supported_parameters", []),
                    "verified": True,
                    "source": url,
                },
                "strict_schema_guarantee": {
                    "verified": False,
                    "features": [],
                    "source": "https://openrouter.ai/docs/guides/features/structured-outputs",
                    "reason": "router capability does not establish native strict enforcement",
                },
            },
        )
        endpoints.append(record.model_dump(mode="json"))
    return {
        "model": model,
        "discovered_at": datetime.now(UTC).isoformat(),
        "source_url": url,
        "endpoints": endpoints,
        "routing_verification_limit": TAG_LIMITATION,
    }


def flagship_candidates(models: list[dict[str, Any]]) -> list[str]:
    result = []
    for prefix in ("moonshotai/kimi", "deepseek/deepseek", "z-ai/glm"):
        eligible = [
            m
            for m in models
            if str(m.get("id", "")).startswith(prefix)
            and ":" not in str(m["id"])
            and not any(
                label in str(m["id"]).lower()
                for label in ("mini", "flash", "distill", "code", "vision", "lite", "turbo")
            )
        ]
        if eligible:
            result.append(
                str(max(eligible, key=lambda m: (int(m.get("created", 0)), str(m["id"])))["id"])
            )
    return result


async def pick_second(
    client: httpx.AsyncClient,
    models: list[dict[str, Any]],
    *,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    pending = flagship_candidates(models)
    examined: set[str] = set()
    while pending:
        model = pending.pop(0)
        examined.add(model)
        snapshot = await discover_model(client, model, headers=headers)
        active = [e for e in snapshot["endpoints"] if e["metadata"].get("status") == 0]
        count = sum("tools" in e["metadata"].get("supported_parameters", []) for e in active)
        author = re.sub("[^a-z0-9]", "", model.split("/")[0])
        maker = [
            e["id"]
            for e in active
            if re.sub("[^a-z0-9]", "", e["labels"]["provider_name"].lower()) == author
        ]
        candidates.append(
            {
                "model": model,
                "tool_capable_endpoints": count,
                "declared_first_party_endpoints": maker,
                "snapshot": snapshot,
            }
        )
        if not maker:
            remaining = [m for m in models if str(m.get("id")) not in examined]
            alternatives = [
                m for m in flagship_candidates(remaining) if m.split("/")[0] == model.split("/")[0]
            ]
            pending.extend(alternatives)
    eligible = [
        c for c in candidates if c["declared_first_party_endpoints"] and c["tool_capable_endpoints"]
    ]
    if not eligible:
        raise SafeError("no_eligible_second_model_with_declared_first_party_endpoint")
    chosen = sorted(eligible, key=lambda c: (-int(c["tool_capable_endpoints"]), str(c["model"])))[0]
    return {
        **chosen,
        "candidate_counts": [{k: v for k, v in c.items() if k != "snapshot"} for c in candidates],
        "selection_rule": (
            "latest non-small flagship per family with a declared maker endpoint, "
            "then most active tool-capable endpoints "
            "with a router-declared maker endpoint"
        ),
    }


GENERATION_FIELDS = {
    "id",
    "model",
    "provider_name",
    "provider_responses",
    "upstream_id",
    "finish_reason",
    "native_finish_reason",
    "native_tokens_prompt",
    "native_tokens_completion",
    "native_tokens_reasoning",
    "native_tokens_cached",
    "tokens_prompt",
    "tokens_completion",
    "total_cost",
    "cache_discount",
    "service_tier",
    "generation_time",
    "latency",
    "cancelled",
    "is_byok",
}


@dataclass(frozen=True)
class GenerationResult:
    metadata: dict[str, Any]
    status: str
    polls: int


async def fetch_generation(
    client: httpx.AsyncClient,
    base_url: str,
    generation_id: str,
    headers: dict[str, str],
    *,
    sleep: Sleep = asyncio.sleep,
    max_polls: int = 5,
    base_delay: float = 0.25,
) -> GenerationResult:
    if not re.fullmatch(r"gen-[0-9A-Za-z-]{1,124}", generation_id):
        return GenerationResult({}, "meta_unavailable", 0)
    for index in range(max_polls):
        try:
            value = await _metadata_get(
                client, base_url.rstrip("/") + "/generation", headers, {"id": generation_id}
            )
            data = value.get("data")
            if isinstance(data, dict) and data.get("id") == generation_id:
                clean = {key: data[key] for key in GENERATION_FIELDS if key in data}
                if isinstance(clean.get("provider_responses"), list):
                    clean["provider_responses"] = [
                        {
                            key: row[key]
                            for key in (
                                "provider_name",
                                "status",
                                "latency",
                                "upstream_id",
                                "finish_reason",
                            )
                            if key in row
                        }
                        for row in clean["provider_responses"]
                        if isinstance(row, dict)
                    ]
                return GenerationResult(clean, "available", index + 1)
        except SafeError:
            pass
        if index + 1 < max_polls:
            await sleep(base_delay * 2**index)
    return GenerationResult({}, "meta_unavailable", max_polls)


@dataclass(frozen=True)
class GenerationVerification:
    status: str
    tag_verified: bool = False
    exclude_from_scoring: bool = False
    true_fallback: bool | None = None


def verify_generation(endpoint: EndpointSpec, metadata: dict[str, Any]) -> GenerationVerification:
    actual = metadata.get("provider_name")
    expected = endpoint.labels.get("provider_name")
    if not isinstance(actual, str) or not expected:
        return GenerationVerification("meta_unavailable")
    if actual.strip().casefold() != expected.strip().casefold():
        return GenerationVerification("ROUTING_MISMATCH", exclude_from_scoring=True)
    responses = metadata.get("provider_responses")
    fallback = None
    if (
        isinstance(responses, list)
        and responses
        and all(isinstance(r, dict) and isinstance(r.get("provider_name"), str) for r in responses)
    ):
        fallback = len({r["provider_name"].strip().casefold() for r in responses}) > 1
    return GenerationVerification("verified_provider", true_fallback=fallback)


async def fetch_external(
    client: httpx.AsyncClient,
    *,
    headers: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    value = await _metadata_get(client, API_BASE + "/benchmarks", headers, {"source": "openrouter"})
    rows = value.get("data", [])
    if not isinstance(rows, list):
        raise SafeError("benchmark_schema_unverified")
    fields = {
        "source",
        "citation",
        "license",
        "model",
        "provider_name",
        "tag",
        "benchmark",
        "score",
        "n",
        "ci_low",
        "ci_high",
    }
    return [
        {
            **{key: row[key] for key in fields if key in row},
            "external_only": True,
            "drives_verdict": False,
            "license_status": "unverified",
        }
        for row in rows
        if isinstance(row, dict) and row.get("source") == "openrouter" and row.get("citation")
    ]
