"""Resolve explicit endpoint identities; unknown live prices fail in the estimator."""

from pathlib import Path
from urllib.parse import urlsplit

import yaml

from servegauge.core.hashing import content_hash
from servegauge.core.models import EndpointSpec


def resolve_endpoint(
    url: str,
    model: str,
    profile: str,
    *,
    api_key_env: str | None = None,
    identifier: str = "target",
    root: Path = Path("."),
) -> EndpointSpec:
    if not url.startswith(("https://", "http://")):
        matches: list[EndpointSpec] = []
        for path in (root / "configs/endpoints").glob("*.yaml"):
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
            matches.extend(
                EndpointSpec.model_validate(row)
                for row in value.get("endpoints", [])
                if row.get("id") == url
            )
        if len(matches) != 1 or matches[0].model != model or matches[0].claimed_profile != profile:
            raise ValueError("unique_matching_endpoint_ID_required")
        endpoint = matches[0]
        if api_key_env:
            endpoint.api_key_env = api_key_env
        return endpoint
    parts = urlsplit(url)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username
        or parts.query
        or parts.fragment
    ):
        raise ValueError("HTTP_endpoint_without_credentials_required")
    normalized = url.rstrip("/")
    if model == "mock" and parts.hostname in {"127.0.0.1", "localhost", "::1"}:
        return EndpointSpec(
            id="mock:" + identifier + "-" + content_hash(normalized)[:8],
            kind="mock",
            base_url=normalized,
            model=model,
            claimed_profile=profile,
            api_key_env=api_key_env,
            metadata={
                "context_length": 131072,
                "metadata_status": "default mock assumption; refreshed before dispatch",
            },
        )
    matches = []
    for path in (root / "configs/endpoints").glob("*.yaml"):
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        for row in value.get("endpoints", []):
            endpoint = EndpointSpec.model_validate(row)
            if endpoint.base_url.rstrip("/") == normalized and endpoint.model == model:
                matches.append(endpoint)
    if len(matches) > 1:
        raise ValueError("ambiguous_endpoint_pins; use_an_explicit_run_plan")
    if matches:
        endpoint = matches[0]
        if endpoint.claimed_profile != profile:
            raise ValueError("endpoint_profile_mismatch")
        if api_key_env:
            endpoint.api_key_env = api_key_env
        return endpoint
    if parts.hostname == "openrouter.ai":
        raise ValueError("OpenRouter_requires_a_discovered_exact_pin")
    return EndpointSpec(
        id="direct:" + identifier + "-" + content_hash([normalized, model])[:8],
        kind="openai_compat",
        base_url=normalized,
        model=model,
        claimed_profile=profile,
        api_key_env=api_key_env,
    )
