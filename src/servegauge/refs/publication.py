"""Reusable public reference copies; calibration and observations stay unchanged."""

from __future__ import annotations

import copy
import re
from typing import Any

import numpy as np

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.models import ReferencePack
from servegauge.core.redaction import Redactor, sensitive_key
from servegauge.core.secrets import PATTERNS
from servegauge.refs.pack import _pack_digest

VERSION = "reference-publication-v1"
OBSERVATION_METADATA = {
    "requested_output_cap",
    "effective_output_cap",
    "output_capped",
    "timeout_scope",
    "timeout_seconds",
    "request_not_sent",
    "verdict",
}
PRIVATE_FIELDS = {
    "headers",
    "providermeta",
    "providerresponses",
    "accountid",
    "upstreamid",
    "generationid",
    "apikeyenv",
    "baseurl",
    "workspace",
    "localpath",
}
PRIVATE_PATH = re.compile(
    r"(?:^|\s)(?:/(?:Users|home|Volumes|private|var|tmp)/|[A-Za-z]:[\\/]|\\\\)"
)
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def clean_public(value: Any, *, secrets: tuple[str, ...] = ()) -> None:
    """Refuse known private fields/paths/secrets; this is not an anonymity guarantee."""

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
                if sensitive_key(str(key)) or normalized in PRIVATE_FIELDS:
                    raise ValueError("reference_private_semantic_field_refused")
                if normalized == "metadata" and (
                    not isinstance(child, dict)
                    or (
                        set(child) != {"mock_declaration"}
                        and not set(child) <= OBSERVATION_METADATA
                    )
                ):
                    raise ValueError("reference_opaque_semantic_metadata_refused")
                walk(str(key))
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)
        elif isinstance(item, str) and (PRIVATE_PATH.search(item) or EMAIL.search(item)):
            raise ValueError("reference_private_semantic_string_refused")

    walk(value)
    blob = canonical_json(value)
    if (
        Redactor(values=secrets).redact(value) != value
        or any(pattern.search(blob) for pattern in PATTERNS)
        or any(secret.encode() in blob for secret in secrets)
    ):
        raise ValueError("reference_secret_scan_failed")


def _declaration(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("reference_endpoint_declaration_invalid")
    result = {
        key: copy.deepcopy(value[key])
        for key in ["id", "kind", "model", "claimed_profile", "profile", "seed", "clean"]
        if key in value
    }
    pin = value.get("openrouter_pin")
    if isinstance(pin, dict):
        result["openrouter_pin"] = {
            key: copy.deepcopy(pin[key])
            for key in ["tag", "require_parameters", "allow_fallbacks"]
            if key in pin
        }
    metadata = value.get("metadata")
    recipe = metadata.get("mock_declaration") if isinstance(metadata, dict) else None
    if isinstance(recipe, dict):
        result["metadata"] = {
            "mock_declaration": {
                key: copy.deepcopy(recipe[key])
                for key in ["seed", "backend", "sampling_version", "healthy_logit_jitter", "clean"]
                if key in recipe
            }
        }
    return result


def _token(value: Any) -> Any:
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    result = {
        key: copy.deepcopy(value[key]) for key in ["token", "logprob", "bytes"] if key in value
    }
    if "top_logprobs" in value:
        rows = value["top_logprobs"]
        if rows is not None and not isinstance(rows, list):
            raise ValueError("reference_logprob_shape_unreviewed")
        result["top_logprobs"] = [_token(row) for row in rows] if rows is not None else None
    return result


def _response(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("reference_response_shape_unreviewed")
    result: dict[str, Any] = {}
    if "choices" not in value:
        return result
    if not isinstance(value["choices"], list):
        raise ValueError("reference_response_shape_unreviewed")
    choices = []
    for choice in value["choices"]:
        if not isinstance(choice, dict):
            choices.append(copy.deepcopy(choice))
            continue
        selected = {"finish_reason": choice["finish_reason"]} if "finish_reason" in choice else {}
        if "message" in choice:
            message = choice["message"]
            if message is not None and not isinstance(message, dict):
                raise ValueError("reference_response_message_unreviewed")
            selected["message"] = (
                {"content": copy.deepcopy(message["content"])}
                if isinstance(message, dict) and "content" in message
                else None
                if message is None
                else {}
            )
        if "logprobs" in choice:
            logs = choice["logprobs"]
            if logs is not None and not isinstance(logs, dict):
                raise ValueError("reference_logprob_shape_unreviewed")
            if logs is None:
                selected["logprobs"] = None
            else:
                selected["logprobs"] = {}
                if "content" in logs:
                    rows = logs["content"]
                    if rows is not None and not isinstance(rows, list):
                        raise ValueError("reference_logprob_shape_unreviewed")
                    selected["logprobs"]["content"] = (
                        [_token(row) for row in rows] if rows is not None else None
                    )
        choices.append(selected)
    result["choices"] = choices
    return result


def _sample(value: dict[str, Any]) -> None:
    details = value.get("details")
    if not isinstance(details, dict) or "observations" not in details:
        return
    observations = details["observations"]
    if details.get("observation_hash") != content_hash(observations):
        raise ValueError("reference_source_observation_hash_mismatch")
    details["source_observation_hash"] = details["observation_hash"]
    for observation in observations:
        metadata = observation.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("reference_observation_metadata_unreviewed")
        observation["metadata"] = {
            key: copy.deepcopy(metadata[key]) for key in OBSERVATION_METADATA if key in metadata
        }
        if "response" in observation:
            observation["response"] = _response(observation["response"])
    details["observation_hash"] = content_hash(observations)


def public_pack(original: ReferencePack) -> ReferencePack:
    """Give the reduced provenance a distinct identity; never rewrite its source pack."""
    if original.manifest.get("pack_content_hash") != _pack_digest(original):
        raise ValueError("reference_source_pack_content_mismatch")
    value = original.model_dump(mode="json")
    source = value["manifest"]
    source_manifest = source.get("source_manifest", {})
    if not isinstance(source_manifest, dict):
        raise ValueError("reference_source_manifest_invalid")
    manifest = {
        key: copy.deepcopy(source[key])
        for key in [
            "source_run",
            "configuration",
            "configuration_hash",
            "item_set_hashes",
            "n_per_check",
            "method_version",
            "unavailable_views",
            "fingerprints",
            "qualification",
            "origin_review",
        ]
        if key in source
    }
    manifest.update(
        publication_version=VERSION,
        source_pack_id=original.pack_id,
        source_pack_content_hash=source["pack_content_hash"],
        source_manifest_hash=content_hash(source_manifest),
        source_manifest={
            **{
                key: copy.deepcopy(source_manifest[key])
                for key in [
                    "created_at",
                    "git_sha",
                    "git_dirty",
                    "source_tree_hash",
                    "source_identity",
                    "git_state",
                    "config_hash",
                    "packages",
                    "suite_versions",
                    "item_set_hashes",
                ]
                if key in source_manifest
            },
            "endpoint_metadata": [
                _declaration(row) for row in source_manifest.get("endpoint_metadata", [])
            ],
        },
        endpoint_declarations=[
            _declaration(row) for row in source.get("endpoint_declarations", [])
        ],
    )
    value["manifest"] = manifest
    fingerprints = value["fingerprints"]
    for control in fingerprints.get("controls", []):
        control["declarations"] = [_declaration(row) for row in control.get("declarations", [])]
    for reference in fingerprints.get("reference_sources", {}).values():
        reference["declaration"] = _declaration(reference["declaration"])
    for view in fingerprints.get("views", {}).values():
        for samples in view.get("references", {}).values():
            for sample in samples.values():
                _sample(sample)
    value.pop("pack_id")
    clean_public(value)
    value["pack_id"] = "ref-" + content_hash(value)[:20]
    result = ReferencePack.model_validate(value)
    result.manifest["pack_content_hash"] = _pack_digest(result)
    return result


def outcomes_match(pack: ReferencePack, source: dict[str, Any]) -> bool:
    """Check retained ordinary view measurements against current raw-derived outcomes."""
    try:
        if source["profile"] != pack.profile:
            return False
        for target, checks in pack.outcomes.items():
            for check, view in checks.items():
                items, peers = view["item_ids"], view["peer_ids"]
                if (
                    not items
                    or items != sorted(set(items))
                    or not peers
                    or len(set(peers)) != len(peers)
                    or (pack.tier == "R3" and (target in peers or len(peers) < 3))
                ):
                    return False
                raw = [source["checks"][check][peer] for peer in peers]
                if any(set(row) != set(items) for row in raw):
                    return False
                matrix = np.asarray([[float(np.mean(row[item])) for item in items] for row in raw])
                pooled = np.median(matrix, axis=0) if pack.tier == "R3" else np.mean(matrix, axis=0)
                if (
                    not np.all(np.isfinite(matrix))
                    or matrix.tolist() != view["peer_item_means"]
                    or pooled.tolist() != view["values"]
                    or pack.calibration[target][check]["n"] != len(items)
                ):
                    return False
        return True
    except (ValueError, TypeError, KeyError, OverflowError):
        return False
