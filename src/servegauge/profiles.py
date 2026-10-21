"""Sourced model facts. Missing provenance is a configuration error, never model evidence."""

from pathlib import Path

import yaml

from servegauge.core.models import ModelProfile


def load_profile(identifier: str, directory: Path = Path("profiles")) -> ModelProfile:
    if not identifier or Path(identifier).name != identifier or identifier in {".", ".."}:
        raise ValueError("profile_identifier_required")
    profile = ModelProfile.model_validate(
        yaml.safe_load((directory / f"{identifier}.yaml").read_text(encoding="utf-8"))
    )
    if profile.id != identifier:
        raise ValueError("profile_id_mismatch")
    for field, value in profile.model_dump().items():
        if field in {"sources", "field_sources", "unverified"} or value in (None, {}, []):
            continue
        source = profile.field_sources.get(field)
        if not source or source not in profile.sources or not source.startswith("https://"):
            raise ValueError("profile_field_requires_official_source")
    return profile


def field_verified(profile: ModelProfile, field: str) -> bool:
    return field in profile.field_sources and not any(
        field == unknown or field.startswith(unknown + ".") for unknown in profile.unverified
    )
