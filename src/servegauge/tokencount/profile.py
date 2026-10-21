"""Finite verified counter dispatch; unknown identities never inherit a tokenizer."""

from typing import Literal

from servegauge.core.models import ModelProfile
from servegauge.profiles import field_verified
from servegauge.tokencount.cache import HF_MODEL, HF_REVISION
from servegauge.tokencount.glm_assets import MODEL, REVISION


def counter_kind(profile: ModelProfile) -> Literal["gpt-oss", "glm-5.3"] | None:
    if (
        profile.family == "gpt-oss"
        and field_verified(profile, "tokenizer")
        and profile.tokenizer.get("hf_model") == HF_MODEL
        and profile.tokenizer.get("hf_revision") == HF_REVISION
    ):
        # Preserve the exact historical gpt-oss admission predicate.
        return "gpt-oss"
    if (
        profile.id == "glm-5.3"
        and profile.family == "glm-5.3"
        and profile.chat_format == "glm-5.3"
        and profile.weights == MODEL
        and profile.tokenizer.get("kind") == "glm-5.3"
        and profile.tokenizer.get("hf_model") == MODEL
        and profile.tokenizer.get("hf_revision") == REVISION
        and all(
            field_verified(profile, field)
            for field in (
                "id",
                "weights",
                "family",
                "chat_format",
                "tokenizer",
                "tokenizer.kind",
                "tokenizer.hf_model",
                "tokenizer.hf_revision",
            )
        )
    ):
        return "glm-5.3"
    return None
