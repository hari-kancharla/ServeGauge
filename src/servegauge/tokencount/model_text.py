"""Verified GLM overrides; existing gpt-oss corpus/count defaults stay unchanged."""

from collections.abc import Callable
from dataclasses import dataclass

from servegauge.core.models import ModelProfile
from servegauge.tokencount.glm_assets import MODEL, REVISION, verified_assets
from servegauge.tokencount.glm_counter import glm_decode, glm_text_ids
from servegauge.tokencount.profile import counter_kind


@dataclass(frozen=True)
class TextCodec:
    profile: str
    revision: str
    encode: Callable[[str], list[int]]
    decode: Callable[[list[int]], str]

    def count(self, text: str) -> int:
        return len(self.encode(text))


def uses_glm_profile(profile: ModelProfile | None) -> bool:
    return profile is not None and (
        profile.id == "glm-5.3"
        or profile.family == "glm-5.3"
        or profile.weights == MODEL
        or profile.tokenizer.get("hf_model") == MODEL
    )


def profile_codec(profile: ModelProfile | None) -> TextCodec | None:
    if not uses_glm_profile(profile):
        return None
    if profile is None or counter_kind(profile) != "glm-5.3":
        raise ValueError("verified_glm_text_codec_required")
    verified_assets()
    return TextCodec(profile.id, REVISION, glm_text_ids, glm_decode)
