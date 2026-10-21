"""Explicit GLM text projections for protocol/JSON checks; no legacy fallback."""

from servegauge.core.models import Exchange, Item, ModelProfile
from servegauge.scoring.completion_accounting import (
    AccountingUnavailable,
    _complete,
    _count,
    completion_channels,
)
from servegauge.tokencount.model_text import TextCodec, profile_codec


class GLMObservationUnavailable(LookupError):
    """A controlled missing/ambiguous channel, distinct from a malformed envelope."""


def _codec(profile: ModelProfile | None) -> TextCodec:
    try:
        codec = profile_codec(profile)
    except ValueError:
        raise GLMObservationUnavailable("verified GLM text counter unavailable") from None
    if codec is None:
        raise GLMObservationUnavailable("verified GLM text counter unavailable")
    return codec


def cap_violation(item: Item, record: Exchange, profile: ModelProfile | None) -> bool:
    codec = _codec(profile)
    cap = record.request.get("max_completion_tokens", record.request.get("max_tokens"))
    if type(cap) is not int or cap < 1:
        raise GLMObservationUnavailable("positive integer output cap unavailable")
    try:
        visible, reasoning, _ = completion_channels(record, allow_length=True)
        reported = _count((record.usage or {}).get("completion_tokens"))
    except AccountingUnavailable as error:
        raise GLMObservationUnavailable("GLM output count unavailable: " + str(error)) from None
    local = codec.count(visible) + codec.count(reasoning)
    requested = item.expected["full_text"]
    known_cut = bool(visible) and len(visible) < len(requested) and requested.startswith(visible)
    finish = (record.response or {})["choices"][0]["finish_reason"]
    return local > cap or reported > cap or (known_cut and finish != "length")


def reasoning_count(record: Exchange, profile: ModelProfile | None) -> tuple[int, str]:
    try:
        _complete(record)
        choices = (record.response or {}).get("choices")
        if (
            not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(choices[0], dict)
            or choices[0].get("finish_reason") != "stop"
            or not isinstance(choices[0].get("message"), dict)
            or type(choices[0].get("index", 0)) is not int
            or choices[0].get("index", 0) != 0
            or type(record.request.get("n", 1)) is not int
            or record.request.get("n", 1) != 1
        ):
            raise AccountingUnavailable("one_uncensored_reasoning_observation_required")
        details = (record.usage or {}).get("completion_tokens_details", {})
        if not isinstance(details, dict):
            raise AccountingUnavailable("invalid_reasoning_count_channel")
        if details.get("reasoning_tokens") is not None:
            count = _count(details["reasoning_tokens"])
            total = (record.usage or {}).get("completion_tokens")
            if total is not None and count > _count(total):
                raise AccountingUnavailable("reasoning_count_exceeds_completion")
            return count, "reported-reasoning-tokens"
        _, reasoning, _ = completion_channels(record)
    except AccountingUnavailable as error:
        raise GLMObservationUnavailable(
            "GLM reasoning observation unavailable: " + str(error)
        ) from None
    return _codec(profile).count(reasoning), "complete-typed-reasoning-text-projection"
