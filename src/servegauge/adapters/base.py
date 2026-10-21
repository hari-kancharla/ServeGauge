from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from servegauge.core.errors import SafeError
from servegauge.core.models import ChatRequest, EndpointSpec


class Adapter(Protocol):
    def build_request(self, request: ChatRequest, endpoint: EndpointSpec) -> dict[str, Any]: ...


class UnsupportedFeature(SafeError):
    def __init__(self, code: str, features: list[str], *, uncertain: bool = False) -> None:
        super().__init__(code)
        self.features = features
        self.verdict: Literal["NOT_SUPPORTED", "INCONCLUSIVE"] = (
            "INCONCLUSIVE" if uncertain else "NOT_SUPPORTED"
        )

    def __str__(self) -> str:
        return self.code + ": " + ", ".join(self.features)


@dataclass(frozen=True)
class PreparedRequest:
    body: dict[str, Any] | None
    verdict: Literal["NOT_SUPPORTED", "INCONCLUSIVE"] | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def prepare_request(request: ChatRequest, endpoint: EndpointSpec) -> PreparedRequest:
    from servegauge.adapters.openai_compat import OpenAICompatAdapter

    try:
        body = OpenAICompatAdapter().build_request(request, endpoint)
    except UnsupportedFeature as error:
        return PreparedRequest(
            None, error.verdict, str(error), {"unsupported_features": error.features}
        )
    requested = request.params.get(
        "max_completion_tokens",
        request.params.get(
            "max_tokens",
            endpoint.defaults.get(
                "max_completion_tokens", endpoint.defaults.get("max_tokens", 256)
            ),
        ),
    )
    effective = body.get("max_completion_tokens", body.get("max_tokens"))
    if (
        not isinstance(requested, int)
        or not isinstance(effective, int)
        or min(requested, effective) < 1
    ):
        raise ValueError("explicit_positive_output_cap_required")
    return PreparedRequest(
        body,
        metadata={
            "output_capped": effective < requested,
            "requested_output_cap": requested,
            "effective_output_cap": effective,
        },
    )
