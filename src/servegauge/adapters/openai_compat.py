"""Pure Chat Completions request construction, with endpoint identity fixed last."""

from typing import Any

from servegauge.adapters.base import UnsupportedFeature
from servegauge.core.models import ChatRequest, EndpointSpec
from servegauge.core.redaction import sensitive_key


class OpenAICompatAdapter:
    def build_request(self, request: ChatRequest, endpoint: EndpointSpec) -> dict[str, Any]:
        if endpoint.kind == "openrouter":
            from servegauge.adapters.openrouter import OpenRouterAdapter

            return OpenRouterAdapter().build_request(request, endpoint)
        body = {**endpoint.defaults, **request.params}
        if any(sensitive_key(key) for key in body):
            raise ValueError("credentials_must_come_from_environment_headers")
        body.update({"model": endpoint.model, "messages": request.messages})
        if "max_tokens" in body and "max_completion_tokens" in body:
            raise ValueError("choose_one_output_cap_parameter")
        if "max_tokens" not in body and "max_completion_tokens" not in body:
            body["max_tokens"] = 256
        cap = endpoint.metadata.get("max_completion_tokens")
        if isinstance(cap, int) and cap > 0:
            if request.minimum_output_tokens and request.minimum_output_tokens > cap:
                raise UnsupportedFeature(
                    "required_output_exceeds_endpoint_cap", ["max_completion_tokens"]
                )
            key = "max_completion_tokens" if "max_completion_tokens" in body else "max_tokens"
            body[key] = min(body[key], cap)
        body["stream"] = (
            request.stream
            if "stream" in request.model_fields_set
            else bool(body.get("stream", False))
        )
        for key in ("tools", "tool_choice", "response_format"):
            value = getattr(request, key)
            if value is not None:
                body[key] = value
        if body["stream"]:
            body["stream_options"] = {**body.get("stream_options", {}), "include_usage": True}
        return body
