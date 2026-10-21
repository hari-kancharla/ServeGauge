"""OpenAI-compatible JSON/SSE wire service for stochastic and replay tests."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from servegauge.core.errors import SafeError
from servegauge.core.models import Exchange
from servegauge.items.schemas import LADDER
from servegauge.mock.backends import ProxyBackend, ReplayBackend, ReservedForward, ScriptedBackend
from servegauge.mock.faults import FaultConfig


def stream_frames(response: dict[str, Any], faults: FaultConfig) -> list[bytes]:
    frames = [b": keep-alive\n\n"]
    choice = response.get("choices", [{}])[0]
    message = choice.get("message", {})
    base = {
        "id": response.get("id"),
        "object": "chat.completion.chunk",
        "model": response.get("model"),
    }

    def add(delta: dict[str, Any], finish: str | None = None) -> None:
        chunk = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        frames.append(("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode())

    add({"role": "assistant"})
    if message.get("reasoning_details"):
        add({"reasoning_details": message["reasoning_details"]})
    for field in ("reasoning", "content"):
        text = str(message.get(field) or "")
        for start in range(0, len(text), 64):
            add({field: text[start : start + 64]})
    for index, call in enumerate(message.get("tool_calls") or []):
        info: dict[str, Any] = {
            "id": call["id"],
            "type": "function",
            "function": {"name": call["function"]["name"], "arguments": ""},
        }
        if not faults.stream_delta_without_index:
            info["index"] = index
        add({"tool_calls": [info]})
        arguments = call["function"]["arguments"]
        for start in range(0, len(arguments), 7):
            info = {"function": {"arguments": arguments[start : start + 7]}}
            if not faults.stream_delta_without_index:
                info["index"] = index
            add({"tool_calls": [info]})
    add({}, choice.get("finish_reason"))
    frames.append(
        (
            "data: " + json.dumps({**base, "choices": [], "usage": response.get("usage")}) + "\n\n"
        ).encode()
    )
    frames.append(b"data: [DONE]\n\n")
    return frames


def create_app(
    *,
    seed: int = 1,
    faults: FaultConfig | None = None,
    backend: str = "scripted",
    chunk_bytes: int = 0,
    reserved_forward: ReservedForward | None = None,
    context_length: int = 131072,
    max_completion_tokens: int = 32768,
    sampling_version: str = "v1",
    healthy_logit_jitter: float = 0.0,
) -> FastAPI:
    config = faults or FaultConfig()
    engine: ScriptedBackend | ReplayBackend | ProxyBackend
    if backend not in {"scripted", "other:control-a", "other:control-b"} and (
        sampling_version != "v1" or healthy_logit_jitter != 0
    ):
        raise ValueError("sampling_options_require_scripted_backend")
    if backend.startswith("replay:"):
        engine = ReplayBackend(Path(backend[7:]))
    elif backend.startswith("proxy:"):
        engine = ProxyBackend(backend[6:], reserved_forward, config, seed)
    elif backend == "scripted" or backend in {"other:control-a", "other:control-b"}:
        engine = ScriptedBackend(
            seed,
            config,
            backend[6:] if backend.startswith("other:") else None,
            context_length,
            sampling_version=sampling_version,
            healthy_logit_jitter=healthy_logit_jitter,
        )
    else:
        raise ValueError("Unknown mock backend")
    app = FastAPI(docs_url=None, redoc_url=None)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ready"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "data": [
                {
                    "id": "mock",
                    "context_length": context_length,
                    "max_prompt_tokens": context_length,
                    "max_completion_tokens": max_completion_tokens,
                    "declared_json_schema": {
                        "value": True,
                        "verified": True,
                        "source": "mock:/v1/models",
                    },
                    "strict_schema_guarantee": {
                        "verified": True,
                        "source": "mock:/v1/models",
                        "features": list(LADDER),
                    },
                    "supported_parameters": [
                        "max_tokens",
                        "temperature",
                        "top_p",
                        "top_k",
                        "stop",
                        "seed",
                        "tools",
                        "tool_choice",
                        "parallel_tool_calls",
                        "response_format",
                        "structured_outputs",
                        "logprobs",
                        "top_logprobs",
                        "reasoning_effort",
                    ],
                    "supports_tool_choice": {
                        "none": True,
                        "auto": True,
                        "required": True,
                        "function": True,
                    },
                    "mock_declaration": {
                        **(
                            {
                                "sampling_version": sampling_version,
                                "healthy_logit_jitter": healthy_logit_jitter,
                            }
                            if sampling_version != "v1"
                            else {}
                        ),
                        "seed": config.seed if config.seed is not None else seed,
                        "backend": backend,
                        "clean": backend == "scripted"
                        and not any(
                            v
                            for k, v in config.model_dump().items()
                            if k not in {"seed", "fault_id"}
                        ),
                    },
                }
            ]
        }

    @app.post("/v1/chat/completions", response_model=None)
    async def completions(request: Request) -> JSONResponse | StreamingResponse:
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": {"code": "invalid_json"}}, status_code=400)
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            return JSONResponse({"error": {"code": "invalid_messages"}}, status_code=400)
        if isinstance(engine, ScriptedBackend) and engine.rng.random() < config.error_rate:
            return JSONResponse({"error": {"code": "temporarily_unavailable"}}, status_code=503)
        recorded: Exchange | None = None
        try:
            if isinstance(engine, ProxyBackend):
                response = await engine.respond(body)
            elif isinstance(engine, ReplayBackend):
                recorded = engine.select(body)
                response = recorded.response or {}
            else:
                response = engine.respond(body)
        except (ValueError, KeyError, TypeError):
            return JSONResponse(
                {"error": {"code": "invalid_or_unmatched_request"}}, status_code=422
            )
        except RuntimeError:
            return JSONResponse({"error": {"code": "reserved_proxy_required"}}, status_code=503)
        except SafeError:
            return JSONResponse({"error": {"code": "proxy_upstream_failed"}}, status_code=502)
        if "error" in response:
            return JSONResponse(
                response, status_code=recorded.status if recorded and recorded.status else 400
            )
        if not body.get("stream"):
            return JSONResponse(
                response, status_code=recorded.status if recorded and recorded.status else 200
            )

        if recorded and recorded.sse_events:
            frames = [
                (
                    (
                        "\n".join(event.raw_lines)
                        if event.raw_lines
                        else (": " + event.data if event.is_comment else "data: " + event.data)
                    )
                    + "\n\n"
                ).encode()
                for event in recorded.sse_events
            ]
        else:
            frames = stream_frames(response, config)

        async def chunks() -> AsyncIterator[bytes]:
            for frame in frames:
                if chunk_bytes > 0:
                    for index in range(0, len(frame), chunk_bytes):
                        yield frame[index : index + chunk_bytes]
                else:
                    yield frame

        return StreamingResponse(chunks(), media_type="text/event-stream")

    return app
