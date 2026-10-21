import json
from collections import Counter
from typing import Any

import httpx
import pytest

from servegauge.mock.faults import FaultConfig
from servegauge.mock.server import create_app
from servegauge.tokencount.counters import encode
from servegauge.transport.sse import SseParser


def request(task: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    return {
        "model": "mock",
        "messages": [
            {
                "role": "user",
                "content": "TW_TASK: "
                + json.dumps(
                    task or {"type": "sample", "choices": ["red", "blue", "green", "gold"]}
                ),
            }
        ],
        **kwargs,
    }


async def sample(payload: dict[str, Any], faults: dict[str, Any] | None = None) -> dict[str, Any]:
    transport = httpx.ASGITransport(app=create_app(seed=7, faults=FaultConfig(**(faults or {}))))
    async with httpx.AsyncClient(transport=transport, base_url="http://mock") as client:
        response = await client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    return response.json()


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
]
CALL = {"type": "tool", "name": "weather", "arguments": {"city": "München"}}


@pytest.mark.parametrize("key", ["ignore_temperature", "ignore_top_p"])
async def test_parameter_effects_visible_over_http(key: str) -> None:
    async def draws(faults: dict[str, Any], value: float) -> list[str]:
        app = create_app(seed=2, faults=FaultConfig(**faults))
        param = "temperature" if key == "ignore_temperature" else "top_p"
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://mock"
        ) as client:
            results = []
            for _ in range(80):
                response = await client.post("/v1/chat/completions", json=request(**{param: value}))
                results.append(response.json()["choices"][0]["message"]["content"])
            return results

    assert len(set(await draws({}, 0))) <= 2
    assert len(set(await draws({key: True}, 0))) >= 3


async def test_max_stop_and_effort_faults() -> None:
    task = {"type": "echo", "text": "word " * 100}
    clean = await sample(request(task, max_tokens=3))
    bad = await sample(request(task, max_tokens=3), {"ignore_max_tokens": True})
    assert clean["usage"]["completion_tokens"] <= 3
    assert bad["usage"]["completion_tokens"] > 3
    assert clean["choices"][0]["finish_reason"] == "length"
    clean = await sample(request({"type": "echo", "text": "before STOP after"}, stop="STOP"))
    bad = await sample(
        request({"type": "echo", "text": "before STOP after"}, stop="STOP"), {"ignore_stop": True}
    )
    assert "STOP" not in clean["choices"][0]["message"]["content"]
    assert "STOP" in bad["choices"][0]["message"]["content"]
    low = await sample(request(reasoning={"effort": "low"}, max_tokens=1024))
    high = await sample(request(reasoning={"effort": "high"}, max_tokens=1024))
    ignored = await sample(
        request(reasoning={"effort": "high"}, max_tokens=1024), {"ignore_reasoning_effort": True}
    )
    assert (
        high["usage"]["completion_tokens_details"]["reasoning_tokens"]
        > low["usage"]["completion_tokens_details"]["reasoning_tokens"] * 2
    )
    assert (
        ignored["usage"]["completion_tokens_details"]["reasoning_tokens"]
        < high["usage"]["completion_tokens_details"]["reasoning_tokens"]
    )


async def test_parser_template_and_schema_faults() -> None:
    payload = request(CALL, tools=TOOLS)
    drop = await sample(payload, {"drop_tool_calls": 1})
    assert drop["choices"][0]["finish_reason"] == "tool_calls"
    assert not drop["choices"][0]["message"].get("tool_calls")
    leak = await sample(payload, {"leak_markers": 1})
    assert "<tool_call>" in leak["choices"][0]["message"]["content"]
    bad = await sample(payload, {"corrupt_arguments": 1})
    with pytest.raises(ValueError):
        json.loads(bad["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    payload["messages"].append({"role": "tool", "tool_call_id": "call_0", "content": "sunny"})
    bad = await sample(payload, {"plain_text_after_tool_result": 1})
    assert not bad["choices"][0]["message"].get("tool_calls")


async def test_noindex_stream_and_arbitrary_chunking() -> None:
    app = create_app(seed=2, faults=FaultConfig(stream_delta_without_index=True), chunk_bytes=3)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://mock"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=request(CALL, tools=TOOLS, stream=True, stream_options={"include_usage": True}),
        )
    parser = SseParser()
    events = parser.feed(response.content, 0)
    assert parser.terminal
    assert events[0].is_comment
    data = [json.loads(e.data) for e in events if not e.is_comment and e.data != "[DONE]"]
    deltas = [d["choices"][0]["delta"] for d in data if d["choices"]]
    calls = [call for delta in deltas for call in delta.get("tool_calls", [])]
    assert calls and all("index" not in call for call in calls)
    assert data[-1]["usage"]["prompt_tokens"] > 0


async def test_context_output_usage_and_error_faults() -> None:
    content = (
        "head_canary = FIRST\n"
        + "hay " * 20000
        + '\ntail_canary = LAST\nTW_TASK: {"type":"retrieve","keys":["head_canary","tail_canary"]}'
    )
    payload = {
        "model": "mock",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 100,
    }
    clean = await sample(payload)
    bad = await sample(payload, {"truncate_prompt_head_over": 16000})
    observed = json.loads(bad["choices"][0]["message"]["content"])
    assert observed["head_canary"] is None and observed["tail_canary"] == "LAST"
    assert bad["usage"]["prompt_tokens"] < clean["usage"]["prompt_tokens"]
    output = request({"type": "sequence", "count": 4000}, max_tokens=16000)
    bad = await sample(output, {"truncate_output_after": 6000})
    assert len(encode(bad["choices"][0]["message"]["content"])) == 6000
    good = await sample(request())
    usage = await sample(request(), {"misreport_usage_pct": 15})
    assert usage["usage"]["prompt_tokens"] > good["usage"]["prompt_tokens"]
    app = create_app(faults=FaultConfig(error_rate=1))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://mock"
    ) as client:
        assert (await client.post("/v1/chat/completions", json=request())).status_code == 503


async def test_cache_swap_mix_noise_and_distinct_seed_distributions() -> None:
    async def draws(seed: int, faults: FaultConfig) -> list[str]:
        app = create_app(seed=seed, faults=faults)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://mock"
        ) as client:
            return [
                (await client.post("/v1/chat/completions", json=request())).json()["choices"][0][
                    "message"
                ]["content"]
                for _ in range(200)
            ]

    a, b = await draws(1, FaultConfig()), await draws(2, FaultConfig())
    assert a != b
    assert max(abs(Counter(a)[x] - Counter(b)[x]) for x in set(a + b)) < 40
    cached = await draws(1, FaultConfig(stale_cache=1))
    assert len(set(cached)) == 1
    swapped = await draws(1, FaultConfig(route_to_other_model=1))
    mixed = await draws(1, FaultConfig(route_to_other_model=0.5))
    noisy = await draws(1, FaultConfig(logit_noise=0.3))
    assert swapped != a and mixed != a and noisy != a


async def test_replay_and_reserved_proxy_contract(tmp_path: Any) -> None:
    from tests_support import exchange

    from servegauge.mock.backends import ProxyBackend

    cassette = tmp_path / "cassette.json"
    cassette.write_text(exchange().model_dump_json())
    app = create_app(backend=f"replay:{cassette}")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://mock"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "other", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert response.json()["choices"][0]["message"]["content"] == "hello"
        assert (await client.post("/v1/chat/completions", json=request())).status_code == 422
    proxy = ProxyBackend("http://local/v1")
    with pytest.raises(RuntimeError, match="reserved runner"):
        await proxy.respond(request())

    async def reserved_forward(url: str, body: dict[str, Any]) -> dict[str, Any]:
        assert url == "http://local/v1" and body["messages"]
        return {"choices": []}

    assert await ProxyBackend("http://local/v1", reserved_forward).respond(request()) == {
        "choices": []
    }


async def test_other_model_controls_and_replayed_streams(tmp_path: Any) -> None:
    from tests_support import exchange

    from servegauge.core.models import SseEvent

    outputs = []
    for backend in ("scripted", "other:control-a", "other:control-b"):
        app = create_app(seed=1, backend=backend)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://mock"
        ) as client:
            outputs.append(
                [
                    (await client.post("/v1/chat/completions", json=request())).json()["choices"][
                        0
                    ]["message"]["content"]
                    for _ in range(50)
                ]
            )
    assert len({tuple(output) for output in outputs}) == 3
    record = exchange(
        sse_events=[
            SseEvent(t_ms=1, data="kept", is_comment=True, raw_lines=[": kept"]),
            SseEvent(t_ms=2, data="[DONE]", raw_lines=["data: [DONE]"]),
        ]
    )
    cassette = tmp_path / "stream.json"
    cassette.write_text(record.model_dump_json())
    app = create_app(backend=f"replay:{cassette}")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://mock"
    ) as client:
        result = await client.post(
            "/v1/chat/completions", json={"messages": record.request["messages"], "stream": True}
        )
        assert result.content == b": kept\n\ndata: [DONE]\n\n"
