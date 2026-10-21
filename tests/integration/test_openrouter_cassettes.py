import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from servegauge.adapters.base import UnsupportedFeature, prepare_request
from servegauge.adapters.openrouter import (
    OpenRouterAdapter,
    discover_model,
    fetch_external,
    fetch_generation,
    flagship_candidates,
    pick_second,
    verify_generation,
)
from servegauge.core.models import ChatRequest, EndpointSpec, OpenRouterPin
from servegauge.runner.cost import actual_cost, worst_case_cost


def cassette(name: str) -> dict[str, Any]:
    return json.loads(Path(f"tests/cassettes/{name}.json").read_text())["payload"]


def endpoint() -> EndpointSpec:
    data = cassette("openrouter_contract")["data"]["endpoints"][0]
    return EndpointSpec(
        id="or:example/model@provider-a/fp4/us",
        kind="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="example/model",
        claimed_profile="example",
        openrouter_pin=OpenRouterPin(tag=data["tag"]),
        labels={"provider_name": data["provider_name"], "quantization": data["quantization"]},
        metadata=data,
    )


async def test_discovery_replays_public_capture_and_preserves_disabled_endpoints() -> None:
    public = cassette("openrouter_endpoints_public")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/models/openai/gpt-oss-120b/endpoints"
        return httpx.Response(200, json=public)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        discovered = await discover_model(client, "openai/gpt-oss-120b")
    assert len(discovered["endpoints"]) == len(public["data"]["endpoints"])
    assert any(e["metadata"]["status"] != 0 for e in discovered["endpoints"])
    for record, original in zip(discovered["endpoints"], public["data"]["endpoints"], strict=True):
        assert record["openrouter_pin"]["tag"] == original["tag"]
        assert record["metadata"]["supported_parameters"] == original["supported_parameters"]
        assert record["metadata"]["pricing"] == original["pricing"]


def test_pinning_compression_parameter_and_output_guards() -> None:
    ep = endpoint()
    adapter = OpenRouterAdapter()
    request = ChatRequest(
        messages=[{"role": "user", "content": "hello"}], params={"max_tokens": 5000}
    )
    prepared = prepare_request(request, ep)
    body = prepared.body
    assert body and body["provider"] == {
        "only": ["provider-a/fp4/us"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert body["plugins"] == [{"id": "context-compression", "enabled": False}]
    assert body["max_tokens"] == 4096 and prepared.metadata["output_capped"]
    envelope = {"model", "messages", "stream", "stream_options", "provider", "plugins"}
    assert set(body) - envelope <= set(ep.metadata["supported_parameters"])
    with pytest.raises(UnsupportedFeature, match="required_output_exceeds_endpoint_cap"):
        adapter.build_request(request.model_copy(update={"minimum_output_tokens": 5000}), ep)
    with pytest.raises(UnsupportedFeature, match="parallel_tool_calls"):
        adapter.build_request(ChatRequest(messages=[], params={"parallel_tool_calls": False}), ep)
    tools = [{"type": "function", "function": {"name": "x", "parameters": {"type": "object"}}}]
    with pytest.raises(UnsupportedFeature, match="tool_choice"):
        adapter.build_request(ChatRequest(messages=[], tools=tools, tool_choice="required"), ep)
    parallel_probe = adapter.build_request(ChatRequest(messages=[], tools=tools), ep)
    assert "parallel_tool_calls" not in parallel_probe
    assert "temperature" not in parallel_probe and "seed" not in parallel_probe


def test_unsupported_probe_is_explicit_and_missing_metadata_is_uncertain() -> None:
    ep = endpoint()
    body = OpenRouterAdapter().build_request(
        ChatRequest(messages=[], params={"not_a_parameter": 1}, unsupported_probe="P13"), ep
    )
    assert body["provider"]["require_parameters"] is False
    assert body["plugins"] == [{"id": "context-compression", "enabled": False}]
    unknown = ep.model_copy(update={"metadata": {}})
    prepared = prepare_request(ChatRequest(messages=[]), unknown)
    assert prepared.body is None and prepared.verdict == "INCONCLUSIVE"
    price = worst_case_cost(
        OpenRouterAdapter().build_request(ChatRequest(messages=[], params={"max_tokens": 20}), ep),
        ep,
    )
    assert price == Decimal("0.0131112")


async def test_generation_polling_verification_cost_and_privacy() -> None:
    polls = 0
    waits: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal polls
        polls += 1
        assert request.url.params["id"] == "gen-fixture-1"
        return (
            httpx.Response(404)
            if polls == 1
            else httpx.Response(200, json=cassette("openrouter_generation_contract"))
        )

    async def sleep(delay: float) -> None:
        waits.append(delay)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await fetch_generation(
            client, "https://openrouter.ai/api/v1", "gen-fixture-1", {}, sleep=sleep
        )
    assert polls == 2 and waits
    assert "account_id" not in result.metadata and "external_user" not in result.metadata
    verified = verify_generation(endpoint(), result.metadata)
    assert verified.status == "verified_provider"
    assert verified.tag_verified is False
    assert verified.true_fallback is False
    assert actual_cost({}, endpoint(), Decimal("1"), generation=result.metadata) == (
        Decimal("0.000003"),
        "generation_record",
    )
    mismatch = verify_generation(endpoint(), {**result.metadata, "provider_name": "Other Provider"})
    assert mismatch.status == "ROUTING_MISMATCH" and mismatch.exclude_from_scoring


async def test_generation_unavailable_is_capped_and_never_a_routing_fail() -> None:
    polls = 0

    async def sleep(delay: float) -> None:
        pass

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal polls
        polls += 1
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await fetch_generation(
            client, "https://openrouter.ai/api/v1", "gen-missing", {}, sleep=sleep, max_polls=3
        )
    assert polls == 3 and result.status == "meta_unavailable"
    assert verify_generation(endpoint(), result.metadata).status == "meta_unavailable"


async def test_second_model_selection_and_attributed_external_column() -> None:
    models = [
        {"id": "moonshotai/kimi-k3", "created": 3},
        {"id": "deepseek/deepseek-v4-pro", "created": 2},
        {"id": "z-ai/glm-5.3", "created": 4},
        {"id": "z-ai/glm-5.3-flash", "created": 5},
    ]
    assert flagship_candidates(models) == [
        "moonshotai/kimi-k3",
        "deepseek/deepseek-v4-pro",
        "z-ai/glm-5.3",
    ]
    models.append({"id": "z-ai/glm-5.3-prime", "created": 6})

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/benchmarks"):
            assert request.url.params["source"] == "openrouter"
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "source": "openrouter",
                            "citation": "Synthetic contract citation",
                            "model": "example/model",
                            "score": 0.5,
                        }
                    ]
                },
            )
        model = request.url.path.split("/models/", 1)[1].removesuffix("/endpoints")
        count = (
            5
            if model == "z-ai/glm-5.3"
            else 1
            if model.endswith("prime")
            else 4
            if model.startswith("deepseek/")
            else 2
        )
        template = cassette("openrouter_contract")["data"]["endpoints"][0]
        data = [{**template, "tag": f"provider-{i}"} for i in range(count)]
        maker_name = {
            "deepseek": "DeepSeek",
            "moonshotai": "Moonshot AI",
            "z-ai": "Z.AI",
        }[model.split("/")[0]]
        if not model.endswith("prime"):
            data[0]["provider_name"] = maker_name
        return httpx.Response(200, json={"data": {"id": model, "endpoints": data}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        choice = await pick_second(client, models)
        external = await fetch_external(client)
    assert choice["model"] == "z-ai/glm-5.3"
    assert choice["tool_capable_endpoints"] == 5
    assert any(
        c["model"].endswith("prime") and not c["declared_first_party_endpoints"]
        for c in choice["candidate_counts"]
    )
    assert external[0]["citation"] == "Synthetic contract citation"
    assert external[0]["external_only"] and not external[0]["drives_verdict"]


@pytest.mark.parametrize("mismatch", [False, True])
async def test_runner_integrates_verification_and_generation_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: bool
) -> None:
    from servegauge.runner.plan import RunPlan
    from servegauge.runner.scheduler import run_plan
    from servegauge.storage.raw_store import RawStore

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/budgets.yaml").write_bytes(
        Path("tests/fixtures/simulated-budgets.yaml").read_bytes()
    )
    ep = endpoint()
    ep.api_key_env = "TEST_API_KEY"
    monkeypatch.setenv("TEST_API_KEY", "sk-" + "fixturevalue" * 4)
    disabled = ep.model_copy(deep=True)
    disabled.id = "disabled"
    disabled.metadata["status"] = 1
    plan = RunPlan(
        plan_id="fixture",
        profile="example",
        level="smoke",
        seed=1,
        reference={"tier": "MOCK"},
        max_usd=0.5,
        endpoints=[ep, disabled],
        suites=["S0"],
        overrides={"S0": {"items": 1}},
    )
    posts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.url.path.endswith("/key"):
            return httpx.Response(200, json={"data": {"limit": 5, "limit_remaining": 5}})
        if request.url.path.endswith("/generation"):
            data = cassette("openrouter_generation_contract")
            if mismatch:
                data["data"]["provider_name"] = "Other Provider"
            return httpx.Response(200, json=data)
        assert request.method == "POST"
        posts += 1
        body = json.loads(request.content)
        assert body["plugins"] == [{"id": "context-compression", "enabled": False}]
        return httpx.Response(
            200,
            json={
                "id": "gen-fixture-1",
                "choices": [{"message": {"content": "ready"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 4},
            },
        )

    result = await run_plan(
        plan, root=tmp_path, approved=True, transports={ep.id: httpx.MockTransport(handler)}
    )
    assert posts == 1 and result.completed == 1
    assert result.errors == int(mismatch)
    record = next(RawStore(tmp_path / "data/raw").exchanges())
    assert record.provider_meta and record.cost_source == "generation_record"
    assert record.cost_usd == 0.000003
    assert record.metadata["tag_verified"] is False
    assert record.metadata["exclude_from_scoring"] is mismatch
    assert record.error == ("ROUTING_MISMATCH" if mismatch else None)


@pytest.mark.parametrize("shared_key", [True, False])
async def test_payment_rejection_stops_following_dispatch_for_the_same_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shared_key: bool
) -> None:
    from servegauge.runner.plan import RunPlan
    from servegauge.runner.scheduler import run_plan
    from servegauge.storage.raw_store import RawStore

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/budgets.yaml").write_bytes(
        Path("tests/fixtures/simulated-budgets.yaml").read_bytes()
    )
    first, second = endpoint(), endpoint()
    first.api_key_env = "TEST_API_KEY"
    second.id = "second"
    second.api_key_env = "TEST_API_KEY" if shared_key else "OTHER_API_KEY"
    monkeypatch.setenv("TEST_API_KEY", "fixture-payment-key")
    monkeypatch.setenv("OTHER_API_KEY", "other-fixture-key")
    plan = RunPlan(
        plan_id="payment-block",
        profile="example",
        level="smoke",
        seed=1,
        reference={"tier": "MOCK"},
        max_usd=0.5,
        concurrency_per_endpoint=1,
        requests_per_minute=60000,
        endpoints=[first, second],
        suites=["S0"],
        overrides={"S0": {"items": 3}},
    )
    posts = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/key"):
            return httpx.Response(200, json={"data": {"limit": 5, "limit_remaining": 5}})
        assert request.method == "POST"
        posts.append(request)
        return httpx.Response(
            402,
            json={
                "error": {
                    "code": 402,
                    "message": "Insufficient credits",
                    "metadata": {"limit_source": "openrouter_credits"},
                }
            },
        )

    result = await run_plan(
        plan,
        root=tmp_path,
        approved=True,
        transports={ep.id: httpx.MockTransport(handler) for ep in plan.endpoints},
    )
    expected_posts = 1 if shared_key else 2
    records = list(RawStore(tmp_path / "data/raw").exchanges())
    assert len(posts) == result.errors == expected_posts
    assert result.unmeasured == 6 - expected_posts
    assert sum(bool(r.metadata.get("account_credit_blocked")) for r in records) == result.unmeasured
    assert result.spent_usd > 0  # Missing provider cost retains the conservative charge.


async def test_unsupported_check_produces_no_http_or_auth_lookup(tmp_path: Path) -> None:
    from servegauge.core.models import Item
    from servegauge.runner.plan import RunPlan
    from servegauge.runner.scheduler import run_plan
    from servegauge.storage.raw_store import RawStore

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/budgets.yaml").write_bytes(
        Path("tests/fixtures/simulated-budgets.yaml").read_bytes()
    )
    ep = endpoint()
    ep.api_key_env = "MISSING_CREDENTIAL_FOR_UNSENT_CHECK"
    plan = RunPlan(
        plan_id="unsupported",
        profile="example",
        level="smoke",
        seed=1,
        reference={"tier": "MOCK"},
        max_usd=0,
        endpoints=[ep],
        suites=["S1"],
    )
    item = Item(
        item_id="P10",
        suite_id="S1",
        version="1",
        expected={},
        payload=ChatRequest(messages=[], params={"parallel_tool_calls": False}).model_dump(
            mode="json"
        ),
    )

    def forbidden(request: httpx.Request) -> httpx.Response:
        raise AssertionError("An unsupported check must not issue HTTP requests")

    result = await run_plan(
        plan,
        root=tmp_path,
        approved=True,
        transports={ep.id: httpx.MockTransport(forbidden)},
        items_override=[item],
    )
    record = next(RawStore(tmp_path / "data/raw").exchanges())
    assert result.unmeasured == 1 and result.errors == 0
    assert record.attempts == [] and record.cost_usd == 0
    assert record.metadata["verdict"] == "NOT_SUPPORTED"
