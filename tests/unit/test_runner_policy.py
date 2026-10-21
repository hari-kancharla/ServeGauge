import random
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from servegauge.cli import app
from servegauge.core.errors import SafeError
from servegauge.core.models import Attempt, EndpointSpec, Timings
from servegauge.runner.budget import BudgetExceeded
from servegauge.runner.cost import actual_cost, conservative_paired_mde, prices, worst_case_cost
from servegauge.runner.estimate import estimate_plan
from servegauge.runner.plan import RunPlan, load_plan, load_requirements
from servegauge.runner.ratelimit import TokenBucket
from servegauge.runner.retry import delay_seconds, retryable
from servegauge.runner.scheduler import key_preflight


def priced() -> EndpointSpec:
    return EndpointSpec(
        id="priced",
        kind="openai_compat",
        base_url="https://example.com/v1",
        model="x",
        claimed_profile="x",
        metadata={
            "pricing": {"prompt": "0.001", "completion": "0.002"},
            "max_prompt_tokens": 100,
            "max_completion_tokens": 50,
        },
    )


def test_pricing_bounds_missing_usage_and_actual_cost() -> None:
    endpoint = priced()
    assert worst_case_cost({"messages": [], "max_tokens": 10}, endpoint) == Decimal("0.12")
    assert actual_cost(
        {"usage": {"prompt_tokens": 20, "completion_tokens": 10}}, endpoint, Decimal("0.12")
    ) == (Decimal("0.04"), "estimate")
    assert actual_cost({"usage": {"cost": 0.03}}, endpoint, Decimal("0.12")) == (
        Decimal("0.03"),
        "provider",
    )
    assert actual_cost(None, endpoint, Decimal("0.12")) == (Decimal("0.12"), "estimate")
    for prompt_tokens, completion_tokens in ((True, 10), (20, False)):
        assert actual_cost(
            {"usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}},
            endpoint,
            Decimal("0.12"),
        ) == (Decimal("0.12"), "estimate")
    with pytest.raises(ValueError, match="explicit_positive_output_cap"):
        worst_case_cost({"messages": []}, endpoint)
    with pytest.raises(ValueError, match="output_cap_exceeds"):
        worst_case_cost({"messages": [], "max_tokens": 51}, endpoint)
    for metadata in (
        {},
        {"pricing": {"prompt": "-1", "completion": "0"}},
        {"pricing": {"prompt": "0", "completion": "0", "image": "1"}},
    ):
        with pytest.raises(ValueError):
            prices(endpoint.model_copy(update={"metadata": metadata}))
    with_fee = endpoint.model_copy(deep=True)
    with_fee.metadata["pricing"]["request"] = "1"
    assert worst_case_cost({"messages": [], "max_tokens": 10}, with_fee) == Decimal("1.12")
    assert actual_cost(
        {"usage": {"prompt_tokens": 20, "completion_tokens": 10}}, with_fee, Decimal("1.12")
    ) == (Decimal("1.04"), "estimate")
    with pytest.raises(ValueError, match="prompt_token_bound"):
        worst_case_cost(
            {"messages": [], "max_tokens": 10},
            endpoint.model_copy(update={"metadata": {"pricing": endpoint.metadata["pricing"]}}),
        )


def test_validation_estimate_and_cli_gates(tmp_path: Path) -> None:
    plan = load_plan(Path("configs/plans/transport_smoke.yaml"))
    assert load_requirements(Path("configs/requirements/example_app.yaml")).app == "support-bot"
    estimate = estimate_plan(plan)
    assert estimate["total_reserved_cost_usd"] == "0"
    assert estimate["cost_table"][0]["n_requests"] == 8
    assert estimate["mde_table"][0]["reference_tier"] == "MOCK"
    duplicate = plan.model_dump()
    duplicate["endpoints"].append(duplicate["endpoints"][0])
    with pytest.raises(ValueError):
        RunPlan.model_validate(duplicate)
    for url in (
        "https://remote.example/v1",
        "http://user:password@localhost/v1",
        "http://localhost/v1?key=private",
    ):
        broken = plan.model_dump()
        broken["endpoints"][0]["base_url"] = url
        with pytest.raises(ValueError):
            RunPlan.model_validate(broken)
    bad = tmp_path / "invalid.yaml"
    bad.write_text("endpoints: [")
    with pytest.raises(ValueError, match="Invalid plan"):
        load_plan(bad)
    with pytest.raises(ValueError, match="Invalid requirements"):
        load_requirements(bad)
    cli = CliRunner()
    assert cli.invoke(app, ["run", "--plan", "configs/plans/mock_smoke.yaml"]).exit_code == 4
    assert (
        cli.invoke(
            app, ["run", "--plan", "configs/plans/mock_smoke.yaml", "--max-usd", "1"]
        ).exit_code
        == 4
    )
    assert (
        cli.invoke(app, ["run", "--plan", "configs/plans/mock_smoke.yaml", "--dry-run"]).exit_code
        == 0
    )
    assert (
        cli.invoke(
            app, ["run", "--plan", "configs/plans/mock_smoke.yaml", "--yes", "--dry-run"]
        ).exit_code
        == 3
    )
    assert cli.invoke(app, ["estimate", "--plan", str(bad)]).exit_code == 3


async def test_key_credit_limit_and_unavailable_metadata_fail_closed() -> None:
    endpoint = priced().model_copy(update={"kind": "openrouter"})

    async def check(data: Any, owner: bool = False, status: int = 200) -> None:
        def response(request: httpx.Request) -> httpx.Response:
            assert request.method == "GET" and request.url.path.endswith("/key")
            return httpx.Response(status, json={"data": data})

        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
            await key_preflight(client, endpoint, {}, owner, Decimal("0.5"))

    with pytest.raises(BudgetExceeded, match="requires_credit_limit"):
        await check({"limit": None, "limit_remaining": 5})
    with pytest.raises(BudgetExceeded, match="below_planned"):
        await check({"limit": 1, "limit_remaining": 0.1})
    with pytest.raises(BudgetExceeded, match="credit_unavailable"):
        await check({"limit": 1, "limit_remaining": None})
    with pytest.raises(SafeError, match="key_preflight_unavailable"):
        await check({}, status=401)
    with pytest.raises(SafeError, match="key_preflight_unavailable"):
        await check([])
    await check({"limit": 2, "limit_remaining": 1})
    await check({"limit": None, "limit_remaining": None}, owner=True)


def test_doctor_does_not_present_a_key_limit_as_account_balance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "example-credential-for-local-test")
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *args, **kwargs: httpx.Response(
            200, json={"data": {"limit": 100, "limit_remaining": 100}}
        ),
    )
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Key spending limit: configured; remaining $100.00" in result.stdout
    assert "Account balance: unverified" in result.stdout
    assert "example-credential-for-local-test" not in result.stdout


async def test_rate_limiter_uses_virtual_time_and_retry_respects_server_wait() -> None:
    now = 0.0

    async def sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    bucket = TokenBucket(60, 1, clock=lambda: now, sleep=sleep)
    await bucket.acquire()
    await bucket.acquire()
    assert now == 1
    with pytest.raises(ValueError):
        await bucket.acquire(2)
    attempt = Attempt(
        n=2,
        status=429,
        timings=Timings(start_unix=0, total_ms=0),
        response_headers={"retry-after": "120"},
    )
    assert delay_seconds(attempt, 0.5, 30, random.Random(1)) == 120
    attempt.response_headers = {"retry-after": "not a date"}
    assert 0 <= delay_seconds(attempt, 0.5, 30, random.Random(1)) <= 30
    attempt.response_headers = {"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}
    assert delay_seconds(attempt, 0.5, 30, random.Random(1)) >= 0
    assert retryable(attempt)
    attempt.status = 400
    assert not retryable(attempt)
    assert conservative_paired_mde(100, 2) > conservative_paired_mde(100, 1)
    with pytest.raises(ValueError):
        conservative_paired_mde(0)
