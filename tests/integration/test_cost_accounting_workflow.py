"""Local priced adapter fixture; synthetic ledger dollars never reach a public API."""

import json
import shutil
from datetime import UTC, datetime
from decimal import Decimal

import httpx

from servegauge.adapters.openrouter import price_provenance
from servegauge.core.hashing import content_hash
from servegauge.core.models import EndpointSpec, OpenRouterPin
from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.refs.pack import build_pack, load_pack, save_pack
from servegauge.runner.evaluate import evaluate, reference_source
from servegauge.runner.execute import execute_plan
from servegauge.runner.plan import RunPlan
from servegauge.scoring.cost_accounting import cost_samples
from servegauge.storage.analysis import reconstruct
from servegauge.verdicts.config import load_policy

PRICING = {
    "prompt": ".00000001",
    "completion": ".00000002",
    "input_cache_read": ".000000002",
    "input_cache_write": ".0000000125",
    "request": ".0000001",
    "overrides": [{"min_prompt_tokens": 60, "prompt": ".000000013"}],
}


def priced_transport(seed, factor, rate_factor="1"):
    engine = ScriptedBackend(seed, FaultConfig())
    generations, seen = {}, set()

    def handler(request):
        assert not request.headers.get("authorization")
        if request.url.path.endswith("/key"):
            return httpx.Response(200, json={"data": {"limit": 1, "limit_remaining": 1}})
        if request.url.path.endswith("/generation"):
            return httpx.Response(200, json={"data": generations[request.url.params["id"]]})
        assert request.url.path.endswith("/chat/completions")
        body = json.loads(request.content)
        key = content_hash(body)
        retry = key in seen
        seen.add(key)
        response = engine.respond(body)
        response["id"] = f"gen-priced-{seed}-{engine.sequence}"
        usage = response["usage"]
        usage["is_byok"] = False
        prompt, completion = usage["prompt_tokens"], usage["completion_tokens"]
        mode = int(key[:8], 16) % 3
        read = prompt // 3 if mode == 1 else 0
        write = prompt // 4 if mode == 2 else 0
        usage["prompt_tokens_details"] = {"cached_tokens": read, "cache_write_tokens": write}
        # Independent explicit arithmetic, not the scorer or tariff resolver.
        p = Decimal(".000000013" if prompt > 60 else ".00000001")
        input_cost = (
            (prompt - read - write) * p
            + read * Decimal(".000000002")
            + write * Decimal(".0000000125")
        )
        expected = input_cost + completion * Decimal(".00000002") + Decimal(".0000001")
        charge = expected * Decimal(factor) * Decimal(rate_factor)
        usage["cost"] = float(charge)
        generations[response["id"]] = {
            "id": response["id"],
            "model": "mock",
            "provider_name": "Fixture",
            "cancelled": False,
            "is_byok": False,
            "native_tokens_prompt": prompt,
            "native_tokens_completion": completion,
            "native_tokens_cached": read,
            "tokens_prompt": prompt,
            "tokens_completion": completion,
            "total_cost": float(charge),
            "cache_discount": float((prompt * p - input_cost) * Decimal(rate_factor)),
        }
        return httpx.Response(200 if retry else 503, json=response)

    return httpx.MockTransport(handler)


async def test_reserved_cost_cache_retry_reference_and_raw_reconstruction(tmp_path):
    (tmp_path / "configs").mkdir()
    shutil.copyfile("tests/fixtures/simulated-budgets.yaml", tmp_path / "configs/budgets.yaml")
    endpoints = []
    for index, name in enumerate(("baseline", "clean", "charge-shift")):
        base = f"http://127.0.0.1:{8011 + index}/api/v1"
        scale = 2 if name == "clean" else 1
        pricing = {
            key: str(Decimal(value) * scale) for key, value in PRICING.items() if key != "overrides"
        }
        pricing["overrides"] = [
            {"min_prompt_tokens": 60, "prompt": str(Decimal(".000000013") * scale)}
        ]
        endpoints.append(
            EndpointSpec(
                id=name,
                kind="openrouter",
                model="mock",
                base_url=base,
                claimed_profile="gpt-oss-120b",
                openrouter_pin=OpenRouterPin(tag="fixture"),
                labels={"provider_name": "Fixture"},
                metadata={
                    "tag": "fixture",
                    "pricing": pricing,
                    "max_prompt_tokens": 10000,
                    "context_length": 10000,
                    "max_completion_tokens": 4096,
                    "supported_parameters": ["max_tokens", "reasoning"],
                    "price_provenance": price_provenance(
                        base + "/models/mock/endpoints",
                        "mock",
                        "fixture",
                        pricing,
                        captured_at=datetime(1970, 1, 1, tzinfo=UTC).isoformat(),
                    ),
                },
            )
        )
    plan = RunPlan(
        plan_id="offline-priced-accounting",
        profile="gpt-oss-120b",
        level="core",
        seed=1911,
        reference={"tier": "BASELINE"},
        endpoints=endpoints,
        suites=["S7"],
        max_usd=0.5,
        retries=1,
        retry_base_seconds=0,
        retry_max_seconds=0,
        requests_per_minute=600000,
    )
    result = await execute_plan(
        plan,
        root=tmp_path,
        approved=True,
        transports={
            "baseline": priced_transport(51, "1"),
            "clean": priced_transport(52, "1", "2"),
            "charge-shift": priced_transport(53, "1.3"),
        },
    )
    analysis = reconstruct(tmp_path / "data/raw", result.run_id)
    assert len(analysis.records) == 2400
    assert all(len(r.attempts) == 2 and r.cost_source == "estimate" for r in analysis.records)
    policy = load_policy()
    direct = evaluate(
        plan,
        result.run_id,
        analysis.items,
        analysis.samples,
        policy,
        target_ids=["clean", "charge-shift"],
        baseline_id="baseline",
    )
    cards = {(c.endpoint_id, c.check_id): c for c in direct}
    assert cards["clean", "B04"].verdict == "PASS"
    shifted = cards["charge-shift", "B04"]
    assert shifted.verdict == "FAIL" and shifted.n == 200 and shifted.delta_ci_low > 0.29
    assert shifted.reference_tier == "BASELINE" and shifted.margin == 0.03
    source = reference_source(
        plan,
        result.run_id,
        analysis.items,
        analysis.samples,
        policy,
        analysis.manifest,
        baseline_id="baseline",
    )
    pack = load_pack(
        str(save_pack(build_pack(source, "BASELINE", seed=plan.seed), tmp_path / "refs"))
    )
    packed = evaluate(
        plan,
        result.run_id,
        analysis.items,
        analysis.samples,
        policy,
        target_ids=["clean", "charge-shift"],
        reference_pack=pack,
    )
    assert [(c.endpoint_id, c.verdict) for c in packed if c.check_id == "B04"] == [
        ("clean", "PASS"),
        ("charge-shift", "FAIL"),
    ]
    raw = tmp_path / "raw-only"
    shutil.copytree(tmp_path / "data/raw", raw)
    shutil.rmtree(tmp_path / "data/runs")
    assert reconstruct(raw, result.run_id).samples == analysis.samples
    damaged = [r.model_copy(deep=True) for r in analysis.records]
    affected = next(
        r for r in damaged if r.endpoint_id == "clean" and r.item_id.startswith("cost-accounting-")
    )
    affected.attempts[0].metadata["cost_source"] = "estimate"
    assert cost_samples(analysis.items, damaged, "clean")["B04"].status == "INCONCLUSIVE"
    fields = {
        "check_id",
        "endpoint_id",
        "n",
        "estimate",
        "ci_low",
        "ci_high",
        "delta",
        "delta_ci_low",
        "delta_ci_high",
        "reference_tier",
        "margin",
        "p_value",
        "adjusted_p",
        "verdict",
    }
    receipt = {
        "scope": "offline local fixtures; all ledger dollars are simulated, no public API",
        "exchanges": len(analysis.records),
        "attempts": sum(len(r.attempts) for r in analysis.records),
        "n_per_role_cost_group": 200,
        "new_real_spend_usd": 0,
        "model_api_calls": 0,
        "raw_only_samples_equal": True,
        "matched_pack_decisions_equal": True,
        "cards": [
            {k: v for k, v in cards[ep, "B04"].model_dump(mode="json").items() if k in fields}
            for ep in ("clean", "charge-shift")
        ],
        "raw_record_hash": content_hash(
            [
                r.model_dump(mode="json")
                for r in sorted(analysis.records, key=lambda r: r.exchange_id)
            ]
        ),
    }
    (tmp_path / "cost-acceptance.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
