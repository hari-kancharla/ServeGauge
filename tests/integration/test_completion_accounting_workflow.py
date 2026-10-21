"""Reserved, socket-free adapter captures, references and raw-only accounting replay."""

import json
import shutil

import httpx
import pytest

from servegauge.core.hashing import content_hash
from servegauge.core.models import EndpointSpec, OpenRouterPin
from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.profiles import load_profile
from servegauge.refs.pack import build_pack, load_pack, save_pack
from servegauge.runner.check import normalize_tests, selected_check
from servegauge.runner.evaluate import collect, evaluate, reference_source
from servegauge.runner.execute import execute_plan
from servegauge.runner.plan import RunPlan
from servegauge.scoring.routing import ROUTING_CHECK
from servegauge.storage.analysis import reconstruct
from servegauge.verdicts.config import load_policy


def fixture_transport(seed, *, misreport=0, generation_shift=0):
    engine = ScriptedBackend(seed, FaultConfig(misreport_usage_pct=misreport))
    generations = {}

    def handler(request):
        assert not request.headers.get("authorization")
        if request.url.path.endswith("/key"):
            return httpx.Response(200, json={"data": {"limit": 0, "limit_remaining": 0}})
        if request.url.path.endswith("/generation"):
            return httpx.Response(200, json={"data": generations[request.url.params["id"]]})
        assert request.url.path.endswith("/chat/completions")
        body = json.loads(request.content)
        assert body["plugins"] == [{"id": "context-compression", "enabled": False}]
        response = engine.respond(body)
        response["id"] = f"gen-fixture-{seed}-{engine.sequence}"
        usage = response["usage"]
        generations[response["id"]] = {
            "id": response["id"],
            "model": response["model"],
            "provider_name": "Fixture",
            "cancelled": False,
            "finish_reason": "stop",
            "total_cost": 0,
            "tokens_prompt": round(usage["prompt_tokens"] * (1.1 + generation_shift)),
            "native_tokens_prompt": usage["prompt_tokens"],
            "tokens_completion": round(usage["completion_tokens"] * (1.1 + generation_shift)),
            "native_tokens_completion": usage["completion_tokens"],
        }
        return httpx.Response(200, json=response)

    return httpx.MockTransport(handler)


async def test_reserved_accounting_adapter_pack_and_raw_only_roundtrip(tmp_path):
    (tmp_path / "configs").mkdir()
    shutil.copyfile("configs/budgets.yaml", tmp_path / "configs/budgets.yaml")
    endpoints = [
        EndpointSpec(
            id=name,
            kind="openrouter",
            base_url=f"http://127.0.0.1:{8011 + index}/api/v1",
            model="mock",
            claimed_profile="gpt-oss-120b",
            openrouter_pin=OpenRouterPin(tag="fixture"),
            labels={"provider_name": "Fixture"},
            metadata={
                "tag": "fixture",
                "provider_name": "Fixture",
                "pricing": {"prompt": "0", "completion": "0"},
                "context_length": 131072,
                "max_completion_tokens": 4096,
                "supported_parameters": ["max_tokens", "reasoning"],
            },
        )
        for index, name in enumerate(("clean", "baseline", "usage-shift", "generation-shift"))
    ]
    plan = RunPlan(
        plan_id="accounting-capture",
        profile="gpt-oss-120b",
        level="core",
        seed=911,
        reference={"tier": "BASELINE"},
        endpoints=endpoints,
        suites=["S7"],
        max_usd=0,
        requests_per_minute=60000,
        retries=0,
    )
    result = await execute_plan(
        plan,
        root=tmp_path,
        approved=True,
        transports={
            "clean": fixture_transport(31),
            "baseline": fixture_transport(32),
            "usage-shift": fixture_transport(33, misreport=30),
            "generation-shift": fixture_transport(34, generation_shift=0.3),
        },
    )
    analysis = reconstruct(tmp_path / "data/raw", result.run_id)
    assert len(analysis.records) == 3200
    assert all(record.provider_meta and record.cost_usd == 0 for record in analysis.records)
    policy = load_policy()
    scores = evaluate(
        plan,
        result.run_id,
        analysis.items,
        analysis.samples,
        policy,
        target_ids=["clean", "usage-shift", "generation-shift"],
        baseline_id="baseline",
    )
    checks = {(check.endpoint_id, check.check_id): check for check in scores}
    assert checks["clean", "B02"].verdict == "PASS"
    assert checks["usage-shift", "B02"].verdict == "FAIL"
    assert checks["usage-shift", "B02"].n == 200
    assert checks["generation-shift", "B03.completion"].verdict == "FAIL"
    assert checks["clean", "B03.completion"].verdict == "PASS"
    # A nonzero normalized/native offset is calibrated, not treated as a violation.
    assert checks["clean", "B03.completion"].estimate > 0.08
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
        target_ids=["clean", "usage-shift", "generation-shift"],
        reference_pack=pack,
    )
    by_pack = {(check.endpoint_id, check.check_id): check for check in packed}
    for identity in (
        ("clean", "B02"),
        ("usage-shift", "B02"),
        ("generation-shift", "B03.completion"),
        ("clean", "B03.completion"),
    ):
        assert by_pack[identity].verdict == checks[identity].verdict
    fields = (
        "endpoint_id",
        "check_id",
        "verdict",
        "n",
        "estimate",
        "ci_low",
        "ci_high",
        "delta",
        "delta_ci_low",
        "delta_ci_high",
        "margin",
        "p_value",
        "adjusted_p",
        "reference_tier",
        "reason",
    )
    selected = [
        ("clean", "B02"),
        ("usage-shift", "B02"),
        ("generation-shift", "B03.completion"),
        ("clean", "B03.completion"),
    ]
    receipt = {
        "scope": "zero-price offline adapter fixtures; not a provider billing validation",
        "request_count": len(analysis.records),
        "n_per_role_probe_group": 200,
        "api_calls": 0,
        "spend_usd": 0,
        "seeds": {"inventory": plan.seed, "roles": [31, 32, 33, 34]},
        "cards": [
            {k: v for k, v in checks[identity].model_dump(mode="json").items() if k in fields}
            for identity in selected
        ],
        "reference_pack_id": pack.pack_id,
        "raw_record_hash": content_hash(
            [
                r.model_dump(mode="json")
                for r in sorted(analysis.records, key=lambda r: r.exchange_id)
            ]
        ),
    }
    (tmp_path / "accounting-acceptance.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    raw_copy = tmp_path / "raw-only"
    shutil.copytree(tmp_path / "data/raw", raw_copy)
    shutil.rmtree(tmp_path / "data/runs")
    rebuilt = reconstruct(raw_copy, result.run_id)
    assert rebuilt.samples == analysis.samples
    # Derivation uses sealed contracts rather than current inventory/config defaults.
    assert collect(plan, analysis.items, analysis.records, load_profile(plan.profile), policy) == {
        endpoint: {check: sample for check, sample in scores.items() if check != ROUTING_CHECK}
        for endpoint, scores in rebuilt.samples.items()
    }
    assert all(ROUTING_CHECK in scores for scores in rebuilt.samples.values())
    # A matched item ID does not excuse different request settings or visibility modes.
    analysis.samples["baseline"]["B02"].details["input_hashes"] = {}
    mismatch = evaluate(
        plan,
        result.run_id,
        analysis.items,
        analysis.samples,
        policy,
        target_ids=["clean"],
        baseline_id="baseline",
    )
    assert next(c for c in mismatch if c.check_id == "B02").verdict == "INCONCLUSIVE"
    for token in ("B02", "B03", "B03.completion", "billing"):
        assert selected_check("B03.completion" if token != "B02" else "B02", normalize_tests(token))


@pytest.mark.parametrize("check", ["B02", "B03.prompt", "B03.completion"])
def test_completion_pack_requires_matching_layouts(check):
    from servegauge.refs.pack import reference_for
    from tests.unit.test_refs import reference_run

    source = reference_run()
    source.checks[check] = source.checks["T02"]
    source.inputs[check] = {
        endpoint: {item: ["same"] for item in values}
        for endpoint, values in source.checks[check].items()
    }
    pack = build_pack(source, "R3", seed=3)
    with pytest.raises(ValueError, match="identical_input"):
        reference_for(
            pack,
            target="a",
            check=check,
            item_ids=sorted(source.checks[check]["a"]),
            configuration=source.configuration,
        )
    source.inputs[check]["b"]["item00"] = ["different"]
    assert check not in build_pack(source, "R3", seed=3).outcomes["a"]
