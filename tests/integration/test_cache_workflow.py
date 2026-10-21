"""Normal cache collection uses reservations and reconstructs from sealed raw inputs."""

import importlib
import json
import shutil

import httpx
import pytest

from servegauge.diagnose.bundle import write_bundle
from servegauge.diagnose.repro import replay_bundle
from servegauge.diagnose.rules import diagnoses
from servegauge.mock.faults import FaultConfig
from servegauge.mock.server import create_app
from servegauge.runner.check import normalize_tests, run_check
from servegauge.runner.estimate import estimate_plan
from servegauge.runner.execute import execute_plan
from servegauge.runner.plan import RunPlan
from servegauge.storage.analysis import reconstruct
from servegauge.suites.registry import item_checks, plan_items
from servegauge.suites.s1_cache import CACHE_CHECKS, VERSION, cache_items
from servegauge.tokencount.counters import text_tokens
from servegauge.validation.capture import endpoint
from servegauge.validation.common import named_seed


def cache_transports(plan, *, changed=False):
    """Native cache stimulus; optional explicit wire-level deterministic control."""
    output = {}
    for ep, target in [(plan.endpoints[0], True), (plan.endpoints[1], False)]:
        seed = ep.metadata["mock_declaration"]["seed"]
        native = httpx.ASGITransport(
            app=create_app(
                seed=seed, faults=FaultConfig(seed=seed, stale_cache=0.20 if target else 0)
            )
        )

        async def handler(request, native=native, target=target):
            response = await native.handle_async_request(request)
            if changed and target and request.method == "POST":
                body = json.loads(request.content)
                if any(
                    str(m.get("content", "")).startswith("Observation cache-")
                    for m in body["messages"]
                ):
                    # Deliberately freeze BOTH conditions after native generation.
                    # This is a disclosed changed-control fixture, not another
                    # qualification of the stock stale-cache fault.
                    value = json.loads(await response.aread())
                    message = value["choices"][0]["message"]
                    change = text_tokens("red") - text_tokens(message["content"])
                    message["content"] = "red"
                    value["usage"]["completion_tokens"] += change
                    value["usage"]["total_tokens"] += change
                    return httpx.Response(200, json=value)
            return response

        output[ep.id] = httpx.MockTransport(handler)
    return output


def fixture(tmp_path):
    (tmp_path / "configs").mkdir()
    shutil.copyfile("configs/budgets.yaml", tmp_path / "configs/budgets.yaml")
    endpoints = [
        endpoint("mock:target", 201, clean=True),
        endpoint("mock:baseline", 202, clean=True),
    ]
    plan = RunPlan(
        plan_id="normal-cache",
        profile="gpt-oss-120b",
        level="smoke",
        seed=42,
        endpoints=endpoints,
        suites=["S1"],
        reference={"tier": "BASELINE"},
        max_usd=0,
        retries=0,
        requests_per_minute=600000,
        concurrency_per_endpoint=1,
        overrides={"CACHE": {"groups": 3, "draws": 4}},
    )
    transports = {
        e.id: httpx.ASGITransport(
            app=create_app(seed=e.metadata["mock_declaration"]["seed"], faults=FaultConfig())
        )
        for e in endpoints
    }
    return plan, transports


def test_cache_selector_batch_inventory_and_estimator_account_for_every_request(tmp_path):
    plan, _ = fixture(tmp_path)
    assert normalize_tests("cache") == {"P03.cache"}
    items = plan_items(plan)
    cache = [i for i in items if i.version == VERSION]
    assert cache == cache_items("smoke", plan.seed, plan.overrides["CACHE"])
    assert set(c for i in cache for c in item_checks(i)) == set(CACHE_CHECKS)
    estimate = estimate_plan(plan, items_override=cache)
    assert all(row["n_requests"] == 24 for row in estimate["cost_table"])


async def test_normal_cache_check_complete_cohort_and_raw_only_equality(tmp_path):
    plan, transports = fixture(tmp_path)
    result = await run_check(
        plan,
        tests="cache",
        baseline_id="mock:baseline",
        root=tmp_path,
        approved=True,
        transports=transports,
    )
    analysis = reconstruct(tmp_path / "data/raw", result.run_id)
    assert analysis.checks == result.checks
    cards = {c.check_id: c for c in result.checks if c.check_id in CACHE_CHECKS}
    assert set(cards) == set(CACHE_CHECKS)
    assert all(c.n == 3 and c.reference_tier == "BASELINE" for c in cards.values())
    assert all(c.details["cache_contract"]["version"] == VERSION for c in cards.values())
    assert all(c.verdict != "ERROR" for c in cards.values())
    assert all(
        v.status is None for k, v in analysis.samples["mock:target"].items() if k in CACHE_CHECKS
    )
    raw = tmp_path / "raw-only"
    shutil.copytree(tmp_path / "data/raw", raw)
    shutil.rmtree(tmp_path / "data/runs")
    rebuilt = reconstruct(raw, result.run_id)
    assert rebuilt.checks == analysis.checks and rebuilt.samples == analysis.samples
    ledger = json.loads((tmp_path / "budget/dev_spend.json").read_text())
    assert ledger is not None


async def test_batch_cache_collection_reconstructs_without_separate_preflight(tmp_path):
    plan, transports = fixture(tmp_path)
    plan.endpoints = plan.endpoints[:1]
    plan.overrides["S1"] = {
        "invariant_trials": 1,
        "diversity_prompts": 1,
        "diversity_repeats": 2,
        "effort_items": 1,
        "seed_trials": 1,
    }
    result = await execute_plan(
        plan, root=tmp_path, approved=True, transports=transports, run_id="cache-batch"
    )
    analysis = reconstruct(tmp_path / "data/raw", result.run_id)
    cards = [c for c in analysis.checks if c.check_id in CACHE_CHECKS]
    assert len(cards) == 2 and all(c.n == 3 and c.verdict == "INCONCLUSIVE" for c in cards)
    assert analysis.metadata["capability_run_id"] is None
    assert any(i.version == VERSION for i in analysis.items)


async def test_different_endpoint_sampling_defaults_cannot_drive_cache_failure(tmp_path):
    plan, transports = fixture(tmp_path)
    plan.endpoints[1].defaults["seed"] = 9
    result = await run_check(
        plan,
        tests="cache",
        baseline_id="mock:baseline",
        root=tmp_path,
        approved=True,
        transports=transports,
    )
    cards = [c for c in result.checks if c.check_id in CACHE_CHECKS]
    assert len(cards) == 2 and all(c.verdict == "INCONCLUSIVE" and c.n == 0 for c in cards)
    assert all("request layouts" in c.reason for c in cards)
    analysis = reconstruct(tmp_path / "data/raw", result.run_id)
    assert analysis.checks == result.checks


async def test_changed_control_fixture_freezes_both_conditions_after_native_generation(tmp_path):
    plan, _ = fixture(tmp_path)
    result = await run_check(
        plan,
        tests="cache",
        baseline_id="mock:baseline",
        root=tmp_path,
        approved=True,
        transports=cache_transports(plan, changed=True),
    )
    analysis = reconstruct(tmp_path / "data/raw", result.run_id)
    assert all(
        set(sample.values.values()) == {0.0} and sample.status is None
        for key, sample in analysis.samples["mock:target"].items()
        if key in CACHE_CHECKS
    )


@pytest.mark.slow
async def test_original_strength_cache_full_witness_replay_and_changed_fresh_control(
    tmp_path, monkeypatch
):
    plan, _ = fixture(tmp_path)
    plan.level = "core"
    plan.seed = named_seed(202610200040, "cache-items")
    plan.overrides["CACHE"] = {"groups": 400, "draws": 20}
    target_seed = named_seed(202610200040, "cache-target")
    baseline_seed = named_seed(202610200040, "cache-baseline")
    plan.endpoints = [
        endpoint("mock:target", target_seed, clean=False),
        endpoint("mock:baseline", baseline_seed, clean=True),
    ]

    result = await run_check(
        plan,
        tests="cache",
        baseline_id="mock:baseline",
        root=tmp_path,
        approved=True,
        transports=cache_transports(plan),
    )
    source = reconstruct(tmp_path / "data/raw", result.run_id)
    assert source.checks == result.checks
    found = [d for d in diagnoses(source.checks) if d.code == "CACHE_ANOMALY"]
    assert len(found) == 1, [
        (c.check_id, c.verdict, c.reason) for c in source.checks if c.check_id in CACHE_CHECKS
    ]
    assert found[0].confidence == "low"
    assert found[0].check_ids == list(CACHE_CHECKS)
    assert [e.metadata["mock_declaration"]["seed"] for e in source.plan.endpoints] == [
        target_seed,
        baseline_seed,
    ]
    bundle = write_bundle(source, found[0], root=tmp_path)
    redacted = write_bundle(source, found[0], root=tmp_path, redact_content=True)
    assert (redacted / "bundle.json").stat().st_size < 5 * 1024 * 1024

    def no_regeneration(*args, **kwargs):
        raise AssertionError("cache replay must use exact captured paired inventories")

    module = importlib.import_module("servegauge.runner.check")
    for name in ["cache_items", "capability_items", "protocol_items"]:
        monkeypatch.setattr(module, name, no_regeneration)
    repeated_result = await replay_bundle(
        str(bundle), root=tmp_path, approved=True, transports=cache_transports(plan)
    )
    assert repeated_result["reproduced"] and repeated_result["budget_scope"] == result.run_id
    repeated = reconstruct(tmp_path / "data/raw", repeated_result["run_id"])
    assert (
        next(d for d in diagnoses(repeated.checks) if d.code == "CACHE_ANOMALY").confidence
        == "medium"
    )
    assert [i.model_dump() for i in repeated.items] == [i.model_dump() for i in source.items]
    changed_result = await replay_bundle(
        str(bundle), root=tmp_path, approved=True, transports=cache_transports(plan, changed=True)
    )
    assert not changed_result["reproduced"]
    changed = reconstruct(tmp_path / "data/raw", changed_result["run_id"])
    assert all(c.verdict == "FAIL" for c in changed.checks if c.check_id in CACHE_CHECKS)
    assert not any(d.code == "CACHE_ANOMALY" for d in diagnoses(changed.checks))
    raw = tmp_path / "raw-only"
    shutil.copytree(tmp_path / "data/raw", raw)
    shutil.rmtree(tmp_path / "data/runs")
    for analysis in [source, repeated, changed]:
        rebuilt = reconstruct(raw, analysis.manifest["run_id"])
        assert rebuilt.checks == analysis.checks and rebuilt.samples == analysis.samples
    (tmp_path / "cache-acceptance.json").write_text(
        json.dumps(
            {
                "scope": (
                    "original 0.20 native stale-cache stimulus plus explicit frozen-text control; "
                    "not power or provider qualification"
                ),
                "groups": 400,
                "draws_per_condition": 20,
                "actual_role_seeds": [target_seed, baseline_seed],
                "captured_records": sum(len(a.records) for a in [source, repeated, changed]),
                "captured_attempts": sum(
                    len(r.attempts) for a in [source, repeated, changed] for r in a.records
                ),
                "reproduced": repeated_result["reproduced"],
                "changed_fresh_rejected": not changed_result["reproduced"],
                "raw_only_equal": True,
                "exact_original_inventory_replayed": True,
                "source_confidence": found[0].confidence,
                "replayed_confidence": "medium",
                "source_cards": [
                    c.model_dump(mode="json") for c in source.checks if c.check_id in CACHE_CHECKS
                ],
                "model_api_calls": 0,
                "new_spend_usd": 0,
            },
            indent=2,
        )
        + "\n"
    )
