"""Optional reserved synthetic wire/pack/raw-replay flow with the actual GLM counter."""

import copy
import json
import os
import shutil

import pytest
from integration.test_completion_accounting_workflow import fixture_transport

from servegauge.core.models import EndpointSpec, OpenRouterPin
from servegauge.refs.pack import build_pack, load_pack, save_pack
from servegauge.runner.evaluate import evaluate, reference_source
from servegauge.runner.execute import execute_plan
from servegauge.runner.ledger import Ledger
from servegauge.runner.plan import RunPlan
from servegauge.storage.accounting_identity import accounting_counter_identity
from servegauge.storage.analysis import reconstruct
from servegauge.storage.workspace import initialize
from servegauge.verdicts.config import load_policy


@pytest.mark.slow
@pytest.mark.parametrize("item_count", [200, 800])
async def test_actual_glm_counter_reserved_synthetic_capture_pack_and_raw_replay(
    tmp_path, item_count
):
    if not os.environ.get("SERVEGAUGE_GLM_REFERENCE_PYTHON"):
        pytest.skip("explicit local GLM tokenizer acceptance runtime/assets required")
    initialize(tmp_path)
    endpoints = [
        EndpointSpec(
            id=name,
            kind="openrouter",
            base_url=f"http://127.0.0.1:{8011 + index}/api/v1",
            model="mock",
            claimed_profile="glm-5.3",
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
        for index, name in enumerate(("clean", "baseline", "usage-shift"))
    ]
    plan = RunPlan(
        plan_id="glm-synthetic-accounting",
        profile="glm-5.3",
        level="core",
        seed=911,
        reference={"tier": "BASELINE"},
        endpoints=endpoints,
        suites=["S7"],
        max_usd=0,
        requests_per_minute=60000,
        retries=0,
        overrides={"S7": {"items": item_count}},
    )
    result = await execute_plan(
        plan,
        root=tmp_path,
        approved=True,
        transports={
            "clean": fixture_transport(31),
            "baseline": fixture_transport(32),
            "usage-shift": fixture_transport(33, misreport=30),
        },
    )
    assert result.spent_usd == 0 and not result.errors
    analysis = reconstruct(tmp_path / "data/raw", result.run_id)
    policy = load_policy()
    checks = evaluate(
        plan,
        result.run_id,
        analysis.items,
        analysis.samples,
        policy,
        target_ids=["clean", "usage-shift"],
        baseline_id="baseline",
    )
    selected = {(row.endpoint_id, row.check_id): row for row in checks}
    for check in ("B01", "B02", "B05"):
        control = selected["clean", check]
        assert control.n == item_count and control.reference_tier == "BASELINE"
        assert control.ci_low is not None and control.ci_high is not None
        assert control.verdict in {"PASS", "INCONCLUSIVE"}
        shifted = selected["usage-shift", check]
        assert shifted.n == item_count and shifted.reference_tier == "BASELINE"
        assert shifted.delta_ci_low is not None and shifted.delta_ci_low > 0.03
        if item_count == 200 and check in {"B01", "B05"}:
            assert control.verdict == shifted.verdict == "INCONCLUSIVE"
            assert "reference noise exceeds declared margin" in shifted.reason
        else:
            assert control.verdict == "PASS" and shifted.verdict == "FAIL"
    assert analysis.samples["clean"]["B01"].details["counter_version"] == "glm-hf-api-adapter-v1"
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
    identity = accounting_counter_identity("glm-5.3")
    for check in ("B01", "B02", "B05"):
        assert pack.manifest["configuration"]["checks"][check]["local_counter"] == identity
    packed = evaluate(
        plan,
        result.run_id,
        analysis.items,
        analysis.samples,
        policy,
        target_ids=["clean", "usage-shift"],
        reference_pack=pack,
    )
    packed_selected = {(row.endpoint_id, row.check_id): row for row in packed}
    fields = (
        "n",
        "estimate",
        "ci_low",
        "ci_high",
        "reference_tier",
        "reference_estimate",
        "delta",
        "delta_ci_low",
        "delta_ci_high",
        "p_value",
        "adjusted_p",
        "verdict",
    )
    for key, expected in selected.items():
        if key[1] in {"B01", "B02", "B05"}:
            assert {field: getattr(expected, field) for field in fields} == {
                field: getattr(packed_selected[key], field) for field in fields
            }
    incompatible = copy.deepcopy(pack)
    for check in ("B01", "B02", "B05"):
        incompatible.manifest["configuration"]["checks"][check]["local_counter"][
            "runtime_versions"
        ]["tokenizers"] = "unknown"
    refused = evaluate(
        plan,
        result.run_id,
        analysis.items,
        analysis.samples,
        policy,
        target_ids=["usage-shift"],
        reference_pack=incompatible,
    )
    assert all(
        row.verdict == "INCONCLUSIVE" for row in refused if row.check_id in {"B01", "B02", "B05"}
    )
    raw_copy = tmp_path / "raw-only"
    shutil.copytree(tmp_path / "data/raw", raw_copy)
    shutil.rmtree(tmp_path / "data/runs")
    rebuilt = reconstruct(raw_copy, result.run_id)
    assert rebuilt.samples == analysis.samples
    ledger = Ledger(tmp_path)
    ledger.refresh()
    assert not ledger.pending and ledger.development_spend() == 0
    receipt = {
        "version": "model-b-synthetic-accounting-v1",
        "scope": "synthetic zero-price local adapters; no GLM provider or bill validation",
        "counter": identity,
        "records": len(analysis.records),
        "n_per_role_probe_group": item_count,
        "checks": [
            row.model_dump(mode="json") for row in checks if row.check_id in {"B01", "B02", "B05"}
        ],
        "raw_only_reconstruction_equal": True,
        "changed_runtime_reference_refused": True,
        "reference_pack_id": pack.pack_id,
        "model_api_calls": 0,
        "new_spend_usd": 0,
    }
    (tmp_path / "accounting-acceptance.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
