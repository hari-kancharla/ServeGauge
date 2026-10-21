"""B05 normal reserved mock checks, matched packs and sealed raw reconstruction."""

import json
import shutil

import httpx

from servegauge.diagnose.rules import diagnoses
from servegauge.mock.faults import FaultConfig
from servegauge.mock.server import create_app
from servegauge.refs.pack import build_pack, load_pack, save_pack
from servegauge.runner.check import run_check
from servegauge.runner.endpoints import resolve_endpoint
from servegauge.runner.evaluate import evaluate, reference_source
from servegauge.runner.plan import RunPlan
from servegauge.storage.analysis import reconstruct
from servegauge.verdicts.config import load_policy


async def test_reserved_tokenizer_check_pack_and_raw_only_reconstruction(tmp_path):
    (tmp_path / "configs").mkdir()
    shutil.copyfile("configs/budgets.yaml", tmp_path / "configs/budgets.yaml")
    endpoints = [
        resolve_endpoint(
            f"http://127.0.0.1:{8011 + index}/v1", "mock", "gpt-oss-120b", identifier=name
        )
        for index, name in enumerate(("count-shift", "baseline", "clean"))
    ]
    plan = RunPlan(
        plan_id="tokenizer-hint",
        profile="gpt-oss-120b",
        level="core",
        seed=2911,
        reference={"tier": "BASELINE"},
        endpoints=endpoints,
        suites=["S7"],
        max_usd=0,
        requests_per_minute=600000,
        retries=0,
    )
    baseline = endpoints[1].id
    captured = await run_check(
        plan,
        root=tmp_path,
        approved=True,
        tests="B05",
        baseline_id=baseline,
        transports={
            e.id: httpx.ASGITransport(
                app=create_app(
                    seed=81 + i, faults=FaultConfig(misreport_usage_pct=30 if i == 0 else 0)
                )
            )
            for i, e in enumerate(endpoints)
        },
    )
    assert captured.code == 1
    actual = next(c for c in captured.checks if c.check_id == "B05")
    assert actual.verdict == "FAIL" and actual.n == 200 and actual.delta_ci_low > 0.03
    assert actual.reference_tier == "BASELINE" and actual.margin == 0.03
    assert not any(d.code == "DIFFERENT_MODEL_LIKE" for d in diagnoses(captured.checks))
    analysis = reconstruct(tmp_path / "data/raw", captured.run_id)
    assert sum(r.item_id.startswith("tokenizer-hint-") for r in analysis.records) == 600
    policy = load_policy()
    compared = evaluate(
        analysis.plan,
        captured.run_id,
        analysis.items,
        analysis.samples,
        policy,
        target_ids=[e.id for e in endpoints if e.id != baseline],
        baseline_id=baseline,
    )
    cards = {(c.endpoint_id, c.check_id): c for c in compared}
    assert cards[endpoints[2].id, "B05"].verdict == "PASS"
    source = reference_source(
        analysis.plan,
        captured.run_id,
        analysis.items,
        analysis.samples,
        policy,
        analysis.manifest,
        baseline_id=baseline,
    )
    pack = load_pack(
        str(save_pack(build_pack(source, "BASELINE", seed=plan.seed), tmp_path / "refs"))
    )
    packed = evaluate(
        analysis.plan,
        captured.run_id,
        analysis.items,
        analysis.samples,
        policy,
        target_ids=[endpoints[0].id, endpoints[2].id],
        reference_pack=pack,
    )
    assert [(c.endpoint_id, c.verdict) for c in packed if c.check_id == "B05"] == [
        (endpoints[0].id, "FAIL"),
        (endpoints[2].id, "PASS"),
    ]
    raw = tmp_path / "raw-only"
    shutil.copytree(tmp_path / "data/raw", raw)
    shutil.rmtree(tmp_path / "data/runs")
    rebuilt = reconstruct(raw, captured.run_id)
    assert rebuilt.samples == analysis.samples and rebuilt.checks == analysis.checks
    analysis.samples[baseline]["B05"].details["input_hashes"] = {}
    mismatch = evaluate(
        analysis.plan,
        captured.run_id,
        analysis.items,
        analysis.samples,
        policy,
        target_ids=[endpoints[0].id],
        baseline_id=baseline,
    )
    assert next(c for c in mismatch if c.check_id == "B05").verdict == "INCONCLUSIVE"
    fields = {
        "check_id",
        "endpoint_id",
        "verdict",
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
    }
    receipt = {
        "scope": "local scripted count controls; no model/tokenizer identity attestation",
        "captured_tokenizer_exchanges": 600,
        "n_per_role": 200,
        "model_api_calls": 0,
        "new_spend_usd": 0,
        "raw_only_checks_and_samples_equal": True,
        "matched_pack_decisions_equal": True,
        "missing_layout_abstains": True,
        "no_model_diagnosis_without_fingerprints": True,
        "cards": [
            {k: v for k, v in cards[e.id, "B05"].model_dump(mode="json").items() if k in fields}
            for e in (endpoints[0], endpoints[2])
        ],
    }
    (tmp_path / "tokenizer-acceptance.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
