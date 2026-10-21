"""Original-strength faults through normal reserved native product workflows."""

import json
import shutil

import httpx
import pytest
import yaml

from servegauge.core.hashing import content_hash
from servegauge.diagnose.bundle import read_bundle, write_bundle
from servegauge.diagnose.rules import diagnoses
from servegauge.mock.faults import FaultConfig
from servegauge.mock.server import create_app
from servegauge.runner.check import run_check
from servegauge.runner.plan import RunPlan
from servegauge.storage.analysis import reconstruct
from servegauge.validation.capture import endpoint
from servegauge.validation.common import named_seed

pytestmark = pytest.mark.slow

# Independent acceptance oracle. No label enters the scorer or returned messages.
CASES = [
    ("F-TEMP", "ignore_temperature", "P03", ["S1"], "PARAM_IGNORED", "API_INCOMPATIBILITY"),
    ("F-TOPP", "ignore_top_p", "P04.top_p", ["S1"], "PARAM_IGNORED", "API_INCOMPATIBILITY"),
    ("F-MAXTOK", "ignore_max_tokens", "P01", ["S1"], "PARAM_IGNORED", "API_INCOMPATIBILITY"),
    ("F-STOP", "ignore_stop", "P02", ["S1"], "PARAM_IGNORED", "API_INCOMPATIBILITY"),
    ("F-EFFORT", "ignore_reasoning_effort", "P07", ["S1"], "PARAM_IGNORED", "API_INCOMPATIBILITY"),
    ("F-TOOLDROP", "tool_drop", "T01,T02,T06", ["S2"], "PARSER_DROP", "API_INCOMPATIBILITY"),
    ("F-LEAK", "leak_markers", "T05", ["S2"], "PARSER_LEAK", "API_INCOMPATIBILITY"),
    (
        "F-NOINDEX",
        "noindex",
        "P11,T04.parallel.stream",
        ["S1", "S2"],
        "PARSER_DROP",
        "API_INCOMPATIBILITY",
    ),
    (
        "F-ARGJSON",
        "corrupt_arguments",
        "T01,T02",
        ["S2"],
        "GUIDED_DECODING_ABSENT",
        "API_INCOMPATIBILITY",
    ),
    ("F-TEMPLATE", "template", "T07", ["S2"], "TEMPLATE_MISMATCH", "API_INCOMPATIBILITY"),
    ("F-HEADTRUNC", "headtrunc", "L04.32000", ["S4"], "CONTEXT_TRUNCATION", "API_INCOMPATIBILITY"),
    ("F-OUTTRUNC", "outtrunc", "R03.4000,R03.8000", ["S5"], "LONG_OUTPUT_FAULT", "MODEL_BEHAVIOR"),
    ("F-USAGE", "usage", "B01", ["S7"], "ACCOUNTING_MISMATCH", "ACCOUNTING"),
]


@pytest.mark.parametrize(
    "fault_id,stem,selectors,suites,expected_code,expected_class",
    CASES,
    ids=[case[0] for case in CASES],
)
async def test_original_native_fault_has_complete_code_class_and_raw_evidence(
    tmp_path, fault_id, stem, selectors, suites, expected_code, expected_class
):
    from pathlib import Path

    (tmp_path / "configs").mkdir()
    shutil.copyfile("configs/budgets.yaml", tmp_path / "configs/budgets.yaml")
    original = yaml.safe_load(Path(f"configs/faults/{stem}.yaml").read_text())
    assert original["fault_id"] == fault_id
    target_seed = named_seed(202610200100, "native-api", fault_id, "target")
    baseline_seed = named_seed(202610200100, "native-api", fault_id, "baseline")
    assert target_seed != baseline_seed and 7 not in {target_seed, baseline_seed}
    assigned = {**original, "fault_id": None, "seed": target_seed}
    target = endpoint("mock:native-target", target_seed, clean=False)
    baseline = endpoint("mock:native-baseline", baseline_seed, clean=True)
    plan = RunPlan(
        plan_id="native-api-fault-acceptance",
        profile="gpt-oss-120b",
        level="core",
        seed=named_seed(202610200100, "native-api", fault_id, "items"),
        endpoints=[target, baseline],
        suites=suites,
        reference={"tier": "BASELINE"},
        max_usd=0,
        retries=0,
        requests_per_minute=600000,
        concurrency_per_endpoint=1,
    )
    result = await run_check(
        plan,
        tests=selectors,
        baseline_id=baseline.id,
        root=tmp_path,
        approved=True,
        transports={
            target.id: httpx.ASGITransport(
                app=create_app(seed=target_seed, faults=FaultConfig(**assigned))
            ),
            baseline.id: httpx.ASGITransport(
                app=create_app(seed=baseline_seed, faults=FaultConfig(seed=baseline_seed))
            ),
        },
    )
    analysis = reconstruct(tmp_path / "data/raw", result.run_id)
    assert analysis.checks == result.checks
    found = [d for d in diagnoses(analysis.checks) if d.code == expected_code]
    assert len(found) == 1, [
        (c.check_id, c.verdict, c.n, c.reason) for c in analysis.checks if c.suite_id != "S0"
    ]
    diagnosis = found[0]
    assert diagnosis.failure_class == expected_class
    assert diagnosis.evidence and diagnosis.check_ids
    assert [e.metadata["mock_declaration"]["seed"] for e in analysis.plan.endpoints] == [
        target_seed,
        baseline_seed,
    ]
    evidence = {r.exchange_id for r in analysis.records}
    assert set(diagnosis.evidence) <= evidence
    private = write_bundle(analysis, diagnosis, root=tmp_path)
    bundle = read_bundle(str(private), tmp_path)
    assert bundle["diagnosis"] == diagnosis.model_dump(mode="json")
    assert {r["exchange_id"] for r in bundle["exchanges"]} == set(diagnosis.evidence)
    assert {c["check_id"] for c in bundle["expected_vs_observed"]} == set(diagnosis.check_ids)
    assert all(
        c["n"] > 0 and c["ci_low"] is not None and c["ci_high"] is not None
        for c in bundle["expected_vs_observed"]
    )
    issue = (private / "issue.md").read_text().replace("\\", "")
    assert expected_code in issue and expected_class in issue
    assert all(text in issue for text in diagnosis.alternatives + diagnosis.falsifiers)
    marker = "PRIVATE_NATIVE_API_PAYLOAD_MARKER"
    analysis.plan.endpoints[0].labels[marker] = marker
    redacted = write_bundle(analysis, diagnosis, root=tmp_path, redact_content=True)
    assert all(marker not in p.read_text() for p in redacted.iterdir() if p.is_file())
    raw = tmp_path / "raw-only"
    shutil.copytree(tmp_path / "data/raw", raw)
    shutil.rmtree(tmp_path / "data/runs")
    rebuilt = reconstruct(raw, result.run_id)
    assert rebuilt.checks == analysis.checks and rebuilt.samples == analysis.samples
    assert diagnoses(rebuilt.checks) == diagnoses(analysis.checks)
    receipt = {
        "scope": (
            "single original-strength native ASGI product workflow; "
            "not power/FPR or provider qualification"
        ),
        "fault_id": fault_id,
        "injected_original_configuration": original,
        "actual_seed_override": assigned,
        "actual_role_seeds": [target_seed, baseline_seed],
        "selectors": selectors,
        "run_id": result.run_id,
        "expected_code": expected_code,
        "expected_class": expected_class,
        "diagnosis": diagnosis.model_dump(mode="json"),
        "all_observed_codes": sorted(d.code for d in diagnoses(analysis.checks)),
        "witness_checks": bundle["expected_vs_observed"],
        "captured_records": len(analysis.records),
        "captured_attempts": sum(len(r.attempts) for r in analysis.records),
        "raw_records_hash": content_hash([r.model_dump(mode="json") for r in analysis.records]),
        "raw_only_checks_samples_and_diagnoses_equal": True,
        "private_witness_complete": True,
        "redacted_private_marker_absent": True,
        "private_bundle": str(private.relative_to(tmp_path)),
        "redacted_bundle": str(redacted.relative_to(tmp_path)),
        "model_api_calls": 0,
        "new_spend_usd": 0,
    }
    (tmp_path / "native-api-acceptance.json").write_text(json.dumps(receipt, indent=2) + "\n")
