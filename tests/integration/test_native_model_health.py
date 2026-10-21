"""Native original-strength health/swap witnesses and complete reserved replay."""

import importlib
import json
import shutil
from pathlib import Path

import httpx
import pytest
import yaml

from servegauge.diagnose.bundle import write_bundle
from servegauge.diagnose.repro import replay_bundle
from servegauge.diagnose.rules import diagnoses
from servegauge.mock.backends import task_from_messages
from servegauge.mock.faults import FaultConfig
from servegauge.mock.math_tasks import math_answer
from servegauge.mock.server import create_app
from servegauge.refs.pack import ReferenceRun, build_pack
from servegauge.runner.check import run_check
from servegauge.runner.plan import RunPlan
from servegauge.storage.analysis import reconstruct
from servegauge.suites.s8_operations import HEALTH_CHECKS
from servegauge.tokencount.counters import text_tokens
from servegauge.validation.capture import endpoint
from servegauge.validation.common import named_seed

pytestmark = pytest.mark.slow


def budget(root):
    (root / "configs").mkdir()
    shutil.copyfile("configs/budgets.yaml", root / "configs/budgets.yaml")


def no_regeneration(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("derived witness replay must use complete captured inventories")

    module = importlib.import_module("servegauge.runner.check")
    for name in [
        "capability_items",
        "protocol_items",
        "operational_items",
        "reasoning_items",
        "fingerprint_items",
    ]:
        monkeypatch.setattr(module, name, forbidden)


def raw_only(root, analyses):
    raw = root / "raw-only"
    shutil.copytree(root / "data/raw", raw)
    shutil.rmtree(root / "data/runs")
    for actual in analyses:
        rebuilt = reconstruct(raw, actual.manifest["run_id"])
        assert rebuilt.checks == actual.checks and rebuilt.samples == actual.samples
        assert diagnoses(rebuilt.checks) == diagnoses(actual.checks)


async def test_native_original_error_strength_complete_inventory_replay_and_healthy_control(
    tmp_path, monkeypatch
):
    budget(tmp_path)
    original = yaml.safe_load(Path("configs/faults/errors.yaml").read_text())
    # Reuse the declared native prototype seeds; this is integration/replay, not
    # a fresh independent population-rate study. The full core S8 design is used.
    target_seed, baseline_seed = 1301, 1302
    target = endpoint("mock:native-target", target_seed, clean=False)
    baseline = endpoint("mock:native-baseline", baseline_seed, clean=True)
    plan = RunPlan(
        plan_id="native-core-health",
        profile="gpt-oss-120b",
        level="core",
        seed=1931,
        endpoints=[target, baseline],
        suites=["S8"],
        reference={"tier": "BASELINE"},
        max_usd=0,
        retries=0,
        requests_per_minute=600000,
        concurrency_per_endpoint=1,
    )

    def transports(*, healthy=False):
        return {
            target.id: httpx.ASGITransport(
                app=create_app(
                    seed=target_seed,
                    faults=FaultConfig(seed=target_seed)
                    if healthy
                    else FaultConfig(**{**original, "fault_id": None, "seed": target_seed}),
                )
            ),
            baseline.id: httpx.ASGITransport(
                app=create_app(seed=baseline_seed, faults=FaultConfig(seed=baseline_seed))
            ),
        }

    result = await run_check(
        plan,
        tests="operations",
        baseline_id=baseline.id,
        root=tmp_path,
        approved=True,
        transports=transports(),
    )
    source = reconstruct(tmp_path / "data/raw", result.run_id)
    assert source.checks == result.checks
    health = {c.check_id: c for c in source.checks if c.check_id in HEALTH_CHECKS}
    assert health["S8.health.requests"].n == 212
    assert all(health[c].details["operational_alarm"] for c in HEALTH_CHECKS[:2])
    assert health["S8.health.nonempty"].verdict == "INCONCLUSIVE"
    assert all(c.check_class == "report_only" and c.verdict != "FAIL" for c in health.values())
    diagnosis = next(d for d in diagnoses(source.checks) if d.code == "INFRA_UNRELIABLE")
    assert diagnosis.confidence == "low"
    private = write_bundle(source, diagnosis, root=tmp_path)
    write_bundle(source, diagnosis, root=tmp_path, redact_content=True)
    no_regeneration(monkeypatch)
    replay = await replay_bundle(
        str(private), root=tmp_path, approved=True, transports=transports()
    )
    assert replay["reproduced"] and replay["budget_scope"] == result.run_id
    repeated = reconstruct(tmp_path / "data/raw", replay["run_id"])
    assert (
        next(d for d in diagnoses(repeated.checks) if d.code == diagnosis.code).confidence
        == "medium"
    )
    changed_result = await replay_bundle(
        str(private), root=tmp_path, approved=True, transports=transports(healthy=True)
    )
    assert not changed_result["reproduced"]
    changed = reconstruct(tmp_path / "data/raw", changed_result["run_id"])
    assert not any(d.code == diagnosis.code for d in diagnoses(changed.checks))
    assert all(
        c.details.get("operational_alarm") is not True
        for c in changed.checks
        if c.check_id in HEALTH_CHECKS
    )
    raw_only(tmp_path, [source, repeated, changed])
    (tmp_path / "native-health-acceptance.json").write_text(
        json.dumps(
            {
                "scope": (
                    "native 0.05-error core health witness/replay/healthy falsifier; "
                    "not new power or provider qualification"
                ),
                "original_fault": original,
                "actual_role_seeds": [target_seed, baseline_seed],
                "seed_scope": (
                    "reused predeclared native prototype seeds; not a fresh population replicate"
                ),
                "source_run": result.run_id,
                "replay_run": replay["run_id"],
                "changed_run": changed_result["run_id"],
                "n_root_procedures": 212,
                "reproduced": True,
                "healthy_control_rejected": True,
                "raw_only_equal": True,
                "captured_records": sum(len(a.records) for a in [source, repeated, changed]),
                "captured_attempts": sum(
                    len(r.attempts) for a in [source, repeated, changed] for r in a.records
                ),
                "source_cards": [c.model_dump(mode="json") for c in health.values()],
                "model_api_calls": 0,
                "new_spend_usd": 0,
            },
            indent=2,
        )
        + "\n"
    )


async def test_native_original_usage_full_replay_retains_non_witness_controls(
    tmp_path, monkeypatch
):
    budget(tmp_path)
    original = yaml.safe_load(Path("configs/faults/usage.yaml").read_text())
    target_seed = named_seed(202610200300, "native-usage-target")
    baseline_seed = named_seed(202610200300, "native-usage-baseline")
    target = endpoint("mock:native-target", target_seed, clean=False)
    baseline = endpoint("mock:native-baseline", baseline_seed, clean=True)
    plan = RunPlan(
        plan_id="native-accounting-replay",
        profile="gpt-oss-120b",
        level="core",
        seed=named_seed(202610200300, "native-usage-items"),
        endpoints=[target, baseline],
        suites=["S1", "S7"],
        reference={"tier": "BASELINE"},
        max_usd=0,
        retries=0,
        requests_per_minute=600000,
        concurrency_per_endpoint=1,
    )

    def transports(*, matched_usage=False):
        return {
            target.id: httpx.ASGITransport(
                app=create_app(
                    seed=target_seed,
                    faults=FaultConfig(seed=target_seed)
                    if matched_usage
                    else FaultConfig(**{**original, "fault_id": None, "seed": target_seed}),
                )
            ),
            baseline.id: httpx.ASGITransport(
                app=create_app(seed=baseline_seed, faults=FaultConfig(seed=baseline_seed))
            ),
        }

    result = await run_check(
        plan,
        tests="B01,P05",
        baseline_id=baseline.id,
        root=tmp_path,
        approved=True,
        transports=transports(),
    )
    source = reconstruct(tmp_path / "data/raw", result.run_id)
    assert source.checks == result.checks
    diagnosis = next(d for d in diagnoses(source.checks) if d.code == "ACCOUNTING_MISMATCH")
    assert diagnosis.check_ids == ["B01", "B01.P05"]
    assert "P05" not in diagnosis.check_ids and any(i.suite_id == "S1" for i in source.items)
    private = write_bundle(source, diagnosis, root=tmp_path)
    write_bundle(source, diagnosis, root=tmp_path, redact_content=True)
    no_regeneration(monkeypatch)
    replay = await replay_bundle(
        str(private), root=tmp_path, approved=True, transports=transports()
    )
    assert replay["reproduced"] and replay["budget_scope"] == result.run_id
    repeated = reconstruct(tmp_path / "data/raw", replay["run_id"])
    assert [i.model_dump() for i in repeated.items] == [i.model_dump() for i in source.items]
    assert (
        next(d for d in diagnoses(repeated.checks) if d.code == diagnosis.code).confidence
        == "medium"
    )
    changed_result = await replay_bundle(
        str(private), root=tmp_path, approved=True, transports=transports(matched_usage=True)
    )
    assert not changed_result["reproduced"]
    changed = reconstruct(tmp_path / "data/raw", changed_result["run_id"])
    assert next(c for c in changed.checks if c.check_id == "B01").verdict == "PASS"
    assert not any(d.code == diagnosis.code for d in diagnoses(changed.checks))
    raw_only(tmp_path, [source, repeated, changed])
    (tmp_path / "native-accounting-replay-acceptance.json").write_text(
        json.dumps(
            {
                "scope": (
                    "native 15% usage fault, complete primary/preflight replay "
                    "and matched-usage control"
                ),
                "original_fault": original,
                "actual_role_seeds": [target_seed, baseline_seed],
                "source_run": result.run_id,
                "replay_run": replay["run_id"],
                "changed_run": changed_result["run_id"],
                "reproduced": True,
                "matched_usage_rejected": True,
                "raw_only_equal": True,
                "non_witness_control_card": "P05",
                "accounting_facets": diagnosis.check_ids,
                "captured_records": sum(len(a.records) for a in [source, repeated, changed]),
                "captured_attempts": sum(
                    len(r.attempts) for a in [source, repeated, changed] for r in a.records
                ),
                "source_witness": [
                    c.model_dump(mode="json")
                    for c in source.checks
                    if c.check_id in diagnosis.check_ids
                ],
                "model_api_calls": 0,
                "new_spend_usd": 0,
            },
            indent=2,
        )
        + "\n"
    )


@pytest.mark.qualified_runtime
async def test_native_original_swap_with_accuracy_complete_replay_and_neutral_accuracy_control(
    tmp_path, monkeypatch
):
    budget(tmp_path)
    frozen = json.loads(Path("docs/report/fingerprint-wire-v1/wire.json").read_text())["value"]
    original = json.loads(
        Path("docs/report/fingerprint-wire-v1/calibration/pack-01.json").read_text()
    )["value"]
    inputs = [ReferenceRun.model_validate(v) for v in original["sources"]]
    pack = build_pack(
        inputs[0],
        "MOCK",
        seed=named_seed(frozen["configuration"]["wire"]["seed"], "pack", 1),
        controls=inputs[1:],
        fingerprint_qualification="captured-scripted-text-v1",
    )
    stimulus = yaml.safe_load(Path("configs/faults/swap.yaml").read_text())
    target_seed = named_seed(202610200200, "native-swap-target")
    baseline_seed = named_seed(202610200200, "native-swap-baseline")
    target = endpoint("mock:native-target", target_seed, clean=False)
    baseline = endpoint("mock:native-baseline", baseline_seed, clean=True)
    for ep in [target, baseline]:
        ep.metadata["fingerprint_sampling"] = {
            "independent_sessions": True,
            "matched_sampling_design": True,
            "independent_reference_captures": True,
            "independent_prompt_strata": True,
            "session_size": 1,
        }
    plan = RunPlan(
        plan_id="native-qualified-swap",
        profile="gpt-oss-120b",
        level="core",
        seed=named_seed(202610200200, "native-swap-items"),
        endpoints=[target, baseline],
        suites=["S5", "S6"],
        reference={"tier": "BASELINE"},
        max_usd=0,
        retries=0,
        requests_per_minute=600000,
        concurrency_per_endpoint=1,
        overrides={"S6": frozen["configuration"]["design"]["design"]},
    )

    def transports(*, neutral_accuracy=False):
        output = {}
        for ep, seed, is_target in [(target, target_seed, True), (baseline, baseline_seed, False)]:
            faults = (
                FaultConfig(**{**stimulus, "fault_id": None, "seed": seed})
                if is_target
                else FaultConfig(seed=seed)
            )
            native = httpx.ASGITransport(
                app=create_app(
                    seed=seed, faults=faults, sampling_version="v2", healthy_logit_jitter=0.3
                )
            )

            async def handler(request, native=native, is_target=is_target):
                response = await native.handle_async_request(request)
                if neutral_accuracy and is_target and request.method == "POST":
                    body = json.loads(request.content)
                    task = task_from_messages(body["messages"])
                    if task.get("type") == "math":
                        # Disclosed control: retain the native swap FP process,
                        # but remove its accuracy loss on the returned wire.
                        assert not body.get("stream")
                        value = json.loads(await response.aread())
                        message = value["choices"][0]["message"]
                        correct = str(math_answer(task))
                        change = text_tokens(correct) - text_tokens(message["content"])
                        message["content"] = correct
                        value["usage"]["completion_tokens"] += change
                        value["usage"]["total_tokens"] += change
                        return httpx.Response(200, json=value)
                return response

            output[ep.id] = httpx.MockTransport(handler)
        return output

    result = await run_check(
        plan,
        tests="FP1,FP2,FP3,R01",
        baseline_id=baseline.id,
        reference_pack=pack,
        root=tmp_path,
        approved=True,
        transports=transports(),
    )
    source = reconstruct(tmp_path / "data/raw", result.run_id)
    assert source.checks == result.checks
    problems = [i for i in source.items if i.expected.get("check_id") == "R01"]
    assert len(problems) == 200 and {i.expected["samples"] for i in problems} == {2}
    assert next(c for c in source.checks if c.check_id == "R01").n == 200
    found = [d for d in diagnoses(source.checks) if d.code == "DIFFERENT_MODEL_LIKE"]
    assert len(found) == 1, [
        (c.check_id, c.verdict, c.reason, c.details.get("fingerprint_position"))
        for c in source.checks
        if c.suite_id != "S0"
    ]
    diagnosis = found[0]
    assert {"FP1", "FP2", "R01"} <= set(diagnosis.check_ids) and diagnosis.confidence == "low"
    assert not any(d.code == "QUANTIZATION_LIKE_DRIFT" for d in diagnoses(source.checks))
    private = write_bundle(source, diagnosis, root=tmp_path)
    write_bundle(source, diagnosis, root=tmp_path, redact_content=True)
    no_regeneration(monkeypatch)
    replay = await replay_bundle(
        str(private), root=tmp_path, approved=True, transports=transports()
    )
    assert replay["reproduced"] and replay["budget_scope"] == result.run_id
    repeated = reconstruct(tmp_path / "data/raw", replay["run_id"])
    assert (
        next(d for d in diagnoses(repeated.checks) if d.code == diagnosis.code).confidence == "high"
    )
    assert [i.model_dump() for i in repeated.items] == [i.model_dump() for i in source.items]
    changed_result = await replay_bundle(
        str(private), root=tmp_path, approved=True, transports=transports(neutral_accuracy=True)
    )
    assert not changed_result["reproduced"]
    changed = reconstruct(tmp_path / "data/raw", changed_result["run_id"])
    assert not any(d.code == diagnosis.code for d in diagnoses(changed.checks))
    cards = {c.check_id: c for c in changed.checks}
    assert cards["R01"].verdict == "PASS"
    assert cards["FP1"].verdict == cards["FP2"].verdict == "FAIL"
    assert not any(c.details.get("bundle_replay_reproduced") for c in changed.checks)
    raw_only(tmp_path, [source, repeated, changed])
    (tmp_path / "native-swap-acceptance.json").write_text(
        json.dumps(
            {
                "scope": (
                    "native 1.0-swap qualified text/accuracy witness and disclosed "
                    "neutral-accuracy control; not hardware/provider qualification or new power"
                ),
                "original_fault": stimulus,
                "actual_role_seeds": [target_seed, baseline_seed],
                "reference_pack_id": pack.pack_id,
                "qualification": "captured-scripted-text-v1",
                "accuracy_design": {"items": 200, "draws_per_item": 2},
                "reference_scope": (
                    "retained independent original pack/control captures; no new reference calls"
                ),
                "source_run": result.run_id,
                "replay_run": replay["run_id"],
                "changed_run": changed_result["run_id"],
                "reproduced": True,
                "neutral_accuracy_rejected": True,
                "raw_only_equal": True,
                "captured_records": sum(len(a.records) for a in [source, repeated, changed]),
                "captured_attempts": sum(
                    len(r.attempts) for a in [source, repeated, changed] for r in a.records
                ),
                "source_witness": [
                    c.model_dump(mode="json")
                    for c in source.checks
                    if c.check_id in diagnosis.check_ids
                ],
                "model_api_calls": 0,
                "new_spend_usd": 0,
            },
            indent=2,
        )
        + "\n"
    )
