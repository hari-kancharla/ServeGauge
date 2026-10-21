"""Selected retained mixture counterexample, controlled context, reserved full replay."""

import importlib
import json
import shutil
from fractions import Fraction
from pathlib import Path

import httpx
import pytest

from servegauge.core.hashing import content_hash
from servegauge.diagnose.bundle import write_bundle
from servegauge.diagnose.drift import DRIFT_CONTRACT, REQUIRED_CONTROLS
from servegauge.diagnose.repro import replay_bundle
from servegauge.diagnose.rules import diagnoses
from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.mock.server import create_app, stream_frames
from servegauge.refs.pack import ReferenceRun, build_pack
from servegauge.runner.check import run_check
from servegauge.runner.plan import RunPlan
from servegauge.storage.analysis import reconstruct
from servegauge.suites.registry import prepare_item
from servegauge.suites.s5_longout import reasoning_items
from servegauge.suites.s6_fingerprint import fingerprint_items
from servegauge.tokencount.counters import text_tokens
from servegauge.validation.capture import endpoint
from servegauge.validation.common import named_seed

pytestmark = pytest.mark.slow


def request_key(body):
    normal = {k: v for k, v in body.items() if k not in {"model", "stream_options"}}
    # Sealed inventories sort mapping keys. Embedded mock-task JSON is still
    # equivalent after reconstruction; fixture dispatch must ignore that order.
    normal["messages"] = [dict(m) for m in normal["messages"]]
    for message in normal["messages"]:
        text = message.get("content")
        if isinstance(text, str) and "\nTW_TASK: " in text:
            prefix, task = text.rsplit("\nTW_TASK: ", 1)
            message["content"] = (
                prefix
                + "\nTW_TASK: "
                + json.dumps(json.loads(task), sort_keys=True, ensure_ascii=False)
            )
    return content_hash(normal)


@pytest.mark.qualified_runtime
async def test_selected_captured_mixture_guarded_witness_full_replay_and_changed_control(
    tmp_path, monkeypatch
):
    """A mixed serving process can meet this signature; it is not quantization truth."""
    root = Path.cwd()
    (tmp_path / "configs").mkdir()
    shutil.copyfile("configs/budgets.yaml", tmp_path / "configs/budgets.yaml")
    frozen = json.loads((root / "docs/report/fingerprint-wire-v1/wire.json").read_text())["value"]
    original = json.loads(
        (root / "docs/report/fingerprint-wire-v1/calibration/pack-01.json").read_text()
    )["value"]
    inputs = [ReferenceRun.model_validate(v) for v in original["sources"]]
    pack = build_pack(
        inputs[0],
        "MOCK",
        seed=named_seed(frozen["configuration"]["wire"]["seed"], "pack", 1),
        controls=inputs[1:],
        fingerprint_qualification="captured-scripted-text-v1",
    )
    selected_seed = 50313406623473649
    seed = named_seed(selected_seed, "inference")
    target = endpoint(
        "mock:wire-" + content_hash([selected_seed, "endpoint"])[:20], selected_seed, clean=False
    )
    baseline_seed = named_seed(selected_seed, "guarded-context-baseline")
    baseline = endpoint("mock:clinical-baseline", baseline_seed, clean=True)
    target.labels["quantization"] = "fp4"
    for ep in (target, baseline):
        ep.metadata["fingerprint_sampling"] = {
            "independent_sessions": True,
            "matched_sampling_design": True,
            "independent_reference_captures": True,
            "independent_prompt_strata": True,
            "session_size": 1,
        }
    fp_cfg = frozen["configuration"]["design"]["design"]
    math_cfg = {"items": 1000, "samples": 1}
    plan = RunPlan(
        plan_id="selected-guarded-mixture",
        profile="gpt-oss-120b",
        level="core",
        seed=seed,
        reference={"tier": "BASELINE"},
        endpoints=[target, baseline],
        suites=["S1", "S2", "S5", "S6"],
        max_usd=0,
        retries=0,
        requests_per_minute=600000,
        concurrency_per_endpoint=1,
        overrides={
            "S6": fp_cfg,
            "S1": {"diversity_prompts": 200, "effort_items": 200},
            "S5": math_cfg,
        },
    )
    fp_items = fingerprint_items("core", seed, fp_cfg)
    math_items = [
        i for i in reasoning_items("core", seed, math_cfg) if i.expected["check_id"] == "R01"
    ]
    # The sampling protocol is the original native ScriptedBackend process. Other
    # suites use independent scripted streams and an explicit controlled math loss.
    # This defines a composite counterexample, not the stock single F-MIX fault.
    requests = [prepare_item(i, target).body for i in fp_items]
    assert all(b is not None for b in requests)
    fp_keys = {request_key(b) for b in requests}
    math = {
        request_key(prepare_item(i, target).body): (i.expected["answer"], index < 10)
        for index, i in enumerate(math_items)
    }
    counts = {"posts": 0}

    def transports(*, remove_loss=False):
        output = {}
        for ep, actual_seed, is_target in [
            (target, selected_seed, True),
            (baseline, baseline_seed, False),
        ]:
            fp_backend = ScriptedBackend(
                actual_seed,
                FaultConfig(route_to_other_model=0.5 if is_target else 0),
                sampling_version="v2",
                healthy_logit_jitter=0.3,
            )
            protocol = ScriptedBackend(named_seed(actual_seed, "protocol"), FaultConfig())
            metadata = httpx.ASGITransport(
                app=create_app(
                    seed=actual_seed,
                    faults=FaultConfig(route_to_other_model=0.5 if is_target else 0),
                    sampling_version="v2",
                    healthy_logit_jitter=0.3,
                )
            )

            async def handler(
                request,
                fp_backend=fp_backend,
                protocol=protocol,
                metadata=metadata,
                is_target=is_target,
            ):
                if request.method == "GET":
                    return await metadata.handle_async_request(request)
                assert request.url.path.endswith("/chat/completions")
                body = json.loads(request.content)
                key = request_key(body)
                engine = fp_backend if key in fp_keys else protocol
                # Match capture_run and the mock HTTP server's pre-response
                # error draw, including when error_rate is zero.
                assert engine.rng.random() >= engine.faults.error_rate
                response = engine.respond(body)
                counts["posts"] += 1
                if key in math:
                    answer, wrong = math[key]
                    actual = (
                        str(Fraction(str(answer)) + 1)
                        if is_target and wrong and not remove_loss
                        else str(answer)
                    )
                    response["choices"][0]["message"]["content"] = actual
                    reasoning = response["choices"][0]["message"].get("reasoning") or ""
                    response["usage"]["completion_tokens"] = text_tokens(actual) + text_tokens(
                        reasoning
                    )
                if not body.get("stream"):
                    return httpx.Response(200, json=response)
                return httpx.Response(
                    200,
                    headers={"Content-Type": "text/event-stream"},
                    content=b"".join(stream_frames(response, FaultConfig())),
                )

            output[ep.id] = httpx.MockTransport(handler)
        return output

    tests = ",".join([*REQUIRED_CONTROLS, "R01", "FP1", "FP2", "FP3"])
    checked = await run_check(
        plan,
        root=tmp_path,
        approved=True,
        tests=tests,
        baseline_id=baseline.id,
        reference_pack=pack,
        transports=transports(),
    )
    source = reconstruct(tmp_path / "data/raw", checked.run_id)
    assert source.checks == checked.checks
    assert source.metadata["drift_contract"] == DRIFT_CONTRACT
    found = [d for d in diagnoses(source.checks) if d.code == "QUANTIZATION_LIKE_DRIFT"]
    assert len(found) == 1, [
        (c.check_id, c.verdict, c.delta, c.delta_ci_low, c.delta_ci_high, c.reason)
        for c in source.checks
        if c.check_id in {*REQUIRED_CONTROLS, "FP1", "FP2", "R01"}
    ]
    assert found[0].confidence == "low" and found[0].failure_class == "MODEL_BEHAVIOR"
    fp2 = next(c for c in source.checks if c.check_id == "FP2")
    assert fp2.verdict == "INCONCLUSIVE" and fp2.delta < fp2.margin
    assert any("mixed routing" in a for a in found[0].alternatives)
    bundle = write_bundle(source, found[0], root=tmp_path)
    redacted = write_bundle(source, found[0], root=tmp_path, redact_content=True)
    blob = (redacted / "bundle.json").read_text()
    assert all(
        m["content"] not in blob
        for i in source.items[:30]
        for m in i.payload.get("messages", [])
        if isinstance(m.get("content"), str) and len(m["content"]) > 16
    )

    def regeneration_forbidden(*args, **kwargs):
        raise AssertionError("guarded drift replay needs the exact original full witness inventory")

    module = importlib.import_module("servegauge.runner.check")
    for name in [
        "capability_items",
        "fingerprint_items",
        "protocol_items",
        "tool_items",
        "reasoning_items",
    ]:
        monkeypatch.setattr(module, name, regeneration_forbidden)
    replay = await replay_bundle(str(bundle), root=tmp_path, approved=True, transports=transports())
    assert replay["reproduced"] and replay["budget_scope"] == checked.run_id
    repeated = reconstruct(tmp_path / "data/raw", replay["run_id"])
    assert (
        next(d for d in diagnoses(repeated.checks) if d.code == found[0].code).confidence
        == "medium"
    )
    assert [i.model_dump() for i in source.items] == [i.model_dump() for i in repeated.items]
    changed = await replay_bundle(
        str(bundle), root=tmp_path, approved=True, transports=transports(remove_loss=True)
    )
    assert not changed["reproduced"]
    neutral = reconstruct(tmp_path / "data/raw", changed["run_id"])
    assert next(c for c in neutral.checks if c.check_id == "R01").verdict == "PASS"
    assert not any(d.code == found[0].code for d in diagnoses(neutral.checks))
    raw = tmp_path / "raw-only"
    shutil.copytree(tmp_path / "data/raw", raw)
    shutil.rmtree(tmp_path / "data/runs")
    for actual in [source, repeated, neutral]:
        rebuilt = reconstruct(raw, actual.manifest["run_id"])
        assert rebuilt.checks == actual.checks and rebuilt.samples == actual.samples
    (tmp_path / "drift-acceptance.json").write_text(
        json.dumps(
            {
                "scope": (
                    "selected retained mixture geometry plus controlled 10/1000 accuracy loss; "
                    "not power, quantization truth or provider validation"
                ),
                "new_spend_usd": 0,
                "model_api_calls": 0,
                "selected_pack_index": 1,
                "selected_target_index": 7,
                "selected_seed": selected_seed,
                "seed_selection": (
                    "post-hoc diagnostic case selected from retained geometry, "
                    "not held-out evaluation"
                ),
                "captured_records": sum(len(a.records) for a in [source, repeated, neutral]),
                "actual_http_completion_attempts": counts["posts"],
                "fp2_practical_verdict_unchanged": fp2.verdict,
                "full_witness_reproduced": replay["reproduced"],
                "changed_accuracy_pass_rejected": not changed["reproduced"],
                "raw_only_equal": True,
                "source_confidence": found[0].confidence,
                "replayed_confidence": "medium",
                "cards": [
                    c.model_dump(mode="json")
                    for c in source.checks
                    if c.check_id in {"FP1", "FP2", "R01"}
                ],
                "diagnosis": found[0].model_dump(mode="json"),
            },
            indent=2,
        )
        + "\n"
    )
