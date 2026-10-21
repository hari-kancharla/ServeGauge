import json
from pathlib import Path

import httpx
import pytest
from tests_support import fingerprint_cards
from typer.testing import CliRunner

from servegauge.cli import app
from servegauge.core.hashing import content_hash
from servegauge.core.ids import safe_filename
from servegauge.mock.server import create_app
from servegauge.refs.pack import load_pack
from servegauge.runner.check import _mock_metadata
from servegauge.runner.execute import execute_plan
from servegauge.runner.plan import Requirements, RunPlan
from servegauge.storage.analysis import export_outcomes, reconstruct
from servegauge.storage.scientific import score_run
from servegauge.validation.capture import endpoint


async def capture_run(
    tmp_path,
    name,
    seeds,
    backend="scripted",
    *,
    known_design=True,
    backend_by_seed=None,
    s6=None,
    jitter=0.0,
):
    backends = {seed: (backend_by_seed or {}).get(seed, backend) for seed in seeds}
    endpoints = [
        endpoint(f"mock:{name}-{seed}", seed, clean=backends[seed] == "scripted") for seed in seeds
    ]
    for ep in endpoints if known_design else []:
        ep.metadata["fingerprint_sampling"] = {
            "independent_sessions": True,
            "matched_sampling_design": True,
            "independent_reference_captures": True,
            "independent_prompt_strata": True,
            "session_size": 1,
        }
    plan = RunPlan(
        plan_id=name,
        profile="gpt-oss-120b",
        level="core",
        seed=88,
        reference={"tier": "MOCK"},
        endpoints=endpoints,
        suites=["S6"],
        max_usd=0,
        requests_per_minute=60000,
        overrides={
            "S6": s6
            or {
                "methods": ["FP1", "FP2", "FP3"],
                "fp1_cells": 2,
                "fp1_samples": 16,
                "fp2_cells": 2,
                "fp2_samples": 16,
                "fp3_samples": 32,
            }
        },
    )
    transports = {
        ep.id: httpx.ASGITransport(
            app=create_app(
                seed=seed,
                backend=backends[seed],
                sampling_version="v2",
                healthy_logit_jitter=jitter,
            )
        )
        for ep, seed in zip(endpoints, seeds, strict=True)
    }
    await _mock_metadata(plan, transports)
    result = await execute_plan(plan, root=tmp_path, approved=True, transports=transports)
    assert not result.errors and result.spent_usd == 0
    return result.run_id


async def test_sealed_raw_cli_pack_attachment_and_reconstruction(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "profiles").mkdir()
    for name in ["suites.yaml", "margins.yaml", "fingerprints.yaml", "budgets.yaml"]:
        (tmp_path / "configs" / name).write_bytes((Path("configs") / name).read_bytes())
    (tmp_path / "profiles/gpt-oss-120b.yaml").write_bytes(
        Path("profiles/gpt-oss-120b.yaml").read_bytes()
    )
    reference = await capture_run(tmp_path, "reference", [1, 2, 3])
    controls = [
        await capture_run(tmp_path, "control-a", [11], "other:control-a"),
        await capture_run(tmp_path, "control-b", [12], "other:control-b"),
    ]
    target = await capture_run(tmp_path, "fresh-target", [104])
    unknown = await capture_run(tmp_path, "unknown-design", [205], known_design=False)
    raw = tmp_path / "data/raw"
    original = reconstruct(raw, reference)
    exported = export_outcomes(original)
    assert not exported["checks"] and len(exported["fingerprints"]) == 3
    assert exported["fingerprint_policy"] == original.metadata["policy"]
    assert all(
        ep["metadata"]["mock_declaration"]["sampling_version"] == "v2"
        for ep in exported["manifest"]["endpoint_metadata"]
    )
    # A corrupt, contradictory derived file must never become a reference input.
    for identifier in [reference, *controls]:
        (tmp_path / "data/runs" / safe_filename(identifier) / "outcomes.json").write_text(
            "corrupt derived cache"
        )
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "refs",
            "build",
            "--profile",
            "gpt-oss-120b",
            "--tier",
            "MOCK",
            "--from-run",
            reference,
            "--control-run",
            controls[0],
            "--control-run",
            controls[1],
        ],
    )
    assert result.exit_code == 0, result.output
    pack = load_pack(json.loads(result.output)["pack_id"])
    assert len(pack.fingerprints["controls"]) == 2
    refused = CliRunner().invoke(
        app,
        ["refs", "build", "--profile", "gpt-oss-120b", "--tier", "MOCK", "--from-run", reference],
    )
    assert refused.exit_code == 3
    with pytest.raises(ValueError, match="no_matching"):
        score_run(tmp_path, unknown, reference=pack.pack_id)
    assert not (raw / safe_filename(unknown) / "reference.json").exists()
    analysis, found = score_run(tmp_path, target, reference=pack.pack_id)
    fingerprints = fingerprint_cards(analysis.checks)
    assert not found and {c.check_id for c in fingerprints} == {"FP1", "FP2", "FP3"}
    assert all(c.n == 32 and c.reference_tier == "MOCK" for c in fingerprints)
    assert all(
        c.estimate is not None and c.delta_ci_low is not None and c.adjusted_p is not None
        for c in fingerprints
    )
    assert {c.verdict for c in fingerprints} <= {"INCONCLUSIVE", "INFO"}
    assert all("observations" not in c.details for c in analysis.checks)
    before = content_hash([r.model_dump(mode="json") for r in analysis.records])
    directory = tmp_path / "data/runs" / safe_filename(target)
    (directory / "checks.json").write_text("corrupt derived cache")
    for name in ["margins.yaml", "fingerprints.yaml"]:
        (tmp_path / "configs" / name).write_text("invalid current config")
    replay = reconstruct(raw, target)
    assert replay.checks == analysis.checks
    assert content_hash([r.model_dump(mode="json") for r in replay.records]) == before
    envelope_path = raw / safe_filename(reference) / "analysis.json"
    damaged = json.loads(envelope_path.read_text())
    damaged["value"]["run_id"] = "tampered"
    envelope_path.write_text(json.dumps(damaged))
    refused = CliRunner().invoke(
        app,
        ["refs", "build", "--profile", "gpt-oss-120b", "--tier", "MOCK", "--from-run", reference],
    )
    assert refused.exit_code == 3
    envelope_path.write_text("{}")
    refused = CliRunner().invoke(
        app,
        ["refs", "build", "--profile", "gpt-oss-120b", "--tier", "MOCK", "--from-run", reference],
    )
    assert refused.exit_code == 3


async def test_fingerprint_requirements_tighten_and_unknown_checks_refuse(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/budgets.yaml").write_bytes(Path("configs/budgets.yaml").read_bytes())
    identifier = await capture_run(tmp_path, "requirements-target", [104])
    analysis = reconstruct(tmp_path / "data/raw", identifier)
    from servegauge.runner.evaluate import evaluate
    from servegauge.verdicts.config import DecisionPolicy

    policy = DecisionPolicy.model_validate(analysis.metadata["policy"])
    requirements = Requirements(
        app="app", requires=[{"id": "identity", "tests": ["FP1"], "margin": 0.02}]
    )
    checks = evaluate(
        analysis.plan,
        identifier,
        analysis.items,
        analysis.samples,
        policy,
        target_ids=[analysis.plan.endpoints[0].id],
        requirements=requirements,
    )
    fp1 = next(c for c in checks if c.check_id == "FP1")
    assert (
        fp1.requirement_ids == ["identity"] and fp1.margin == 0.02 and fp1.verdict == "INCONCLUSIVE"
    )
    requirements.requires[0].tests = ["FP3"]
    with pytest.raises(ValueError, match="log_unit"):
        evaluate(
            analysis.plan,
            identifier,
            analysis.items,
            analysis.samples,
            policy,
            target_ids=[analysis.plan.endpoints[0].id],
            requirements=requirements,
        )
    requirements.requires[0].tests = ["unmeasured"]
    with pytest.raises(ValueError, match="not_measured"):
        evaluate(
            analysis.plan,
            identifier,
            analysis.items,
            analysis.samples,
            policy,
            target_ids=[analysis.plan.endpoints[0].id],
            requirements=requirements,
        )


async def test_original_mock_build_command_and_explicit_trusted_captures(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "profiles").mkdir()
    for name in ["suites.yaml", "margins.yaml", "fingerprints.yaml", "budgets.yaml"]:
        (tmp_path / "configs" / name).write_bytes((Path("configs") / name).read_bytes())
    (tmp_path / "profiles/gpt-oss-120b.yaml").write_bytes(
        Path("profiles/gpt-oss-120b.yaml").read_bytes()
    )
    combined = await capture_run(
        tmp_path,
        "combined",
        [1, 2, 3, 11, 12],
        backend_by_seed={11: "other:control-a", 12: "other:control-b"},
    )
    trusted = await capture_run(tmp_path, "trusted", [1])
    repeats = [
        await capture_run(tmp_path, "repeat-a", [2]),
        await capture_run(tmp_path, "repeat-b", [3]),
    ]
    controls = [
        await capture_run(tmp_path, "control-a", [11], "other:control-a"),
        await capture_run(tmp_path, "control-b", [12], "other:control-b"),
    ]
    target = await capture_run(tmp_path, "fresh", [104])
    raw = tmp_path / "data/raw"
    hashes = {
        identifier: content_hash(
            [r.model_dump(mode="json") for r in reconstruct(raw, identifier).records]
        )
        for identifier in [combined, trusted, *repeats, *controls, target]
    }
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    original_command = [
        "refs",
        "build",
        "--profile",
        "gpt-oss-120b",
        "--tier",
        "MOCK",
        "--from-run",
        combined,
    ]
    result = runner.invoke(app, original_command)
    assert result.exit_code == 0, result.output
    mock_pack = load_pack(json.loads(result.output)["pack_id"])
    assert len(mock_pack.fingerprints["controls"]) == 2
    assert len(mock_pack.fingerprints["views"]["*"]["references"]) == 3
    args = [
        "refs",
        "build",
        "--profile",
        "gpt-oss-120b",
        "--tier",
        "BASELINE",
        "--from-run",
        trusted,
        "--baseline-endpoint",
        "mock:trusted-1",
        "--control-run",
        controls[0],
        "--control-run",
        controls[1],
    ]
    assert runner.invoke(app, args).exit_code == 3
    result = runner.invoke(
        app, [*args, "--reference-run", repeats[0], "--reference-run", repeats[1]]
    )
    assert result.exit_code == 0, result.output
    pack = load_pack(json.loads(result.output)["pack_id"])
    assert len(pack.fingerprints["reference_sources"]) == 3
    analysis, _ = score_run(tmp_path, target, reference=pack.pack_id)
    assert all(
        c.n == 32 and c.reference_tier == "BASELINE" for c in fingerprint_cards(analysis.checks)
    )
    assert analysis.checks == reconstruct(raw, target).checks
    assert runner.invoke(app, [*original_command, "--reference-run", repeats[0]]).exit_code == 3
    assert (
        runner.invoke(app, [*original_command, "--baseline-endpoint", "mock:combined-1"]).exit_code
        == 3
    )
    assert (
        runner.invoke(
            app, [*args, "--reference-run", repeats[0], "--reference-run", repeats[0]]
        ).exit_code
        == 3
    )
    for identifier, expected in hashes.items():
        assert (
            content_hash([r.model_dump(mode="json") for r in reconstruct(raw, identifier).records])
            == expected
        )
