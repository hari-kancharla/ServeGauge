"""Synthetic ASGI capture traverses runner/raw/CLI; it is not creator API evidence."""

import json
import os
from pathlib import Path

import httpx
import pytest
from integration.test_fingerprint_pack_workflow import capture_run
from typer.testing import CliRunner
from unit.test_owner_references import inputs

from servegauge.cli import app
from servegauge.core.hashing import content_hash
from servegauge.mock.server import create_app
from servegauge.refs.owner import bind_endpoint
from servegauge.refs.pack import ReferenceRun, build_pack, load_pack
from servegauge.runner.execute import execute_plan
from servegauge.runner.ledger import Ledger
from servegauge.runner.plan import RunPlan
from servegauge.storage.analysis import export_outcomes, reconstruct
from servegauge.storage.owner_reference import captured_reference
from servegauge.storage.workspace import initialize


@pytest.mark.slow
async def test_synthetic_direct_origin_fingerprints_need_three_captures_and_distinct_controls(
    tmp_path, monkeypatch
):
    initialize(tmp_path)
    monkeypatch.setenv("SERVEGAUGE_TOKENIZER_CACHE", str(Path.cwd() / ".cache/tokenizers"))
    request, profile, endpoint, _ = inputs(maker="moonshotai")
    profile.sources = [request.source_url]
    profile.field_sources = {
        k: request.source_url
        for k, v in profile.model_dump().items()
        if v not in (None, [], {}) and k not in {"sources", "field_sources", "unverified"}
    }
    request.profile_hash = content_hash(profile.model_dump(mode="json"))
    endpoint.metadata = {
        "context_length": 131072,
        "max_completion_tokens": 65536,
        "pricing": {"prompt": "0", "completion": "0"},
    }
    request.endpoint_hash = content_hash(endpoint.model_dump(mode="json"))
    endpoint = bind_endpoint(request, profile, endpoint)
    (tmp_path / "profiles" / (profile.id + ".yaml")).write_text(profile.model_dump_json())
    monkeypatch.chdir(tmp_path)
    s6 = {
        "methods": ["FP1", "FP2", "FP3"],
        "fp1_cells": 2,
        "fp1_samples": 16,
        "fp2_cells": 2,
        "fp2_samples": 16,
        "fp3_samples": 32,
    }
    sources = []
    for seed in [411, 412, 413]:
        plan = RunPlan(
            plan_id="synthetic-origin",
            profile=profile.id,
            level="core",
            seed=88,
            reference={"tier": "BASELINE"},
            endpoints=[endpoint],
            suites=["S6"],
            max_usd=0,
            requests_per_minute=60000,
            overrides={"S6": s6},
        )
        result = await execute_plan(
            plan,
            root=tmp_path,
            approved=True,
            transports={
                endpoint.id: httpx.ASGITransport(app=create_app(seed=seed, sampling_version="v2"))
            },
        )
        assert not result.errors and result.spent_usd == 0
        sources.append(result.run_id)
    controls = [
        await capture_run(tmp_path, "origin-control-a", [431], "other:control-a", s6=s6),
        await capture_run(tmp_path, "origin-control-b", [432], "other:control-b", s6=s6),
    ]
    capture = captured_reference(tmp_path, sources[0], profile, request)
    assert capture.source.fingerprints
    baseline_source = capture.source.model_copy(
        update={"baseline_endpoint_id": endpoint.id}, deep=True
    )
    control_sources = [
        ReferenceRun.model_validate(export_outcomes(reconstruct(tmp_path / "data/raw", run)))
        for run in controls
    ]
    baseline = build_pack(
        baseline_source,
        "BASELINE",
        seed=8,
        controls=control_sources,
        reference_runs=[
            captured_reference(tmp_path, run, profile, request).source for run in sources[1:]
        ],
    )
    review = tmp_path / "configs/references"
    review.mkdir()
    (review / "review.json").write_text(request.model_dump_json())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "servegauge.cli._reference_environment", lambda: {"SERVEGAUGE_OWNER_RUN": "1"}
    )
    args = [
        "refs",
        "build",
        "--profile",
        profile.id,
        "--tier",
        "R1",
        "--from-run",
        sources[0],
        "--seed",
        "8",
        "--owner-qualification",
        "configs/references/review.json",
        "--confirm",
    ]
    for run in controls:
        args.extend(["--control-run", run])
    assert CliRunner().invoke(app, args).exit_code == 3
    for run in sources[1:]:
        args.extend(["--reference-run", run])
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    pack = load_pack(json.loads(result.output)["path"])
    assert pack.tier == "R1" and pack.fingerprints == baseline.fingerprints
    assert len(pack.manifest["owner_qualification"]["captures"]) == 3
    assert all(
        not metric["equivalence_qualified"]
        for view in pack.fingerprints["views"].values()
        for metric in view["metrics"].values()
    )
    ledger = Ledger(tmp_path)
    ledger.refresh()
    assert ledger.development_spend() == 0 and not ledger.pending
    assert (
        os.environ.get("SERVEGAUGE_OWNER_RUN") is None
        and os.environ.get("TRUEWEIGHTS_OWNER_RUN") is None
    )
