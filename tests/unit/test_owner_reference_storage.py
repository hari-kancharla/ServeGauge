"""Real sealed fixture reconstruction and CLI origin gates; no provider transport."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner
from unit.test_owner_references import inputs, records, session_proof
from unit.test_selfhost_runtime import prepared, receipt
from unit.test_selfhost_storage import SID

from servegauge.cli import app
from servegauge.core.hashing import content_hash
from servegauge.core.ids import safe_filename
from servegauge.refs.owner import Qualification, bind_endpoint
from servegauge.refs.pack import build_pack, load_pack
from servegauge.refs.publication import public_pack
from servegauge.runner.ledger import Ledger
from servegauge.runner.plan import RunPlan
from servegauge.storage.analysis import capture_context, seal_analysis
from servegauge.storage.owner_reference import captured_reference, read_review
from servegauge.storage.raw_store import RawStore
from servegauge.storage.reference_export import source_state
from servegauge.storage.selfhost import prepare_session
from servegauge.storage.selfhost_runtime import save_runtime
from servegauge.suites.config import suite_config
from servegauge.suites.s1_protocol import probe
from servegauge.verdicts.config import load_policy


def sealed_fixture(root, tier="R1", *, observed_session=None):
    request, profile, endpoint, source = inputs(tier)
    if observed_session is not None:
        request.origin.session_id = observed_session["session_id"]
        request.origin.plan_hash = observed_session["plan"]["plan_hash"]
        request.origin.runtime_sha256 = observed_session["facts"]["runtime_receipt_sha256"]
        endpoint.base_url = observed_session["facts"]["endpoint_url"]
        endpoint.metadata.clear()
        request.endpoint_hash = content_hash(endpoint.model_dump(mode="json"))
        source.manifest["created_at"] = datetime.now(UTC).isoformat()
    profile.sources = [request.source_url]
    profile.field_sources = {
        k: request.source_url
        for k, v in profile.model_dump().items()
        if v not in (None, [], {}) and k not in {"sources", "field_sources", "unverified"}
    }
    request.profile_hash = content_hash(profile.model_dump(mode="json"))
    endpoint.metadata.clear()
    endpoint = bind_endpoint(request, profile, endpoint)
    source.manifest["endpoint_metadata"] = [endpoint.model_dump(mode="json")]
    plan = RunPlan(
        plan_id="origin-fixture",
        profile=profile.id,
        endpoints=[endpoint],
        suites=["S1"],
        level="smoke",
        seed=8,
        reference={"tier": "BASELINE"},
        max_usd=0,
    )
    items = [probe("P03", 0, v, "generated local fixture", {}, samples=3) for v in ["low", "high"]]
    manifest = source.manifest | {
        "config": plan.model_dump(mode="json"),
        "config_hash": content_hash(plan.model_dump(mode="json")),
        "item_set_hashes": {"S1": content_hash([i.model_dump(mode="json") for i in items])},
    }
    capture_context(
        root,
        source.run_id,
        {"manifest": manifest, "items": [i.model_dump(mode="json") for i in items]},
    )
    store = RawStore(root / "data/raw")
    for item in items:
        for index in range(3):
            record = records(source, endpoint)[0]
            if observed_session is not None:
                record.timings.start_unix = datetime.now(UTC).timestamp()
                record.timings.total_ms = 0
            record.item_id, record.suite_id, record.sample_idx = item.item_id, item.suite_id, index
            record.exchange_id = item.item_id + str(index)
            record.response["choices"] = [
                {
                    "message": {
                        "content": str(index) if item.expected["variant"] == "high" else "fixed"
                    },
                    "finish_reason": "stop",
                }
            ]
            record.attempts[0].response = record.response
            assert store.append(record)
    metadata = {
        "run_id": source.run_id,
        "profile": profile.model_dump(mode="json"),
        "policy": load_policy().model_dump(mode="json"),
        "suite_config": suite_config(),
        "target_ids": [endpoint.id],
        "tests": "all",
        "baseline_id": None,
    }
    seal_analysis(root, source.run_id, metadata)
    if tier == "R2" and observed_session is None:
        journal = prepare_session(root, SID, prepared())
        digest = save_runtime(root, SID, prepared(), receipt())
        assert digest == request.origin.runtime_sha256
        reservation = session_proof(request, endpoint)["reservation"]
        reservation.pop("dispatched")
        ledger = Ledger(root)
        ledger.ensure_origin()
        ledger.append(reservation | {"type": "reserve", "month": "2026-10"})
        ledger.append({"type": "dispatch", "reservation_id": reservation["reservation_id"]})
        with patch("servegauge.storage.selfhost.datetime") as clock:
            clock.now.return_value = datetime.fromisoformat("2026-10-20T13:00:00+00:00")
            journal.record("reserved", {"reservation_id": reservation["reservation_id"]})
            journal.record("dispatching", {})
            journal.record(
                "running",
                {
                    "remote_id": "f" * 64,
                    "endpoint_url": endpoint.base_url,
                    "runtime_receipt_sha256": digest,
                },
            )
    (root / "profiles").mkdir(exist_ok=True)
    (root / "profiles" / (profile.id + ".yaml")).write_text(profile.model_dump_json())
    reviews = root / "configs/references"
    reviews.mkdir(parents=True)
    (root / "configs/suites.yaml").write_bytes(Path("configs/suites.yaml").read_bytes())
    (reviews / "review.json").write_text(request.model_dump_json())
    (reviews / "endpoint.json").write_text(endpoint.model_dump_json())
    derived = root / "data/runs" / safe_filename(source.run_id)
    derived.mkdir(parents=True)
    (derived / "outcomes.json").write_text("corrupt derived cache fixture")
    return request, profile, endpoint, source.run_id


@pytest.mark.parametrize("tier", ["R1", "R2"])
def test_sealed_cli_build_is_confirmed_owner_only_and_keeps_exact_baseline_numbers(
    tmp_path, monkeypatch, tier
):
    request, profile, endpoint, run = sealed_fixture(tmp_path, tier)
    before = {
        str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    }
    capture = captured_reference(tmp_path, run, profile, request)
    assert len(capture.records) == 6 and "P03" in capture.source.checks
    baseline_source = capture.source.model_copy(
        update={"baseline_endpoint_id": endpoint.id}, deep=True
    )
    baseline = build_pack(baseline_source, "BASELINE", seed=8)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    args = [
        "refs",
        "build",
        "--profile",
        profile.id,
        "--tier",
        tier,
        "--from-run",
        run,
        "--seed",
        "8",
        "--owner-qualification",
        "configs/references/review.json",
    ]
    assert runner.invoke(app, args).exit_code == 3
    assert runner.invoke(app, args + ["--confirm"]).exit_code == 3
    monkeypatch.setattr(
        "servegauge.cli._reference_environment", lambda: {"SERVEGAUGE_OWNER_RUN": "1"}
    )
    assert runner.invoke(app, args).exit_code == 3
    result = runner.invoke(app, args + ["--confirm"])
    assert result.exit_code == 0, result.output
    pack = load_pack(json.loads(result.output)["path"])
    assert (
        pack.tier == tier
        and pack.outcomes == baseline.outcomes
        and pack.calibration == baseline.calibration
    )
    assert pack.manifest["owner_qualification"]["captures"][0]["raw_requests"] == 6
    assert runner.invoke(app, args + ["--confirm"]).exit_code == 0
    public = public_pack(pack)
    assert public.manifest["origin_review"] == pack.manifest["origin_review"]
    assert "owner_qualification" not in public.manifest
    assert public.outcomes == pack.outcomes and public.calibration == pack.calibration
    assert "source_owner_origin_unverified" not in source_state(tmp_path, pack)["blockers"]
    if tier == "R2":
        runtime = next((tmp_path / "data/raw/selfhost-runtime").rglob("*.json"))
        old = runtime.read_bytes()
        runtime.write_bytes(old + b" ")
        assert "source_owner_origin_unverified" in source_state(tmp_path, pack)["blockers"]
        runtime.write_bytes(old)
    assert all((tmp_path / name).read_bytes() == blob for name, blob in before.items())
    assert (
        os.environ.get("SERVEGAUGE_OWNER_RUN") is None
        and os.environ.get("TRUEWEIGHTS_OWNER_RUN") is None
    )
    # A changed raw envelope refuses even with explicit injected authorization.
    path = tmp_path / "data/raw" / safe_filename(run) / "capture.json"
    value = json.loads(path.read_text())
    value["value"]["manifest"]["created_at"] = datetime.now(UTC).isoformat()
    path.write_text(json.dumps(value))
    assert runner.invoke(app, args + ["--confirm"]).exit_code == 3


def test_cli_template_prepare_never_grants_a_tier_or_writes_a_pack(tmp_path, monkeypatch):
    request, profile, endpoint, _ = sealed_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    args = ["--profile", profile.id, "--endpoint", "configs/references/endpoint.json"]
    template = runner.invoke(
        app, ["refs", "template", *args, "--tier", "R1", "--maker", "deepseek"]
    )
    assert template.exit_code == 0, template.output
    with pytest.raises(ValueError):
        Qualification.model_validate(json.loads(template.output))
    prepared = runner.invoke(
        app, ["refs", "prepare", *args, "--qualification", "configs/references/review.json"]
    )
    assert prepared.exit_code == 0, prepared.output
    value = json.loads(prepared.output)
    assert value["endpoint"] == endpoint.model_dump(mode="json")
    assert value["reference_tier_granted"] is False and value["model_api_calls"] == 0
    assert not (tmp_path / "refs").exists()
    assert not (tmp_path / "data/ledger").exists()


def test_review_paths_and_runtime_tampering_refuse_before_any_build(tmp_path):
    request, profile, _, run = sealed_fixture(tmp_path, "R2")
    with pytest.raises(ValueError):
        read_review(tmp_path, tmp_path / "elsewhere.json")
    runtime = next((tmp_path / "data/raw/selfhost-runtime").rglob("*.json"))
    runtime.write_bytes(runtime.read_bytes() + b" ")
    with pytest.raises(ValueError, match="runtime_receipt_bytes_changed"):
        captured_reference(tmp_path, run, profile, request)
    assert not (tmp_path / "refs").exists()


def test_nested_raw_directory_refuses_before_recursive_capture_read(tmp_path):
    request, profile, endpoint, run = sealed_fixture(tmp_path)
    nested = tmp_path / "data/raw" / safe_filename(run) / safe_filename(endpoint.id) / "unexpected"
    nested.mkdir()
    with pytest.raises(ValueError, match="flat_endpoint_raw_layout"):
        captured_reference(tmp_path, run, profile, request)
