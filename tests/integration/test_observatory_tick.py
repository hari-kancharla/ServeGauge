"""Offline windows preserve identities and refuse unsafe recovery before dispatch."""

import asyncio
import hashlib
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from filelock import FileLock
from typer.testing import CliRunner

from servegauge.cli import app
from servegauge.core.ids import safe_filename
from servegauge.observatory.schedule import ObservatoryConfig, cadence_window, window_id
from servegauge.observatory.tick import execution_lock, prepare_jobs, tick
from servegauge.storage.monitor import MonitorJournal
from servegauge.storage.raw_store import RawStore
from servegauge.validation.async_offline import SocketlessLoop

NOW = datetime(2026, 10, 20, 9, tzinfo=UTC)


@pytest.fixture
def monitor(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/budgets.yaml").write_bytes(Path("configs/budgets.yaml").read_bytes())
    cfg = ObservatoryConfig.model_validate(
        {
            "id": "local-test",
            "enabled": True,
            "requests_per_minute": 600000,
            "models": [
                {
                    "id": "sample-model",
                    "profile": "gpt-oss-120b",
                    "endpoints": [
                        {
                            "id": "mock:monitor",
                            "kind": "mock",
                            "base_url": "http://127.0.0.1:8011/v1",
                            "model": "mock",
                            "claimed_profile": "gpt-oss-120b",
                        }
                    ],
                }
            ],
        }
    )
    cfg.cadences["daily"].tests = ["S0", "P01"]
    cfg.cadences["daily"].level = "smoke"
    cfg.cadences["daily"].overrides = {"S1": {"invariant_trials": 4}}
    return cfg


def records(root):
    return list(RawStore(root / "data/raw").exchanges())


async def test_native_window_is_immutable_and_second_tick_does_nothing(tmp_path, monitor):
    first = await tick(monitor, "daily", root=tmp_path, now=NOW)
    assert first["status"] == "completed" and first["spent_usd"] == "0"
    assert first["resumed"] is False
    assert first["measurement_exit_code"] == 2  # smoke evidence remains underpowered
    assert len(first["jobs"]) == 1
    saved = records(tmp_path)
    assert len(saved) >= 4
    spec = MonitorJournal(tmp_path / "data/raw", first["window_id"]).spec()
    binding = spec["binding"]
    assert binding["source_hash"] == binding["loaded_source_hash"]
    assert binding["runtime"]["complete"] is True
    assert binding["generator"]["complete"] is True
    from servegauge.core.hashing import content_hash

    for field, key in [("runtime", "runtime_hash"), ("generator", "generator_hash")]:
        assert binding[field][key] == content_hash(
            {k: v for k, v in binding[field].items() if k != key}
        )
    before = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (tmp_path / "data").rglob("*")
        if p.is_file()
    }
    again = await tick(monitor, "daily", root=tmp_path, now=NOW + timedelta(hours=2))
    assert again["status"] == "already_completed" and again["jobs"] == first["jobs"]
    assert records(tmp_path) == saved
    assert before == {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (tmp_path / "data").rglob("*")
        if p.is_file()
    }
    scope = first["window_id"]
    manifests = list((tmp_path / "data/runs").glob("*/manifest.json"))
    assert manifests and all(json.loads(p.read_text())["budget_scope"] == scope for p in manifests)


async def test_changed_config_or_missing_sealed_evidence_refuses_without_recollection(
    tmp_path, monitor
):
    result = await tick(monitor, "daily", root=tmp_path, now=NOW)
    saved = records(tmp_path)
    changed = monitor.model_copy(deep=True)
    changed.seed += 1
    with pytest.raises(ValueError, match="binding"):
        await tick(changed, "daily", root=tmp_path, now=NOW, resume=True)
    analysis = tmp_path / "data/raw" / safe_filename(result["jobs"][0]["run_id"]) / "analysis.json"
    analysis.rename(analysis.with_suffix(".retained"))
    with pytest.raises((ValueError, FileNotFoundError)):
        await tick(monitor, "daily", root=tmp_path, now=NOW, resume=True)
    assert records(tmp_path) == saved


async def test_disabled_dry_run_owner_and_busy_gates_have_no_model_calls(
    tmp_path, monitor, monkeypatch
):
    disabled = monitor.model_copy(update={"enabled": False})
    assert (await tick(disabled, "daily", root=tmp_path, now=NOW))["status"] == "disabled"
    assert not (tmp_path / "data").exists()
    preview = await tick(monitor, "daily", root=tmp_path, now=NOW, dry_run=True)
    assert preview["status"] == "dry_run" and preview["jobs"][0]["estimate"]["cost_table"]
    assert not (tmp_path / "data").exists()
    paid = monitor.model_copy(update={"mode": "owner"}, deep=True)
    paid.models[0].reference.tier = "R3"
    paid.models[0].endpoints[0].kind = "openai_compat"
    monkeypatch.delenv("SERVEGAUGE_OWNER_RUN", raising=False)
    monkeypatch.delenv("TRUEWEIGHTS_OWNER_RUN", raising=False)
    with pytest.raises(ValueError, match="owner"):
        await tick(paid, "daily", root=tmp_path, now=NOW, approved=True)
    path = execution_lock(tmp_path, monitor.id)
    path.parent.mkdir(parents=True)
    with FileLock(str(path), timeout=0):
        assert (await tick(monitor, "daily", root=tmp_path, now=NOW))["status"] == "in_progress"
    assert not records(tmp_path)


async def test_interrupted_job_needs_resume_and_reuses_preflight_without_duplicate_calls(
    tmp_path, monitor, monkeypatch
):
    import servegauge.runner.execute as runner

    ordinary = runner.run_plan

    async def interrupted(plan, **kwargs):
        if not kwargs["run_id"].endswith("-capabilities"):
            raise RuntimeError("controlled after preflight")
        return await ordinary(plan, **kwargs)

    monkeypatch.setattr(runner, "run_plan", interrupted)
    with pytest.raises(RuntimeError):
        await tick(monitor, "daily", root=tmp_path, now=NOW)
    before = records(tmp_path)
    assert before and all(r.run_id.endswith("-capabilities") for r in before)
    with pytest.raises(ValueError, match="resume"):
        await tick(monitor, "daily", root=tmp_path, now=NOW)
    monkeypatch.setattr(runner, "run_plan", ordinary)
    result = await tick(monitor, "daily", root=tmp_path, now=NOW, resume=True)
    assert result["status"] == "completed"
    assert result["resumed"] is True
    after = records(tmp_path)
    assert len(after) == len(before) + 4
    assert {r.exchange_id for r in before} <= {r.exchange_id for r in after}
    identifier = window_id(monitor.id, cadence_window("daily", NOW))
    journal = MonitorJournal(tmp_path / "data/raw", identifier)
    assert len([e for e in journal.events() if e["type"] == "job_started"]) == 1
    assert len([e for e in journal.events() if e["type"] == "window_resumed"]) == 1


async def test_next_window_has_distinct_job_and_mock_seed(tmp_path, monitor):
    a = await prepare_jobs(monitor, cadence_window("daily", NOW))
    b = await prepare_jobs(monitor, cadence_window("daily", NOW + timedelta(days=1)))
    assert a[0]["run_id"] != b[0]["run_id"]
    assert a[0]["plan"]["seed"] != b[0]["plan"]["seed"]
    assert (
        a[0]["plan"]["endpoints"][0]["metadata"]["mock_declaration"]["seed"]
        != b[0]["plan"]["endpoints"][0]["metadata"]["mock_declaration"]["seed"]
    )


async def test_lock_symlink_and_source_binding_change_refuse_before_new_captures(
    tmp_path, monitor, monkeypatch
):
    lock = execution_lock(tmp_path, monitor.id)
    lock.parent.mkdir(parents=True)
    target = tmp_path / "owner.txt"
    target.write_text("preserve")
    lock.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        await tick(monitor, "daily", root=tmp_path, now=NOW)
    assert target.read_text() == "preserve" and not records(tmp_path)
    lock.unlink()  # disposable test link only
    await tick(monitor, "daily", root=tmp_path, now=NOW)
    before = records(tmp_path)
    monkeypatch.setattr(
        "servegauge.observatory.identity.implementation_hash", lambda: "different-source"
    )
    with pytest.raises(ValueError, match="binding"):
        await tick(monitor, "daily", root=tmp_path, now=NOW)
    assert records(tmp_path) == before


async def test_started_window_with_lost_runner_context_cannot_restart(
    tmp_path, monitor, monkeypatch
):
    import servegauge.observatory.tick as observer

    async def stopped(*args, **kwargs):
        raise RuntimeError("before manifest")

    monkeypatch.setattr(observer, "execute_plan", stopped)
    with pytest.raises(RuntimeError):
        await tick(monitor, "daily", root=tmp_path, now=NOW)
    with pytest.raises(ValueError, match="context_missing"):
        await tick(monitor, "daily", root=tmp_path, now=NOW, resume=True)
    assert not records(tmp_path)


async def test_changed_measurement_file_refuses_same_window(tmp_path, monitor, monkeypatch):
    await tick(monitor, "daily", root=tmp_path, now=NOW)
    before = records(tmp_path)
    for source in Path("configs").glob("*"):
        if source.is_file() and source.suffix in {".yaml", ".json"}:
            shutil.copyfile(source, tmp_path / "configs" / source.name)
    (tmp_path / "profiles").mkdir()
    shutil.copyfile("profiles/gpt-oss-120b.yaml", tmp_path / "profiles/gpt-oss-120b.yaml")
    shutil.copyfile("uv.lock", tmp_path / "uv.lock")
    monkeypatch.chdir(tmp_path)
    assert (await tick(monitor, "daily", root=tmp_path, now=NOW))["status"] == "already_completed"
    source = tmp_path / "configs/cache_probes_v1.json"
    original = source.read_text()
    source.write_text(original + "\n")  # disposable copy, exact bytes are part of binding
    with pytest.raises(ValueError, match="binding"):
        await tick(monitor, "daily", root=tmp_path, now=NOW)
    assert records(tmp_path) == before


async def test_multiple_models_resume_skips_completed_job_and_keeps_parent_scope(
    tmp_path, monitor, monkeypatch
):
    import servegauge.runner.execute as runner

    second = monitor.models[0].model_copy(deep=True)
    second.id = "second-model"
    second.endpoints[0].id = "mock:second"
    monitor.models.append(second)
    jobs = await prepare_jobs(monitor, cadence_window("daily", NOW))
    original = runner.run_plan

    async def interrupted(plan, **kwargs):
        if kwargs["run_id"] == jobs[1]["run_id"]:
            raise RuntimeError("second primary interrupted")
        return await original(plan, **kwargs)

    monkeypatch.setattr(runner, "run_plan", interrupted)
    with pytest.raises(RuntimeError):
        await tick(monitor, "daily", root=tmp_path, now=NOW)
    first_records = [r for r in records(tmp_path) if r.run_id.startswith(jobs[0]["run_id"])]
    assert first_records
    monkeypatch.setattr(runner, "run_plan", original)
    result = await tick(monitor, "daily", root=tmp_path, now=NOW, resume=True)
    assert len(result["jobs"]) == 2 and result["spent_usd"] == "0"
    assert [r for r in records(tmp_path) if r.run_id.startswith(jobs[0]["run_id"])] == first_records
    assert all(
        json.loads(p.read_text())["budget_scope"] == result["window_id"]
        for p in (tmp_path / "data/runs").glob("*/manifest.json")
    )


async def test_failing_job_is_not_hidden_by_an_inconclusive_second_job(tmp_path, monitor):
    from servegauge.mock.faults import FaultConfig

    second = monitor.models[0].model_copy(deep=True)
    second.id = "second-model"
    second.endpoints[0].id = "mock:second"
    monitor.models.append(second)
    monitor.models[0].faults = FaultConfig(ignore_max_tokens=True)
    result = await tick(monitor, "daily", root=tmp_path, now=NOW)
    assert result["status"] == "completed"
    assert [job["measurement_exit_code"] for job in result["jobs"]] == [1, 2]
    assert result["measurement_exit_code"] == 1


def test_cli_disabled_configuration_and_no_error_details(tmp_path, monkeypatch):
    def run_offline(coroutine):
        with asyncio.Runner(loop_factory=SocketlessLoop) as runner:
            return runner.run(coroutine)

    monkeypatch.setattr("servegauge.cli.asyncio.run", run_offline)
    config = tmp_path / "observer.yaml"
    config.write_text("id: disabled\n")
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app, ["monitor", "tick", "--cadence", "daily", "--config", str(config)]
    )
    assert result.exit_code == 0 and json.loads(result.stdout)["status"] == "disabled"
    config.write_text("private malformed value\n")
    result = CliRunner().invoke(
        app, ["monitor", "tick", "--cadence", "daily", "--config", str(config)]
    )
    assert result.exit_code == 3 and "private malformed value" not in result.output
