"""Installed workspaces receive offline defaults without replacing owner files."""

import asyncio
import json
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from servegauge.cli import app
from servegauge.config import load_budget
from servegauge.profiles import load_profile
from servegauge.runner.budget import BudgetExceeded, BudgetManager
from servegauge.storage.workspace import configure_workspace, initialize, template_payloads
from servegauge.validation.async_offline import SocketlessLoop


def test_init_has_profiles_runtime_defaults_and_zero_live_caps(tmp_path, monkeypatch):
    result = initialize(tmp_path)
    assert result["network_writes"] == 0 and result["live_budget_usd"] == 0
    monkeypatch.chdir(tmp_path)
    assert load_profile("gpt-oss-120b").id == "gpt-oss-120b"
    assert load_profile("glm-5.3").weights == "zai-org/GLM-5.3"
    policy = load_budget(tmp_path)
    assert policy.monthly_usd == policy.per_run_usd == policy.dev_live_usd == 0
    assert policy.selfhost_usd_per_session == 0
    manager = BudgetManager(tmp_path, "fixture", policy)
    with pytest.raises(BudgetExceeded):
        manager.reserve(Decimal("0.01"), "fixture", "fixture", "a" * 64)
    assert manager.reserve(Decimal(0), "fixture", "free-mock", "b" * 64).amount == 0
    before = {name: (tmp_path / name).read_bytes() for name in template_payloads()}
    assert initialize(tmp_path)["status"] == "unchanged"
    assert all((tmp_path / n).read_bytes() == b for n, b in before.items())


def test_conflicting_owner_file_refuses_before_creating_any_other_files(tmp_path):
    (tmp_path / "configs").mkdir()
    existing = tmp_path / "configs/budgets.yaml"
    existing.write_text("owner content must stay")
    with pytest.raises(ValueError, match="conflict"):
        initialize(tmp_path)
    assert existing.read_text() == "owner content must stay"
    assert not (tmp_path / "profiles").exists()
    assert not (tmp_path / ".servegauge-workspace.json").exists()


@pytest.mark.parametrize("name", ["Starter Files", ".git/unsafe", "../unsafe", "linked"])
def test_protected_or_linked_destination_refuses(tmp_path, name):
    if name == "linked":
        (tmp_path / name).symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        initialize(tmp_path / name)


def test_only_known_runtime_files_are_packaged_not_endpoints_or_paid_plans():
    payloads = template_payloads()
    assert "profiles/gpt-oss-120b.yaml" in payloads
    assert "profiles/glm-5.3.yaml" in payloads
    assert "docs/METHODS.md" in payloads
    assert not any(n.startswith("configs/endpoints/") for n in payloads)
    assert "configs/fingerprint_qualification.json" not in payloads
    registry = json.loads(payloads["configs/monitor_transport_admission_v1.json"])
    assert registry["qualifications"] == []


@pytest.mark.qualified_runtime
def test_optional_evidence_root_is_verified_without_changing_frozen_loader(tmp_path):
    import servegauge.refs.fingerprint_qualification as qualification

    original = qualification.ROOT, qualification.REGISTRY
    result = initialize(tmp_path, evidence_root=Path.cwd())
    assert result["qualification_evidence"] == "verified_scripted_scope"
    assert original == (qualification.ROOT, qualification.REGISTRY)
    marker = json.loads((tmp_path / ".servegauge-workspace.json").read_text())
    assert marker["evidence"]["id"] == "captured-scripted-text-v1"


def test_missing_evidence_refuses_before_writing(tmp_path):
    target = tmp_path / "new"
    with pytest.raises((OSError, ValueError)):
        initialize(target, evidence_root=tmp_path / "absent")
    assert not target.exists()


def test_workspace_context_without_evidence_cannot_inherit_repo_qualification(
    tmp_path, monkeypatch
):
    import servegauge.refs.fingerprint_qualification as qualification

    monkeypatch.setattr(qualification, "ROOT", qualification.ROOT)
    monkeypatch.setattr(qualification, "REGISTRY", qualification.REGISTRY)
    initialize(tmp_path)
    assert configure_workspace(tmp_path) == "unqualified"
    assert tmp_path == qualification.ROOT
    with pytest.raises(ValueError):
        qualification.load_qualification("captured-scripted-text-v1")


def test_cli_init_then_core_dry_run_needs_no_repository_or_key(tmp_path, monkeypatch):
    def offline_run(coroutine):
        with asyncio.Runner(loop_factory=SocketlessLoop) as loop:
            return loop.run(coroutine)

    monkeypatch.setattr(asyncio, "run", offline_run)
    monkeypatch.setenv("SERVEGAUGE_TOKENIZER_CACHE", str(Path.cwd() / ".cache/tokenizers"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        app,
        [
            "check",
            "--endpoint",
            "http://127.0.0.1:8011/v1",
            "--model",
            "mock",
            "--profile",
            "gpt-oss-120b",
            "--tests",
            "P01",
            "--level",
            "core",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "data/raw").exists()


def test_parent_file_conflict_refuses_before_any_other_write(tmp_path):
    (tmp_path / "profiles").write_text("owner file")
    with pytest.raises(ValueError, match="conflict"):
        initialize(tmp_path)
    assert not (tmp_path / "configs").exists()
    assert (tmp_path / "profiles").read_text() == "owner file"


@pytest.mark.qualified_runtime
def test_verified_context_rechecks_evidence_and_stale_context_abstains(tmp_path, monkeypatch):
    import servegauge.refs.fingerprint_qualification as qualification

    monkeypatch.setattr(qualification, "ROOT", qualification.ROOT)
    monkeypatch.setattr(qualification, "REGISTRY", qualification.REGISTRY)
    initialize(tmp_path, evidence_root=Path.cwd())
    assert configure_workspace(tmp_path) == "verified_scripted_scope"
    assert qualification.load_qualification("captured-scripted-text-v1")["scope"].startswith(
        "Text equivalence only"
    )
    marker = tmp_path / ".servegauge-workspace.json"
    value = json.loads(marker.read_bytes())
    value["evidence"]["registry_sha256"] = "0" * 64
    marker.write_text(json.dumps(value))
    assert configure_workspace(tmp_path) == "unqualified"
    with pytest.raises(ValueError):
        qualification.load_qualification("captured-scripted-text-v1")


def test_cli_conflict_has_safe_error_without_paths(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/budgets.yaml").write_text("owner content")
    result = CliRunner().invoke(app, ["init", "--path", str(tmp_path)])
    assert result.exit_code == 3
    assert "refused" in result.output and str(tmp_path) not in result.output
