"""Capture code identity belongs to the imported package, never an unrelated CWD."""

import json
import subprocess
from pathlib import Path

import pytest

from servegauge.core.errors import SafeError
from servegauge.core.hashing import content_hash
from servegauge.core.manifest import build_manifest
from servegauge.observatory.provenance import implementation_hash
from servegauge.storage.source_identity import _snapshot, capture_source


def package(root):
    base = root / "src/servegauge"
    for name in [
        "__init__.py",
        "core/manifest.py",
        "storage/source_identity.py",
        "observatory/tick.py",
    ]:
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('"""DO_NOT_EMIT_SOURCE_TEXT_CANARY."""\n')
    return base


def test_manifest_uses_package_inventory_outside_git_without_leaking_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    value = build_manifest("provenance", {}, {}, [])
    assert value["source_tree_hash"] == implementation_hash()
    assert value["source_tree_hash"] != content_hash({})
    identity = value["source_identity"]
    assert identity["scope"] == "loaded-package-python-v1"
    assert identity["source_hash"] == identity["import_source_hash"] == value["source_tree_hash"]
    assert identity["python_files"] > 1
    assert str(tmp_path) not in json.dumps(value)


def test_unrelated_checkout_and_git_redirection_cannot_supply_capture_source(tmp_path, monkeypatch):
    expected = capture_source()
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / "src").mkdir()
    unrelated = tmp_path / "src/private.py"
    unrelated.write_text("private-source-canary")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path))
    first = capture_source()
    unrelated.write_text("different-private-source-canary")
    assert capture_source() == first
    assert first["source_tree_hash"] == implementation_hash()
    # A source archive or new checkout can have no commit; redirection must not change that.
    assert first == expected
    assert "private-source-canary" not in json.dumps(first)


def test_git_is_optional_and_exception_messages_never_enter_a_manifest(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("private-process-error-canary")

    monkeypatch.setattr(subprocess, "run", missing)
    value = build_manifest("git-unavailable", {}, {}, [])
    assert value["source_tree_hash"] == implementation_hash()
    assert value["git_state"] == "unavailable"
    assert value["git_sha"] == "uncommitted"
    assert "private-process-error-canary" not in json.dumps(value)


def test_snapshot_retains_source_hash_but_never_source_text_or_absolute_paths(tmp_path):
    base = package(tmp_path)
    value = _snapshot(base, implementation_hash(base))
    assert value["source_tree_hash"] == implementation_hash(base)
    assert value["git_state"] == "not_checkout"
    assert value["source_identity"]["python_files"] == 4
    encoded = json.dumps(value)
    assert "DO_NOT_EMIT_SOURCE_TEXT_CANARY" not in encoded and str(tmp_path) not in encoded


def test_changed_after_import_package_refuses_instead_of_reporting_a_fresh_loaded_hash(tmp_path):
    base = package(tmp_path)
    before = implementation_hash(base)
    (base / "new_module.py").write_text("value = 1\n")
    with pytest.raises(SafeError, match="capture_source_changed_since_import"):
        _snapshot(base, before)


def test_missing_or_symlinked_sources_refuse_without_reading_the_target(tmp_path, monkeypatch):
    base = package(tmp_path)
    outside = tmp_path / "private-target.py"
    outside.write_text("DO_NOT_READ_PRIVATE_CANARY")
    (base / "escape.py").symlink_to(outside)
    original = Path.read_text

    def guarded(path, *args, **kwargs):
        if path == outside:
            raise AssertionError("private target was read")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    with pytest.raises(SafeError, match="capture_source_inventory_unavailable"):
        _snapshot(base, "a" * 64)
    (base / "escape.py").unlink()
    (base / "core/manifest.py").unlink()
    with pytest.raises(SafeError, match="capture_source_inventory_unavailable"):
        _snapshot(base, "a" * 64)


def test_git_queries_are_bound_read_only_and_ignore_inherited_git_environment(
    tmp_path, monkeypatch
):
    base = package(tmp_path)
    (tmp_path / ".git").mkdir()
    monkeypatch.setenv("GIT_INDEX_FILE", "private-index-canary")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    calls = []

    def git(args, **kwargs):
        assert Path(kwargs["cwd"]) == tmp_path
        assert kwargs["env"]["GIT_OPTIONAL_LOCKS"] == "0"
        assert "GIT_INDEX_FILE" not in kwargs["env"] and "GIT_CONFIG_COUNT" not in kwargs["env"]
        assert "core.fsmonitor=false" in args
        assert kwargs["timeout"] > 0
        calls.append(args)
        output = (
            "git version 2.50.0"
            if "--version" in args
            else str(tmp_path)
            if "--show-toplevel" in args
            else "a" * 40
            if "rev-parse" in args
            else " M src/servegauge/core/manifest.py\n"
        )
        return subprocess.CompletedProcess(args, 0, stdout=output, stderr="private-stderr-canary")

    monkeypatch.setattr(subprocess, "run", git)
    value = _snapshot(base, implementation_hash(base))
    assert value["git_sha"] == "a" * 40 and value["git_dirty"] is True
    assert len(calls) == 4
    assert "private-stderr-canary" not in json.dumps(value)


def test_ancestor_repository_is_not_an_identity_for_a_nested_package_copy(tmp_path, monkeypatch):
    base = package(tmp_path)
    (tmp_path / ".git").mkdir()

    def git(args, **kwargs):
        return subprocess.CompletedProcess(
            args,
            0,
            stdout="git version 2.50.0" if "--version" in args else str(tmp_path.parent),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", git)
    value = _snapshot(base, implementation_hash(base))
    assert value["git_state"] == "not_checkout" and value["git_sha"] == "uncommitted"


def test_old_git_cannot_launch_an_fsmonitor_hook_or_claim_checked_context(tmp_path, monkeypatch):
    base = package(tmp_path)
    (tmp_path / ".git").mkdir()

    def git(args, **kwargs):
        assert "--version" in args
        return subprocess.CompletedProcess(args, 0, stdout="git version 2.35.1", stderr="")

    monkeypatch.setattr(subprocess, "run", git)
    value = _snapshot(base, implementation_hash(base))
    assert value["git_state"] == "unavailable" and value["source_tree_hash"] == implementation_hash(
        base
    )


def test_source_change_during_git_observation_refuses_a_mixed_snapshot(tmp_path, monkeypatch):
    base = package(tmp_path)
    original = implementation_hash(base)
    (tmp_path / ".git").mkdir()

    def git(args, **kwargs):
        if "--version" in args:
            output = "git version 2.50.0"
        elif "--show-toplevel" in args:
            output = str(tmp_path)
        elif "rev-parse" in args:
            output = "a" * 40
        else:
            (base / "core/manifest.py").write_text("changed = True\n")
            output = ""
        return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")

    monkeypatch.setattr(subprocess, "run", git)
    with pytest.raises(SafeError, match="capture_source_changed_during_snapshot"):
        _snapshot(base, original)
