"""Generator equivalence is narrower than a package inventory or version label."""

import json

import pytest

from servegauge.observatory.identity import generator_manifest, runtime_envelope


def package(root):
    base = root / "servegauge"
    base.mkdir()
    (base / "__init__.py").write_text("")
    (base / "driver.py").write_text("from . import helper\nfrom servegauge.nested import leaf\n")
    (base / "helper.py").write_text("VALUE = 1\n")
    (base / "nested").mkdir()
    (base / "nested/__init__.py").write_text("")
    (base / "nested/leaf.py").write_text("VALUE = 2\n")
    return base


def test_closure_includes_relative_submodules_and_parent_initializers(tmp_path):
    base = package(tmp_path)
    result = generator_manifest(base, roots=("servegauge.driver",))
    assert result["complete"] is True
    assert set(result["pins"]) == {
        "src/servegauge/__init__.py",
        "src/servegauge/driver.py",
        "src/servegauge/helper.py",
        "src/servegauge/nested/__init__.py",
        "src/servegauge/nested/leaf.py",
    }
    assert result["unresolved"] == []


def test_unused_python_leaf_does_not_change_generator_but_imported_code_does(tmp_path):
    base = package(tmp_path)
    first = generator_manifest(base, roots=("servegauge.driver",))
    (base / "unused.py").write_text("VALUE = 3\n")
    assert generator_manifest(base, roots=("servegauge.driver",)) == first
    (base / "helper.py").write_text("VALUE = 4\n")
    assert (
        generator_manifest(base, roots=("servegauge.driver",))["generator_hash"]
        != first["generator_hash"]
    )


def test_unresolved_dynamic_local_import_is_not_declared_compatible(tmp_path):
    base = package(tmp_path)
    (base / "driver.py").write_text(
        "import importlib\nmodule = importlib.import_module(selected)\n"
    )
    result = generator_manifest(base, roots=("servegauge.driver",))
    assert not result["complete"] and result["unresolved"]
    (base / "driver.py").write_text(
        'import importlib\nmodule = importlib.import_module("servegauge.helper")\n'
    )
    result = generator_manifest(base, roots=("servegauge.driver",))
    assert result["complete"] and "src/servegauge/helper.py" in result["pins"]


def test_missing_root_and_symlink_refuse(tmp_path):
    base = package(tmp_path)
    with pytest.raises(ValueError):
        generator_manifest(base, roots=("servegauge.absent",))
    (base / "linked.py").symlink_to(base / "helper.py")
    with pytest.raises(ValueError):
        generator_manifest(base, roots=("servegauge.driver",))


def test_runtime_is_recorded_from_actual_interpreter_dependencies_and_lock(tmp_path):
    lock = tmp_path / "lock"
    lock.write_text("fixed-test-lock")
    result = runtime_envelope(lock_path=lock)
    assert result["complete"] is True
    assert result["python"]["implementation"] == "CPython"
    assert result["python"]["version"].startswith("3.12.")
    assert result["platform"]["system"] and result["platform"]["machine"]
    assert result["dependencies"]["numpy"] and result["dependencies"]["httpx"]
    assert (
        result["lock_sha256"]
        and result["stdlib_random_sha256"]
        and result["rng_implementation_sha256"]
    )
    assert len(result["runtime_hash"]) == 64
    assert json.loads(json.dumps(result)) == result
    assert runtime_envelope(lock_path=lock) == result
    lock.write_text("changed-test-lock")
    assert runtime_envelope(lock_path=lock)["runtime_hash"] != result["runtime_hash"]


def test_missing_or_linked_lock_does_not_invent_runtime_qualification(tmp_path):
    result = runtime_envelope(lock_path=tmp_path / "absent")
    assert not result["complete"] and result["lock_sha256"] is None
    (tmp_path / "lock").write_text("test-lock")
    (tmp_path / "linked").symlink_to(tmp_path / "lock")
    assert not runtime_envelope(lock_path=tmp_path / "linked")["complete"]


def test_producer_separates_actual_code_runtime_and_generator_without_guessing(monkeypatch):
    from servegauge.observatory import identity

    first = identity.producer_identity()
    assert first["source_hash"] == first["loaded_source_hash"]
    assert first["runtime"]["complete"] and first["generator"]["complete"]
    monkeypatch.setattr(identity, "_IMPORTED_SOURCE_HASH", "a" * 64)
    changed = identity.producer_identity()
    assert changed["loaded_source_hash"] != changed["source_hash"]
    assert changed["runtime"] == first["runtime"]
