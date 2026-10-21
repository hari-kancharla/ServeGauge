"""Public image contexts are explicit and refuse hidden npm/account influence."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest


def module():
    return runpy.run_path(str(Path("scripts/comparison_image.py")))


def test_recipe_pins_public_cpu_bases_and_uses_only_locked_offline_project_install():
    value = module()
    recipe = value["dockerfile"]("servegauge-test.whl")
    assert "node@sha256:" in recipe and "astral-sh/uv@sha256:" in recipe
    assert "COPY . " not in recipe and "COPY .env" not in recipe
    assert "--ignore-scripts" in recipe and "--no-audit" in recipe
    assert "--frozen" in recipe and "--no-install-project" in recipe
    assert "--no-index" in recipe and "--no-deps" in recipe
    assert "USER" not in recipe  # Actual execution fixes an unprivileged UID.


def test_invalid_or_account_bearing_public_dependency_sources_refuse():
    validate = module()["validate_public_dependencies"]
    project = {"project": {"name": "servegauge"}, "tool": {}}
    lock = {
        "package": [
            {"name": "servegauge", "source": {"editable": "."}},
            {
                "name": "package",
                "source": {"registry": "https://pypi.org/simple"},
                "wheels": [{"url": "https://files.pythonhosted.org/package.whl"}],
            },
        ]
    }
    validate(project, lock)
    for url in [
        "https://example.invalid/package.whl",
        "https://user:password@files.pythonhosted.org/package.whl",
        "https://files.pythonhosted.org/package.whl?credential=synthetic",
    ]:
        lock["package"][1]["wheels"][0]["url"] = url
        with pytest.raises(ValueError):
            validate(project, lock)


def test_exact_context_inventory_refuses_project_npmrc_before_any_build(tmp_path, monkeypatch):
    value = module()
    context = tmp_path / "context"
    context.mkdir()
    (context / "manifest.json").write_text("{}")
    (context / ".npmrc").write_text("synthetic-project-influence")

    def forbidden(*args, **kwargs):
        raise AssertionError("no Docker may start")

    monkeypatch.setattr("subprocess.run", forbidden)
    with pytest.raises(ValueError):
        value["build_image"](
            context,
            context_hash="0" * 64,
            docker=Path("/missing"),
            config=tmp_path / "anonymous",
            host="unix:///var/run/docker.sock",
            out=tmp_path / "receipt.json",
        )
    assert (context / ".npmrc").read_text() == "synthetic-project-influence"


def test_dockerignore_excludes_everything_except_the_named_public_payloads():
    ignore = module()["dockerignore"](
        ["Dockerfile", ".dockerignore", "known.whl", "tokenizers/tokenizer.json"]
    )
    assert ignore.splitlines()[0] == "**"
    assert "!known.whl" in ignore and "!tokenizers/tokenizer.json" in ignore
    assert "!*" not in ignore


def test_public_code_scan_allows_credential_handling_code_but_refuses_actual_credential_patterns():
    scan = module()["require_public_text"]
    scan(b"headers = {'Authorization': 'Bearer ' + environment_value}")
    fake = b"sk-" + b"x" * 40
    with pytest.raises(ValueError) as failure:
        scan(fake)
    assert fake.decode() not in str(failure.value)
