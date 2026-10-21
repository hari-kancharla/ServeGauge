import hashlib
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from servegauge.cli import app
from servegauge.core.errors import SafeError
from servegauge.tokencount import glm_assets


@pytest.fixture
def assets(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    payloads = {
        "tokenizer.json": b'{"original":"test data, no weights"}',
        "tokenizer_config.json": b"{}",
        "chat_template.jinja": b"original local test template",
        "LICENSE": b"original local test notice",
    }
    for name, raw in payloads.items():
        (source / name).write_bytes(raw)
    monkeypatch.setattr(
        glm_assets, "FILES", {k: hashlib.sha256(v).hexdigest() for k, v in payloads.items()}
    )
    monkeypatch.setenv("SERVEGAUGE_TOKENIZER_CACHE", str(tmp_path / "cache"))
    return source, payloads


def test_explicit_local_setup_retains_license_and_revalidates_after_loading(assets):
    source, payloads = assets
    result = glm_assets.setup_glm_cache(source=source)
    assert result.name == "hf-glm-5.3"
    assert glm_assets.verified_assets() == payloads
    (result / "tokenizer.json").write_bytes(b"corrupt")
    with pytest.raises(SafeError, match="glm_tokenizer_cache_missing_or_invalid"):
        glm_assets.verified_assets()
    glm_assets.setup_glm_cache(source=source)
    assert glm_assets.verified_assets() == payloads
    (result / "LICENSE").unlink()
    with pytest.raises(SafeError):
        glm_assets.verified_assets()


def test_missing_or_changed_source_is_rejected_before_writing_any_output(assets):
    source, _ = assets
    (source / "LICENSE").write_bytes(b"changed")
    with pytest.raises(SafeError):
        glm_assets.setup_glm_cache(source=source)
    assert not (source.parent / "cache").exists()


def test_public_setup_is_explicit_credential_free_bounded_and_idempotent(assets, respx_mock):
    _, payloads = assets
    routes = []
    for name, raw in payloads.items():
        route = respx_mock.get(glm_assets.asset_url(name)).mock(
            return_value=httpx.Response(200, content=raw)
        )
        routes.append(route)
    glm_assets.setup_glm_cache()
    assert glm_assets.verified_assets() == payloads
    assert all(route.call_count == 1 for route in routes)
    glm_assets.setup_glm_cache()
    assert all(route.call_count == 1 for route in routes)
    for call in respx_mock.calls:
        assert "authorization" not in call.request.headers
        assert "cookie" not in call.request.headers
        assert glm_assets.REVISION in str(call.request.url)


@pytest.mark.parametrize("failure", ["hash", "status", "redirect", "large", "credentials"])
def test_failed_downloads_never_materialize_partial_assets(assets, respx_mock, failure):
    source, payloads = assets
    first = next(iter(payloads))
    response = {
        "hash": httpx.Response(200, content=b"wrong"),
        "status": httpx.Response(403, text="private upstream message"),
        "redirect": httpx.Response(302, headers={"location": "http://127.0.0.1/private"}),
        "large": httpx.Response(200, headers={"content-length": "999999999"}),
        "credentials": httpx.Response(
            302, headers={"location": "https://user:password@huggingface.co/file"}
        ),
    }[failure]
    respx_mock.get(glm_assets.asset_url(first)).mock(return_value=response)
    with pytest.raises(SafeError) as error:
        glm_assets.setup_glm_cache()
    assert "private" not in str(error.value) and "password" not in str(error.value)
    assert not (source.parent / "cache").exists()


def test_protected_paths_are_refused_before_any_filesystem_operation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("filesystem must not be consulted")

    monkeypatch.setenv("SERVEGAUGE_TOKENIZER_CACHE", "/unused/Starter Files/cache")
    monkeypatch.setattr(Path, "is_symlink", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    with pytest.raises(SafeError):
        glm_assets.verified_assets()
    with pytest.raises(SafeError):
        glm_assets.setup_glm_cache()
    with pytest.raises(SafeError):
        glm_assets.setup_glm_cache(source=Path("/unused/Starter Files/source"))


def test_symlink_assets_are_refused_without_following_them(assets):
    source, _ = assets
    result = glm_assets.setup_glm_cache(source=source)
    target = result / "tokenizer.json"
    target.unlink()
    target.symlink_to(source / "tokenizer.json")
    with pytest.raises(SafeError):
        glm_assets.verified_assets()


def test_installed_cli_exposes_explicit_profile_setup_and_safe_failures(monkeypatch):
    calls = []
    monkeypatch.setattr(glm_assets, "setup_glm_cache", lambda: calls.append("glm") or Path("cache"))
    result = CliRunner().invoke(app, ["setup-cache", "--profile", "glm-5.3"])
    assert result.exit_code == 0 and calls == ["glm"]
    unknown = CliRunner().invoke(app, ["setup-cache", "--profile", "unknown"])
    assert unknown.exit_code == 3 and calls == ["glm"]

    def failed():
        raise SafeError("fixed_setup_reason")

    monkeypatch.setattr(glm_assets, "setup_glm_cache", failed)
    assert CliRunner().invoke(app, ["setup-cache", "--profile", "glm-5.3"]).exit_code == 5
