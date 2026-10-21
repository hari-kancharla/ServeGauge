import socket
from pathlib import Path

import httpx
import pytest
from pytest_socket import SocketBlockedError
from typer.testing import CliRunner

from servegauge.cli import app
from servegauge.core.secrets import scan_paths

runner = CliRunner()


def test_commands_registered() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in (
        "setup-cache",
        "doctor",
        "discover",
        "estimate",
        "check",
        "run",
        "score",
        "diagnose",
        "report",
        "replay",
        "refs",
        "monitor",
        "validate",
        "mock",
        "selfhost",
        "site",
        "publish",
        "rebuild",
    ):
        assert name in result.output
    assert runner.invoke(app, ["check"]).exit_code == 3


def test_network_is_blocked() -> None:
    with pytest.raises(SocketBlockedError):
        socket.socket()


def test_empty_selector_does_not_call_windows_incompatible_native_select(monkeypatch) -> None:
    import selectors

    from conftest import EmptySafeSelector

    def fail(*args, **kwargs):
        raise AssertionError("empty native select would fail on Windows")

    monkeypatch.setattr(selectors.SelectSelector, "select", fail)
    with EmptySafeSelector() as selector:
        assert selector.select(0) == []


def test_doctor_does_not_disclose_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "budget").mkdir()
    (tmp_path / "budget/dev_spend.json").write_text('{"spent_usd":0.37}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    secret = "sk-" + "testcredential" * 4
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *args, **kwargs: httpx.Response(
            200, json={"data": {"limit": 5, "limit_remaining": 5}}
        ),
    )
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert secret not in result.output
    assert "Development spend: $0.37" in result.output


def test_secret_scan_suppresses_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "sk-" + "testcredential" * 4
    monkeypatch.setenv("TEST_API_KEY", secret)
    path = tmp_path / "cassette.json"
    path.write_text(secret)
    findings = scan_paths([path])
    assert len(findings) == 1
    assert secret not in str(findings)


def test_unexpected_cli_error_cannot_dump_credentials(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from servegauge import cli

    secret = "sk-" + "privatevalue" * 4

    def fail() -> None:
        raise RuntimeError(secret)

    monkeypatch.setattr(cli, "app", fail)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 5
    output = capsys.readouterr()
    assert secret not in output.out + output.err


def test_secret_scan_stays_offline_with_key_and_doctor_refuses_uncapped_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-" + "syntheticvalue" * 4
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)

    def forbidden(*args, **kwargs):
        raise AssertionError("offline scan must not call metadata endpoints")

    monkeypatch.setattr(httpx, "get", forbidden)
    assert runner.invoke(app, ["doctor", "--scan-secrets"]).exit_code == 0
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *args, **kwargs: httpx.Response(
            200, json={"data": {"limit": None, "limit_remaining": None}}
        ),
    )
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 4 and secret not in result.output
