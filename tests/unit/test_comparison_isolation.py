"""Account and outbound isolation is mandatory for the optional comparison tool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from servegauge.comparison.isolation import ToolRuntime, prepare_process


def runtime(tmp_path: Path) -> ToolRuntime:
    tool = tmp_path / "tool"
    package = tool / "node_modules/promptfoo"
    (package / "dist/src").mkdir(parents=True)
    (package / "package.json").write_text(json.dumps({"name": "promptfoo", "version": "0.124.1"}))
    (package / "dist/src/entrypoint.js").write_text("// synthetic unit fixture\n")
    (tool / "package-lock.json").write_text(
        json.dumps({"packages": {"node_modules/promptfoo": {"version": "0.124.1"}}})
    )
    node = tmp_path / "node"
    node.touch()
    python = tmp_path / "python"
    python.touch()
    package_root = tmp_path / "python-package"
    package_root.mkdir()
    return ToolRuntime(node=node, tool=tool, python=python, package_root=package_root)


def test_environment_drops_all_inherited_account_and_process_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in [
        "OPENROUTER_API_KEY",
        "SERVEGAUGE_OWNER_RUN",
        "TRUEWEIGHTS_OWNER_RUN",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NODE_OPTIONS",
        "PYTHONSTARTUP",
        "PROMPTFOO_CONFIG_DIR",
        "DOTENV_PATH",
        "SYNTHETIC_ACCOUNT_CANARY",
    ]:
        monkeypatch.setenv(name, "synthetic-inherited-value")
    run = tmp_path / "run"
    run.mkdir()
    process = prepare_process(runtime(tmp_path), run, platform="darwin")
    assert not set(process.environment.values()).intersection({"synthetic-inherited-value"})
    assert not any(
        name in process.environment
        for name in [
            "HOME",
            "OPENROUTER_API_KEY",
            "SERVEGAUGE_OWNER_RUN",
            "TRUEWEIGHTS_OWNER_RUN",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NODE_OPTIONS",
            "SYNTHETIC_ACCOUNT_CANARY",
        ]
    )
    assert process.environment["PROMPTFOO_CONFIG_DIR"] == str(run / "tool-state")
    assert process.environment["PROMPTFOO_PYTHON"] == str(tmp_path / "python")
    assert process.environment["PYTHONNOUSERSITE"] == "1"
    assert process.environment["DOTENV_PATH"] == str(run / "empty.env")
    assert (run / "empty.env").read_bytes() == b""


def test_generated_eval_is_serial_uncached_local_and_never_owner(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    process = prepare_process(runtime(tmp_path), run, platform="darwin")
    argv = process.eval_command()
    assert argv[:2] == ["/usr/bin/sandbox-exec", "-f"]
    assert "--no-cache" in argv and "--no-write" in argv and "--no-share" in argv
    assert argv[argv.index("--max-concurrency") + 1] == "1"
    assert "--grader" not in argv and "--remote" not in argv
    assert all(
        process.environment[name]
        for name in [
            "PROMPTFOO_DISABLE_TELEMETRY",
            "PROMPTFOO_DISABLE_UPDATE",
            "PROMPTFOO_DISABLE_REMOTE_GENERATION",
            "PROMPTFOO_DISABLE_SHARING",
            "PROMPTFOO_SELF_HOSTED",
        ]
    )
    profile = process.profile.read_text()
    assert "(deny default)" in profile and "(deny network*)" in profile
    assert "(allow file-write*" in profile
    assert json.dumps(str(run)) in profile
    assert '(subpath "/")' not in profile


@pytest.mark.parametrize("name", ["Starter Files", "starter files", "STARTER FILES"])
def test_protected_path_refuses_before_any_filesystem_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    tool = ToolRuntime(
        node=tmp_path / "node",
        tool=tmp_path / name,
        python=tmp_path / "python",
        package_root=tmp_path / "package",
    )

    def forbidden(*args: object, **kwargs: object) -> bool:
        raise AssertionError("protected path reached filesystem")

    monkeypatch.setattr(Path, "is_symlink", forbidden)
    monkeypatch.setattr(Path, "resolve", forbidden)
    with pytest.raises(ValueError, match="comparison_path_refused"):
        prepare_process(tool, tmp_path / "run", platform="darwin")


@pytest.mark.parametrize("platform", ["linux", "win32", "other"])
def test_unavailable_os_boundary_refuses_before_mutating_work(
    tmp_path: Path, platform: str
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    tool = runtime(tmp_path)
    with pytest.raises(ValueError, match="comparison_os_boundary_unavailable"):
        prepare_process(tool, run, platform=platform)
    assert list(run.iterdir()) == []


@pytest.mark.parametrize("name", ["tool-state", "tool-tmp", "empty.env", "boundary.sb"])
def test_existing_isolated_state_or_profile_is_never_replaced(tmp_path: Path, name: str) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / name).write_text("foreign bytes")
    with pytest.raises(ValueError, match="comparison_new_isolated_state_required"):
        prepare_process(runtime(tmp_path), run, platform="darwin")
    assert (run / name).read_text() == "foreign bytes" and len(list(run.iterdir())) == 1


def test_symlink_work_refuses_before_writing_target(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "run"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="comparison_symlink_refused"):
        prepare_process(runtime(tmp_path), link, platform="darwin")
    assert not list(target.iterdir())


def test_wrong_tool_version_refuses_without_creating_state(tmp_path: Path) -> None:
    tool = runtime(tmp_path)
    (tool.tool / "node_modules/promptfoo/package.json").write_text(
        '{"name":"promptfoo","version":"0.0.0"}'
    )
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(ValueError, match="comparison_pinned_tool_required"):
        prepare_process(tool, run, platform="darwin")
    assert not list(run.iterdir())


def test_path_controls_and_overbroad_roots_refuse(tmp_path: Path) -> None:
    tool = runtime(tmp_path)
    run = tmp_path / "run\nextra"
    with pytest.raises(ValueError, match="comparison_path_refused"):
        prepare_process(tool, run, platform="darwin")
    tool = ToolRuntime(node=tool.node, tool=tool.tool, python=tool.python, package_root=Path("/"))
    with pytest.raises(ValueError, match="comparison_read_scope_too_broad"):
        prepare_process(tool, tmp_path / "run", platform="darwin")


def test_executable_alias_cannot_follow_protected_target(tmp_path: Path) -> None:
    tool = runtime(tmp_path)
    tool.python.unlink()
    tool.python.symlink_to(tmp_path / "Starter Files" / "python")
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(ValueError, match="comparison_path_refused"):
        prepare_process(tool, run, platform="darwin")
    assert not list(run.iterdir())


def test_account_home_cannot_be_an_allowed_package_root(tmp_path: Path) -> None:
    tool = runtime(tmp_path)
    tool = ToolRuntime(node=tool.node, tool=tool.tool, python=tool.python, package_root=Path.home())
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(ValueError, match="comparison_read_scope_too_broad"):
        prepare_process(tool, run, platform="darwin")
    assert not list(run.iterdir())
