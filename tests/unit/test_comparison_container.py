"""Portable CPU boundary binds only explicit private inputs and fresh output."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from servegauge.comparison.container import prepare_container


def paths(tmp_path):
    config = tmp_path / "anonymous"
    config.mkdir()
    (config / "config.json").write_text('{"auths":{}}')
    inputs = tmp_path / "private-input"
    inputs.mkdir()
    (inputs / "synthetic.json").write_text("{}")
    output = tmp_path / "fresh-output"
    return config, inputs, output


def executable(tmp_path):
    path = tmp_path / "docker"
    path.write_text("synthetic executable path; never launched by unit tests")
    return path


def prepare(tmp_path, **kwargs):
    config, inputs, output = paths(tmp_path)
    return prepare_container(
        docker=executable(tmp_path),
        config=config,
        host="unix:///var/run/docker.sock",
        image="sha256:" + "a" * 64,
        inputs=inputs,
        output=output,
        user="501:20",
        arguments=["/usr/bin/true"],
        **kwargs,
    )


def test_boundary_has_no_network_account_home_or_socket_mount_or_inherited_environment(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SYNTHETIC_ACCOUNT_CANARY", "must-not-inherit")
    command = prepare(tmp_path)
    args = command.arguments
    assert args[args.index("--network") + 1] == "none"
    for flag in [
        "--read-only",
        "--cap-drop",
        "--security-opt",
        "--pids-limit",
        "--memory",
        "--cpus",
        "--pull",
    ]:
        assert flag in args
    assert args[args.index("--pull") + 1] == "never"
    mounts = [args[i + 1] for i, value in enumerate(args) if value == "--mount"]
    assert len(mounts) == 2 and "readonly" in mounts[0]
    assert all("docker.sock" not in mount and "/Users/" not in mount for mount in mounts)
    assert (
        not {"HOME", "DOCKER_CONTEXT", "HTTP_PROXY", "SYNTHETIC_ACCOUNT_CANARY"}
        & command.environment.keys()
    )
    assert "HOME=" not in " ".join(args) and "/usr/bin/env" in args and "-i" in args
    assert command.output.is_dir()


@pytest.mark.parametrize(
    "kind",
    [
        "account",
        "proxy",
        "extra_config",
        "remote",
        "tag",
        "existing_output",
        "overlap",
        "user",
        "symlink",
    ],
)
def test_unsafe_client_image_or_mount_scope_refuses_before_launch(tmp_path, kind):
    config, inputs, output = paths(tmp_path)
    options = dict(
        docker=executable(tmp_path),
        config=config,
        host="unix:///var/run/docker.sock",
        image="sha256:" + "a" * 64,
        inputs=inputs,
        output=output,
        user="501:20",
        arguments=["/usr/bin/true"],
    )
    if kind == "account":
        (config / "config.json").write_text(
            json.dumps({"auths": {"registry": {"auth": "synthetic"}}})
        )
    elif kind == "proxy":
        (config / "config.json").write_text('{"auths":{},"proxies":{}}')
    elif kind == "extra_config":
        (config / "influence").write_text("refuse")
    elif kind == "remote":
        options["host"] = "tcp://example.invalid:2375"
    elif kind == "tag":
        options["image"] = "arbitrary:latest"
    elif kind == "existing_output":
        output.mkdir()
    elif kind == "overlap":
        options["output"] = inputs / "output"
    elif kind == "user":
        options["user"] = "0:0"
    else:
        (inputs / "escape").symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError):
        prepare_container(**options)


def test_protected_input_path_is_refused_before_any_filesystem_access(tmp_path, monkeypatch):
    accesses = []

    def forbidden(*args, **kwargs):
        accesses.append(args)
        raise AssertionError("must not access any path")

    monkeypatch.setattr(Path, "is_symlink", forbidden)
    with pytest.raises(ValueError, match="path_refused"):
        prepare_container(
            docker=Path("/usr/local/bin/docker"),
            config=tmp_path / "config",
            host="unix:///var/run/docker.sock",
            image="sha256:" + "a" * 64,
            inputs=tmp_path / "Starter Files" / "input",
            output=tmp_path / "out",
            user="501:20",
            arguments=["/usr/bin/true"],
        )
    assert not accesses


def test_windows_local_named_pipe_and_escaped_mount_spaces_are_explicit(tmp_path):
    config, inputs, output = paths(tmp_path)
    spaced = tmp_path / "private input with spaces"
    inputs.rename(spaced)
    process = prepare_container(
        docker=executable(tmp_path),
        config=config,
        host="npipe:////./pipe/docker_engine",
        image="sha256:" + "a" * 64,
        inputs=spaced,
        output=output,
        user="65534:65534",
        arguments=["/usr/bin/true"],
    )
    assert (
        process.arguments[process.arguments.index("--host") + 1] == "npipe:////./pipe/docker_engine"
    )
    assert any("private input with spaces" in value for value in process.arguments)
