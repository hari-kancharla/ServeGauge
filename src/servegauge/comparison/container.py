"""Explicit local CPU-container boundary for private, finite comparison inputs."""

from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path

from servegauge.comparison.isolation import _executable, _lexical
from servegauge.storage.comparison import _files, _guard, _private_json


@dataclass(frozen=True)
class ContainerProcess:
    arguments: list[str]
    environment: dict[str, str]
    output: Path


def _mount(source: Path, target: str, *, readonly: bool) -> str:
    stream = io.StringIO(newline="")
    fields = ["type=bind", "src=" + str(source.absolute()), "dst=" + target]
    if readonly:
        fields.append("readonly")
    csv.writer(stream, lineterminator="").writerow(fields)
    return stream.getvalue()


def prepare_container(
    *,
    docker: Path,
    config: Path,
    host: str,
    image: str,
    inputs: Path,
    output: Path,
    user: str,
    arguments: list[str],
) -> ContainerProcess:
    """Prepare but never launch; the caller must verify the selected input manifest.

    Requires a local daemon, anonymous client config, immutable local image and
    fresh output. Docker must use Linux containers, including on Windows/macOS.
    The container receives only the explicit read-only inputs and writable output;
    no account directory, daemon socket or host source tree is implicitly mounted.
    """
    _lexical([docker, config, inputs, output])
    if (
        host not in {"unix:///var/run/docker.sock", "npipe:////./pipe/docker_engine"}
        or re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None
        or re.fullmatch(r"[1-9][0-9]*:[1-9][0-9]*", user) is None
        or not arguments
        or any(not isinstance(value, str) or "\x00" in value for value in arguments)
    ):
        raise ValueError("comparison_local_pinned_unprivileged_container_required")
    for path in [config, inputs, output]:
        _guard(path)
        if len(path.absolute().parts) < 4:
            raise ValueError("comparison_container_scope_too_broad")
    executable = _executable(docker)
    if (
        not config.is_dir()
        or {path.name for path in config.iterdir()} != {"config.json"}
        or not inputs.is_dir()
        or not output.parent.is_dir()
        or output.exists()
        or inputs.absolute().is_relative_to(output.absolute())
        or output.absolute().is_relative_to(inputs.absolute())
        or Path.home().absolute().is_relative_to(inputs.absolute())
    ):
        raise ValueError("comparison_fresh_separate_private_container_paths_required")
    _guard(config / "config.json")
    if _private_json((config / "config.json").read_bytes()) != {"auths": {}}:
        raise ValueError("comparison_anonymous_client_configuration_required")
    _files(inputs)  # Refuse nested links/special/protected paths before any mount.
    environment = {"PATH": os.pathsep.join([str(executable.parent), os.defpath])}
    if os.name == "nt":
        environment["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
    child = {
        "PATH": "/opt/node/bin:/opt/runtime/.venv/bin:/usr/local/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": "/opt/node-libs",
        "LANG": "C.UTF-8",
        "TZ": "UTC",
        "NO_COLOR": "1",
        "TMPDIR": "/tmp",
        "PROMPTFOO_CONFIG_DIR": "/work/tool-state",
        "DOTENV_PATH": "/work/empty.env",
        "PROMPTFOO_DISABLE_TELEMETRY": "1",
        "PROMPTFOO_DISABLE_UPDATE": "1",
        "PROMPTFOO_DISABLE_REMOTE_GENERATION": "true",
        "PROMPTFOO_DISABLE_SHARING": "1",
        "PROMPTFOO_SELF_HOSTED": "1",
        "PROMPTFOO_PYTHON_WORKERS": "1",
        "PROMPTFOO_PYTHON": "/opt/runtime/.venv/bin/python",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONSAFEPATH": "1",
        "SERVEGAUGE_TOKENIZER_CACHE": "/opt/tokenizers",
        "TIKTOKEN_ENCODINGS_BASE": "/opt/tokenizers",
        "TIKTOKEN_CACHE_DIR": "/opt/tokenizers",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    command = [
        str(executable),
        "--config",
        str(config.absolute()),
        "--host",
        host,
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges=true",
        "--pids-limit",
        "128",
        "--memory",
        "2g",
        "--cpus",
        "2",
        "--user",
        user,
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=256m",
        "--mount",
        _mount(inputs, "/inputs", readonly=True),
        "--mount",
        _mount(output, "/work", readonly=False),
        "--workdir",
        "/work",
        "--entrypoint",
        "/usr/bin/env",
        image,
        "-i",
        *(key + "=" + value for key, value in child.items()),
        *arguments,
    ]
    output.mkdir(mode=0o700)
    (output / "tool-state").mkdir(mode=0o700)
    (output / "empty.env").touch(mode=0o600)
    return ContainerProcess(command, environment, output.absolute())
