"""Prepare/build a pinned public CPU runtime; never include private comparison data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn
from urllib.parse import urlsplit
from zipfile import ZipFile

from servegauge.comparison.isolation import _executable, _lexical
from servegauge.core.hashing import canonical_json, content_hash
from servegauge.core.secrets import PATTERNS, configured_secrets
from servegauge.storage.comparison import _files, _guard, _private_json
from servegauge.storage.comparison_kit import _write
from servegauge.tokencount.cache import HF_FILES, VOCAB_SHA256, VOCAB_URL

NODE_IMAGE = (
    "docker.io/library/node@sha256:4db36457f406501e6f608802e5da617e5fbd0e80b75901b6a09de1ae5a667d32"
)
PYTHON_IMAGE = (
    "ghcr.io/astral-sh/uv@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58"
)


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ValueError("comparison_image_command_usage")


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def require_public_text(blob: bytes) -> None:
    """Scan decoded code like the development secret gate, not like an exchange.

    Capture redaction also recognizes credential-handling expressions as field
    values. Public source must retain that code verbatim. Real credential patterns
    and configured values still refuse, including decoded wheel members.
    """
    if any(pattern.search(blob) for pattern in PATTERNS) or any(
        value.encode() in blob for value in configured_secrets()
    ):
        raise ValueError("comparison_credential_like_public_input_refused")


def _public_url(value: str, hosts: set[str]) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in hosts
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.port not in {None, 443}
    ):
        raise ValueError("comparison_public_dependency_source_required")


def validate_public_dependencies(project: dict[str, Any], lock: dict[str, Any]) -> None:
    if project.get("tool", {}).get("uv"):
        raise ValueError("comparison_explicit_public_uv_configuration_required")
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages:
        raise ValueError("comparison_complete_public_dependency_lock_required")
    for package in packages:
        source = package["source"]
        if source != {"registry": "https://pypi.org/simple"} and not (
            source == {"editable": "."} and package["name"] == project["project"]["name"]
        ):
            raise ValueError("comparison_public_registry_or_project_only")
        for distribution in [package.get("sdist", {}), *package.get("wheels", [])]:
            if "url" in distribution:
                _public_url(distribution["url"], {"files.pythonhosted.org"})


def dockerfile(wheel: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.-]+\.whl", wheel) is None:
        raise ValueError("comparison_simple_wheel_filename_required")
    lines = [
        f"FROM {NODE_IMAGE} AS node",
        f"FROM {PYTHON_IMAGE}",
        "COPY --from=node /usr/local/bin/node /opt/node/bin/node",
        "COPY --from=node /usr/local/lib/node_modules/npm /opt/npm",
        "COPY --from=node /usr/lib/*-linux-gnu/libstdc++.so.6* /opt/node-libs/",
        "COPY --from=node /usr/lib/*-linux-gnu/libatomic.so.1* /opt/node-libs/",
        "COPY --from=node /usr/lib/*-linux-gnu/libgcc_s.so.1 /opt/node-libs/",
        'ENV PATH="/opt/node/bin:/usr/local/bin:/usr/bin:/bin" LD_LIBRARY_PATH="/opt/node-libs" '
        'UV_PYTHON_DOWNLOADS="never" UV_PYTHON_PREFERENCE="only-system" '
        'UV_PYTHON="python3.12" UV_LINK_MODE="copy"',
        "COPY pyproject.toml uv.lock README.md LICENSE CITATION.cff /opt/runtime/",
        "WORKDIR /opt/runtime",
        "RUN "
        + json.dumps(["/usr/local/bin/uv", "sync", "--frozen", "--no-dev", "--no-install-project"]),
        "COPY package.json package-lock.json /opt/promptfoo/",
        "COPY empty-user.npmrc empty-global.npmrc /opt/",
        "RUN "
        + json.dumps(
            [
                "/opt/node/bin/node",
                "/opt/npm/bin/npm-cli.js",
                "ci",
                "--prefix",
                "/opt/promptfoo",
                "--ignore-scripts",
                "--no-audit",
                "--no-fund",
                "--omit=dev",
                "--cache",
                "/opt/npm-cache",
                "--userconfig",
                "/opt/empty-user.npmrc",
                "--globalconfig",
                "/opt/empty-global.npmrc",
            ]
        ),
        f"COPY {wheel} /opt/install/{wheel}",
        "RUN "
        + json.dumps(
            [
                "/usr/local/bin/uv",
                "pip",
                "install",
                "--python",
                "/opt/runtime/.venv/bin/python",
                "--no-index",
                "--no-deps",
                "/opt/install/" + wheel,
            ]
        ),
        "COPY tokenizers /opt/tokenizers",
        "RUN " + json.dumps(["/usr/bin/chmod", "-R", "a+rX", "/opt/tokenizers"]),
    ]
    return "\n".join(lines) + "\n"


def dockerignore(names: list[str]) -> str:
    allowed = {"Dockerfile", ".dockerignore"}
    for name in names:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("comparison_public_context_path_required")
        allowed.add(name)
        allowed.update(
            parent.as_posix() + "/" for parent in relative.parents if parent.as_posix() != "."
        )
    return "**\n" + "\n".join("!" + name for name in sorted(allowed)) + "\n"


def _wheel(root: Path, wheel: Path) -> tuple[bytes, str]:
    _guard(wheel)
    resources = root / "src/servegauge/resources/workspace-v1/manifest.json"
    _guard(resources)
    resource_names: set[str] = set()
    if resources.is_file():
        declared = _private_json(resources.read_bytes())
        resource_names = {"resources/workspace-v1/manifest.json"}
        for name in declared["files"]:
            relative = PurePosixPath(name)
            if (
                not relative.parts
                or relative.is_absolute()
                or ".." in relative.parts
                or "\\" in name
                or relative.parts[0] not in {"configs", "profiles", "docs"}
            ):
                raise ValueError("comparison_public_workspace_resource_required")
            resource_names.add("resources/workspace-v1/" + name)
    public_assets = {
        "report/README.md",
        "report/site/report.css",
        "report/site/report.js",
        "report/site/site.css",
        "report/site/site.js",
        "report/templates/report.html.jinja",
        "report/templates/site.html.jinja",
        "report/vendor/manifest.json",
        "report/vendor/vega-6.4.0.min.js",
        "report/vendor/vega-LICENSE.txt",
    }
    python = {}
    blob = wheel.read_bytes()
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("comparison_unique_wheel_members_required")
        metadata_prefixes = {name.split("/")[0] for name in names if ".dist-info/" in name}
        if len(metadata_prefixes) != 1:
            raise ValueError("comparison_single_distribution_wheel_required")
        metadata_prefix = next(iter(metadata_prefixes))
        if not metadata_prefix.startswith("servegauge-"):
            raise ValueError("comparison_project_wheel_required")
        for info in archive.infolist():
            relative = PurePosixPath(info.filename)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in info.filename
                or any(part.casefold() == "starter files" for part in relative.parts)
                or stat.S_ISLNK(info.external_attr >> 16)
            ):
                raise ValueError("comparison_wheel_member_refused")
            if info.is_dir():
                continue
            name = info.filename
            data = archive.read(info)
            if name.startswith("servegauge/"):
                tail = name.removeprefix("servegauge/")
                if not (name.endswith(".py") or tail in public_assets or tail in resource_names):
                    raise ValueError("comparison_public_package_asset_required")
            elif name not in {"trueweights/__init__.py", "trueweights/cli.py"} and name not in {
                metadata_prefix + "/" + member
                for member in [
                    "METADATA",
                    "WHEEL",
                    "entry_points.txt",
                    "licenses/LICENSE",
                    "RECORD",
                ]
            }:
                raise ValueError("comparison_public_wheel_member_required")
            if name.startswith(("servegauge/", "trueweights/")):
                path = root / "src" / name
                _guard(path)
                if not path.is_file() or path.read_bytes() != data:
                    raise ValueError("comparison_wheel_current_source_mismatch")
            if name.startswith("servegauge/") and name.endswith(".py"):
                python["src/" + name] = content_hash(data.decode())
            data.decode()  # Package members must be inspectable UTF-8 text.
            require_public_text(data)
    if not python:
        raise ValueError("comparison_package_source_inventory_required")
    return blob, content_hash(dict(sorted(python.items())))


def prepare_context(root: Path, *, wheel: Path, tokenizer_cache: Path, out: Path) -> dict[str, Any]:
    _lexical([root, wheel, tokenizer_cache, out])
    for path in [root, wheel, tokenizer_cache, out]:
        _guard(path)
    private = root.absolute() / ".cache/tools"
    if not out.absolute().is_relative_to(private) or out.absolute() == private or out.exists():
        raise ValueError("comparison_new_private_tool_context_required")
    wheel_blob, source_hash = _wheel(root, wheel)
    if re.fullmatch(r"[A-Za-z0-9_.-]+\.whl", wheel.name) is None:
        raise ValueError("comparison_wheel_name_invalid")
    payloads = {wheel.name: wheel_blob}
    for name in ["pyproject.toml", "uv.lock", "README.md", "LICENSE", "CITATION.cff"]:
        path = root / name
        _guard(path)
        blob = path.read_bytes()
        blob.decode()
        require_public_text(blob)
        payloads[name] = blob
    validate_public_dependencies(
        tomllib.loads(payloads["pyproject.toml"].decode()),
        tomllib.loads(payloads["uv.lock"].decode()),
    )
    inputs = root / "scripts/comparison_runtime_inputs"
    _guard(inputs)
    _guard(inputs / "promptfoo-pin.json")
    pin = _private_json((inputs / "promptfoo-pin.json").read_bytes())
    for name, target, hash_key in [
        ("promptfoo-package.json", "package.json", "package_sha256"),
        ("promptfoo-package-lock.json", "package-lock.json", "lock_sha256"),
    ]:
        path = inputs / name
        _guard(path)
        blob = path.read_bytes()
        if _sha(blob) != pin[hash_key]:
            raise ValueError("comparison_pinned_npm_inputs_changed")
        payloads[target] = blob
    lock = _private_json(payloads["package-lock.json"])
    for package in lock["packages"].values():
        if "resolved" in package:
            _public_url(package["resolved"], {"registry.npmjs.org"})
    payloads.update({"empty-user.npmrc": b"", "empty-global.npmrc": b""})
    assets = {
        "o200k_base.tiktoken": VOCAB_SHA256,
        hashlib.sha1(VOCAB_URL.encode(), usedforsecurity=False).hexdigest(): VOCAB_SHA256,
        **{"hf-gpt-oss/" + name: digest for name, digest in HF_FILES.items()},
    }
    for name, digest in assets.items():
        path = tokenizer_cache / name
        _guard(path)
        blob = path.read_bytes()
        if _sha(blob) != digest:
            raise ValueError("comparison_pinned_public_tokenizer_changed")
        payloads["tokenizers/" + name] = blob
    payloads["Dockerfile"] = dockerfile(wheel.name).encode()
    payloads[".dockerignore"] = dockerignore([*payloads, ".dockerignore"]).encode()
    out.mkdir(mode=0o700, parents=True)
    for name, blob in payloads.items():
        _write(out / name, blob)
    manifest = {
        "version": "comparison-image-context-v1",
        "wheel": wheel.name,
        "source_hash": source_hash,
        "node_image": NODE_IMAGE,
        "python_image": PYTHON_IMAGE,
        "private_captures_included": False,
        "files_sha256": {name: _sha(blob) for name, blob in sorted(payloads.items())},
    }
    _write(out / "manifest.json", canonical_json(manifest) + b"\n")
    return manifest


def build_image(
    context: Path, *, context_hash: str, docker: Path, config: Path, host: str, out: Path
) -> dict[str, Any]:
    _lexical([context, docker, config, out])
    for path in [context, config, out]:
        _guard(path)
    manifest = _private_json((context / "manifest.json").read_bytes())
    if (
        not isinstance(manifest, dict)
        or content_hash(manifest) != context_hash
        or manifest.get("version") != "comparison-image-context-v1"
        or manifest.get("node_image") != NODE_IMAGE
        or manifest.get("python_image") != PYTHON_IMAGE
        or manifest.get("private_captures_included") is not False
    ):
        raise ValueError("comparison_external_context_anchor_changed")
    actual = {
        path.relative_to(context).as_posix(): _sha(path.read_bytes())
        for path in _files(context)
        if path != context / "manifest.json"
    }
    if (
        actual != manifest["files_sha256"]
        or (context / "Dockerfile").read_text() != dockerfile(manifest["wheel"])
        or (context / ".dockerignore").read_text() != dockerignore(list(actual))
    ):
        raise ValueError("comparison_exact_public_context_required")
    expected = {
        "Dockerfile",
        ".dockerignore",
        manifest["wheel"],
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "package.json",
        "package-lock.json",
        "empty-user.npmrc",
        "empty-global.npmrc",
        "tokenizers/o200k_base.tiktoken",
        "tokenizers/" + hashlib.sha1(VOCAB_URL.encode(), usedforsecurity=False).hexdigest(),
        *("tokenizers/hf-gpt-oss/" + name for name in HF_FILES),
    }
    if set(actual) != expected:
        raise ValueError("comparison_explicit_public_payload_inventory_required")
    if host not in {"unix:///var/run/docker.sock", "npipe:////./pipe/docker_engine"}:
        raise ValueError("comparison_local_daemon_required")
    _guard(config / "config.json")
    if {path.name for path in config.iterdir()} != {"config.json"} or _private_json(
        (config / "config.json").read_bytes()
    ) != {"auths": {}}:
        raise ValueError("comparison_anonymous_docker_config_required")
    if out.exists() or not out.parent.is_dir() or out.absolute().is_relative_to(context.absolute()):
        raise ValueError("comparison_new_separate_build_receipt_required")
    executable = _executable(docker)
    iid = out.with_suffix(out.suffix + ".iid")
    _guard(iid)
    for path in [out.with_suffix(out.suffix + ".stdout"), out.with_suffix(out.suffix + ".stderr")]:
        _guard(path)
        if path.exists():
            raise ValueError("comparison_new_build_log_destination_required")
    if iid.exists():
        raise ValueError("comparison_new_build_identity_required")
    environment = {"PATH": os.pathsep.join([str(executable.parent), os.defpath])}
    if os.name == "nt":
        environment["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
    command = [
        str(executable),
        "--config",
        str(config.absolute()),
        "--host",
        host,
        "build",
        "--iidfile",
        str(iid.absolute()),
        str(context.absolute()),
    ]
    result = subprocess.run(command, env=environment, capture_output=True, timeout=1800)
    _write(out.with_suffix(out.suffix + ".stdout"), result.stdout)
    _write(out.with_suffix(out.suffix + ".stderr"), result.stderr)
    if result.returncode != 0:
        raise ValueError("comparison_public_image_build_failed")
    image = iid.read_text().strip()
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None:
        raise ValueError("comparison_immutable_local_image_identity_required")
    receipt = {
        "version": "comparison-image-v1",
        "context_hash": context_hash,
        "image": image,
        "source_hash": manifest["source_hash"],
        "public_inputs_only": True,
        "model_api_calls": 0,
        "new_spend_usd": 0,
    }
    _write(out, canonical_json(receipt) + b"\n")
    return receipt


def main() -> None:
    parser = Parser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "build", "init-client"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--tokenizer-cache", type=Path)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--context-hash")
    parser.add_argument("--docker", type=Path)
    parser.add_argument("--docker-config", type=Path)
    parser.add_argument(
        "--docker-host",
        default="npipe:////./pipe/docker_engine"
        if os.name == "nt"
        else "unix:///var/run/docker.sock",
    )
    try:
        args = parser.parse_args()
        _lexical(
            [
                path
                for path in [
                    args.root,
                    args.out,
                    args.wheel,
                    args.tokenizer_cache,
                    args.context,
                    args.docker,
                    args.docker_config,
                ]
                if path is not None
            ]
        )
        if args.command == "init-client":
            _guard(args.out)
            private = args.root.absolute() / ".cache/tools"
            if not args.out.absolute().is_relative_to(private) or args.out.exists():
                raise ValueError
            args.out.mkdir(mode=0o700, parents=True)
            _write(args.out / "config.json", b'{"auths":{}}\n')
            value = {"state": "anonymous_local_client_prepared", "model_api_calls": 0}
        elif args.command == "prepare":
            if args.wheel is None:
                raise ValueError
            manifest = prepare_context(
                args.root,
                wheel=args.wheel,
                tokenizer_cache=args.tokenizer_cache or args.root / ".cache/tokenizers",
                out=args.out,
            )
            value = {
                "state": "public_context_prepared",
                "context_hash": content_hash(manifest),
                "source_hash": manifest["source_hash"],
                "model_api_calls": 0,
            }
        else:
            if (
                args.context is None
                or args.context_hash is None
                or args.docker is None
                or args.docker_config is None
            ):
                raise ValueError
            value = build_image(
                args.context,
                context_hash=args.context_hash,
                docker=args.docker,
                config=args.docker_config,
                host=args.docker_host,
                out=args.out,
            )
        print(canonical_json(value).decode())
    except Exception:
        parser.exit(3, "Invalid or changed public image inputs; no model or comparison executed.\n")


if __name__ == "__main__":
    main()
