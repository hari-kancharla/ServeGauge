"""Owner session transports, usable only behind a persisted plan and reserved dispatch."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import re
import shlex
import subprocess
from collections.abc import Mapping
from decimal import Decimal
from importlib.resources import files
from pathlib import Path
from typing import Any

from servegauge.core.hashing import canonical_json
from servegauge.observatory.provenance import implementation_hash
from servegauge.runner.ledger import Ledger
from servegauge.selfhost.controller import SessionDriver, _authorize, request_hash
from servegauge.selfhost.runtime import validate_runtime
from servegauge.storage.notices import guard_path
from servegauge.storage.selfhost import SessionJournal


def _receiver_source(name: str) -> str:
    """Assemble fixed package sources into a standalone standard-library receiver."""
    if name not in {"remote_ssh.py", "remote_modal.py"}:
        raise ValueError("selfhost_known_remote_receiver_required")
    package = files("servegauge.selfhost")
    body: list[ast.stmt] = []
    allowed = {
        "hashlib",
        "importlib",
        "json",
        "os",
        "re",
        "subprocess",
        "sys",
        "time",
        "collections",
        "datetime",
        "pathlib",
        "typing",
    }
    for filename in ["remote_recipe.py", name]:
        tree = ast.parse(package.joinpath(filename).read_text())
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                if node.module in {"__future__", "servegauge.selfhost.remote_recipe"}:
                    continue
                if (node.module or "").split(".")[0] not in allowed:
                    raise ValueError("selfhost_remote_dependency_not_standalone")
            if isinstance(node, ast.Import) and any(
                alias.name.split(".")[0] not in allowed for alias in node.names
            ):
                raise ValueError("selfhost_remote_dependency_not_standalone")
            body.append(node)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    combined = ast.Module(body=[future, *body], type_ignores=[])
    return ast.unparse(ast.fix_missing_locations(combined))


def _bound(
    root: Path,
    identifier: str,
    plan: dict[str, Any],
    environment: Mapping[str, str],
    *,
    launch: bool,
) -> dict[str, Any]:
    journal = SessionJournal(root, identifier)
    status = journal.status()
    if status["plan"] != plan:
        raise ValueError("selfhost_adapter_reviewed_plan_changed")
    _authorize(plan, environment, launch=launch)
    if launch:
        if (
            status["state"] != "dispatching"
            or journal.spec()["implementation_hash"] != implementation_hash()
        ):
            raise ValueError("selfhost_adapter_reserved_dispatch_required")
        guard_path(root / "data/ledger/spend.jsonl")
        ledger = Ledger(root)
        ledger.refresh()
        row = ledger.pending.get(status["facts"]["reservation_id"])
        if (
            not row
            or not row["dispatched"]
            or row["owner"] is not True
            or row["run_id"] != identifier
            or row["request_hash"] != request_hash(identifier, plan)
            or Decimal(row["amount"]) != Decimal(plan["reserved_upper_usd"])
            or row["endpoint_id"] != "selfhost:" + plan["configuration"]["backend"]
            or row["exchange_id"] != identifier
        ):
            raise ValueError("selfhost_adapter_shared_reservation_required")
    elif status["state"] not in {"dispatching", "running", "stopping", "outcome_unverified"}:
        raise ValueError("selfhost_adapter_active_session_required")
    return status


def _normalize(identifier: str, plan: dict[str, Any], value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "state",
        "remote_id",
        "endpoint_url",
        "remote_exit_code",
        "runtime_receipt",
    }:
        raise ValueError("selfhost_remote_protocol_invalid")
    if value["state"] not in {"running", "terminated"}:
        raise ValueError("selfhost_remote_state_unverified")
    if value["state"] == "running" and not isinstance(value["runtime_receipt"], dict):
        raise ValueError("selfhost_remote_runtime_pending")
    if value["runtime_receipt"] is not None:
        value = {
            **value,
            "runtime_receipt": validate_runtime(identifier, plan, value["runtime_receipt"]),
        }
    return dict(value)


class SSHDriver:
    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def _key(plan: dict[str, Any], environment: Mapping[str, str]) -> Path:
        key = Path(environment[plan["configuration"]["ssh_key_env"]])
        if ".." in key.parts or any(
            p.casefold() == ".git" or p.casefold().startswith(".env") for p in key.parts
        ):
            raise ValueError("selfhost_canonical_ssh_key_path_required")
        guard_path(key)
        if not key.is_file():
            raise ValueError("selfhost_ssh_key_file_required")
        return key

    def preflight(
        self, identifier: str, plan: dict[str, Any], environment: Mapping[str, str]
    ) -> None:
        if (
            SessionJournal(self.root, identifier).status()["plan"] != plan
            or plan["configuration"]["backend"] != "ssh"
        ):
            raise ValueError("selfhost_ssh_reviewed_plan_required")
        _authorize(plan, environment, launch=True)
        self._key(plan, environment)

    def _rpc(
        self, action: str, identifier: str, plan: dict[str, Any], environment: Mapping[str, str]
    ) -> dict[str, Any]:
        status = _bound(self.root, identifier, plan, environment, launch=action == "up")
        config = plan["configuration"]
        if config["backend"] != "ssh":
            raise ValueError("selfhost_ssh_backend_required")
        key = self._key(plan, environment)
        source = _receiver_source("remote_ssh.py")
        remote = shlex.join(
            ["env", "SERVEGAUGE_REMOTE_OWNER_DISPATCH=1", "python3.12", "-c", source]
        )
        args = [
            "ssh",
            "-F",
            "none",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "ForwardAgent=no",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "SendEnv=-*",
            "-p",
            str(config["ssh_port"]),
            "-i",
            str(key),
            "--",
            config["ssh_host"],
            remote,
        ]
        payload = {
            "action": action,
            "session_id": identifier,
            "plan": plan,
            "endpoint_key": environment[config["api_key_env"]] if action == "up" else None,
            "expected_remote_id": status["facts"].get("remote_id"),
        }
        result = subprocess.run(
            args,
            input=canonical_json(payload).decode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=plan["maximum_duration_seconds"] + 60 if action == "up" else 60,
        )
        if result.returncode != 0 or len(result.stdout) > 1048576:
            raise ValueError("selfhost_ssh_outcome_unverified")
        return _normalize(identifier, plan, json.loads(result.stdout))

    def launch(
        self, identifier: str, plan: dict[str, Any], environment: Mapping[str, str]
    ) -> dict[str, Any]:
        return self._rpc("up", identifier, plan, environment)

    def inspect(
        self, identifier: str, plan: dict[str, Any], environment: Mapping[str, str]
    ) -> dict[str, Any]:
        return self._rpc("status", identifier, plan, environment)

    def stop(
        self, identifier: str, plan: dict[str, Any], environment: Mapping[str, str]
    ) -> dict[str, Any]:
        return self._rpc("down", identifier, plan, environment)


class ModalDriver:
    def __init__(self, root: Path) -> None:
        self.root = root

    def preflight(
        self, identifier: str, plan: dict[str, Any], environment: Mapping[str, str]
    ) -> None:
        if (
            SessionJournal(self.root, identifier).status()["plan"] != plan
            or plan["configuration"]["backend"] != "modal"
        ):
            raise ValueError("selfhost_modal_reviewed_plan_required")
        _authorize(plan, environment, launch=True)
        try:
            sdk = importlib.import_module("modal")
            inspect.signature(sdk.Client.from_credentials).bind("placeholder", "placeholder")
            inspect.signature(sdk.App.lookup).bind("app", client=None, create_if_missing=False)
            inspect.signature(sdk.Image.from_id).bind("im-placeholder", client=None)
            inspect.signature(sdk.Sandbox.from_name).bind(
                "app", "name", client=None, include_terminated=True
            )
            inspect.signature(sdk.Sandbox.create).bind(
                "python3",
                "-c",
                "pass",
                app=None,
                client=None,
                image=None,
                name="name",
                tags={},
                timeout=1,
                gpu="H100",
                encrypted_ports=[8000],
                inbound_cidr_allowlist=["198.51.100.7/32"],
                secrets=[],
                env={},
            )
        except (ImportError, AttributeError, TypeError, ValueError):
            raise ValueError("selfhost_modal_supported_optional_sdk_required") from None

    def _context(
        self, identifier: str, plan: dict[str, Any], environment: Mapping[str, str], *, launch: bool
    ) -> tuple[Any, Any, dict[str, Any]]:
        status = _bound(self.root, identifier, plan, environment, launch=launch)
        if plan["configuration"]["backend"] != "modal":
            raise ValueError("selfhost_modal_backend_required")
        sdk = importlib.import_module("modal")
        client = sdk.Client.from_credentials(
            environment["MODAL_TOKEN_ID"], environment["MODAL_TOKEN_SECRET"]
        )
        return sdk, client, status

    @staticmethod
    def _snapshot(sandbox: Any, identifier: str, plan: dict[str, Any]) -> dict[str, Any]:
        tags = sandbox.get_tags()
        if (
            tags.get("servegauge_session") != identifier
            or tags.get("servegauge_plan") != plan["plan_hash"]
        ):
            raise ValueError("selfhost_modal_remote_binding_changed")
        remote_id = sandbox.object_id
        if not isinstance(remote_id, str) or not re.fullmatch(
            r"sb-[A-Za-z0-9_-]{1,128}", remote_id
        ):
            raise ValueError("selfhost_modal_remote_identity_invalid")
        code = sandbox.poll()
        if code is not None:
            if type(code) is not int:
                raise ValueError("selfhost_modal_exit_code_invalid")
            return {
                "state": "terminated",
                "remote_id": remote_id,
                "endpoint_url": None,
                "remote_exit_code": code,
                "runtime_receipt": None,
            }
        raw = sandbox.filesystem.read_text("/tmp/servegauge-runtime-v1.json")
        if len(raw) > 65536:
            raise ValueError("selfhost_modal_runtime_response_too_large")
        receipt = {**json.loads(raw), "modal_sandbox_id": remote_id}
        receipt["modal_image_id"] = plan["configuration"]["modal_image_id"]
        endpoint = sandbox.tunnels()[8000].url
        return _normalize(
            identifier,
            plan,
            {
                "state": "running",
                "remote_id": remote_id,
                "endpoint_url": endpoint.rstrip("/") + "/v1",
                "remote_exit_code": None,
                "runtime_receipt": receipt,
            },
        )

    def launch(
        self, identifier: str, plan: dict[str, Any], environment: Mapping[str, str]
    ) -> dict[str, Any]:
        sdk, client, _ = self._context(identifier, plan, environment, launch=True)
        config = plan["configuration"]
        app = sdk.App.lookup(config["modal_app"], client=client, create_if_missing=False)
        # Image build/import work is a separate owner preparation action. It must
        # not introduce an unbounded setup job before the Sandbox lifetime starts.
        image = sdk.Image.from_id(config["modal_image_id"], client=client)
        source = _receiver_source("remote_modal.py")
        secret = sdk.Secret.from_dict({"VLLM_API_KEY": environment[config["api_key_env"]]})
        sandbox = sdk.Sandbox.create(
            "python3",
            "-c",
            source,
            app=app,
            client=client,
            image=image,
            name="servegauge-" + identifier.removeprefix("selfhost-"),
            tags={"servegauge_session": identifier, "servegauge_plan": plan["plan_hash"]},
            timeout=plan["shutdown_minutes"] * 60,
            gpu=config["gpu"],
            encrypted_ports=[8000],
            inbound_cidr_allowlist=config["inbound_cidrs"],
            secrets=[secret],
            env={
                "SERVEGAUGE_RUNTIME_PLAN": canonical_json(
                    {"session_id": identifier, "plan": plan}
                ).decode()
            },
        )
        return self._snapshot(sandbox, identifier, plan)

    def _lookup(self, identifier: str, plan: dict[str, Any], environment: Mapping[str, str]) -> Any:
        sdk, client, status = self._context(identifier, plan, environment, launch=False)
        sandbox = sdk.Sandbox.from_name(
            plan["configuration"]["modal_app"],
            "servegauge-" + identifier.removeprefix("selfhost-"),
            include_terminated=True,
            client=client,
        )
        if status["facts"].get("remote_id", sandbox.object_id) != sandbox.object_id:
            raise ValueError("selfhost_modal_original_resource_changed")
        tags = sandbox.get_tags()
        if (
            tags.get("servegauge_session") != identifier
            or tags.get("servegauge_plan") != plan["plan_hash"]
        ):
            raise ValueError("selfhost_modal_remote_binding_changed")
        return sandbox

    def inspect(
        self, identifier: str, plan: dict[str, Any], environment: Mapping[str, str]
    ) -> dict[str, Any]:
        return self._snapshot(self._lookup(identifier, plan, environment), identifier, plan)

    def stop(
        self, identifier: str, plan: dict[str, Any], environment: Mapping[str, str]
    ) -> dict[str, Any]:
        sandbox = self._lookup(identifier, plan, environment)
        if sandbox.poll() is None:
            sandbox.terminate(wait=True)
        return self._snapshot(sandbox, identifier, plan)


def driver_for(root: Path, plan: dict[str, Any]) -> SessionDriver:
    if plan["configuration"]["backend"] == "ssh":
        return SSHDriver(root)
    if plan["configuration"]["backend"] == "modal":
        return ModalDriver(root)
    raise ValueError("selfhost_supported_backend_required")
