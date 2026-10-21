"""Run the actual remote Python job against fake system commands, with no GPU or SSH."""

import json
from decimal import Decimal

import pytest
from unit.test_selfhost_plan import config, policy
from unit.test_selfhost_storage import SID

from servegauge.selfhost.plan import build_plan
from servegauge.selfhost.remote_ssh import handle


def payload(action="up"):
    return {
        "action": action,
        "session_id": SID,
        "plan": build_plan(config(), policy(), max_usd=Decimal("2"), shutdown_minutes=10),
        "endpoint_key": "fixture-endpoint-key" if action == "up" else None,
    }


class Commands:
    def __init__(self, fail_shutdown=False, gpu="NVIDIA H100 80GB HBM3"):
        self.calls = []
        self.fail_shutdown = fail_shutdown
        self.gpu = gpu
        self.running = True
        self.bad_labels = False

    def __call__(self, args, *, environment=None, timeout=60):
        self.calls.append((args, environment))
        if args[:3] == ["sudo", "-n", "shutdown"]:
            if self.fail_shutdown:
                raise RuntimeError("permission denied")
            return ""
        if args[0] == "nvidia-smi":
            return self.gpu + ", 570.26\n"
        if args[:2] == ["docker", "run"]:
            if "--detach" in args:
                assert environment["VLLM_API_KEY"] == "fixture-endpoint-key"
                return "c" * 64 + "\n"
            return "0.11.0\n"
        if args[:3] == ["docker", "image", "inspect"]:
            return json.dumps(["vllm/vllm-openai@sha256:" + "a" * 64])
        if args[:2] == ["docker", "inspect"]:
            request = payload()
            return json.dumps(
                {
                    "Id": "c" * 64,
                    "Config": {
                        "Labels": {
                            "servegauge.session_id": SID,
                            "servegauge.plan_hash": "wrong"
                            if self.bad_labels
                            else request["plan"]["plan_hash"],
                        }
                    },
                    "State": {"Running": self.running, "ExitCode": 0},
                }
            )
        if args[:2] == ["docker", "stop"]:
            self.running = False
            return "servegauge-" + "a" * 32
        raise AssertionError("Unexpected fake command")


def test_shutdown_is_first_and_pinned_authenticated_gpu_launch_is_loopback_only(tmp_path):
    commands = Commands()
    result = handle(payload(), home=tmp_path, execute=commands)
    assert commands.calls[0][0] == ["sudo", "-n", "shutdown", "-h", "+10"]
    gpu = [row for row in commands.calls if "--detach" in row[0]]
    assert len(gpu) == 1
    args, env = gpu[0]
    assert args[args.index("--publish") + 1] == "127.0.0.1:8000:8000"
    assert args[args.index("--gpus") + 1] == "device=0"
    assert "fixture-endpoint-key" not in str(args)
    assert env["VLLM_API_KEY"] == "fixture-endpoint-key"
    assert result["state"] == "running"
    assert result["runtime_receipt"]["gpu_observed"] == "NVIDIA H100 80GB HBM3"
    assert result["runtime_receipt"]["engine_version"] == "0.11.0"
    assert result["runtime_receipt"]["shutdown_armed"] is True
    assert all(
        b"fixture-endpoint-key" not in p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    )


def test_shutdown_failure_prevents_even_the_image_probe(tmp_path):
    commands = Commands(fail_shutdown=True)
    with pytest.raises(RuntimeError):
        handle(payload(), home=tmp_path, execute=commands)
    assert len(commands.calls) == 1


def test_repeated_up_never_creates_a_second_container(tmp_path):
    commands = Commands()
    handle(payload(), home=tmp_path, execute=commands)
    before = len(commands.calls)
    with pytest.raises(ValueError):
        handle(payload(), home=tmp_path, execute=commands)
    assert len(commands.calls) == before


def test_status_and_stop_require_exact_saved_plan_and_container_labels(tmp_path):
    commands = Commands()
    handle(payload(), home=tmp_path, execute=commands)
    assert handle(payload("status"), home=tmp_path, execute=commands)["state"] == "running"
    commands.bad_labels = True
    with pytest.raises(ValueError):
        handle(payload("down"), home=tmp_path, execute=commands)
    assert not any(row[0][:2] == ["docker", "stop"] for row in commands.calls)
    commands.bad_labels = False
    assert handle(payload("down"), home=tmp_path, execute=commands)["state"] == "terminated"
    assert sum(row[0][:2] == ["docker", "stop"] for row in commands.calls) == 1
    assert handle(payload("down"), home=tmp_path, execute=commands)["state"] == "terminated"
    assert sum(row[0][:2] == ["docker", "stop"] for row in commands.calls) == 1


def test_wrong_gpu_refuses_before_gpu_container(tmp_path):
    commands = Commands(gpu="NVIDIA A10")
    with pytest.raises(ValueError):
        handle(payload(), home=tmp_path, execute=commands)
    assert not any("--detach" in row[0] for row in commands.calls)


def test_cleanup_rejects_a_recreated_container_even_with_the_same_labels(tmp_path):
    commands = Commands()
    handle(payload(), home=tmp_path, execute=commands)
    value = payload("down") | {"expected_remote_id": "f" * 64}
    with pytest.raises(ValueError):
        handle(value, home=tmp_path, execute=commands)
    assert not any(row[0][:2] == ["docker", "stop"] for row in commands.calls)


def test_remote_payload_tamper_and_protected_home_refuse_before_commands(tmp_path):
    commands = Commands()
    value = payload()
    value["plan"]["configuration"]["image"] = "evil; anything"
    with pytest.raises(ValueError):
        handle(value, home=tmp_path, execute=commands)
    with pytest.raises(ValueError):
        handle(payload(), home=tmp_path / "Starter Files", execute=commands)
    assert commands.calls == []
