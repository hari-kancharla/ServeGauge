"""Runtime observations bind the exact prepared session without granting R2 status."""

from decimal import Decimal

import pytest
from unit.test_selfhost_plan import config, policy
from unit.test_selfhost_storage import SID, file_hashes

from servegauge.selfhost.plan import build_plan
from servegauge.selfhost.runtime import validate_runtime
from servegauge.storage.selfhost import prepare_session
from servegauge.storage.selfhost_runtime import save_runtime


def prepared():
    return build_plan(config(), policy(), max_usd=Decimal("2"), shutdown_minutes=10)


def receipt():
    plan = prepared()
    from servegauge.selfhost.remote_ssh import container_args

    return {
        "version": "selfhost-runtime-v1",
        "session_id": SID,
        "plan_hash": plan["plan_hash"],
        "backend": "ssh",
        "recorded_at": "2026-10-20T13:00:00+00:00",
        "engine_version": "0.11.0",
        "image": config().image,
        "model_revision": "b" * 40,
        "gpu_requested": "H100",
        "gpu_observed": "NVIDIA H100 80GB HBM3",
        "driver_version": "570.26",
        "engine_args": plan["engine_args"],
        "shutdown_armed": True,
        "shutdown_minutes": 10,
        "container_command": container_args(SID, plan),
    }


def test_runtime_binds_pins_flags_and_observed_machine_without_reference_certificate():
    value = validate_runtime(SID, prepared(), receipt())
    assert value["gpu_observed"] == "NVIDIA H100 80GB HBM3"
    assert "reference_tier" not in value
    assert "equivalence_qualified" not in value


@pytest.mark.parametrize(
    "change",
    [
        {"session_id": "selfhost-" + "c" * 32},
        {"plan_hash": "c" * 64},
        {"engine_version": "0.10.1"},
        {"image": "vllm/vllm-openai:latest"},
        {"model_revision": "c" * 40},
        {"gpu_requested": "H200"},
        {"gpu_observed": "NVIDIA A10"},
        {"shutdown_armed": False},
        {"shutdown_armed": 1},
        {"shutdown_minutes": 60},
        {"engine_args": ["vllm", "serve", "other"]},
        {"recorded_at": "2026-10-20T13:00:00"},
        {"container_command": ["docker", "run", "other"]},
        {"api_key": "fixture"},
    ],
)
def test_changed_or_unverified_runtime_observations_refuse(change):
    with pytest.raises(ValueError):
        validate_runtime(SID, prepared(), receipt() | change)


def test_observation_storage_is_immutable_guarded_private_and_read_only_on_repeat(tmp_path):
    plan = prepared()
    prepare_session(tmp_path, SID, plan)
    digest = save_runtime(tmp_path, SID, plan, receipt())
    before = file_hashes(tmp_path)
    assert save_runtime(tmp_path, SID, plan, receipt()) == digest
    assert file_hashes(tmp_path) == before
    assert not (tmp_path / "data/ledger").exists()
    assert not (tmp_path / "budget").exists()
    assert len(list((tmp_path / "data/raw/selfhost-runtime").rglob("*.json"))) == 1


def test_credential_bearing_observation_never_creates_runtime_files(tmp_path, monkeypatch):
    plan = prepared()
    prepare_session(tmp_path, SID, plan)
    monkeypatch.setenv("SELFHOST_FIXTURE_SECRET", "fixture-runtime-secret")
    with pytest.raises(ValueError):
        save_runtime(tmp_path, SID, plan, receipt() | {"driver_version": "fixture-runtime-secret"})
    assert not (tmp_path / "data/raw/selfhost-runtime").exists()
