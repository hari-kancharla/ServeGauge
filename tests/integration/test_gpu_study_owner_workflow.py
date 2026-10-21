"""Real controller/receiver/runner wiring with simulated SSH/SDK/HTTP and no owner flags."""

import json
import math
import os
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest
from integration.test_gpu_study_collection_workflow import setup_kit
from integration.test_selfhost_controls_workflow import Machine
from unit.test_gpu_study_recipe import reviewed
from unit.test_selfhost_adapters import environment, modal_sdk
from unit.test_selfhost_plan import policy as fixture_policy

from servegauge.adapters.selfhosted import ModalDriver, SSHDriver
from servegauge.config import load_budget
from servegauge.runner.ledger import Ledger
from servegauge.selfhost.controller import launch_session, stop_session
from servegauge.selfhost.plan import build_plan
from servegauge.selfhost.study_recipe import VARIANTS
from servegauge.storage.gpu_capture import collect_slot, load_slot
from servegauge.storage.selfhost import SessionJournal, prepare_session
from servegauge.storage.selfhost_runtime import save_runtime


@pytest.mark.parametrize("backend", ["ssh", "modal"])
@pytest.mark.parametrize("variant", VARIANTS)
async def test_source_bound_owner_path_all_variants_retains_shutdown_and_raw_after_cleanup(
    tmp_path, monkeypatch, backend, variant
):
    assert os.environ.get("SERVEGAUGE_OWNER_RUN") is None
    assert os.environ.get("TRUEWEIGHTS_OWNER_RUN") is None
    spec, kit, digest, matrix = setup_kit(
        tmp_path,
        monkeypatch,
        budget_policy=fixture_policy(),
        scope="owner_gpu",
        backend=backend,
        engine_images={
            "0.31.0": "vllm/vllm-openai@sha256:" + "b" * 64,
            "0.32.0": "vllm/vllm-openai@sha256:" + "a" * 64,
        },
        modal_image_ids={"0.31.0": "im-fixture-old", "0.32.0": "im-fixture"},
    )
    slot = next(row for row in matrix if row["role"] == "target" and row["recipe"] == variant)
    settings = reviewed(
        variant, backend, **({"modal_app": "servegauge-owner"} if backend == "modal" else {})
    )
    settings = settings.model_copy(
        update={
            "image": spec.engine_images[settings.engine_version],
            **(
                {"modal_image_id": spec.modal_image_ids[settings.engine_version]}
                if backend == "modal"
                else {}
            ),
        }
    )
    policy = load_budget(tmp_path)
    plan = build_plan(settings, policy, max_usd=Decimal("2"), shutdown_minutes=10)
    prepare_session(tmp_path, slot["session_id"], plan)
    key = tmp_path / "fixture-key"
    key.write_text("fixture")
    owner = environment(key)
    owner["SELFHOST_API_KEY"] = "native-fixture-endpoint-key"
    monkeypatch.setenv("SELFHOST_API_KEY", owner["SELFHOST_API_KEY"])
    remote_actions = []
    if backend == "ssh":
        machine = Machine()
        original_execute, original_run = machine.execute, subprocess.run

        def execute(args, **kwargs):
            if args[:3] == ["docker", "run", "--rm"]:
                machine.commands.append(args)
                return settings.engine_version + "\n"
            return original_execute(args, **kwargs)

        def ssh(args, **kwargs):
            if args[0] != "ssh":
                assert args[0] == "git"
                return original_run(args, **kwargs)
            request = json.loads(kwargs["input"])
            ledger = Ledger(tmp_path)
            ledger.refresh()
            if request["action"] == "up":
                row = next(r for r in ledger.pending.values() if r["run_id"] == slot["session_id"])
                assert row["owner"] and row["dispatched"]
            namespace = {"__name__": "local_study_receiver"}
            exec(compile(shlex.split(args[-1])[-1], "local-study-receiver", "exec"), namespace)
            value = namespace["handle"](request, home=tmp_path / "remote-home", execute=execute)
            remote_actions.append(request["action"])
            return SimpleNamespace(returncode=0, stdout=json.dumps(value))

        monkeypatch.setattr(subprocess, "run", ssh)
        driver = SSHDriver(tmp_path)
    else:
        sdk, state = modal_sdk(plan, remote_actions, identifier=slot["session_id"])
        original = sdk.Sandbox.create

        def create(*args, **kwargs):
            ledger = Ledger(tmp_path)
            ledger.refresh()
            row = next(r for r in ledger.pending.values() if r["run_id"] == slot["session_id"])
            assert row["owner"] and row["dispatched"]
            namespace = {"__name__": "local_study_modal_receiver"}
            exec(compile(args[2], "local-study-modal-receiver", "exec"), namespace)
            receipt = namespace["runtime_observation"](
                slot["session_id"],
                plan,
                engine_version=settings.engine_version,
                gpu="NVIDIA H100 80GB HBM3, 570.26",
            )
            box = original(*args, **kwargs)
            box.filesystem = SimpleNamespace(read_text=lambda path: json.dumps(receipt))
            return box

        sdk.Sandbox.create = create
        monkeypatch.setitem(sys.modules, "modal", sdk)
        driver = ModalDriver(tmp_path)
    launched = launch_session(
        tmp_path, slot["session_id"], policy, driver=driver, environment=owner
    )
    assert launched["state"] == "running"
    before = len(remote_actions)
    calls = []

    def http(request):
        assert request.headers["authorization"] == "Bearer " + owner["SELFHOST_API_KEY"]
        body = json.loads(request.content)
        assert body["model"] == "qwen3-0.6b-study"
        calls.append(body)
        text = "OK" if "Reply with exactly OK" in str(body["messages"]) else "red"
        logs = (
            {
                "content": [
                    {
                        "token": text,
                        "logprob": math.log(0.5),
                        "bytes": list(text.encode()),
                        "top_logprobs": [
                            {
                                "token": token,
                                "logprob": math.log(0.5),
                                "bytes": list(token.encode()),
                            }
                            for token in ["red", "blue"]
                        ],
                    }
                ]
            }
            if body.get("logprobs")
            else None
        )
        return httpx.Response(
            200,
            json={
                "id": "original-native-fixture",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                        "logprobs": logs,
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    preview = await collect_slot(tmp_path, kit, digest, slot["slot_id"], dry_run=True)
    assert preview["estimate"]["total_reserved_cost_usd"] == "0"
    assert not calls and len(remote_actions) == before
    result = await collect_slot(
        tmp_path,
        kit,
        digest,
        slot["slot_id"],
        approved=True,
        environment=owner,
        transport=httpx.MockTransport(http),
    )
    assert result["state"] == "captured" and len(calls) >= 28
    loaded = load_slot(tmp_path, kit, digest, slot["slot_id"])
    assert loaded["runtime_interval_valid"] is True
    assert loaded["analysis"].plan.reference.tier == "BASELINE"
    assert len(remote_actions) == before
    # Refreshing an observation does not retarget the original capture to a newer receipt.
    journal = SessionJournal(tmp_path, slot["session_id"])
    original_digest = journal.status()["facts"]["runtime_receipt_sha256"]
    copied = json.loads(
        (
            tmp_path
            / "data/raw/selfhost-runtime"
            / journal.directory.name
            / (original_digest + ".json")
        ).read_bytes()
    )
    copied["recorded_at"] = datetime.now(UTC).isoformat()
    next_digest = save_runtime(tmp_path, slot["session_id"], plan, copied)
    journal.record(
        "running",
        {
            **{key: journal.status()["facts"][key] for key in ["remote_id", "endpoint_url"]},
            "runtime_receipt_sha256": next_digest,
        },
    )
    assert (
        load_slot(tmp_path, kit, digest, slot["slot_id"])["binding_hash"] == loaded["binding_hash"]
    )
    stopped = stop_session(tmp_path, slot["session_id"], driver=driver, environment=owner)
    assert stopped["state"] == "reconciled"
    assert load_slot(tmp_path, kit, digest, slot["slot_id"])["samples"] == loaded["samples"]
    book = Ledger(tmp_path)
    book.refresh()
    assert not book.pending and book.development_spend() == 0
    assert book.by_run[slot["session_id"]] == Decimal("1.02")
    assert os.environ.get("SERVEGAUGE_OWNER_RUN") is None
    assert os.environ.get("TRUEWEIGHTS_OWNER_RUN") is None
