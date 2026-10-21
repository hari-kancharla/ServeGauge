"""Actual zero-cost single-slot collection, immutable intents and raw-only binding."""

import json
from pathlib import Path

import pytest
from unit.test_gpu_study_protocol import specification

from servegauge.core.hashing import content_hash
from servegauge.core.ids import safe_filename
from servegauge.runner.ledger import Ledger
from servegauge.storage.gpu_capture import collect_slot, load_slot
from servegauge.storage.gpu_collection import prepare_collection_study
from servegauge.storage.workspace import initialize


def setup_kit(tmp_path, monkeypatch, *, budget_policy=None, **changes):
    initialize(tmp_path)
    if budget_policy is not None:
        (tmp_path / "configs/budgets.yaml").write_text(
            json.dumps(budget_policy.model_dump(mode="json"))
        )
    monkeypatch.setenv(
        "SERVEGAUGE_TOKENIZER_CACHE",
        str(Path(__file__).absolute().parents[2] / ".cache/tokenizers"),
    )
    monkeypatch.chdir(tmp_path)
    spec = specification(**changes)
    out = tmp_path / ".cache/gpu-studies/full"
    value = prepare_collection_study(tmp_path, spec, out=out)
    matrix = json.loads((out / "matrix.json").read_bytes())
    return spec, out, content_hash(value), matrix


async def test_actual_mock_slot_uses_normal_reservations_reconstructs_and_is_idempotent(
    tmp_path, monkeypatch
):
    spec, kit, digest, matrix = setup_kit(tmp_path, monkeypatch)
    slot = matrix[0]
    preview = await collect_slot(tmp_path, kit, digest, slot["slot_id"], dry_run=True)
    assert preview["state"] == "dry_run" and preview["estimate"]["total_reserved_cost_usd"] == "0"
    assert not (tmp_path / "data").exists()
    with pytest.raises(ValueError):
        await collect_slot(tmp_path, kit, digest, slot["slot_id"])
    result = await collect_slot(tmp_path, kit, digest, slot["slot_id"], approved=True)
    assert result["state"] == "captured" and result["physical_GPU_sessions"] == 0
    capture = load_slot(tmp_path, kit, digest, slot["slot_id"])
    assert capture is not None and capture["analysis"].plan.profile == "qwen3-0.6b-study"
    assert len([r for r in capture["analysis"].records if r.suite_id == "S6"]) == 28
    assert set(capture["samples"]) >= {"FP1", "FP2", "FP3", "FP4"}
    book = Ledger(tmp_path)
    book.refresh()
    assert not book.pending and book.development_spend() == 0
    before = (tmp_path / "data/ledger/spend.jsonl").read_bytes()
    again = await collect_slot(tmp_path, kit, digest, slot["slot_id"], approved=True)
    assert again["state"] == "already_captured" and again["binding_hash"] == result["binding_hash"]
    assert (tmp_path / "data/ledger/spend.jsonl").read_bytes() == before
    # Derived caches are irrelevant; selected raw inputs remain authoritative.
    (tmp_path / "data/runs" / safe_filename(slot["run_id"]) / "checks.json").write_text(
        "corrupt cache"
    )
    assert load_slot(tmp_path, kit, digest, slot["slot_id"])["samples"] == capture["samples"]


async def test_interrupted_intent_requires_explicit_same_run_resume(tmp_path, monkeypatch):
    import servegauge.storage.gpu_capture as capture

    spec, kit, digest, matrix = setup_kit(tmp_path, monkeypatch)
    slot = matrix[0]
    original = capture.execute_plan

    async def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(capture, "execute_plan", interrupt)
    with pytest.raises(KeyboardInterrupt):
        await collect_slot(tmp_path, kit, digest, slot["slot_id"], approved=True)
    assert load_slot(tmp_path, kit, digest, slot["slot_id"]) is None
    with pytest.raises(ValueError):
        await collect_slot(tmp_path, kit, digest, slot["slot_id"], approved=True)
    monkeypatch.setattr(capture, "execute_plan", original)
    result = await collect_slot(tmp_path, kit, digest, slot["slot_id"], approved=True, resume=True)
    assert result["state"] == "captured"
    assert load_slot(tmp_path, kit, digest, slot["slot_id"])["binding"]["run_id"] == slot["run_id"]


async def test_raw_change_and_unknown_slot_refuse_without_new_dispatch(tmp_path, monkeypatch):
    spec, kit, digest, matrix = setup_kit(tmp_path, monkeypatch)
    slot = matrix[0]
    with pytest.raises(ValueError):
        await collect_slot(tmp_path, kit, digest, "unknown", approved=True)
    assert not (tmp_path / "data").exists()
    await collect_slot(tmp_path, kit, digest, slot["slot_id"], approved=True)
    namespace = tmp_path / "data/raw" / safe_filename(slot["run_id"])
    path = namespace / "capture.json"
    path.write_text("{}")
    before = (tmp_path / "data/ledger/spend.jsonl").read_bytes()
    with pytest.raises(ValueError):
        load_slot(tmp_path, kit, digest, slot["slot_id"])
    with pytest.raises(ValueError):
        await collect_slot(tmp_path, kit, digest, slot["slot_id"], approved=True)
    assert (tmp_path / "data/ledger/spend.jsonl").read_bytes() == before


async def test_owner_collection_cannot_use_mock_scope_or_skip_real_session_guards(
    tmp_path, monkeypatch
):
    spec, kit, digest, matrix = setup_kit(tmp_path, monkeypatch, scope="owner_gpu")
    slot = matrix[0]
    with pytest.raises(ValueError):
        await collect_slot(tmp_path, kit, digest, slot["slot_id"], approved=True, environment={})
    assert not (tmp_path / "data").exists()
