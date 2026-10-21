"""Reference publication freezes explicit reusable bytes and rechecks live local evidence."""

import json

import pytest
from typer.testing import CliRunner
from unit.test_refs import reference_run

from servegauge.cli import app
from servegauge.refs.pack import build_pack, load_pack, save_pack
from servegauge.storage.reference_export import (
    reference_preflight,
    stage_reference,
    verify_reference,
)


def source(tmp_path, monkeypatch, *, mock=False):
    run = reference_run()
    if not mock:
        for endpoint in run.endpoints:
            endpoint.kind = "openai_compat"
    pack = build_pack(run, "MOCK" if mock else "R3", seed=8)
    path = save_pack(pack, tmp_path / "refs")
    state = {"binding": "a" * 64, "rights_verified": True, "blockers": []}
    monkeypatch.setattr("servegauge.storage.reference_export.source_state", lambda *args: state)
    return pack, path, state


def test_staged_pack_roundtrip_inventory_and_original_are_immutable(tmp_path, monkeypatch):
    pack, path, _ = source(tmp_path, monkeypatch)
    original = path.read_bytes()
    out = tmp_path / "public-copy"
    staged = stage_reference(tmp_path, pack.pack_id, out)
    assert staged["network_writes"] == 0 and not staged["published"]
    assert staged["publication_blockers"] == []
    manifest = verify_reference(out)
    assert len(manifest["files"]) == 4 and "local-selection.json" not in manifest["files"]
    public_file = next(n for n in manifest["files"] if n.endswith("/pack.json"))
    public = load_pack(str(out / public_file))
    assert public.outcomes == pack.outcomes and public.calibration == pack.calibration
    assert public.pack_id != pack.pack_id and public.manifest["source_pack_id"] == pack.pack_id
    blobs = {n: (out / n).read_bytes() for n in [*manifest["files"], "local-selection.json"]}
    assert stage_reference(tmp_path, pack.pack_id, out) == staged
    assert reference_preflight(tmp_path, pack.pack_id, out) == {
        n: blobs[n] for n in manifest["files"]
    }
    assert path.read_bytes() == original
    assert all((out / n).read_bytes() == b for n, b in blobs.items())


def test_mock_cannot_borrow_real_upload_scope(tmp_path, monkeypatch):
    pack, _, _ = source(tmp_path, monkeypatch, mock=True)
    out = tmp_path / "copy"
    result = stage_reference(tmp_path, pack.pack_id, out)
    assert "scripted_mock_not_public" in result["publication_blockers"]
    with pytest.raises(ValueError):
        reference_preflight(tmp_path, pack.pack_id, out)


@pytest.mark.parametrize(
    "blocker", ["source_rights_unverified", "right_of_reply_required", "source_capture_unavailable"]
)
def test_current_source_or_notice_blockers_cannot_be_bypassed(tmp_path, monkeypatch, blocker):
    pack, _, state = source(tmp_path, monkeypatch)
    out = tmp_path / "copy"
    stage_reference(tmp_path, pack.pack_id, out)
    state["blockers"] = [blocker]
    with pytest.raises(ValueError):
        reference_preflight(tmp_path, pack.pack_id, out)


def test_stale_source_and_extra_files_refuse_without_replacement(tmp_path, monkeypatch):
    pack, _, state = source(tmp_path, monkeypatch)
    out = tmp_path / "copy"
    stage_reference(tmp_path, pack.pack_id, out)
    original = (out / "README.md").read_bytes()
    state["binding"] = "b" * 64
    with pytest.raises(ValueError):
        stage_reference(tmp_path, pack.pack_id, out)
    with pytest.raises(ValueError):
        reference_preflight(tmp_path, pack.pack_id, out)
    assert (out / "README.md").read_bytes() == original
    state["binding"] = "a" * 64
    (out / "extra.txt").write_text("foreign content")
    with pytest.raises(ValueError):
        verify_reference(out)


def test_rehashed_local_forgery_still_needs_exact_regenerated_bytes(tmp_path, monkeypatch):
    import hashlib

    pack, _, _ = source(tmp_path, monkeypatch)
    out = tmp_path / "copy"
    stage_reference(tmp_path, pack.pack_id, out)
    marker = out / "local-selection.json"
    data = json.loads(marker.read_bytes())
    (out / "README.md").write_text("forged card")
    data["files"]["README.md"] = hashlib.sha256(b"forged card").hexdigest()
    marker.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        reference_preflight(tmp_path, pack.pack_id, out)


@pytest.mark.parametrize(
    "name", ["Starter Files", "refs/output", "src/output", ".git/output", "../output", "linked"]
)
def test_protected_or_linked_output_refuses_without_writes(tmp_path, monkeypatch, name):
    pack, _, _ = source(tmp_path, monkeypatch)
    if name == "linked":
        (tmp_path / name).symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        stage_reference(tmp_path, pack.pack_id, tmp_path / name)


def test_cli_dry_run_and_unapproved_confirm_never_import_upload(tmp_path, monkeypatch):
    pack, _, _ = source(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SERVEGAUGE_ALLOW_PUBLISH", raising=False)
    monkeypatch.delenv("TRUEWEIGHTS_ALLOW_PUBLISH", raising=False)
    monkeypatch.setattr(
        "servegauge.adapters.hub.upload_dataset", lambda *a, **k: pytest.fail("upload")
    )
    runner = CliRunner()
    result = runner.invoke(app, ["refs", "publish", pack.pack_id, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["network_writes"] == 0
    refused = runner.invoke(app, ["refs", "publish", pack.pack_id, "--confirm"])
    assert refused.exit_code == 3 and "refused" in refused.output.lower()
    assert "Traceback" not in refused.output


def test_unknown_rights_do_not_get_a_licensed_dataset_card(tmp_path, monkeypatch):
    pack, _, state = source(tmp_path, monkeypatch)
    state["rights_verified"] = False
    out = tmp_path / "copy"
    result = stage_reference(tmp_path, pack.pack_id, out)
    assert "source_rights_unverified" in result["publication_blockers"]
    assert "license: cc-by-4.0" not in (out / "README.md").read_text()


@pytest.mark.parametrize("identifier", ["../pack.json", "Starter Files/pack.json", "ref-invalid"])
def test_invalid_identifier_does_not_discover_or_open_any_pack(tmp_path, monkeypatch, identifier):
    monkeypatch.setattr(
        "servegauge.storage.reference_export.load_pack", lambda *a: pytest.fail("pack read")
    )
    with pytest.raises(ValueError):
        stage_reference(tmp_path, identifier, tmp_path / "copy")
    assert not (tmp_path / "copy").exists()


def test_missing_capture_cannot_gain_rights_from_a_valid_local_pack(tmp_path):
    from servegauge.storage.reference_export import source_state

    run = reference_run()
    for endpoint in run.endpoints:
        endpoint.kind = "openai_compat"
    pack = build_pack(run, "R3", seed=8)
    actual = source_state(tmp_path, pack)
    assert not actual["rights_verified"]
    assert {"source_capture_unavailable", "source_rights_unverified"} <= set(actual["blockers"])


def test_mocked_owner_adapter_gets_exact_frozen_inventory_and_safe_failure(tmp_path, monkeypatch):
    from pathlib import Path

    import yaml

    pack, _, _ = source(tmp_path, monkeypatch)
    settings = yaml.safe_load(Path("configs/observatory.yaml").read_text())
    settings["publishing"] = {
        "enabled": True,
        "repo_id": "owner/reference-fixture",
        "credential_env": "REFERENCE_TEST_TOKEN",
        "license": "cc-by-4.0",
    }
    config = tmp_path / "publication.yaml"
    config.write_text(yaml.safe_dump(settings))
    out = tmp_path / "copy"
    stage_reference(tmp_path, pack.pack_id, out)
    expected = reference_preflight(tmp_path, pack.pack_id, out)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SERVEGAUGE_ALLOW_PUBLISH", "1")
    monkeypatch.setenv("REFERENCE_TEST_TOKEN", "fixture-review-token")
    calls = []

    def upload(repo, token, payloads, **kwargs):
        calls.append((repo, token, payloads, kwargs))
        return "a" * 40

    monkeypatch.setattr("servegauge.adapters.hub.upload_dataset", upload)
    args = [
        "refs",
        "publish",
        pack.pack_id,
        "--out",
        str(out),
        "--config",
        str(config),
        "--confirm",
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["upload_commits"] == 1
    assert len(calls) == 1 and calls[0][0] == "owner/reference-fixture"
    assert calls[0][2] == expected and "local-selection.json" not in calls[0][2]
    assert calls[0][3]["commit_message"] == "Add reviewed ServeGauge reusable reference"

    def failed(*args, **kwargs):
        raise ValueError("fixture-review-token unsafe upstream context")

    monkeypatch.setattr("servegauge.adapters.hub.upload_dataset", failed)
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 3 and "fixture-review-token" not in result.output
    assert "unsafe upstream" not in result.output and "Traceback" not in result.output


@pytest.mark.parametrize(
    "mode", ["valid", "external", "missing", "context", "numeric", "fingerprint"]
)
def test_capture_rights_and_measurement_binding_component(tmp_path, monkeypatch, mode):
    from types import SimpleNamespace

    from servegauge.core.hashing import content_hash
    from servegauge.core.ids import safe_filename
    from servegauge.core.models import Item
    from servegauge.storage.reference_export import source_state

    run = reference_run()
    directory = tmp_path / "data/raw" / safe_filename(run.run_id)
    directory.mkdir(parents=True)
    sealed = {"run_ids": [run.run_id], "metadata": {}}
    (directory / "analysis.json").write_text(
        json.dumps({"value": sealed, "hash": content_hash(sealed)})
    )
    for endpoint in run.endpoints:
        endpoint.kind = "openai_compat"
    pack = build_pack(run, "R3", seed=8)
    items = [
        Item(item_id=f"item{i:02}", suite_id="S2", version="fixture", payload={}, expected={})
        for i in range(50)
    ]
    if mode == "external":
        items[0].source = "external-unreviewed"
    if mode == "missing":
        items.pop()
    exported = run.model_dump(mode="json")
    if mode == "context":
        exported["configuration"] = {"other": "context"}
    if mode == "numeric":
        exported["checks"]["T02"]["b"]["item00"] = [0.125]
    if mode == "fingerprint":
        pack.fingerprints = {
            "views": {
                "a": {
                    "references": {
                        "b": {
                            "FP2": {
                                "evidence": ["unbound"],
                                "details": {
                                    "observations": [],
                                    "observation_hash": content_hash([]),
                                },
                            }
                        }
                    }
                }
            }
        }
    analysis = SimpleNamespace(
        plan=SimpleNamespace(endpoints=run.endpoints),
        items=items,
        records=[],
        samples={},
        checks=[],
        manifest={},
    )
    monkeypatch.setattr("servegauge.storage.reference_export.reconstruct", lambda *a: analysis)
    monkeypatch.setattr("servegauge.storage.reference_export.export_outcomes", lambda *a: exported)
    actual = source_state(tmp_path, pack)
    assert actual["rights_verified"] is (mode == "valid")
    assert bool(actual["blockers"]) is (mode != "valid")


def test_consensus_label_does_not_hide_mock_endpoint_declarations(tmp_path, monkeypatch):
    source(tmp_path, monkeypatch, mock=True)
    consensus_mock = build_pack(reference_run(), "R3", seed=8)
    save_pack(consensus_mock, tmp_path / "refs")
    out = tmp_path / "consensus-copy"
    result = stage_reference(tmp_path, consensus_mock.pack_id, out)
    assert "scripted_mock_not_public" in result["publication_blockers"]
    with pytest.raises(ValueError):
        reference_preflight(tmp_path, consensus_mock.pack_id, out)


def test_linked_supporting_capture_refuses_before_reconstruction(tmp_path, monkeypatch):
    from servegauge.core.hashing import content_hash
    from servegauge.core.ids import safe_filename
    from servegauge.storage.reference_export import source_state

    run = reference_run()
    pack = build_pack(run, "R3", seed=8)
    raw = tmp_path / "data/raw"
    directory = raw / safe_filename(run.run_id)
    directory.mkdir(parents=True)
    other = tmp_path / "other-data"
    other.mkdir()
    (raw / safe_filename("linked-support")).symlink_to(other, target_is_directory=True)
    sealed = {"run_ids": [run.run_id], "metadata": {"capability_run_id": "linked-support"}}
    (directory / "analysis.json").write_text(
        json.dumps({"value": sealed, "hash": content_hash(sealed)})
    )
    monkeypatch.setattr(
        "servegauge.storage.reference_export.reconstruct",
        lambda *a: pytest.fail("linked capture read"),
    )
    actual = source_state(tmp_path, pack)
    assert not actual["rights_verified"] and "source_capture_unavailable" in actual["blockers"]
    assert not list(other.iterdir())
