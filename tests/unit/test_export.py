"""Publishing inventory is a numeric projection, never a copy of private tables."""

import copy
import json

import polars as pl
import pytest
from typer.testing import CliRunner
from unit.test_endpoint_report import PRIVATE, fixture

from servegauge.cli import app
from servegauge.core.models import Item
from servegauge.scoring.common import Samples
from servegauge.storage.export import publication_preflight, stage_export, verify_export
from servegauge.storage.tables import TABLE_NAMES


def parquet(out, month, name):
    return pl.read_parquet(next((out / "data" / month).glob(name + "-*.parquet")))


def selected(monkeypatch):
    analysis = fixture()
    analysis.items = [
        Item(
            item_id=f"probe-{i}",
            suite_id="S1",
            version="fixture",
            payload={"text": PRIVATE},
            expected={},
        )
        for i in range(4)
    ]
    analysis.samples = {
        "mock:target": {
            "P01": Samples(check_id="P01", check_class="invariant", values={"probe-0": 1.0})
        }
    }
    monkeypatch.setattr("servegauge.storage.export.reconstruct", lambda raw, run: analysis)
    return analysis


def test_empty_dry_run_has_ten_typed_tables_and_never_discovers_private_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "servegauge.storage.export.reconstruct", lambda *a: pytest.fail("private discovery")
    )
    result = stage_export(tmp_path, tmp_path / "export")
    assert result["network_writes"] == 0 and not result["published"]
    assert result["publication_blockers"] == ["no_selected_runs"]
    manifest = verify_export(tmp_path / "export")
    assert len(manifest["files"]) == 13
    for table in TABLE_NAMES:
        assert parquet(tmp_path / "export", "empty", table).height == 0
    assert any(n.endswith("/DATASHEET.md") for n in manifest["files"])
    assert "local-selection.json" not in manifest["files"]


def test_monthly_numeric_projection_preserves_exact_checks_and_removes_private_fields(
    tmp_path, monkeypatch
):
    analysis = selected(monkeypatch)
    out = tmp_path / "export"
    result = stage_export(tmp_path, out, runs=["report-run"])
    assert "scripted_mock_not_public" in result["publication_blockers"]
    checks = parquet(out, "2026-10", "checks")
    row = checks.filter(pl.col("check_id") == "P01").row(0, named=True)
    assert row["n"] == 4 and row["violations"] == 4 and row["ci_low"] == 0.5101
    assert row["estimate"] == 1 and row["verdict"] == "FAIL"
    assert parquet(out, "2026-10", "exchanges").height == 4
    for name in verify_export(out)["files"]:
        blob = (out / name).read_bytes()
        assert PRIVATE.encode() not in blob and b"mock:target" not in blob
        if name.endswith(".parquet"):
            text = json.dumps(pl.read_parquet(out / name).to_dicts())
            assert PRIVATE not in text and "request" not in pl.read_parquet(out / name).columns
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    assert stage_export(tmp_path, out, runs=[analysis.manifest["run_id"]]) == result
    assert all(p.read_bytes() == b for p, b in before.items())


def test_external_items_are_aggregate_only(tmp_path, monkeypatch):
    analysis = selected(monkeypatch)
    for item in analysis.items:
        item.source = "external-gated-fixture"
    out = tmp_path / "export"
    stage_export(tmp_path, out, runs=["report-run"])
    assert parquet(out, "2026-10", "checks").height == 3
    for table in ["exchanges", "item_scores", "costs"]:
        assert parquet(out, "2026-10", table).height == 0


def test_changed_or_extra_output_and_stale_source_refuse_without_replacing_files(
    tmp_path, monkeypatch
):
    analysis = selected(monkeypatch)
    out = tmp_path / "export"
    stage_export(tmp_path, out, runs=["report-run"])
    original = (out / "README.md").read_bytes()
    (out / "extra.txt").write_text("not reviewed")
    with pytest.raises(ValueError):
        verify_export(out)
    (out / "extra.txt").unlink()
    analysis.checks[0].n = 5
    with pytest.raises(ValueError):
        stage_export(tmp_path, out, runs=["report-run"])
    assert (out / "README.md").read_bytes() == original


def test_mock_upload_is_blocked_even_after_local_staging(tmp_path, monkeypatch):
    selected(monkeypatch)
    out = tmp_path / "export"
    stage_export(tmp_path, out, runs=["report-run"])
    with pytest.raises(ValueError, match="publication"):
        publication_preflight(tmp_path, out)


def test_real_selection_needs_current_notice_and_unchanged_public_bytes(tmp_path, monkeypatch):
    from servegauge.core.hashing import content_hash

    analysis = selected(monkeypatch)
    analysis.plan.endpoints[0].kind = "openai_compat"
    analysis.plan.reference.tier = "BASELINE"
    analysis.metadata["synthetic_mock"] = False
    out = tmp_path / "export"
    result = stage_export(tmp_path, out, runs=["report-run"])
    assert result["publication_blockers"] == ["right_of_reply_required"]
    # Explicit synthetic policy fixture, not a fabricated provider notice.
    monkeypatch.setattr(
        "servegauge.storage.export.publication_status", lambda *a, **k: {"eligible": True}
    )
    payloads = publication_preflight(tmp_path, out)
    assert "local-selection.json" not in payloads
    assert len(payloads) == 13 and all(isinstance(b, bytes) for b in payloads.values())
    replacement = (out / "README.md").read_bytes() + b"unreviewed alteration"
    (out / "README.md").write_bytes(replacement)
    manifest = json.loads((out / "local-selection.json").read_text())
    import hashlib

    manifest["files"]["README.md"] = hashlib.sha256(replacement).hexdigest()
    (out / "local-selection.json").write_text(json.dumps(manifest))
    verify_export(out)  # Hash consistency alone is insufficient.
    with pytest.raises(ValueError):
        publication_preflight(tmp_path, out)
    assert content_hash(payloads["README.md"].hex()) != content_hash(replacement.hex())


def test_openrouter_public_route_does_not_export_account_metadata(tmp_path, monkeypatch):
    from servegauge.core.models import OpenRouterPin

    analysis = selected(monkeypatch)
    endpoint = analysis.plan.endpoints[0]
    endpoint.kind = "openrouter"
    endpoint.model = "openai/gpt-oss-120b"
    endpoint.openrouter_pin = OpenRouterPin(tag="deepinfra")
    endpoint.metadata = {"account_id": PRIVATE, "upstream_id": PRIVATE}
    out = tmp_path / "export"
    stage_export(tmp_path, out, runs=["report-run"])
    row = parquet(out, "2026-10", "endpoints").row(0, named=True)
    assert row["declared_provider"] == "deepinfra" and row["declared_model"] == endpoint.model
    assert PRIVATE not in json.dumps(row)


def test_confirmed_cli_passes_only_frozen_allowlisted_bytes_to_fake_adapter(tmp_path, monkeypatch):
    import yaml

    analysis = selected(monkeypatch)
    analysis.plan.endpoints[0].kind = "openai_compat"
    analysis.plan.reference.tier = "BASELINE"
    analysis.metadata["synthetic_mock"] = False
    for check in analysis.checks:
        if check.verdict == "FAIL":
            check.verdict = "INCONCLUSIVE"
    out = tmp_path / "export"
    stage_export(tmp_path, out, runs=["report-run"])
    config = tmp_path / "owner.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "id": "owner-publish-fixture",
                "mode": "owner",
                "publishing": {
                    "enabled": True,
                    "repo_id": "fixture-owner/fixture-data",
                    "credential_env": "FIXTURE_HUB_CREDENTIAL",
                },
            }
        )
    )
    seen = []

    def fake_upload(repo, credential, payloads):
        assert repo == "fixture-owner/fixture-data" and credential == "fixture-credential"
        assert "local-selection.json" not in payloads and len(payloads) == 13
        assert all(isinstance(b, bytes) for b in payloads.values())
        seen.append(payloads)
        return "a" * 40

    monkeypatch.setattr("servegauge.adapters.hub.upload_dataset", fake_upload)
    monkeypatch.setenv("SERVEGAUGE_ALLOW_PUBLISH", "1")
    monkeypatch.setenv("FIXTURE_HUB_CREDENTIAL", "fixture-credential")
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app, ["publish", "hf", "--confirm", "--out", str(out), "--config", str(config)]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["published"] and len(seen) == 1
    assert "fixture-credential" not in result.output


def test_two_months_remain_separate_and_only_explicit_runs_are_reconstructed(tmp_path, monkeypatch):
    first = selected(monkeypatch)
    second = copy.deepcopy(first)
    second.manifest.update(run_id="second-run", created_at="2026-11-01T00:00:00+00:00")
    for check in second.checks:
        check.run_id = "second-run"
    seen = []

    def reconstruct(raw, run):
        seen.append(run)
        return {"report-run": first, "second-run": second}[run]

    monkeypatch.setattr("servegauge.storage.export.reconstruct", reconstruct)
    out = tmp_path / "export"
    stage_export(tmp_path, out, runs=["second-run", "report-run", "report-run"])
    assert seen == ["report-run", "second-run"]
    assert len(list((out / "data").glob("*"))) == 2
    assert parquet(out, "2026-11", "checks").height == 3


def test_new_snapshot_preserves_old_paths_and_card_selects_no_duplicate_snapshots(
    tmp_path, monkeypatch
):
    import yaml

    analysis = selected(monkeypatch)
    first, second = tmp_path / "first", tmp_path / "second"
    old = stage_export(tmp_path, first, runs=["report-run"])
    analysis.checks[0].n += 1
    new = stage_export(tmp_path, second, runs=["report-run"])
    assert set(old["files"]) & set(new["files"]) == {"README.md"}
    text = (second / "README.md").read_text()
    card = yaml.safe_load(text.split("---")[1])
    for config in card["configs"]:
        paths = config["data_files"][0]["path"]
        assert all(p in new["files"] and "*" not in p for p in paths)
        assert all((second / p).is_file() for p in paths)


def test_secret_in_public_route_is_rejected_before_creating_output(tmp_path, monkeypatch):
    from servegauge.core.models import OpenRouterPin

    analysis = selected(monkeypatch)
    marker = "fixture-route-private-sentinel"
    monkeypatch.setenv("FIXTURE_SECRET_TOKEN", marker)
    endpoint = analysis.plan.endpoints[0]
    endpoint.kind = "openrouter"
    endpoint.model = "vendor/" + marker
    endpoint.openrouter_pin = OpenRouterPin(tag="fixture-provider")
    out = tmp_path / "export"
    with pytest.raises(ValueError, match="secret"):
        stage_export(tmp_path, out, runs=["report-run"])
    assert not out.exists()


def test_cli_default_is_offline_dry_run_and_confirmation_needs_owner_gates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli = CliRunner()
    result = cli.invoke(app, ["publish", "hf", "--dry-run", "--out", "export"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["network_writes"] == 0
    result = cli.invoke(app, ["publish", "hf", "--confirm", "--out", "export"])
    assert result.exit_code != 0 and "Traceback" not in result.output


@pytest.mark.parametrize(
    "name", ["Starter Files/blocked", "linked/out", "data/raw/export", "src/export", "../export"]
)
def test_protected_or_symlinked_output_is_refused_before_writing(tmp_path, name):
    if name.startswith("linked"):
        (tmp_path / "linked").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        stage_export(tmp_path, tmp_path / name)
