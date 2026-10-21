"""Static site presentation and local preview guards, without opening sockets."""

import copy
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner
from unit.test_endpoint_report import PRIVATE, files, fixture

from servegauge.cli import app
from servegauge.report.site.builder import build_site, site_context, verify_site
from servegauge.report.site.server import site_file


def prepare(root):
    (root / "docs/validation").mkdir(parents=True)
    (root / "docs/METHODS.md").write_text("# Methods\n<unsafe> [text](javascript:alert(1))\n")
    (root / "docs/validation/known.json").write_text(
        json.dumps({"accepted_component": "synthetic fixture only", "source_tests_passed": 17})
    )


def test_site_context_does_not_turn_counts_into_quality_or_missing_cost_into_zero():
    analysis = fixture()
    before = copy.deepcopy(analysis)
    context = site_context([analysis])
    assert context["run_count"] == 1 and len(context["endpoints"]) == 1
    endpoint = context["endpoints"][0]
    assert endpoint["counts"]["FAIL"] == 1 and endpoint["counts"]["INCONCLUSIVE"] == 1
    assert endpoint["cost"]["reported_usd"] is None
    assert endpoint["cost"]["known_records"] == 0 and endpoint["cost"]["sent_records"] == 4
    assert endpoint["change_events"] is None
    assert endpoint["endpoint_id"] == analysis.plan.endpoints[0].id
    assert PRIVATE not in json.dumps(context)
    assert analysis == before


def test_site_retains_fingerprint_band_positions_and_unsupported_missing_values():
    analysis = fixture()
    fp3 = next(c for c in analysis.checks if c.check_id == "FP3")
    fp3.details = {
        "fingerprint_position": "between-bands",
        "noise_band": 0.02,
        "fingerprint_metrics": {"FP3": {"different_model_band": [0.2, 0.3]}},
    }
    row = site_context([analysis])["endpoints"][0]["fingerprints"][0]
    assert row["n"] == 0 and row["estimate_text"] == "unavailable"
    assert row["position"] == "between-bands"
    assert row["noise_text"] == "0.02" and row["different_text"] == "[0.2, 0.3]"


@pytest.mark.parametrize("metrics", [None, [], {"FP3": None}, {"FP3": {"control_positions": None}}])
def test_absent_or_unrecognized_fingerprint_details_remain_unavailable(metrics):
    analysis = fixture()
    fp3 = next(c for c in analysis.checks if c.check_id == "FP3")
    fp3.details = {"fingerprint_metrics": metrics}
    row = site_context([analysis])["endpoints"][0]["fingerprints"][0]
    assert row["position"] == row["noise_text"] == row["different_text"] == "unavailable"
    assert row["controls"] == []


def test_static_site_build_is_redacted_offline_linked_and_idempotent(tmp_path, monkeypatch):
    prepare(tmp_path)
    analysis = fixture()
    monkeypatch.setattr("servegauge.report.site.builder.reconstruct", lambda raw, run: analysis)
    out = tmp_path / "site"
    result = build_site(tmp_path, out, runs=[analysis.manifest["run_id"]])
    assert result["runs"] == 1 and result["endpoints"] == 1
    checked = verify_site(out)
    assert checked["redact_content"] and checked["publication_ready"] is False
    for path in out.rglob("*"):
        if path.is_file():
            assert PRIVATE not in path.read_text(), path.name
    html = (out / "index.html").read_text()
    assert '<meta name="viewport"' in html and "aria-label=" in html
    assert 'id="model-filter"' in html and "Provider / suite matrix" in html
    assert (out / "methods.html").is_file() and (out / "validation.html").is_file()
    assert (out / "timeline.html").is_file() and (out / "replies.html").is_file()
    assert "&lt;unsafe&gt;" in (out / "methods.html").read_text()
    assert "<unsafe>" not in (out / "methods.html").read_text()
    assert 'href="javascript:' not in (out / "methods.html").read_text()
    assert "No calibrated change events" in (out / "timeline.html").read_text()
    before = files(out)
    assert build_site(tmp_path, out, runs=[analysis.manifest["run_id"]]) == result
    assert files(out) == before
    (out / "owner-edit.txt").write_text("keep")
    changed = files(out)
    with pytest.raises(ValueError, match="site_output"):
        build_site(tmp_path, out, runs=[analysis.manifest["run_id"]])
    assert files(out) == changed


def test_empty_site_has_honest_no_data_and_never_discovers_private_runs(tmp_path, monkeypatch):
    prepare(tmp_path)
    monkeypatch.setattr(
        "servegauge.report.site.builder.reconstruct",
        lambda *args: pytest.fail("No implicit private-store read"),
    )
    out = tmp_path / "empty"
    assert build_site(tmp_path, out)["runs"] == 0
    assert "No recorded runs selected" in (out / "index.html").read_text()
    assert verify_site(out)["redact_content"]


def test_site_verification_refuses_changed_missing_and_symlinked_files(tmp_path):
    prepare(tmp_path)
    out = tmp_path / "site"
    build_site(tmp_path, out)
    original = (out / "index.html").read_bytes()
    (out / "index.html").write_bytes(original + b"owner edit")
    with pytest.raises(ValueError, match="site_output"):
        verify_site(out)
    (out / "index.html").write_bytes(original)
    (out / "assets/site.js").unlink()
    with pytest.raises(ValueError, match="site_output"):
        verify_site(out)
    (out / "assets/site.js").symlink_to(tmp_path / "docs/METHODS.md")
    with pytest.raises(ValueError, match="site_output"):
        verify_site(out)


@pytest.mark.parametrize(
    "url",
    [
        "/../private",
        "/%2e%2e/private",
        "/%252e%252e/private",
        "//outside/index.html",
        "/assets/",
        "/owner.txt",
        "/assets\\secret",
        "/%00",
        "https://example.test/index.html",
    ],
)
def test_preview_only_serves_exact_manifest_files(tmp_path, url):
    out = tmp_path / "site"
    out.mkdir()
    (out / "index.html").write_text("safe")
    (out / "owner.txt").write_text("private")
    assert site_file(out, url, {"index.html": "hash"}) is None
    assert site_file(out, "/", {"index.html": "hash"}) == out / "index.html"


def test_site_cli_build_and_preview_dry_run_are_real_offline_paths(tmp_path, monkeypatch):
    prepare(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    built = runner.invoke(app, ["site", "build"])
    assert built.exit_code == 0, built.output
    assert json.loads(built.output)["runs"] == 0
    preview = runner.invoke(app, ["site", "serve", "--dry-run"])
    assert preview.exit_code == 0 and "127.0.0.1" in preview.output
    assert runner.invoke(app, ["site", "serve", "--port", "-1", "--dry-run"]).exit_code != 0
    assert runner.invoke(app, ["site", "serve", "--out", "missing", "--dry-run"]).exit_code == 5


def test_site_does_not_pool_same_endpoint_across_captures_and_separates_known_charges():
    first = fixture()
    second = copy.deepcopy(first)
    second.manifest["run_id"] = "second-run"
    second.records[0].cost_source = "provider"
    second.records[0].cost_usd = 0.1
    second.records[1].cost_source = "estimate"
    second.records[1].cost_usd = 999
    context = site_context([first, second])
    assert len(context["endpoints"]) == 2 and context["run_count"] == 2
    assert len({e["href"] for e in context["endpoints"]}) == 2
    assert all(e["counts"]["FAIL"] == 1 for e in context["endpoints"])
    later = next(e for e in context["endpoints"] if e["run_id"] == "second-run")
    assert later["cost"]["known_records"] == 1 and later["cost"]["reported_usd"] == "0.1"


def test_publication_preflight_requires_owner_flags_and_never_approves_real_captures(
    tmp_path, monkeypatch
):
    prepare(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    assert runner.invoke(app, ["site", "build"]).exit_code == 0
    assert runner.invoke(app, ["site", "verify"]).exit_code == 0
    for variable in ["SERVEGAUGE_ALLOW_PUBLISH", "TRUEWEIGHTS_ALLOW_PUBLISH"]:
        monkeypatch.delenv(variable, raising=False)
    assert runner.invoke(app, ["site", "verify", "--for-publication", "--confirm"]).exit_code == 4
    monkeypatch.setenv("SERVEGAUGE_ALLOW_PUBLISH", "1")
    assert runner.invoke(app, ["site", "verify", "--for-publication"]).exit_code == 4
    permitted = runner.invoke(app, ["site", "verify", "--for-publication", "--confirm"])
    assert permitted.exit_code == 0 and json.loads(permitted.output)["network_writes"] == 0
    analysis = fixture()
    analysis.plan.endpoints[0].kind = "openai_compat"
    monkeypatch.setattr("servegauge.report.site.builder.reconstruct", lambda raw, run: analysis)
    private_out = tmp_path / "provider"
    build_site(tmp_path, private_out, runs=[analysis.manifest["run_id"]])
    refused = runner.invoke(
        app, ["site", "verify", "--out", str(private_out), "--for-publication", "--confirm"]
    )
    assert refused.exit_code == 5 and "reply" in refused.output


def test_preview_refuses_symlinked_root_or_nested_asset(tmp_path):
    out = tmp_path / "site"
    out.mkdir()
    victim = tmp_path / "private"
    victim.mkdir()
    (victim / "index.html").write_text("private")
    root_link = tmp_path / "link"
    root_link.symlink_to(victim, target_is_directory=True)
    assert site_file(root_link, "/", {"index.html": "hash"}) is None
    (out / "assets").symlink_to(victim, target_is_directory=True)
    assert site_file(out, "/assets/index.html", {"assets/index.html": "hash"}) is None


def test_pages_workflow_is_manual_disabled_by_default_and_has_separate_deploy_permissions():
    import yaml

    root = Path(__file__).resolve().parents[2]
    assert not (root / ".github/workflows/pages.yml").exists()
    workflow = yaml.safe_load((root / "configs/workflows/pages.yml.example").read_text())
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["on"]["workflow_dispatch"]["inputs"]["publish"]["default"] is False
    for job in workflow["jobs"].values():
        assert "PAGES_ENABLED == 'true'" in job["if"] and "inputs.publish" in job["if"]
    assert workflow["jobs"]["deploy"]["needs"] == "build"
    assert workflow["jobs"]["deploy"]["environment"]["name"] == "github-pages"
    assert workflow["jobs"]["deploy"]["permissions"] == {
        "contents": "read",
        "pages": "write",
        "id-token": "write",
    }
    steps = workflow["jobs"]["build"]["steps"]
    assert any("--for-publication --confirm" in step.get("run", "") for step in steps)
    build = next(step["run"] for step in steps if "site build" in step.get("run", ""))
    assert "--run" not in build  # No implicit private captures in the documentation workflow.
