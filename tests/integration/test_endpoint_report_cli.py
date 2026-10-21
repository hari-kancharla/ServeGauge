"""Exercise the report command on actual reserved ASGI captures, without sockets."""

import hashlib
import json
import shutil
from html.parser import HTMLParser
from pathlib import Path

import httpx
from typer.testing import CliRunner

from servegauge.cli import app
from servegauge.diagnose.rules import diagnoses
from servegauge.mock.faults import FaultConfig
from servegauge.mock.server import create_app
from servegauge.report.site.builder import verify_site
from servegauge.runner.check import run_check
from servegauge.runner.endpoints import resolve_endpoint
from servegauge.runner.plan import RunPlan
from servegauge.storage.analysis import reconstruct


def hashes(directory):
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(v for k, v in attrs if k in {"href", "src"} and v)


async def test_native_report_cli_preserves_inputs_rebuilds_results_and_links_local_bundles(
    tmp_path, monkeypatch
):
    (tmp_path / "configs").mkdir()
    for name in ["budgets.yaml", "suites.yaml"]:
        shutil.copyfile(Path("configs") / name, tmp_path / "configs" / name)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/METHODS.md").write_text("# Recorded methods\n", encoding="utf-8")
    endpoint = resolve_endpoint("http://127.0.0.1:8011/v1", "mock", "gpt-oss-120b")
    marker = "PRIVATE_PROVIDER_LABEL_REPORT_CANARY"
    endpoint.labels["provider_name"] = marker
    plan = RunPlan(
        plan_id="native-report",
        profile="gpt-oss-120b",
        level="smoke",
        seed=1303,
        endpoints=[endpoint],
        suites=["S1"],
        reference={"tier": "MOCK"},
        max_usd=0,
        requests_per_minute=60000,
        overrides={"S1": {"invariant_trials": 4}},
    )
    result = await run_check(
        plan,
        tests="P01",
        root=tmp_path,
        approved=True,
        transports={
            endpoint.id: httpx.ASGITransport(
                app=create_app(faults=FaultConfig(ignore_max_tokens=True))
            )
        },
    )
    expected = reconstruct(tmp_path / "data/raw", result.run_id)
    before = {name: hashes(tmp_path / name) for name in ["data", "reports", "bundles", "budget"]}
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    for mode in ["private", "redacted"]:
        out = tmp_path / mode
        args = ["report", "--run", result.run_id, "--out", str(out)]
        if mode == "redacted":
            args.append("--redact-content")
        response = runner.invoke(app, args)
        assert response.exit_code == 0, response.output
        report = json.loads(response.output)
        assert report["checks"] == len(expected.checks)
        assert report["diagnoses"] == [
            d.model_dump(mode="json") for d in diagnoses(expected.checks)
        ]
        assert "PARAM_IGNORED" in {d["code"] for d in report["diagnoses"]}
        data = json.loads((out / "checks.json").read_text())
        for actual, original in zip(data["checks"], expected.checks, strict=True):
            original = original.model_dump(mode="json")
            if mode == "redacted":
                actual.pop("details")
                original.pop("details")
            assert actual == original
        parser = Links()
        parser.feed((out / "index.html").read_text())
        local = [link for link in parser.links if not link.startswith(("#", "data:"))]
        assert "methods.md" in local and any(link.endswith("/README.md") for link in local)
        assert all((out / link).is_file() for link in local)
        assert all("://" not in link for link in local)
        if mode == "redacted":
            assert all(marker not in p.read_text() for p in out.rglob("*") if p.is_file())
        saved = hashes(out)
        repeated = runner.invoke(app, args)
        assert repeated.exit_code == 0 and hashes(out) == saved
    assert before == {
        name: hashes(tmp_path / name) for name in ["data", "reports", "bundles", "budget"]
    }
    site = tmp_path / "static-site"
    site_args = ["site", "build", "--run", result.run_id, "--out", str(site)]
    built = runner.invoke(app, site_args)
    assert built.exit_code == 0, built.output
    manifest = verify_site(site)
    assert manifest["runs"] == [result.run_id] and manifest["redact_content"]
    index = json.loads((site / "site.json").read_text())
    assert len(index["endpoints"]) == 1
    assert index["endpoints"][0]["counts"]["FAIL"] == 1
    for html_file in site.rglob("*.html"):
        parser = Links()
        parser.feed(html_file.read_text())
        assert all(
            (html_file.parent / link).is_file()
            for link in parser.links
            if not link.startswith(("#", "data:"))
        ), html_file
    assert all(marker not in p.read_text() for p in site.rglob("*") if p.is_file())
    saved = hashes(site)
    assert runner.invoke(app, site_args).exit_code == 0 and hashes(site) == saved
    assert before == {
        name: hashes(tmp_path / name) for name in ["data", "reports", "bundles", "budget"]
    }
    # Foreign report directories and errors cannot expose raw exception context.
    response = runner.invoke(app, ["report", "--run", result.run_id, "--out", "reports"])
    assert response.exit_code == 5 and "Traceback" not in response.output
    assert runner.invoke(app, ["report", "--run", "missing"]).exit_code == 5
    assert before == {
        name: hashes(tmp_path / name) for name in ["data", "reports", "bundles", "budget"]
    }
