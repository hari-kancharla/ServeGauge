"""Presentation fixtures test integrity/privacy, not measurement qualification."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from servegauge.core.models import Attempt, CheckResult, Exchange, Timings
from servegauge.diagnose.issue import markdown_text
from servegauge.report.endpoint_report import endpoint_context, write_endpoint_report
from servegauge.runner.plan import RunPlan
from servegauge.storage.analysis import Analysis
from servegauge.validation.capture import endpoint

PRIVATE = "PRIVATE_REPORT_PAYLOAD_CANARY"
ATTACK = '</script><img src=x onerror="alert(1)">'


def fixture():
    target = endpoint("mock:target", 1, clean=True)
    target.labels["provider_name"] = ATTACK + PRIVATE
    plan = RunPlan(
        plan_id="report-presentation",
        profile="gpt-oss-120b",
        level="smoke",
        seed=7,
        endpoints=[target],
        suites=["S1"],
        reference={"tier": "MOCK"},
        max_usd=0,
    )
    records = [
        Exchange(
            exchange_id=f"capture-{i}",
            run_id="report-run",
            endpoint_id=target.id,
            suite_id="S1",
            item_id=f"probe-{i}",
            sample_idx=0,
            request={
                "model": "mock",
                "messages": [{"role": "user", "content": PRIVATE}],
                "max_tokens": 1,
            },
            status=200,
            response={"choices": [{"message": {"content": PRIVATE}, "finish_reason": "stop"}]},
            attempts=[Attempt(n=1, status=200, timings=Timings(start_unix=0, total_ms=1))],
        )
        for i in range(4)
    ]
    evidence = [r.exchange_id for r in records]
    checks = [
        CheckResult(
            check_id="P01",
            suite_id="S1",
            check_class="invariant",
            endpoint_id=target.id,
            run_id="report-run",
            verdict="FAIL",
            reason="observed cap violation",
            n=4,
            violations=4,
            estimate=1,
            ci_low=0.5101,
            ci_high=1,
            evidence=evidence,
            details={"replay_confirmed": True, "opaque_response": PRIVATE},
        ),
        CheckResult(
            check_id="FP3",
            suite_id="S6",
            check_class="comparative",
            endpoint_id=target.id,
            run_id="report-run",
            verdict="INCONCLUSIVE",
            reason="logprobs unavailable",
            n=0,
            details={"private_payload": {"text": PRIVATE}},
        ),
        CheckResult(
            check_id="S8.health.requests",
            suite_id="S8",
            check_class="report_only",
            endpoint_id=target.id,
            run_id="report-run",
            verdict="INFO",
            n=4,
            estimate=0.5,
            ci_low=0.1,
            ci_high=0.9,
            delta=-0.5,
            delta_ci_low=-0.9,
            delta_ci_high=-0.1,
            reference_tier="BASELINE",
            reference_estimate=1,
            margin=0.03,
            p_value=0.01,
            adjusted_p=0.02,
            evidence=evidence,
            reason="observed operational comparison",
            details={"operational_alarm": True},
        ),
    ]
    return Analysis(
        plan=plan,
        items=[],
        records=records,
        checks=checks,
        samples={},
        metadata={"run_id": "report-run", "synthetic_mock": True, "private_metadata": PRIVATE},
        manifest={
            "run_id": "report-run",
            "created_at": "2026-10-20T00:00:00Z",
            "git_sha": "abc1234",
            "source_tree_hash": "a" * 64,
            "config_hash": "b" * 64,
            "item_set_hashes": {"S1": "c" * 64},
        },
        reference_pack=None,
    )


def files(path):
    return {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob("*")
        if p.is_file()
    }


def test_context_retains_unavailable_values_alarm_labels_and_exact_outcome_counts():
    analysis = fixture()
    before = copy.deepcopy(analysis)
    context = endpoint_context(analysis)
    rows = {c["check_id"]: c for c in context["checks"]}
    assert rows["FP3"]["estimate_text"] == rows["FP3"]["interval_text"] == "unavailable"
    assert rows["FP3"]["n"] == 0 and rows["FP3"]["verdict"] == "INCONCLUSIVE"
    assert rows["S8.health.requests"]["verdict"] == "INFO"
    assert "alarm" in rows["S8.health.requests"]["verdict_label"].lower()
    assert sum(c["count"] for c in context["summaries"]) == 3
    assert sum(c["count"] for c in context["counts"]) == 3
    assert context["total_checks"] == 3 and context["endpoint_count"] == 1
    assert analysis == before


@pytest.mark.parametrize(
    "link",
    [
        "https://example.com/x",
        "//example.com/x",
        "../source",
        "/source",
        "C:\\source",
        "javascript:alert(1)",
        "bundles/%2e%2e/source",
        ".",
        "bundles/./source",
        "bundles//source",
    ],
)
def test_incoming_links_cannot_escape_the_owned_report_artifact(link):
    with pytest.raises(ValueError, match="report_link"):
        endpoint_context(fixture(), methods_href=link)


def test_html_markdown_json_and_local_assets_are_real_and_payload_text_is_escaped(tmp_path):
    analysis = fixture()
    before = copy.deepcopy(analysis)
    out = tmp_path / "report"
    result = write_endpoint_report(analysis, out)
    assert Path(result["html"]) == out / "index.html"
    html = (out / "index.html").read_text()
    assert '<meta name="viewport"' in html and "aria-label=" in html
    assert 'id="chart-data-body"' in html and "<noscript>" in html
    assert ATTACK not in html and "<img src=x" not in html
    assert "unsafe-eval" in html and "connect-src 'none'" in html
    assert '<script src="assets/vega-6.4.0.min.js"' in html
    data = json.loads((out / "checks.json").read_text())
    assert len(data["checks"]) == 3
    assert any(c["check_id"] == "FP3" and c["estimate"] is None for c in data["checks"])
    assert all(
        markdown_text(value) in (out / "checks.md").read_text()
        for value in ["P01", "FP3", "BASELINE", "PARAM_IGNORED", "INFRA_UNRELIABLE"]
    )
    vendor = json.loads(Path("src/servegauge/report/vendor/manifest.json").read_text())
    for name, meta in vendor["files"].items():
        assert hashlib.sha256((out / "assets" / name).read_bytes()).hexdigest() == meta["sha256"]
    assert (out / "assets/report.css").is_file() and (out / "assets/report.js").is_file()
    assert analysis == before


def test_content_redaction_removes_nested_payloads_metadata_and_bundle_text(tmp_path):
    analysis = fixture()
    before = copy.deepcopy(analysis)
    out = tmp_path / "shared"
    write_endpoint_report(analysis, out, redact_content=True)
    for path in out.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".html", ".md", ".yaml", ".py"}:
            assert PRIVATE not in path.read_text(), path.name
    data = json.loads((out / "checks.json").read_text())
    assert data["metadata"]["redact_content"] is True
    row = next(c for c in data["checks"] if c["check_id"] == "P01")
    assert row["estimate"] == 1 and row["ci_low"] == 0.5101 and row["n"] == 4
    assert all(p.name.endswith("-redacted-v2") for p in (out / "bundles").iterdir())
    assert analysis == before


def test_verified_repeat_is_idempotent_and_privacy_change_cannot_leave_private_files(tmp_path):
    analysis = fixture()
    out = tmp_path / "private"
    first = write_endpoint_report(analysis, out)
    hashes = files(out)
    second = write_endpoint_report(analysis, out)
    assert first["out"] == second["out"] and files(out) == hashes
    with pytest.raises(ValueError, match="report_output"):
        write_endpoint_report(analysis, out, redact_content=True)
    assert files(out) == hashes


def test_unknown_existing_or_changed_report_files_are_preserved_and_refused(tmp_path):
    out = tmp_path / "foreign"
    out.mkdir()
    (out / "owner-file.txt").write_text("keep")
    before = files(out)
    with pytest.raises(ValueError, match="report_output"):
        write_endpoint_report(fixture(), out)
    assert files(out) == before
    known = tmp_path / "known"
    write_endpoint_report(fixture(), known)
    (known / "checks.md").write_text("owner edit")
    before = files(known)
    with pytest.raises(ValueError, match="report_output"):
        write_endpoint_report(fixture(), known)
    assert files(known) == before


def test_report_output_symlink_cannot_replace_an_owner_directory(tmp_path):
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("keep")
    link = tmp_path / "link"
    link.symlink_to(victim, target_is_directory=True)
    with pytest.raises(ValueError, match="report_output"):
        write_endpoint_report(fixture(), link)
    assert files(victim) == {"keep.txt": hashlib.sha256(b"keep").hexdigest()}


@pytest.mark.parametrize("change", ["label", "record", "nested_manifest"])
def test_idempotence_binds_visible_labels_private_bundle_inputs_and_all_owned_files(
    tmp_path, change
):
    analysis = fixture()
    out = tmp_path / "report"
    write_endpoint_report(analysis, out)
    if change == "label":
        analysis.plan.endpoints[0].labels["provider_name"] = "changed label"
    elif change == "record":
        analysis.records[0].response = {"changed": "private evidence"}
    else:
        nested = out / "owner" / "report-manifest.json"
        nested.parent.mkdir()
        nested.write_text("preserve this foreign file")
    before = files(out)
    with pytest.raises(ValueError, match="report_output"):
        write_endpoint_report(analysis, out)
    assert files(out) == before


def test_counts_separate_unsent_logical_rows_from_captures_and_attempts():
    analysis = fixture()
    analysis.records[0].attempts = []
    analysis.records[1].attempts.append(analysis.records[1].attempts[0].model_copy())
    context = endpoint_context(analysis)
    assert context["capture_count"] == 3
    assert "4 recorded attempts; 1 unsent logical records" in context["description"]


def test_markdown_escapes_html_once_and_cannot_create_an_active_link(tmp_path):
    analysis = fixture()
    analysis.checks[0].reason = "<img src=x> [click](https://example.test) | &"
    out = tmp_path / "report"
    write_endpoint_report(analysis, out)
    markdown = (out / "checks.md").read_text()
    assert "&lt;img" in markdown and "&amp;lt;img" not in markdown
    assert "[click](https" not in markdown
