"""Selected sealed events and owner assertions are never promoted into delivery proof."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from unit.test_endpoint_report import PRIVATE, fixture
from unit.test_site import prepare

from servegauge.core.hashing import content_hash
from servegauge.diagnose.bundle import write_bundle
from servegauge.diagnose.rules import diagnoses
from servegauge.observatory.notice_workflow import prepare_notice
from servegauge.report.site.builder import build_site, publication_preflight
from servegauge.storage.notices import NoticeJournal


def real_fixture(root, monkeypatch):
    analysis = fixture()
    analysis.plan.endpoints[0].kind = "openai_compat"
    monkeypatch.setattr("servegauge.report.site.builder.reconstruct", lambda raw, run: analysis)
    monkeypatch.setattr(
        "servegauge.observatory.notice_workflow.reconstruct", lambda raw, run: analysis
    )
    return analysis


def notice(root, analysis):
    check = next(c for c in analysis.checks if c.verdict == "FAIL")
    diagnosis = next(d for d in diagnoses(analysis.checks) if check.check_id in d.check_ids)
    path = write_bundle(analysis, diagnosis, root=root)
    stamp = datetime.now(UTC) - timedelta(days=15)
    value = prepare_notice(
        root,
        check.run_id,
        check.endpoint_id,
        check.check_id,
        (path / "bundle.json").relative_to(root).as_posix(),
        now=stamp,
    )
    return NoticeJournal(root / "data/raw", value["notice_id"]), stamp


def test_real_fail_stays_blocked_without_a_notice_and_private_content_is_absent(
    tmp_path, monkeypatch
):
    prepare(tmp_path)
    analysis = real_fixture(tmp_path, monkeypatch)
    out = tmp_path / "site"
    build_site(tmp_path, out, runs=[analysis.manifest["run_id"]])
    with pytest.raises(ValueError, match="reply"):
        publication_preflight(out, root=tmp_path)
    index = json.loads((out / "site.json").read_text())
    assert index["reply_notes"][0]["state"]["eligible"] is False
    assert PRIVATE not in json.dumps(index)
    assert "Owner assertions" in (out / "replies.html").read_text()


def test_bound_owner_assertion_can_pass_but_withdrawal_blocks_a_stale_site(tmp_path, monkeypatch):
    prepare(tmp_path)
    analysis = real_fixture(tmp_path, monkeypatch)
    journal, stamp = notice(tmp_path, analysis)
    journal.record(
        "sent",
        at=stamp,
        now=stamp,
        owner_assertion=True,
        reference_sha256=content_hash("fixture delivery"),
        channel="https://support.example.invalid/ticket",
    )
    out = tmp_path / "site"
    build_site(tmp_path, out, runs=[analysis.manifest["run_id"]])
    result = publication_preflight(out, root=tmp_path)
    assert result["network_writes"] == 0 and result["published"] is False
    assert result["owner_asserted_notices"] == 1 and result["delivery_verified"] is False
    notes = json.loads((out / "site.json").read_text())["reply_notes"]
    assert notes[0]["state"]["eligible"] and notes[0]["assertions"][0]["owner_asserted"] is True
    assert "ticket" not in json.dumps(notes) and PRIVATE not in json.dumps(notes)
    assert "not independently verified" in (out / "replies.html").read_text()
    journal.record(
        "withdrawn",
        at=datetime.now(UTC),
        owner_assertion=True,
        reference_sha256=content_hash("fixture correction"),
    )
    with pytest.raises(ValueError, match="reply"):
        publication_preflight(out, root=tmp_path)


def test_draft_or_changed_failure_cannot_borrow_eligibility(tmp_path, monkeypatch):
    prepare(tmp_path)
    analysis = real_fixture(tmp_path, monkeypatch)
    journal, stamp = notice(tmp_path, analysis)
    out = tmp_path / "draft"
    build_site(tmp_path, out, runs=[analysis.manifest["run_id"]])
    with pytest.raises(ValueError, match="reply"):
        publication_preflight(out, root=tmp_path)
    journal.record(
        "sent",
        at=stamp,
        now=stamp,
        owner_assertion=True,
        reference_sha256=content_hash("fixture delivery"),
        channel="https://support.example.invalid",
    )
    fresh = tmp_path / "fresh"
    build_site(tmp_path, fresh, runs=[analysis.manifest["run_id"]])
    analysis.checks[0].reason = "new failure revision"
    with pytest.raises(ValueError):
        publication_preflight(fresh, root=tmp_path)
    revised = tmp_path / "revised"
    build_site(tmp_path, revised, runs=[analysis.manifest["run_id"]])
    note = json.loads((revised / "site.json").read_text())["reply_notes"][0]
    assert not note["state"]["eligible"] and note["assertions"] == []
    assert note["earlier_revisions"] == 1


def test_asserted_public_requires_unchanged_full_bundle_at_publication(tmp_path, monkeypatch):
    prepare(tmp_path)
    analysis = real_fixture(tmp_path, monkeypatch)
    journal, stamp = notice(tmp_path, analysis)
    journal.record(
        "already_public",
        at=stamp,
        now=stamp,
        owner_assertion=True,
        reference_sha256=content_hash("fixture public reference"),
        channel="https://support.example.invalid/private-case",
    )
    site = tmp_path / "site"
    build_site(tmp_path, site, runs=[analysis.manifest["run_id"]])
    assert publication_preflight(site, root=tmp_path)["owner_asserted_notices"] == 1
    bundle = tmp_path / journal.spec()["bundle_reference"]
    bundle.write_text("{}")
    with pytest.raises(ValueError, match="reply"):
        publication_preflight(site, root=tmp_path)


def test_real_capture_without_failures_needs_no_fabricated_notice(tmp_path, monkeypatch):
    prepare(tmp_path)
    analysis = real_fixture(tmp_path, monkeypatch)
    for check in analysis.checks:
        if check.verdict == "FAIL":
            check.verdict = "INCONCLUSIVE"
    out = tmp_path / "site"
    build_site(tmp_path, out, runs=[analysis.manifest["run_id"]])
    assert publication_preflight(out, root=tmp_path)["owner_asserted_notices"] == 0
    assert not (tmp_path / "data/raw/notices").exists()


def test_empty_selection_never_reads_observatory_or_notice_history(tmp_path, monkeypatch):
    prepare(tmp_path)
    monkeypatch.setattr(
        "servegauge.report.site.observatory.observatory_rows",
        lambda *a, **k: pytest.fail("private event read"),
    )
    monkeypatch.setattr(
        "servegauge.report.site.observatory.notice_snapshots",
        lambda *a, **k: pytest.fail("private notice read"),
    )
    build_site(tmp_path, tmp_path / "site")
    index = json.loads((tmp_path / "site/site.json").read_text())
    assert index["monitor_events"] == [] and index["reply_notes"] == []
