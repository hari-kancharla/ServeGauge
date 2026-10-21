"""Local records do not assert delivery; exact evidence and UTC time control eligibility."""

from datetime import UTC, datetime, timedelta

import pytest

from servegauge.core.hashing import content_hash
from servegauge.observatory.notices import FailureSubject, notice_state
from servegauge.storage.notices import NoticeJournal

START = datetime(2026, 1, 1, tzinfo=UTC)


def subject(**changes):
    return FailureSubject.model_validate(
        {
            "run_id": "fixture-run",
            "endpoint_id": "provider:fixture",
            "provider_id": "fixture-provider",
            "check_id": "P01",
            "synthetic_mock": False,
            "evidence_sha256": content_hash("failure revision one"),
            **changes,
        }
    )


def prepare(root, value=None):
    value = value or subject()
    return NoticeJournal.prepare(
        root, value, "bundles/fixture/bundle.json", content_hash("bundle"), now=START
    )


def sent(journal, *, at=START, now=START):
    return journal.record(
        "sent",
        at=at,
        now=now,
        owner_assertion=True,
        reference_sha256=content_hash("private delivery reference"),
        channel="https://support.example.invalid/private-ticket",
    )


def test_draft_never_starts_clock_and_repeated_prepare_preserves_initial_time(tmp_path):
    journal = prepare(tmp_path)
    assert journal.events() == []
    assert (
        notice_state(subject(), [journal.snapshot()], now=START + timedelta(days=100))["eligible"]
        is False
    )
    other = NoticeJournal.prepare(
        tmp_path,
        subject(),
        "bundles/fixture/bundle.json",
        content_hash("bundle"),
        now=START + timedelta(days=1),
    )
    assert other.spec() == journal.spec() and other.events() == []


def test_exact_fourteen_full_days_and_reply_does_not_reset_clock(tmp_path):
    journal = prepare(tmp_path)
    assert sent(journal) and not sent(journal)
    journal.record(
        "reply",
        at=START + timedelta(days=2),
        now=START + timedelta(days=2),
        owner_assertion=True,
        reference_sha256=content_hash("private reply"),
    )
    early = notice_state(
        subject(), [journal.snapshot()], now=START + timedelta(days=14, microseconds=-1)
    )
    boundary = notice_state(subject(), [journal.snapshot()], now=START + timedelta(days=14))
    assert not early["eligible"] and boundary["eligible"]
    assert boundary["basis"] == "owner_asserted_send_plus_14_days"
    assert boundary["reply_count"] == 1 and boundary["owner_asserted"] is True
    assert "private-ticket" not in str(journal.snapshot())


@pytest.mark.parametrize(
    "case", ["evidence_sha256", "provider_id", "endpoint_id", "run_id", "check_id"]
)
def test_other_evidence_or_identity_cannot_inherit_waiting_period(tmp_path, case):
    journal = prepare(tmp_path)
    sent(journal)
    value = content_hash("changed") if case == "evidence_sha256" else "different"
    assert not notice_state(
        subject(**{case: value}), [journal.snapshot()], now=START + timedelta(days=99)
    )["eligible"]


@pytest.mark.parametrize(
    "case",
    [
        "no_assertion",
        "naive",
        "future",
        "before_draft",
        "reply_before_send",
        "bad_reference",
        "private_url",
    ],
)
def test_invalid_assertions_fail_before_any_record(tmp_path, case):
    journal = prepare(tmp_path)
    arguments = {
        "at": START,
        "now": START,
        "owner_assertion": True,
        "reference_sha256": content_hash("reference"),
        "channel": "https://support.example.invalid",
    }
    kind = "sent"
    if case == "no_assertion":
        arguments["owner_assertion"] = False
    elif case == "naive":
        arguments["at"] = START.replace(tzinfo=None)
    elif case == "future":
        arguments["at"] = START + timedelta(seconds=1)
    elif case == "before_draft":
        arguments["at"] = START - timedelta(days=1)
    elif case == "reply_before_send":
        kind = "reply"
    elif case == "bad_reference":
        arguments["reference_sha256"] = "missing"
    else:
        arguments["channel"] = "https://user:secret@example.invalid/?token=private"
    with pytest.raises(ValueError):
        journal.record(kind, **arguments)
    assert journal.events() == []


def test_withdrawal_invalidates_sent_and_already_public_state(tmp_path):
    journal = prepare(tmp_path)
    journal.record(
        "already_public",
        at=START,
        now=START,
        owner_assertion=True,
        reference_sha256=content_hash("public source"),
        channel="https://example.invalid/report",
    )
    assert notice_state(subject(), [journal.snapshot()], now=START)["eligible"]
    journal.record(
        "withdrawn",
        at=START,
        now=START,
        owner_assertion=True,
        reference_sha256=content_hash("owner correction"),
    )
    assert not notice_state(subject(), [journal.snapshot()], now=START + timedelta(days=99))[
        "eligible"
    ]
    with pytest.raises(ValueError):
        sent(journal)


def test_generic_append_and_conflicting_send_are_refused(tmp_path):
    journal = prepare(tmp_path)
    sent(journal)
    with pytest.raises(ValueError):
        journal.append({"type": "sent"})
    with pytest.raises(ValueError):
        sent(journal, at=START + timedelta(seconds=1), now=START + timedelta(seconds=1))
    assert len(journal.events()) == 1


def test_boolean_owner_assertion_is_strict_and_missing_attachment_blocks_publication(tmp_path):
    from servegauge.observatory.notice_workflow import publication_status
    from servegauge.observatory.notices import NoticeAssertion

    with pytest.raises(ValueError):
        NoticeAssertion.model_validate(
            {
                "type": "sent",
                "at": START.isoformat(),
                "owner_asserted": 1,
                "reference_sha256": content_hash("ref"),
            }
        )
    journal = prepare(tmp_path)
    sent(journal)
    assert not publication_status(tmp_path, subject(), now=START + timedelta(days=99))["eligible"]


def test_table_rebuild_retains_draft_assertions_and_raw_only_parity(tmp_path):
    import json
    import shutil

    import polars as pl

    from servegauge.storage.tables import rebuild

    journal = prepare(tmp_path)
    sent(journal)
    counts = rebuild(tmp_path / "data/raw", tmp_path / "tables")
    assert counts["notifications"] == 1 and counts["change_events"] == 0
    row = json.loads(pl.read_parquet(tmp_path / "tables/notifications.parquet")["record_json"][0])
    assert row["state"] == "sent" and row["delivery_verified"] is False
    assert "private-ticket" not in json.dumps(row) and "bundle_reference" not in row
    shutil.copytree(tmp_path / "data/raw", tmp_path / "copy/raw")
    rebuild(tmp_path / "copy/raw", tmp_path / "copy/tables")
    assert pl.read_parquet(tmp_path / "tables/notifications.parquet").equals(
        pl.read_parquet(tmp_path / "copy/tables/notifications.parquet")
    )


def test_unbound_arbitrary_bundle_cannot_grant_publication_even_with_owner_assertion(tmp_path):
    import hashlib

    from servegauge.observatory.notice_workflow import publication_status

    file = tmp_path / "bundles/arbitrary/bundle.json"
    file.parent.mkdir(parents=True)
    file.write_bytes(b"{}\n")
    journal = NoticeJournal.prepare(
        tmp_path,
        subject(),
        "bundles/arbitrary/bundle.json",
        hashlib.sha256(file.read_bytes()).hexdigest(),
        now=START,
    )
    sent(journal)
    assert not publication_status(tmp_path, subject(), now=START + timedelta(days=99))["eligible"]


def test_later_recording_of_an_earlier_reply_does_not_lose_history(tmp_path):
    journal = prepare(tmp_path)
    sent(journal)
    for day in [3, 2]:
        journal.record(
            "reply",
            at=START + timedelta(days=day),
            now=START + timedelta(days=5),
            owner_assertion=True,
            reference_sha256=content_hash(["reply", day]),
        )
    state = notice_state(subject(), [journal.snapshot()], now=START + timedelta(days=14))
    assert state["eligible"] and state["reply_count"] == 2 and state["sent_at"] == START.isoformat()


@pytest.mark.parametrize(
    "reference",
    [
        "bundles/..\\outside/bundle.json",
        "bundles/a\\b/bundle.json",
        "bundles/../outside/bundle.json",
        "/bundles/a/bundle.json",
        "bundles//a/bundle.json",
        "bundles/a/./bundle.json",
        "C:/bundles/a/bundle.json",
        "bundles/D:/bundle.json",
        "bundles/a:stream/bundle.json",
        "bundles/CON/bundle.json",
        "bundles/trailing./bundle.json",
        "bundles/wild*/bundle.json",
        "bundles/a/bundle.json\0",
    ],
)
def test_bundle_references_are_canonical_and_portable_before_any_write(tmp_path, reference):
    with pytest.raises(ValueError):
        NoticeJournal.prepare(tmp_path, subject(), reference, content_hash("bundle"), now=START)
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("payload", [[], 42, "PRIVATE_BUNDLE_CANARY"])
def test_malformed_bundle_objects_are_refused_without_echo_or_journal(tmp_path, payload):
    import json

    from typer.testing import CliRunner

    from servegauge.cli import app

    file = tmp_path / "bundles/bad/bundle.json"
    file.parent.mkdir(parents=True)
    file.write_text(json.dumps(payload))
    result = CliRunner().invoke(
        app,
        [
            "notice",
            "prepare",
            "--run",
            "missing",
            "--endpoint",
            "fixture",
            "--check",
            "P01",
            "--bundle",
            "bundles/bad/bundle.json",
            "--root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 3 and "PRIVATE_BUNDLE_CANARY" not in result.output
    assert not (tmp_path / "data").exists()
