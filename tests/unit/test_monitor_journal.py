"""Monitor records must remain immutable, chained, private and recoverable."""

import json

import pytest

from servegauge.storage.monitor import MonitorJournal


def test_specification_is_immutable_and_identical_creation_is_idempotent(tmp_path):
    journal = MonitorJournal(tmp_path / "raw", "window-one")
    spec = {"config_hash": "one", "run_ids": ["job-one"], "budget_scope": "window-one"}
    journal.create(spec)
    before = {str(p): p.read_bytes() for p in journal.directory.glob("*.json")}
    journal.create(dict(spec))
    assert journal.spec() == spec
    assert before == {str(p): p.read_bytes() for p in journal.directory.glob("*.json")}
    with pytest.raises(ValueError, match="conflict"):
        journal.create({**spec, "config_hash": "changed"})
    assert journal.spec() == spec


def test_events_are_chained_and_tampering_or_missing_history_refuses(tmp_path):
    journal = MonitorJournal(tmp_path / "raw", "window-one")
    journal.create({"config_hash": "one"})
    journal.append({"kind": "claimed", "job": "one"})
    journal.append({"kind": "complete", "job": "one"})
    assert [e["kind"] for e in journal.events()] == ["claimed", "complete"]
    path = sorted(journal.directory.glob("event-*.json"))[0]
    preserved = path.read_bytes()
    data = json.loads(preserved)
    data["value"]["data"]["job"] = "changed"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="journal"):
        journal.events()
    path.write_bytes(preserved)
    path.rename(path.with_suffix(".preserved"))
    with pytest.raises(ValueError, match="journal"):
        journal.events()


def test_credentials_are_sanitized_and_caller_mutations_do_not_change_records(
    tmp_path, monkeypatch
):
    private = "journal-test-value-" + "x" * 30
    monkeypatch.setenv("JOURNAL_TEST_KEY", private)
    journal = MonitorJournal(tmp_path / "raw", "window-one")
    spec = {"opaque": private, "run_ids": ["one"]}
    journal.create(spec)
    event = {"kind": "claimed", "opaque": private}
    journal.append(event)
    spec["run_ids"].append("two")
    event["kind"] = "changed"
    assert journal.spec()["run_ids"] == ["one"]
    assert journal.events()[0]["kind"] == "claimed"
    assert all(private not in p.read_text() for p in journal.directory.glob("*.json"))


def test_missing_spec_or_symlinked_committed_record_never_becomes_a_fresh_window(tmp_path):
    journal = MonitorJournal(tmp_path / "raw", "window-one")
    with pytest.raises(ValueError, match="journal"):
        journal.events()
    journal.create({"config_hash": "one"})
    journal.append({"kind": "claimed"})
    path = sorted(journal.directory.glob("event-*.json"))[0]
    target = tmp_path / "owner-file.json"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="journal"):
        journal.events()
    assert target.is_file()


def test_lock_symlink_cannot_truncate_an_owner_file(tmp_path):
    journal = MonitorJournal(tmp_path / "raw", "window-one")
    journal.directory.mkdir(parents=True)
    target = tmp_path / "owner.txt"
    target.write_text("preserve this work")
    (journal.directory / ".record.lock").symlink_to(target)
    with pytest.raises(ValueError, match="journal"):
        journal.create({"config_hash": "one"})
    assert target.read_text() == "preserve this work"


def test_existing_foreign_window_directory_is_preserved_and_refused(tmp_path):
    journal = MonitorJournal(tmp_path / "raw", "window-one")
    journal.directory.mkdir(parents=True)
    owner = journal.directory / "owner.txt"
    owner.write_text("keep")
    with pytest.raises(ValueError, match="journal"):
        journal.create({"config_hash": "one"})
    assert owner.read_text() == "keep" and not (journal.directory / "spec.json").exists()
