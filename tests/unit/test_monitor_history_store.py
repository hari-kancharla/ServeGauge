"""Immutable history projections refuse removal, mutation and operating claims."""

import copy

import pytest

from servegauge.core.hashing import content_hash
from servegauge.storage.transport_history import TransportHistoryJournal


def projection(inputs=None):
    value = {
        "version": "transport-history-v1",
        "observatory_id": "local-one",
        "source_hash": "a" * 64,
        "policy_hash": "b" * 64,
        "qualified": False,
        "verdict": "INCONCLUSIVE",
        "inputs": inputs or {},
        "streams": {},
        "steps": [],
    }
    return {**value, "projection_id": "transport-history-" + content_hash(value)}


def test_identical_projection_is_read_only_and_changes_are_append_only(tmp_path):
    journal = TransportHistoryJournal(tmp_path, "local-one")
    a = projection(
        {
            "window-a": {
                "spec_hash": "a" * 64,
                "state_hash": "b" * 64,
                "event_hashes": [],
                "complete": False,
            }
        }
    )
    assert journal.record(a) is True
    before = {p.name: p.read_bytes() for p in journal.directory.iterdir()}
    assert journal.record(a) is False
    assert before == {p.name: p.read_bytes() for p in journal.directory.iterdir()}
    b = projection(
        {
            "window-a": {
                "spec_hash": "a" * 64,
                "state_hash": "c" * 64,
                "event_hashes": ["d" * 64],
                "complete": True,
            }
        }
    )
    assert journal.record(b) is True
    assert journal.projections() == [a, b]
    assert all((journal.directory / name).read_bytes() == value for name, value in before.items())


@pytest.mark.parametrize(
    "change", ["missing", "changed_spec", "changed_completed", "back_to_incomplete"]
)
def test_previously_observed_complete_evidence_cannot_disappear_or_change(tmp_path, change):
    journal = TransportHistoryJournal(tmp_path, "local-one")
    inputs = {
        "window-a": {
            "spec_hash": "a" * 64,
            "state_hash": "b" * 64,
            "event_hashes": ["d" * 64],
            "complete": True,
        }
    }
    assert journal.record(projection(inputs))
    altered = copy.deepcopy(inputs)
    if change == "missing":
        altered = {}
    elif change == "changed_spec":
        altered["window-a"]["spec_hash"] = "c" * 64
    elif change == "changed_completed":
        altered["window-a"]["state_hash"] = "c" * 64
    else:
        altered["window-a"]["complete"] = False
    with pytest.raises(ValueError, match="history_evidence"):
        journal.record(projection(altered))
    assert len(journal.projections()) == 1


@pytest.mark.parametrize("change", ["id", "qualified", "identity", "step_alarm"])
def test_invalid_or_operating_projections_refuse_without_creating_a_journal(tmp_path, change):
    value = projection()
    if change == "id":
        value["projection_id"] = "other"
    elif change == "qualified":
        value["qualified"] = True
    elif change == "identity":
        value["observatory_id"] = "someone-else"
    else:
        value["steps"] = [{"qualified": True}]
    if change != "id":
        value["projection_id"] = "transport-history-" + content_hash(
            {k: v for k, v in value.items() if k != "projection_id"}
        )
    journal = TransportHistoryJournal(tmp_path, "local-one")
    with pytest.raises(ValueError):
        journal.record(value)
    assert not journal.directory.exists()


def test_symlink_and_tamper_refuse(tmp_path):
    journal = TransportHistoryJournal(tmp_path, "local-one")
    journal.record(projection())
    event = journal.directory / "event-00000001.json"
    event.write_text("{}")
    with pytest.raises(ValueError):
        journal.projections()
    other = TransportHistoryJournal(tmp_path, "other")
    other.directory.symlink_to(journal.directory, target_is_directory=True)
    with pytest.raises(ValueError):
        other.record({**projection(), "observatory_id": "other"})


@pytest.mark.parametrize("hashes", [[], ["c" * 64]])
def test_incomplete_history_cannot_roll_back_or_rewrite_prior_events(tmp_path, hashes):
    journal = TransportHistoryJournal(tmp_path, "local-one")
    first = projection(
        {
            "window-a": {
                "spec_hash": "a" * 64,
                "state_hash": "b" * 64,
                "event_hashes": ["b" * 64],
                "complete": False,
            }
        }
    )
    journal.record(first)
    other = projection(
        {
            "window-a": {
                "spec_hash": "a" * 64,
                "state_hash": "c" * 64,
                "event_hashes": hashes,
                "complete": False,
            }
        }
    )
    with pytest.raises(ValueError, match="history_evidence"):
        journal.record(other)
    assert len(journal.projections()) == 1


def test_private_credentials_and_nested_operating_claims_are_refused(tmp_path, monkeypatch):
    private = "PRIVATE_HISTORY_CREDENTIAL_SENTINEL"
    monkeypatch.setenv("EXAMPLE_API_KEY", private)
    value = projection()
    value["streams"] = {"s": {"endpoint_id": private}}
    value["projection_id"] = "transport-history-" + content_hash(
        {k: v for k, v in value.items() if k != "projection_id"}
    )
    journal = TransportHistoryJournal(tmp_path, "local-one")
    with pytest.raises(ValueError):
        journal.record(value)
    assert not journal.directory.exists()
    value = projection()
    value["steps"] = [{"qualified": False, "verdict": "INCONCLUSIVE", "event": {"qualified": True}}]
    value["projection_id"] = "transport-history-" + content_hash(
        {k: v for k, v in value.items() if k != "projection_id"}
    )
    with pytest.raises(ValueError):
        journal.record(value)
    assert not journal.directory.exists()


def test_json_tuple_normalization_is_not_mistaken_for_credential_redaction(tmp_path):
    value = projection()
    value["steps"] = [
        {
            "qualified": False,
            "verdict": "INCONCLUSIVE",
            "event": {"qualified": False, "input_hashes": ("a" * 64,)},
        }
    ]
    value["projection_id"] = "transport-history-" + content_hash(
        {k: v for k, v in value.items() if k != "projection_id"}
    )
    journal = TransportHistoryJournal(tmp_path, "local-one")
    assert journal.record(value) is True
    assert journal.projections()[0]["steps"][0]["event"]["input_hashes"] == ["a" * 64]
    assert journal.record(value) is False
