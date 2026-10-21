"""Declared owner setup starts remain distinct from total custom engineering work."""

from __future__ import annotations

import pytest

from servegauge.comparison.authorization import SetupRecord, setup_measurement


def record():
    return SetupRecord(
        kit_hash="a" * 64,
        procedure="external",
        started_utc="2026-10-20T12:00:00Z",
        manual_steps=["Installed the pinned public CPU image."],
    )


def test_declared_start_to_observed_result_is_scoped_and_never_claims_total_setup():
    value = setup_measurement(
        record(),
        kit_hash="a" * 64,
        procedure="external",
        first_valid_utc="2026-10-20T12:02:00+00:00",
    )
    assert value["owner_started_to_first_valid_seconds"] == 120
    assert value["total_setup_seconds"] is None
    assert value["prebuilt_custom_work_reused"] is True
    assert value["manual_steps"] == ["Installed the pinned public CPU image."]


def test_missing_or_late_start_does_not_fabricate_zero_setup_time():
    for notes, first in [
        (None, "2026-10-20T12:02:00Z"),
        (record(), None),
        (record(), "2026-10-20T11:59:00Z"),
    ]:
        value = setup_measurement(
            notes, kit_hash="a" * 64, procedure="external", first_valid_utc=first
        )
        assert value["owner_started_to_first_valid_seconds"] is None
        assert value["total_setup_seconds"] is None


@pytest.mark.parametrize("problem", ["kit", "procedure", "timestamp"])
def test_notes_cannot_borrow_another_kit_or_procedure_or_echo_invalid_values(problem):
    notes = record()
    if problem == "kit":
        notes.kit_hash = "b" * 64
    elif problem == "procedure":
        notes.procedure = "ordinary"
    else:
        notes.started_utc = "synthetic-private-value"
    with pytest.raises(ValueError) as failure:
        setup_measurement(notes, kit_hash="a" * 64, procedure="external", first_valid_utc=None)
    assert "synthetic-private-value" not in str(failure.value)
