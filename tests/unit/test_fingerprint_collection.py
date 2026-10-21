import copy

import pytest

from servegauge.core.models import EndpointSpec
from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.scoring.fingerprint_collection import fingerprint_samples, pending_checks
from servegauge.suites.registry import item_request
from servegauge.suites.s6_fingerprint import fingerprint_items
from servegauge.validation.capture import captures, endpoint


@pytest.mark.parametrize(
    "level,counts",
    [
        ("smoke", {"FP1": 120, "FP3": 50}),
        ("core", {"FP1": 1200, "FP3": 200}),
        ("full", {"FP1": 1200, "FP2": 250, "FP3": 400}),
    ],
)
def test_frozen_s6_inventory_matches_spec_and_contains_no_wire_interpreter(level, counts):
    items = fingerprint_items(level, 14)
    assert items == fingerprint_items(level, 14)
    assert len({i.item_id for i in items}) == len(items)
    assert {
        m: sum(i.expected["samples"] for i in items if i.expected["check_id"] == m) for m in counts
    } == counts
    for item in items:
        assert item.suite_id == "S6" and item.expected["design_hash"]
        assert "TW_TASK" not in str(item_request(item).model_dump())
        assert "TW_TASK" in str(
            item_request(item, endpoint("mock:test", 3, clean=True)).model_dump()
        )
    if level == "full":
        assert {i.expected["language"] for i in items if i.expected["check_id"] == "FP1"} == {
            "en",
            "es",
            "fr",
            "de",
        }


@pytest.mark.parametrize(
    "overrides",
    [
        {"fp1_cells": True},
        {"fp1_cells": 41},
        {"fp3_samples": 0},
        {"surprise": 2},
        {"methods": ["unknown"]},
    ],
)
def test_invalid_design_fails_before_dispatch(overrides):
    with pytest.raises(ValueError):
        fingerprint_items("smoke", 1, overrides)


def test_collection_keeps_incomplete_rows_and_exposes_only_aggregate_report_details():
    items = fingerprint_items("smoke", 19, {"fp1_cells": 2, "fp1_samples": 3, "fp3_samples": 4})
    target = endpoint("mock:target", 1, clean=True)
    rows = captures(
        items, ScriptedBackend(1, FaultConfig(), sampling_version="v2"), target, "collection"
    )
    rows[-1].response["choices"][0]["logprobs"] = None
    before = copy.deepcopy(rows)
    samples = fingerprint_samples(items, rows, target)
    assert set(samples) == {"FP1", "FP3"}
    assert samples["FP3"].details["requests"] == 4
    assert len(samples["FP3"].details["observations"]) == 4
    assert samples["FP3"].details["missing_logprobs"] == 1
    checks = pending_checks(samples, target.id, "collection")
    assert all(c.verdict == "INCONCLUSIVE" for c in checks)
    assert all("observations" not in c.details for c in checks)
    assert rows == before


def test_duplicate_and_missing_capture_are_not_complete_case_deleted():
    items = fingerprint_items("smoke", 1, {"methods": ["FP3"], "fp3_samples": 4})
    target = endpoint("mock:target", 1, clean=True)
    rows = captures(
        items, ScriptedBackend(1, FaultConfig(), sampling_version="v2"), target, "collection"
    )
    bad = fingerprint_samples(items, [rows[0], rows[0], *rows[2:]], target)["FP3"]
    assert bad.status == "INCONCLUSIVE"
    assert bad.details["missing_responses"] == 1 and bad.details["duplicate_identities"] == 1
    assert len(bad.details["observations"]) == 5


def test_unsupported_logprobs_remain_separate_from_text_methods():
    items = fingerprint_items("smoke", 1, {"fp1_cells": 2, "fp1_samples": 3, "fp3_samples": 4})
    target = EndpointSpec(
        id="direct:test",
        kind="openai_compat",
        base_url="http://127.0.0.1/v1",
        model="test",
        claimed_profile="gpt-oss-120b",
    )
    rows = captures(items, ScriptedBackend(1, FaultConfig()), target, "collection")
    for row in rows:
        if row.item_id.startswith("FP3"):
            row.error = "NOT_SUPPORTED"
            row.metadata.update(request_not_sent=True, verdict="NOT_SUPPORTED")
            row.status = None
            row.response = None
    found = fingerprint_samples(items, rows, target)
    assert found["FP3"].status == "NOT_SUPPORTED" and found["FP1"].status is None


@pytest.mark.parametrize("fault", ["censored", "transport", "invalid", "extra_identity"])
def test_unusable_capture_never_creates_a_numeric_outcome(fault):
    items = fingerprint_items("smoke", 1, {"methods": ["FP3"], "fp3_samples": 4})
    target = endpoint("mock:target", 1, clean=True)
    rows = captures(items, ScriptedBackend(1, FaultConfig()), target, "collection")
    if fault == "censored":
        rows[0].response["choices"][0]["finish_reason"] = "length"
    elif fault == "transport":
        rows[0].error, rows[0].status = "transport_failure", None
    elif fault == "invalid":
        rows[0].response = {"choices": []}
    else:
        rows[0].sample_idx = 55
    observed = fingerprint_samples(items, rows, target)
    result = pending_checks(observed, target.id, "collection")[0]
    assert result.verdict == ("ERROR" if fault == "transport" else "INCONCLUSIVE")
    assert result.estimate is None and observed["FP3"].values == {}
    assert observed["FP3"].details["captures"] == 4


def test_collected_snapshots_do_not_alias_raw_inputs_or_other_endpoints():
    items = fingerprint_items("smoke", 1, {"methods": ["FP3"], "fp3_samples": 4})
    ep = endpoint("mock:target", 1, clean=True)
    rows = captures(items, ScriptedBackend(1, FaultConfig()), ep, "collection")
    before = copy.deepcopy((items, rows))
    first = fingerprint_samples(items, rows, ep)["FP3"]
    second = fingerprint_samples(items, rows, ep)["FP3"]
    first.details["design"]["output_cap"] += 1
    first.details["observations"][0]["response"]["choices"][0]["logprobs"] = None
    assert (items, rows) == before
    assert first.details != second.details
