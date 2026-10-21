"""Sealed mapping order must not change a captured mock request on replay."""

import json

import pytest

from servegauge.core.models import Exchange, Item
from servegauge.diagnose.repro import _mock_replay_inventory
from servegauge.suites.registry import prepare_item
from servegauge.validation.capture import endpoint


def fixture():
    ep = endpoint("target", 13, clean=False)
    item = Item(
        item_id="case",
        suite_id="S6",
        version="unit-fixture",
        payload={"messages": [{"role": "user", "content": "Return a value."}]},
        expected={"mock_task": {"type": "sample", "choices": ["red", "blue"]}},
    )
    body = prepare_item(item, ep).body
    captured = Exchange(
        exchange_id="capture",
        endpoint_id=ep.id,
        run_id="original",
        suite_id=item.suite_id,
        item_id=item.item_id,
        sample_idx=0,
        request=body,
        attempts=[],
    )
    restored = Item.model_validate_json(json.dumps(item.model_dump(), sort_keys=True))
    assert prepare_item(restored, ep).body != body
    return ep, item, restored, captured


def test_captured_task_order_restores_exact_request_without_changing_inventory_values():
    ep, original, restored, captured = fixture()
    actual = _mock_replay_inventory([restored], [captured], ep.id, "original")
    assert actual[0].model_dump() == original.model_dump()
    assert prepare_item(actual[0], ep).body == captured.request
    assert list(restored.expected["mock_task"]) == ["choices", "type"]


def test_capture_task_values_cannot_replace_changed_inventory_or_cross_run_support():
    ep, _, restored, captured = fixture()
    restored.expected["mock_task"]["choices"] = ["wrong"]
    with pytest.raises(ValueError, match="mock_replay_task_mismatch"):
        _mock_replay_inventory([restored], [captured], ep.id, "original")
    actual = _mock_replay_inventory([restored], [captured], ep.id, "another")
    assert actual[0].model_dump() == restored.model_dump()


def test_conflicting_capture_serialization_abstains_instead_of_choosing_a_repeat():
    ep, _, restored, captured = fixture()
    other = captured.model_copy(deep=True)
    other.request = prepare_item(restored, ep).body
    with pytest.raises(ValueError, match="mock_replay_task_mismatch"):
        _mock_replay_inventory([restored], [captured, other], ep.id, "original")


def test_other_suite_with_same_item_name_cannot_supply_the_wire_task():
    ep, _, restored, captured = fixture()
    captured.suite_id = "S1"
    actual = _mock_replay_inventory([restored], [captured], ep.id, "original")
    assert actual[0].model_dump() == restored.model_dump()
    assert prepare_item(actual[0], ep).body != captured.request
