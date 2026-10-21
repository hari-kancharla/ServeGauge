"""Shareable structure must have a fixed work/output budget and retain value hashes."""

import json

import pytest

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.diagnose.bundle import content_summary


def count_nodes(node):
    return 1 + sum(
        count_nodes(child)
        for child in (
            node.get("items", []) if node["type"] == "array" else node.get("entries", {}).values()
        )
    )


def test_wide_private_dataset_is_bounded_and_payload_keys_values_are_removed():
    marker = "PRIVATE_RESPONSE_MARKER_WITH_UNIQUE_TEXT"
    value = {marker: [{"text": marker, "vector": list(range(40))} for _ in range(10000)]}
    summary = content_summary(value)
    assert summary["version"] == "content-summary-v2"
    assert summary["sha256"] == content_hash(value)
    assert summary["nodes"] == count_nodes(summary["structure"]) <= 256
    assert summary["truncated"] and len(canonical_json(summary)) < 50000
    assert marker not in json.dumps(summary) and '"vector"' not in json.dumps(summary)
    assert value[marker][0]["text"] == marker


def test_deep_payload_folds_with_explicit_truncation_and_full_input_binding():
    value = "PRIVATE_LEAF_VALUE"
    for _ in range(100):
        value = {"nested": value}
    summary = content_summary(value, max_nodes=16, max_depth=4)
    assert summary["sha256"] == content_hash(value)
    assert summary["nodes"] == count_nodes(summary["structure"]) <= 16
    assert summary["truncated"] and len(canonical_json(summary)) < 4000
    assert "PRIVATE_LEAF_VALUE" not in json.dumps(summary)


def test_single_node_budget_keeps_type_length_and_hash_without_child_expansion():
    value = ["private", {"another": "private"}]
    summary = content_summary(value, max_nodes=1)
    assert summary["nodes"] == 1 and summary["truncated"]
    assert summary["structure"] == {
        "type": "array",
        "length": 2,
        "sha256": content_hash(value),
        "truncated": True,
    }


def test_expanded_summary_is_deterministic_under_mapping_order_and_keeps_empty_shapes():
    first = {"z": [], "a": {"last": 3, "first": "private"}, "empty": {}}
    other = {"empty": {}, "a": {"first": "private", "last": 3}, "z": []}
    a, b = content_summary(first), content_summary(other)
    assert a == b and not a["truncated"]
    assert a["nodes"] == count_nodes(a["structure"])
    assert "private" not in json.dumps(a)


@pytest.mark.parametrize("nodes", [1, 2, 3, 6, 10, 32])
@pytest.mark.parametrize("depth", [0, 1, 3])
def test_branching_and_sibling_reservation_always_obey_the_node_budget(nodes, depth):
    value = {
        "branch": [{"private-key": "PRIVATE_MARKER", "list": [1, 2, 3, 4]} for _ in range(4)],
        "leaf": "PRIVATE_MARKER",
        "empty": [],
        "other": {"nested": {"another": [True, None]}},
    }
    summary = content_summary(value, max_nodes=nodes, max_depth=depth)
    assert summary["nodes"] == count_nodes(summary["structure"]) <= nodes
    assert summary["sha256"] == content_hash(value)
    assert "PRIVATE_MARKER" not in json.dumps(summary) and '"private-key"' not in json.dumps(
        summary
    )


@pytest.mark.parametrize("nodes,depth", [(0, 1), (-1, 1), (True, 1), (1, -1), (1, True)])
def test_invalid_rendering_budgets_are_rejected(nodes, depth):
    with pytest.raises(ValueError, match="summary_budget"):
        content_summary({}, max_nodes=nodes, max_depth=depth)


@pytest.mark.parametrize("value", [None, True, 12, 0.25, "private", [], {}])
def test_json_scalar_and_empty_container_shapes_keep_exact_full_value_hash(value):
    summary = content_summary(value, max_nodes=1, max_depth=0)
    assert summary["sha256"] == content_hash(value)
    assert summary["nodes"] == 1 and not summary["truncated"]


def test_nonfinite_values_still_refuse_serialization_instead_of_hiding_invalid_input():
    with pytest.raises(ValueError):
        content_summary({"value": float("nan")})
