"""Bounded, value-bound structure for sharing; these limits never affect inference."""

from __future__ import annotations

from typing import Any

from servegauge.core.hashing import content_hash


def content_summary(value: Any, *, max_nodes: int = 256, max_depth: int = 8) -> dict[str, Any]:
    """Fold oversized subtrees rather than expanding a wrapper for every scalar."""
    if type(max_nodes) is not int or max_nodes < 1 or type(max_depth) is not int or max_depth < 0:
        raise ValueError("positive_summary_budget_required")
    digest = content_hash(value)

    def shape(data: Any, depth: int, allowance: int) -> tuple[dict[str, Any], int, bool]:
        if not isinstance(data, (dict, list)):
            leaf: dict[str, Any] = {"type": type(data).__name__, "sha256": content_hash(data)}
            if isinstance(data, str):
                leaf.update(type="string", characters=len(data))
            return leaf, 1, False
        kind = "object" if isinstance(data, dict) else "array"
        size = len(data)
        if size and (depth >= max_depth or size > allowance - 1):
            return (
                {"type": kind, "length": size, "sha256": content_hash(data), "truncated": True},
                1,
                True,
            )
        children: list[dict[str, Any]] = []
        entries: dict[str, dict[str, Any]] = {}
        used, truncated = 1, False
        values = [(k, data[k]) for k in sorted(data)] if isinstance(data, dict) else enumerate(data)
        for index, (key, child) in enumerate(values):
            # Reserve one node for every remaining sibling before expanding this
            # child. A wide parent itself folds, so no unbounded stub list occurs.
            budget = allowance - used - (size - index - 1)
            nested, count, folded = shape(child, depth + 1, budget)
            if kind == "object":
                entries["key-" + content_hash(key)[:16]] = nested
            else:
                children.append(nested)
            used += count
            truncated |= folded
        result: dict[str, Any] = {"type": kind, "length": size}
        result["entries" if kind == "object" else "items"] = (
            entries if kind == "object" else children
        )
        return result, used, truncated

    structure, nodes, truncated = shape(value, 0, max_nodes)
    return {
        "version": "content-summary-v2",
        "sha256": digest,
        "structure": structure,
        "nodes": nodes,
        "limits": {"max_nodes": max_nodes, "max_depth": max_depth},
        "truncated": truncated,
    }
