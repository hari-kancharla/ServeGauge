"""Versioned repeated/fresh identities and planned request bindings."""

import copy
from collections import Counter

import pytest

from servegauge.core.hashing import content_hash
from servegauge.runner.plan import RunPlan
from servegauge.suites.registry import prepare_item
from servegauge.suites.s1_cache import (
    CACHE_CHECKS,
    cache_check,
    cache_contract,
    cache_items,
    cache_layout,
    request_signature,
)
from servegauge.validation.capture import endpoint


def test_inventory_has_exact_repeat_and_unique_fresh_request_identities():
    items = cache_items("smoke", 41, {"groups": 3, "draws": 4})
    assert len({i.item_id for i in items}) == len(items) == 15
    counts = Counter(cache_check(i) for i in items)
    assert counts == {"P03.cache": 3, "P03.cache.fresh": 12}
    assert len({(i.suite_id, i.expected["check_id"], i.expected["trial"]) for i in items}) == 3
    ep = endpoint("mock:target", 1, clean=True)
    layouts = cache_layout(items, ep)
    assert set(layouts) == set(CACHE_CHECKS)
    for check, groups in layouts.items():
        assert len(groups) == 3
        for rows in groups.values():
            assert len(rows) == 4 and len({(r[0], r[1]) for r in rows}) == 4
            hashes = [r[2] for r in rows]
            assert len(set(hashes)) == (1 if check == "P03.cache" else 4)
    contract = cache_contract(items)
    assert contract["groups"] == 3 and contract["draws"] == 4
    assert sum(i.expected["samples"] for i in items) == 24
    assert all(i.suite_id == "S1" and i.source == "generated" for i in items)
    assert items == cache_items("smoke", 41, {"groups": 3, "draws": 4})


def test_sealed_mapping_order_keeps_identical_prepared_requests_and_layout():
    items = cache_items("smoke", 42, {"groups": 2, "draws": 3})
    import json

    from servegauge.core.models import Item

    restored = [Item.model_validate_json(json.dumps(i.model_dump(), sort_keys=True)) for i in items]
    ep = endpoint("mock:target", 1, clean=True)
    assert [prepare_item(i, ep).body for i in items] == [prepare_item(i, ep).body for i in restored]
    assert cache_layout(items, ep) == cache_layout(restored, ep)


def test_layout_ignores_model_alias_and_retains_sampling_defaults():
    items = cache_items("smoke", 4, {"groups": 2, "draws": 3})
    first = endpoint("mock:first", 1, clean=True)
    other = first.model_copy(deep=True)
    other.id, other.model = "mock:other", "another-alias"
    assert cache_layout(items, first) == cache_layout(items, other)
    other.defaults["seed"] = 31
    assert cache_layout(items, first) != cache_layout(items, other)


def test_transport_aliases_and_endpoint_selection_are_not_sampling_layout_changes():
    original = {
        "model": "first-alias",
        "messages": [{"role": "user", "content": "sample"}],
        "temperature": 1.0,
        "max_tokens": 128,
        "reasoning_effort": "low",
    }
    routed = copy.deepcopy(original)
    routed.update(
        model="other-alias",
        provider={"only": ["other-provider"], "allow_fallbacks": False},
        plugins=[{"id": "context-compression", "enabled": False}],
        max_completion_tokens=routed.pop("max_tokens"),
        reasoning={"effort": routed.pop("reasoning_effort")},
    )
    assert request_signature(original) == request_signature(routed)
    for change in [
        {"temperature": 0},
        {"seed": 1},
        {"plugins": [{"id": "context-compression", "enabled": True}]},
        {"reasoning": {"effort": "low", "exclude": True}},
    ]:
        changed = {**routed, **change}
        assert request_signature(original) != request_signature(changed)


def test_both_cache_checks_are_comparative_and_share_the_existing_declared_margin():
    from servegauge.verdicts.classes import check_class
    from servegauge.verdicts.config import load_policy

    policy = load_policy()
    assert all(check_class(check) == "comparative" for check in CACHE_CHECKS)
    assert policy.margin(CACHE_CHECKS[0]) == policy.margin(CACHE_CHECKS[1]) == 0.03
    policy.entries["P03.cache"]["margin"] = 0.02
    assert policy.margin(CACHE_CHECKS[1]) == 0.02


@pytest.mark.parametrize(
    "overrides", [{"groups": 0}, {"draws": 1}, {"groups": True}, {"unknown": 4}]
)
def test_unknown_or_invalid_cache_design_refuses(overrides):
    with pytest.raises(ValueError, match="cache_design"):
        cache_items("smoke", 1, overrides)


def test_mixed_missing_or_unknown_cache_contract_cannot_fall_back_to_a_legacy_score():
    items = cache_items("smoke", 4, {"groups": 2, "draws": 3})
    for problem in ["missing", "version", "different"]:
        broken = copy.deepcopy(items)
        if problem == "missing":
            broken[0].expected.pop("cache_contract")
        elif problem == "version":
            broken[0].expected["cache_contract"]["version"] = "unknown"
        else:
            broken[0].expected["cache_contract"]["draws"] += 1
        with pytest.raises(ValueError, match="cache_contract"):
            cache_contract(broken)
    assert cache_contract([]) is None


@pytest.mark.parametrize(
    "fault", ["fresh-nonce", "params", "task", "boolean-count", "stream", "trial"]
)
def test_declared_recipe_and_fresh_uniqueness_bind_actual_requests(fault):
    items = cache_items("smoke", 4, {"groups": 2, "draws": 3})
    ep = endpoint("mock:target", 1, clean=True)
    if fault == "fresh-nonce":
        fresh = [i for i in items if cache_check(i) == "P03.cache.fresh"]
        fresh[1].payload = copy.deepcopy(fresh[0].payload)
    elif fault == "params":
        items[0].payload["params"]["temperature"] = 0
    elif fault == "task":
        items[0].expected["mock_task"]["choices"] = ["wrong"]
    elif fault == "stream":
        items[0].payload["stream"] = True
    elif fault == "trial":
        items[0].expected["trial"] = "wrong"
    else:
        next(i for i in items if cache_check(i) == "P03.cache.fresh").expected["samples"] = True
    with pytest.raises(ValueError, match="cache_inventory"):
        cache_layout(items, ep)


def test_per_check_reference_configuration_binds_matching_layout_without_shared_pack_changes():
    from servegauge.runner.evaluate import evaluation_configuration
    from servegauge.verdicts.config import load_policy

    items = cache_items("smoke", 42, {"groups": 2, "draws": 3})
    ep = endpoint("mock:target", 1, clean=True)
    baseline = endpoint("mock:baseline", 2, clean=True)
    plan = RunPlan(
        plan_id="cache",
        profile="gpt-oss-120b",
        level="smoke",
        seed=42,
        endpoints=[ep, baseline],
        suites=["S1"],
        max_usd=0,
        reference={"tier": "BASELINE"},
    )
    cfg = evaluation_configuration(plan, items, load_policy())
    changed = plan.model_copy(deep=True)
    changed.endpoints[1].defaults["seed"] = 9
    bad = evaluation_configuration(changed, items, load_policy())
    for check in CACHE_CHECKS:
        assert cfg["checks"][check]["cache_contract"] == cache_contract(items)
        assert isinstance(cfg["checks"][check]["request_signature"], str)
        assert bad["checks"][check]["request_signature"] is None
        assert content_hash(cfg["checks"][check]) != content_hash(bad["checks"][check])
