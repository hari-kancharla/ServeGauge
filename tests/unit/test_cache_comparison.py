"""Cache comparisons require complete observable request-bound prompt groups."""

import copy

import pytest

from servegauge.core.models import Attempt, Exchange, Timings
from servegauge.scoring.cache_comparison import cache_samples
from servegauge.suites.registry import prepare_item
from servegauge.suites.s1_cache import cache_check, cache_items
from servegauge.validation.capture import endpoint


def fixture(*, groups=2, target_id="mock:target", seed_default=None, role_seed=1):
    ep = endpoint(target_id, role_seed, clean=True)
    if seed_default is not None:
        ep.defaults["seed"] = seed_default
    items = cache_items("smoke", 42, {"groups": groups, "draws": 4})
    records = []
    for item in items:
        for draw in range(item.expected["samples"]):
            index = item.expected.get("cache_draw", draw)
            pattern = (
                ["red", "red", "blue", "blue"]
                if cache_check(item) == "P03.cache"
                else ["red", "blue", "green", "red"]
            )
            response = {
                "id": "response-" + str(len(records)),
                "choices": [{"message": {"content": pattern[index]}, "finish_reason": "stop"}],
            }
            records.append(
                Exchange(
                    exchange_id=ep.id + "-capture-" + str(len(records)),
                    run_id="run",
                    endpoint_id=ep.id,
                    suite_id="S1",
                    item_id=item.item_id,
                    sample_idx=draw,
                    request=prepare_item(item, ep).body,
                    attempts=[
                        Attempt(
                            n=1,
                            status=200,
                            timings=Timings(start_unix=0, total_ms=0),
                            response=response,
                        )
                    ],
                    status=200,
                    response=response,
                )
            )
    return items, records, ep


def test_change_frequency_clusters_adjacent_draws_by_complete_prompt_group():
    items, records, ep = fixture()
    samples = cache_samples(items, records, ep)
    assert set(samples) == {"P03.cache", "P03.cache.fresh"}
    repeated, fresh = samples["P03.cache"], samples["P03.cache.fresh"]
    assert set(repeated.values.values()) == {1 / 3}
    assert set(fresh.values.values()) == {1.0}
    for s in samples.values():
        assert s.status is None and len(s.values) == 2 and not s.binary
        assert len(s.evidence) == 8
        assert set(s.details["input_hashes"]) == set(s.values)
        assert isinstance(s.details["request_signature"], str)
        assert s.details["independent_unit"] == "complete prompt group; adjacent draws clustered"


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "extra-repeat",
        "duplicate-index",
        "duplicate-exchange",
        "wrong-request",
        "empty",
        "refusal",
        "tool",
        "censored",
        "redacted",
        "unsent",
        "bad-shape",
        "failed",
        "no-attempt",
        "attempt-error",
        "attempt-missing",
        "attempt-text",
        "filtered",
        "unknown-finish",
        "attempt-shape",
    ],
)
def test_incomplete_or_unobservable_groups_cannot_become_cache_evidence(fault):
    items, records, ep = fixture()
    original = copy.deepcopy(records)
    row = records[0]
    message = row.response["choices"][0]["message"]
    if fault == "missing":
        records.pop(0)
    elif fault == "extra-repeat":
        records.append(row.model_copy(deep=True))
    elif fault == "duplicate-index":
        records[1].sample_idx = 0
    elif fault == "duplicate-exchange":
        records[1].exchange_id = row.exchange_id
    elif fault == "wrong-request":
        row.request["temperature"] = 0
    elif fault == "empty":
        message["content"] = ""
    elif fault == "refusal":
        message["refusal"] = "refused"
    elif fault == "tool":
        message["tool_calls"] = [{"id": "unexpected"}]
    elif fault == "censored":
        row.response["choices"][0]["finish_reason"] = "length"
    elif fault == "redacted":
        message["content"] = "[REDACTED]"
    elif fault == "unsent":
        row.metadata["request_not_sent"] = True
    elif fault == "bad-shape":
        row.response = {"choices": []}
    elif fault == "failed":
        row.status = 503
    elif fault == "attempt-error":
        row.attempts[-1].error = "timeout"
    elif fault == "attempt-missing":
        row.attempts[-1].response = None
    elif fault == "attempt-text":
        row.attempts[-1].response = copy.deepcopy(row.response)
        row.attempts[-1].response["choices"][0]["message"]["content"] = "different"
    elif fault == "filtered":
        row.response["choices"][0]["finish_reason"] = "content_filter"
    elif fault == "unknown-finish":
        row.response["choices"][0].pop("finish_reason")
    elif fault == "attempt-shape":
        row.attempts[-1].response = {"choices": ["bad"]}
    else:
        row.attempts = []
    samples = cache_samples(items, records, ep)
    assert samples["P03.cache"].status == "INCONCLUSIVE"
    assert not samples["P03.cache"].values
    assert samples["P03.cache.fresh"].status is None
    clean = cache_samples(items, original, ep)
    assert all(v.status is None for v in clean.values())


def test_response_ids_can_repeat_while_capture_identities_must_be_unique():
    items, records, ep = fixture()
    for r in records:
        r.response["id"] = "cached-id"
        r.attempts[-1].response["id"] = "cached-id"
    samples = cache_samples(items, records, ep)
    assert all(s.status is None for s in samples.values())


def test_duplicate_capture_identity_across_repeated_and_fresh_is_not_independent_support():
    items, records, ep = fixture()
    fresh = next(i for i in items if cache_check(i) == "P03.cache.fresh")
    row = next(r for r in records if r.item_id == fresh.item_id)
    row.exchange_id = records[0].exchange_id
    samples = cache_samples(items, records, ep)
    assert all(s.status == "INCONCLUSIVE" and not s.values for s in samples.values())


def test_non_cache_items_and_other_endpoint_records_do_not_enter_the_cohort():
    items, records, ep = fixture()
    other = records[0].model_copy(deep=True)
    other.endpoint_id = "another"
    samples = cache_samples(items, [*records, other], ep)
    assert all(s.status is None and len(s.evidence) == 8 for s in samples.values())
    assert cache_samples([], records, ep) == {}


def test_matched_pack_supports_cache_comparison_and_changed_sampling_inputs_abstain():
    from servegauge.core.hashing import content_hash
    from servegauge.profiles import load_profile
    from servegauge.refs.pack import build_pack
    from servegauge.runner.evaluate import collect, evaluate, reference_source
    from servegauge.runner.plan import RunPlan
    from servegauge.suites.s1_cache import CACHE_CHECKS
    from servegauge.verdicts.config import load_policy

    policy, profile = load_policy(), load_profile("gpt-oss-120b")
    inputs = [
        fixture(groups=20, target_id=f"mock:reference-{i}", role_seed=i + 1) for i in range(3)
    ]
    items = inputs[0][0]
    plan = RunPlan(
        plan_id="cache-reference",
        profile=profile.id,
        level="smoke",
        seed=42,
        endpoints=[v[2] for v in inputs],
        suites=["S1"],
        reference={"tier": "MOCK"},
        max_usd=0,
    )
    records = []
    for _, rows, _ in inputs:
        for row in rows:
            # Known-answer reference: every planned adjacent pair changes.
            row.response["choices"][0]["message"]["content"] = ["red", "blue", "green", "red"][
                next(
                    i.expected.get("cache_draw", row.sample_idx)
                    for i in items
                    if i.item_id == row.item_id
                )
            ]
        records.extend(rows)
    scores = collect(plan, items, records, profile, policy)
    source = reference_source(
        plan,
        "reference",
        items,
        scores,
        policy,
        {"item_set_hashes": {"S1": content_hash([i.model_dump(mode="json") for i in items])}},
    )
    pack = build_pack(source, "MOCK", seed=91)
    mismatch = plan.model_copy(deep=True)
    mismatch.endpoints[0].defaults["seed"] = 9
    unqualified = reference_source(
        mismatch,
        "mismatch",
        items,
        scores,
        policy,
        {"item_set_hashes": {"S1": content_hash([i.model_dump(mode="json") for i in items])}},
    )
    assert all(check not in unqualified.checks for check in CACHE_CHECKS)
    for default in [None, 9]:
        _, rows, ep = fixture(groups=20, seed_default=default)
        target = plan.model_copy(update={"endpoints": [ep]})
        observed = collect(target, items, rows, profile, policy)
        assert all(s.status is None for s in observed[ep.id].values())
        checks = evaluate(
            target, "target", items, observed, policy, target_ids=[ep.id], reference_pack=pack
        )
        cards = {c.check_id: c for c in checks if c.check_id in CACHE_CHECKS}
        assert {k: c.verdict for k, c in cards.items()} == (
            {"P03.cache": "FAIL", "P03.cache.fresh": "PASS"}
            if default is None
            else {k: "INCONCLUSIVE" for k in CACHE_CHECKS}
        )
