"""Aggregate keys must name the exact contributing captured items, per peer/check."""

import copy
import json

import pytest

from servegauge.core.models import EndpointSpec, Exchange, Item, ReferencePack
from servegauge.profiles import load_profile
from servegauge.refs.provenance import OutcomeProvenanceError, outcome_items
from servegauge.scoring.common import Samples, protocol_samples
from servegauge.scoring.json_scoring import json_samples
from servegauge.suites.s1_protocol import probe
from servegauge.suites.s3_structured import structured_items
from servegauge.verdicts.config import load_policy


def protocol(check="P03"):
    variants = ["required", "optional"] if check == "P13" else ["low", "high"]
    items = [
        probe(check, 0, variant, "private prompt canary", {}, samples=3) for variant in variants
    ]
    records = []
    for peer in ["a", "b"]:
        for item in items:
            for index in range(3):
                text = str(index) if item.expected["variant"] == "high" else "fixed"
                records.append(
                    Exchange(
                        exchange_id=f"{peer}-{item.item_id}-{index}",
                        run_id="run",
                        endpoint_id=peer,
                        suite_id="S1",
                        item_id=item.item_id,
                        sample_idx=index,
                        request={},
                        attempts=[],
                        status=200,
                        response={
                            "choices": [{"message": {"content": text}, "finish_reason": "stop"}]
                        },
                        usage={
                            "completion_tokens_details": {
                                "reasoning_tokens": 12 if text != "fixed" else 5
                            }
                        },
                    )
                )
    samples = {
        peer: protocol_samples(items, records, peer, load_profile("gpt-oss-120b"), load_policy())
        for peer in ["a", "b"]
    }
    pack = ReferencePack(
        pack_id="pack",
        profile="p",
        tier="R3",
        manifest={"source_run": "run"},
        outcomes={"target": {check: {"item_ids": ["0"], "peer_ids": ["a", "b"]}}},
    )
    return pack, items, records, samples


@pytest.mark.parametrize("check", ["P03", "P04.top_p", "P04.top_k", "P07", "P13"])
def test_real_protocol_scorer_pairs_are_linked_per_peer(check):
    pack, items, records, samples = protocol(check)
    assert all(s[check].status is None and "0" in s[check].values for s in samples.values())
    links = outcome_items(pack, items, records, samples)
    assert links.item_ids == frozenset(i.item_id for i in items)
    assert len(links.aggregate_links) == 2
    for row in links.aggregate_links:
        assert row.check_id == check and row.outcome_key == "0"
        assert set(row.item_ids) == links.item_ids
        assert set(row.exchange_ids) == {
            r.exchange_id for r in records if r.endpoint_id == row.endpoint_id
        }
    encoded = repr(links)
    assert "private prompt canary" not in encoded and "fixed" not in encoded


@pytest.mark.parametrize(
    "check,source_check", [("J01", "J01"), ("J02.semantic", "J02"), ("J03", "J03")]
)
def test_real_json_scorer_seed_links_do_not_confuse_sibling_checks(check, source_check):
    items = structured_items("smoke", 123, {"items": 1})
    records = [
        Exchange(
            exchange_id=item.item_id,
            run_id="run",
            endpoint_id="peer",
            suite_id="S3",
            item_id=item.item_id,
            sample_idx=0,
            request={},
            attempts=[],
            status=200,
            response={
                "choices": [
                    {
                        "message": {"content": json.dumps(item.expected.get("value", {}))},
                        "finish_reason": "stop",
                    }
                ]
            },
        )
        for item in items
    ]
    endpoint = EndpointSpec(
        id="peer",
        kind="openai_compat",
        base_url="https://unused.invalid/v1",
        model="m",
        claimed_profile="gpt-oss-120b",
    )
    samples = {"peer": json_samples(items, records, endpoint)}
    pack = ReferencePack(
        pack_id="pack",
        profile="p",
        tier="BASELINE",
        manifest={"source_run": "run"},
        outcomes={"*": {check: {"item_ids": ["123"], "peer_ids": ["peer"]}}},
    )
    links = outcome_items(pack, items, records, samples)
    expected = next(i.item_id for i in items if i.expected["check_id"] == source_check)
    assert links.item_ids == frozenset([expected])
    assert links.aggregate_links[0].exchange_ids == (expected,)


@pytest.mark.parametrize(
    "fault",
    [
        "item",
        "variant",
        "version",
        "capture",
        "index",
        "duplicate_id",
        "endpoint",
        "suite",
        "evidence",
        "sample",
        "key",
        "status",
        "class",
        "trial",
        "unknown_check",
    ],
)
def test_aggregate_falsifiers_cannot_borrow_numeric_key_provenance(fault):
    pack, items, records, samples = protocol()
    if fault == "item":
        items.pop()
    elif fault == "variant":
        items[1].expected["variant"] = "low"
    elif fault == "version":
        items[1].version = "unreviewed-future-version"
    elif fault == "capture":
        records.pop()
    elif fault == "index":
        records[-1].sample_idx = 0
    elif fault == "duplicate_id":
        records[-1].exchange_id = records[0].exchange_id
    elif fault == "endpoint":
        records[-1].endpoint_id = "unrelated"
    elif fault == "suite":
        records[-1].suite_id = "S3"
    elif fault == "evidence":
        samples["b"]["P03"].evidence.pop()
    elif fault == "sample":
        samples.pop("b")
    elif fault == "key":
        samples["b"]["P03"].values.clear()
    elif fault == "status":
        samples["b"]["P03"].status = "ERROR"
    elif fault == "class":
        samples["b"]["P03"].check_class = "report_only"
    elif fault == "trial":
        items[1].expected["trial"] = False
    elif fault == "unknown_check":
        pack.outcomes["target"]["future"] = pack.outcomes["target"].pop("P03")
    with pytest.raises(OutcomeProvenanceError, match="outcome_provenance_unverified"):
        outcome_items(pack, items, records, samples)


def test_numeric_item_collision_and_unselected_peer_do_not_change_group_inputs():
    pack, items, records, samples = protocol()
    items.append(
        Item(
            item_id="0",
            suite_id="other",
            version="external",
            payload={},
            expected={},
            source="external-unreviewed",
        )
    )
    samples["unselected"] = copy.deepcopy(samples["a"])
    links = outcome_items(pack, items, records, samples)
    assert "0" not in links.item_ids
    assert {row.endpoint_id for row in links.aggregate_links} == {"a", "b"}


def test_direct_keys_preserve_existing_binding_and_unknown_keys_refuse():
    item = Item(item_id="literal", suite_id="S2", version="v", payload={}, expected={})
    pack = ReferencePack(
        pack_id="pack",
        profile="p",
        tier="BASELINE",
        outcomes={"*": {"T02": {"item_ids": ["literal"], "peer_ids": ["peer"]}}},
    )
    links = outcome_items(pack, [item], [], {})
    assert links.item_ids == frozenset(["literal"]) and links.aggregate_links == ()
    pack.outcomes["*"]["T02"]["item_ids"] = ["missing"]
    with pytest.raises(OutcomeProvenanceError):
        outcome_items(pack, [item], [], {})


def test_duplicate_item_definitions_refuse_ambiguous_ownership():
    pack, items, records, samples = protocol()
    items.append(items[0].model_copy(deep=True))
    with pytest.raises(OutcomeProvenanceError):
        outcome_items(pack, items, records, samples)


@pytest.mark.parametrize("trial", [False, 0.0, "0"])
def test_protocol_python_equal_trial_alias_cannot_hide_an_extra_contributor(trial):
    pack, items, records, samples = protocol()
    extra = items[0].model_copy(deep=True)
    extra.item_id = "extra-input"
    extra.expected["trial"] = trial
    items.append(extra)
    with pytest.raises(OutcomeProvenanceError):
        outcome_items(pack, items, records, samples)


def test_unknown_numeric_check_cannot_borrow_an_existing_item_id():
    pack, items, records, samples = protocol()
    items.append(Item(item_id="0", suite_id="S2", version="v", payload={}, expected={}))
    pack.outcomes["target"]["future"] = pack.outcomes["target"].pop("P03")
    with pytest.raises(OutcomeProvenanceError):
        outcome_items(pack, items, records, samples)


def test_json_stringification_alias_cannot_hide_an_extra_contributor():
    items = structured_items("smoke", 123, {"items": 1})
    duplicate = next(item for item in items if item.expected["check_id"] == "J01").model_copy(
        deep=True
    )
    duplicate.item_id = "extra-json-input"
    duplicate.expected["trial"] = 123
    items.append(duplicate)
    records = [
        Exchange(
            exchange_id=item.item_id,
            run_id="run",
            endpoint_id="peer",
            suite_id="S3",
            item_id=item.item_id,
            sample_idx=0,
            request={},
            attempts=[],
            status=200,
            response={
                "choices": [
                    {
                        "message": {"content": json.dumps(item.expected.get("value", {}))},
                        "finish_reason": "stop",
                    }
                ]
            },
        )
        for item in items
    ]
    endpoint = EndpointSpec(
        id="peer",
        kind="openai_compat",
        base_url="https://unused.invalid/v1",
        model="m",
        claimed_profile="gpt-oss-120b",
    )
    samples = {"peer": json_samples(items, records, endpoint)}
    assert samples["peer"]["J01"].values["123"] == 1
    pack = ReferencePack(
        pack_id="pack",
        profile="p",
        tier="BASELINE",
        manifest={"source_run": "run"},
        outcomes={"*": {"J01": {"item_ids": ["123"], "peer_ids": ["peer"]}}},
    )
    with pytest.raises(OutcomeProvenanceError):
        outcome_items(pack, items, records, samples)


def test_unselected_replay_captures_do_not_replace_primary_trial_evidence():
    pack, items, records, samples = protocol()
    extra = records[-1].model_copy(deep=True)
    extra.run_id = "replay"
    extra.exchange_id = "unselected-replay"
    original = outcome_items(pack, items, records, samples)
    assert outcome_items(pack, items, [*records, extra], samples) == original
    records.pop()
    with pytest.raises(OutcomeProvenanceError):
        outcome_items(pack, items, [*records, extra], samples)


def test_logprob_calibration_resolves_its_structure_probe_without_reading_text():
    item = probe("P06.structure", 1, "logprobs", "private prompt", {})
    record = Exchange(
        exchange_id="logprob",
        run_id="run",
        endpoint_id="peer",
        suite_id="S1",
        item_id=item.item_id,
        sample_idx=0,
        request={},
        attempts=[],
        status=200,
    )
    score = Samples(
        check_id="P06.calibration",
        check_class="comparative",
        values={"1": -0.5},
        evidence=[record.exchange_id],
    )
    pack = ReferencePack(
        pack_id="pack",
        profile="p",
        tier="BASELINE",
        manifest={"source_run": "run"},
        outcomes={"*": {"P06.calibration": {"item_ids": ["1"], "peer_ids": ["peer"]}}},
    )
    links = outcome_items(pack, [item], [record], {"peer": {score.check_id: score}})
    assert links.item_ids == frozenset([item.item_id])


@pytest.mark.parametrize("fault", [None, "external", "license", "missing_capture", "version"])
def test_current_contributor_rights_revalidate_staging_without_replacing_old_bytes(
    tmp_path, monkeypatch, fault
):
    from types import SimpleNamespace

    from servegauge.core.hashing import content_hash
    from servegauge.core.ids import safe_filename
    from servegauge.refs.pack import ReferenceEndpoint, ReferenceRun, build_pack, save_pack
    from servegauge.storage.reference_export import (
        reference_preflight,
        source_state,
        stage_reference,
    )

    _, items, records, _ = protocol()
    originals = [r for r in records if r.endpoint_id == "a"]
    for peer in ["c", "d"]:
        records.extend(
            r.model_copy(update={"endpoint_id": peer, "exchange_id": peer + r.exchange_id})
            for r in originals
        )
    scores = {
        peer: protocol_samples(items, records, peer, load_profile("gpt-oss-120b"), load_policy())
        for peer in "abcd"
    }
    run = ReferenceRun(
        run_id="run",
        profile="p",
        configuration={},
        item_set_hashes={"S1": "sealed"},
        endpoints=[
            ReferenceEndpoint(id=peer, profile="p", kind="openai_compat") for peer in "abcd"
        ],
        checks={
            "P03": {
                peer: {key: [value] for key, value in scores[peer]["P03"].values.items()}
                for peer in "abcd"
            }
        },
    )
    pack = build_pack(run, "R3", seed=1)
    raw = tmp_path / "data/raw" / safe_filename(run.run_id)
    raw.mkdir(parents=True)
    sealed = {"run_ids": [run.run_id], "metadata": {}}
    (raw / "analysis.json").write_text(json.dumps({"value": sealed, "hash": content_hash(sealed)}))
    analysis = SimpleNamespace(
        plan=SimpleNamespace(endpoints=run.endpoints),
        items=items,
        records=records,
        checks=[],
        samples=scores,
        manifest={},
    )
    monkeypatch.setattr("servegauge.storage.reference_export.reconstruct", lambda *a: analysis)
    monkeypatch.setattr(
        "servegauge.storage.reference_export.export_outcomes",
        lambda *a: run.model_dump(mode="json"),
    )
    before = source_state(tmp_path, pack)
    assert before["rights_verified"] and not before["blockers"]
    save_pack(pack, tmp_path / "refs")
    out = tmp_path / "reviewed-copy"
    staged = stage_reference(tmp_path, pack.pack_id, out)
    original = {
        name: (out / name).read_bytes() for name in [*staged["files"], "local-selection.json"]
    }
    if fault == "external":
        items[1].source = "external-unreviewed"
    elif fault == "license":
        items[1].license = "unverified"
    elif fault == "missing_capture":
        records.pop()
    elif fault == "version":
        items[1].version = "unknown"
    actual = source_state(tmp_path, pack)
    assert actual["rights_verified"] is (fault is None)
    if fault:
        assert "source_rights_unverified" in actual["blockers"]
        assert actual["binding"] != before["binding"]
        with pytest.raises(ValueError):
            reference_preflight(tmp_path, pack.pack_id, out)
        blocked = tmp_path / "unverified-copy"
        assert (
            "source_rights_unverified"
            in stage_reference(tmp_path, pack.pack_id, blocked)["publication_blockers"]
        )
        assert "license: cc-by-4.0" not in (blocked / "README.md").read_text()
    else:
        assert reference_preflight(tmp_path, pack.pack_id, out) == {
            name: original[name] for name in staged["files"]
        }
    assert original == {name: (out / name).read_bytes() for name in original}
