import copy

import numpy as np
import pytest

from servegauge.core.hashing import content_hash
from servegauge.embeddings.local import LocalEmbeddingSpec
from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.refs.fingerprint import compare_fingerprint
from servegauge.refs.pack import ReferenceEndpoint, ReferenceRun, build_pack, load_pack, save_pack
from servegauge.scoring.fingerprint_collection import fingerprint_samples
from servegauge.scoring.fingerprint_results import fingerprint_results
from servegauge.suites.s6_fingerprint import fingerprint_items
from servegauge.validation.capture import captures, endpoint
from servegauge.verdicts.config import load_policy
from servegauge.verdicts.engine import decide_family


def embedding_source(name, seeds, control=None):
    policy = load_policy()
    items = fingerprint_items("full", 14, {"methods": ["FP4"], "fp4_cells": 2, "fp4_samples": 16})
    samples, endpoints, declarations = {}, [], []
    for seed in seeds:
        ep = endpoint(f"mock:{name}-{seed}", seed, clean=control is None)
        ep.metadata["mock_declaration"]["backend"] = "other:" + control if control else "scripted"
        ep.metadata["fingerprint_sampling"] = {
            "independent_sessions": True,
            "matched_sampling_design": True,
            "independent_reference_captures": True,
            "independent_prompt_strata": True,
            "session_size": 1,
        }
        rows = captures(
            items,
            ScriptedBackend(seed, FaultConfig(), model_control=control, sampling_version="v2"),
            ep,
            name,
        )
        samples[ep.id] = fingerprint_samples(items, rows, ep, policy)
        endpoints.append(
            ReferenceEndpoint(
                id=ep.id, profile=ep.claimed_profile, kind="mock", seed=seed, clean=control is None
            )
        )
        declarations.append(ep.model_dump(mode="json"))
    return ReferenceRun(
        run_id=name,
        profile="gpt-oss-120b",
        configuration={"policy_hash": content_hash(policy.model_dump(mode="json"))},
        item_set_hashes={"S6": content_hash([i.model_dump(mode="json") for i in items])},
        endpoints=endpoints,
        checks={},
        fingerprints=samples,
        fingerprint_policy=policy,
        manifest={"endpoint_metadata": declarations},
    )


class FixtureEncoder:
    def __init__(self, design):
        self.spec = LocalEmbeddingSpec.model_validate(design["embedding_model"])
        self.identity = {"artifact_hash": self.spec.artifact_hash, "runtime": {"fixture": "v1"}}

    def encode(self, texts):
        return np.asarray([[len(t), sum(map(ord, t)) % 11, 1] for t in texts], dtype=float)


@pytest.fixture
def fixture_runtime(monkeypatch):
    import servegauge.refs.fingerprint as module

    monkeypatch.setattr(
        module, "get_encoder", lambda sample: FixtureEncoder(sample.details["design"])
    )


def test_explicit_fp4_inventory_is_pinned_and_does_not_request_logprobs():
    items = fingerprint_items("full", 14, {"methods": ["FP4"]})
    assert len(items) == 25 and sum(i.expected["samples"] for i in items) == 250
    assert all(
        i.expected["check_id"] == "FP4" and not i.payload["params"].get("logprobs") for i in items
    )
    assert LocalEmbeddingSpec.model_validate(items[0].expected["design"]["embedding_model"])
    assert "FP4" not in {i.expected["check_id"] for i in fingerprint_items("full", 14)}


def test_raw_collection_keeps_full_text_without_loading_an_embedding_runtime():
    source = embedding_source("raw", [51])
    sample = next(iter(source.fingerprints.values()))["FP4"]
    assert sample.status is None and sample.details["captures"] == 32
    assert all(row["text"] for row in sample.details["observations"])
    assert "embedding_identity" not in sample.details


def test_captured_fp4_pack_round_trip_and_family_info(fixture_runtime, tmp_path):
    source = embedding_source("refs", [51, 52, 53])
    controls = [
        embedding_source("ca", [61], "control-a"),
        embedding_source("cb", [62], "control-b"),
    ]
    original = copy.deepcopy((source, controls))
    pack = build_pack(source, "MOCK", seed=11, controls=controls)
    metric = pack.fingerprints["views"]["*"]["metrics"]["FP4"]
    assert metric["embedding_identity"]["runtime"] == {"fixture": "v1"}
    assert metric["calibration"][0]["prompt_aggregation"] == "sum"
    assert metric["equivalence_qualified"] is False and metric["n"] == 32
    assert load_pack(str(save_pack(pack, tmp_path))) == pack
    target = embedding_source("target", [71])
    ep = target.endpoints[0].id
    checks = fingerprint_results(
        target.fingerprints[ep],
        endpoint_id=ep,
        run_id=target.run_id,
        pack=pack,
        policy=load_policy(),
        seed=11,
    )
    results = decide_family([], load_policy(), external=checks)
    assert len(results) == 1 and results[0].verdict == "INFO"
    assert results[0].margin is None and results[0].adjusted_p is not None
    assert results[0].n == 32 and results[0].ci_low is not None
    assert (source, controls) == original


def test_unavailable_model_is_saved_as_unavailable_and_never_invents_a_metric(monkeypatch):
    import servegauge.refs.fingerprint as module

    def missing(sample):
        raise ValueError("local_embedding_assets_required")

    monkeypatch.setattr(module, "get_encoder", missing)
    pack = build_pack(
        embedding_source("refs", [51, 52, 53]),
        "MOCK",
        seed=11,
        controls=[
            embedding_source("ca", [61], "control-a"),
            embedding_source("cb", [62], "control-b"),
        ],
    )
    view = pack.fingerprints["views"]["*"]
    assert (
        "FP4" not in view["metrics"]
        and view["unavailable"]["FP4"] == "local_embedding_assets_required"
    )


def test_embedding_runtime_is_bound_to_pack_identity_and_target_comparison(
    fixture_runtime, monkeypatch
):
    import servegauge.refs.fingerprint as module

    source = embedding_source("refs", [51, 52, 53])
    controls = [
        embedding_source("ca", [61], "control-a"),
        embedding_source("cb", [62], "control-b"),
    ]
    first = build_pack(source, "MOCK", seed=11, controls=controls)

    def changed(sample):
        encoder = FixtureEncoder(sample.details["design"])
        encoder.identity["runtime"] = {"fixture": "v2"}
        return encoder

    monkeypatch.setattr(module, "get_encoder", changed)
    second = build_pack(source, "MOCK", seed=11, controls=controls)
    assert first.pack_id != second.pack_id
    target = embedding_source("target", [71])
    ep = target.endpoints[0].id
    with pytest.raises(ValueError, match="embedding_runtime_mismatch"):
        compare_fingerprint(
            first,
            target.fingerprints[ep]["FP4"],
            target=ep,
            method="FP4",
            policy=load_policy(),
            seed=11,
        )


@pytest.mark.parametrize("case", ["incomplete", "cache", "changed_model", "reused_capture"])
def test_fp4_retains_existing_evidence_guards(fixture_runtime, case):
    from servegauge.scoring.fingerprint_comparison import collection_cells

    sources = [embedding_source(f"s{i}", [80 + i]) for i in range(4)]
    samples = [next(iter(s.fingerprints.values()))["FP4"] for s in sources]
    if case == "incomplete":
        samples[0].details["observations"].pop()
    elif case == "cache":
        samples[0].details["sampling_design"]["independent_sessions"] = False
    elif case == "changed_model":
        samples[0].details["design"]["embedding_model"]["revision"] = "a" * 40
    else:
        samples[0].details["observations"][0]["exchange_id"] = samples[1].details["observations"][
            0
        ]["exchange_id"]
    with pytest.raises(ValueError):
        collection_cells(
            samples,
            method="FP4",
            policy=load_policy(),
            encoder=FixtureEncoder(samples[1].details["design"]),
        )


def test_runtime_factory_requires_explicit_local_assets(monkeypatch):
    from servegauge.embeddings.runtime import get_encoder

    monkeypatch.delenv("SERVEGAUGE_EMBED_ASSETS", raising=False)
    sample = next(iter(embedding_source("raw", [51]).fingerprints.values()))["FP4"]
    with pytest.raises(ValueError, match="local_embedding_assets_required"):
        get_encoder(sample)
