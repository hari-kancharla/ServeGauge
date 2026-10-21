"""Reusable reference copies keep measurements and exclude transport-only context."""

import copy
import json

import pytest
from unit.test_refs import reference_run

from servegauge.core.hashing import canonical_json
from servegauge.refs.pack import build_pack, load_pack, reference_for
from servegauge.refs.publication import public_pack

PRIVATE = "PRIVATE_REFERENCE_TRANSPORT_CONTEXT"


def example():
    source = reference_run()
    source.manifest = {
        "workspace": PRIVATE,
        "endpoint_metadata": [
            {
                "id": "a",
                "kind": "mock",
                "model": "mock",
                "claimed_profile": "p",
                "base_url": "http://127.0.0.1:8011/v1",
                "api_key_env": "OWNER_KEY",
                "metadata": {
                    "account_id": PRIVATE,
                    "mock_declaration": {"seed": 0, "backend": "scripted", "clean": True},
                },
            }
        ],
    }
    return source, build_pack(source, "R3", seed=8)


def test_public_copy_has_new_identity_and_exact_reference_behavior(tmp_path):
    source, original = example()
    before = original.model_dump(mode="json")
    public = public_pack(original)
    assert public.pack_id != original.pack_id
    assert public.manifest["source_pack_id"] == original.pack_id
    assert public.manifest["source_pack_content_hash"] == original.manifest["pack_content_hash"]
    assert original.model_dump(mode="json") == before
    assert public.outcomes == original.outcomes and public.calibration == original.calibration
    assert public.manifest["configuration"] == original.manifest["configuration"]
    path = tmp_path / "pack.json"
    path.write_bytes(canonical_json(public.model_dump(mode="json")) + b"\n")
    loaded = load_pack(str(path))
    for target in "abcd":
        args = {
            "target": target,
            "check": "T02",
            "item_ids": sorted(source.checks["T02"][target]),
            "configuration": source.configuration,
        }
        assert reference_for(loaded, **args) == reference_for(original, **args)
    assert PRIVATE.encode() not in path.read_bytes()
    assert b"base_url" not in path.read_bytes() and b"api_key_env" not in path.read_bytes()
    declaration = loaded.manifest["source_manifest"]["endpoint_metadata"][0]
    assert declaration["metadata"]["mock_declaration"]["seed"] == 0


def test_public_projection_is_deterministic_and_binds_changed_measurements():
    _, original = example()
    assert public_pack(original) == public_pack(original)
    changed = original.model_copy(deep=True)
    changed.outcomes["a"]["T02"]["values"][0] = 0.33
    # A forged source digest is not a valid input to the public projection.
    with pytest.raises(ValueError):
        public_pack(changed)


@pytest.mark.parametrize("value", ["/Users/owner/private/model", "C:\\Users\\owner\\model"])
def test_private_path_in_semantic_configuration_refuses_instead_of_changing_it(value):
    source = reference_run()
    source.configuration["local_asset"] = value
    original = build_pack(source, "R3", seed=8)
    before = copy.deepcopy(original.model_dump(mode="json"))
    with pytest.raises(ValueError):
        public_pack(original)
    assert before == original.model_dump(mode="json")


def test_fingerprint_observations_and_declarations_keep_semantic_payload():
    source = reference_run()
    original = build_pack(source, "R3", seed=8)
    # The real fingerprint fixture/roundtrip is covered separately. This fixture
    # isolates removal of transport metadata without rewriting sample observations.
    from servegauge.refs.pack import _pack_digest

    declaration = example()[0].manifest["endpoint_metadata"][0]
    original.fingerprints = {
        "views": {"a": {"references": {"b": {"FP2": {"text": "generated response"}}}}},
        "controls": [{"declarations": [declaration]}],
        "reference_sources": {"capture/b": {"declaration": declaration}},
    }
    original.manifest["pack_content_hash"] = _pack_digest(original)
    public = public_pack(original)
    assert public.fingerprints["views"] == original.fingerprints["views"]
    assert PRIVATE not in json.dumps(public.model_dump(mode="json"))
    assert (
        original.fingerprints["controls"][0]["declarations"][0]["metadata"]["account_id"] == PRIVATE
    )


def test_embedded_response_ids_and_execution_context_are_removed_without_changing_logprobs():
    from servegauge.core.hashing import content_hash
    from servegauge.refs.pack import _pack_digest
    from servegauge.scoring.fingerprint_evidence import read_response

    _, original = example()
    response = {
        "id": PRIVATE,
        "usage": {"account_id": PRIVATE},
        "choices": [
            {
                "message": {"role": "assistant", "content": "A", "reasoning": PRIVATE},
                "finish_reason": "stop",
                "logprobs": {
                    "content": [
                        {
                            "token": "A",
                            "logprob": -0.7,
                            "bytes": [65],
                            "top_logprobs": [
                                {"token": "A", "logprob": -0.7},
                                {"token": "B", "logprob": -0.8},
                            ],
                        }
                    ]
                },
            }
        ],
    }
    observation = {
        "cell": "one",
        "draw": 0,
        "exchange_id": "a" * 64,
        "text": "A",
        "reason": None,
        "response": response,
        "metadata": {"effective_output_cap": 1, "execution_source": {"workspace": PRIVATE}},
    }
    original.fingerprints = {
        "views": {
            "*": {
                "references": {
                    "peer": {
                        "FP3": {
                            "details": {
                                "observations": [observation],
                                "observation_hash": content_hash([observation]),
                            }
                        }
                    }
                }
            }
        }
    }
    original.manifest["pack_content_hash"] = _pack_digest(original)
    public = public_pack(original)
    details = public.fingerprints["views"]["*"]["references"]["peer"]["FP3"]["details"]
    projected = details["observations"][0]
    assert projected["metadata"] == {"effective_output_cap": 1}
    assert read_response(projected["response"], tolerance=1e-6) == read_response(
        response, tolerance=1e-6
    )
    assert details["source_observation_hash"] == content_hash([observation])
    assert details["observation_hash"] == content_hash(details["observations"])
    assert PRIVATE not in json.dumps(public.model_dump(mode="json"))
    assert (
        original.fingerprints["views"]["*"]["references"]["peer"]["FP3"]["details"]["observations"][
            0
        ]
        == observation
    )


def test_source_numeric_views_require_actual_raw_derived_peer_rows_and_n():
    from servegauge.refs.publication import outcomes_match

    source, pack = example()
    assert outcomes_match(pack, source.model_dump(mode="json"))
    altered = pack.model_copy(deep=True)
    altered.outcomes["a"]["T02"]["peer_item_means"][0][0] += 0.125
    assert not outcomes_match(altered, source.model_dump(mode="json"))
    altered = pack.model_copy(deep=True)
    altered.calibration["a"]["T02"]["n"] += 1
    assert not outcomes_match(altered, source.model_dump(mode="json"))
