"""The experimental profile cannot inherit unsupported accounting/protocol counters."""

import pytest

from servegauge.core.models import EndpointSpec
from servegauge.runner.plan import RunPlan
from servegauge.selfhost.study_collection import study_fingerprint_items
from servegauge.selfhost.study_recipe import STUDY_PROFILE
from servegauge.suites.s6_fingerprint import fingerprint_items

RECIPE = "qwen3-study-collection-v1"


def plan(suites=None, **changes):
    value = {
        "plan_id": "study",
        "profile": STUDY_PROFILE,
        "level": "smoke",
        "seed": 101,
        "reference": {"tier": "BASELINE"},
        "max_usd": 0,
        "endpoints": [
            EndpointSpec(
                id="study",
                kind="mock",
                base_url="http://127.0.0.1:8011/v1",
                model="mock",
                claimed_profile=STUDY_PROFILE,
            ).model_dump(mode="json")
        ],
        "suites": suites or ["S6"],
        "overrides": {"S6": {"collection_recipe": RECIPE}},
    }
    return RunPlan.model_validate(value | changes)


@pytest.mark.parametrize("suite", ["S1", "S2", "S3", "S4", "S5", "S7", "S8"])
def test_unsupported_experiment_suites_refuse_before_generation_or_dispatch(suite):
    with pytest.raises(ValueError):
        plan(["S6", suite])


def test_explicit_collection_recipe_and_claimed_profile_are_required():
    assert plan().suites == ["S6"]
    assert plan(["S0", "S6"]).suites == ["S0", "S6"]
    with pytest.raises(ValueError):
        plan(overrides={})
    endpoint = plan().endpoints[0].model_copy(update={"claimed_profile": "gpt-oss-120b"})
    with pytest.raises(ValueError):
        plan(endpoints=[endpoint])


def test_study_request_controls_and_versions_are_distinct_from_legacy():
    ordinary = fingerprint_items("smoke", 101)
    study = study_fingerprint_items(
        "smoke", 101, {"collection_recipe": RECIPE}, profile_id=STUDY_PROFILE
    )
    assert len(ordinary) == len(study)
    for original, actual in zip(ordinary, study, strict=True):
        assert actual.expected["samples"] == original.expected["samples"]
        assert actual.payload["messages"] == original.payload["messages"]
        assert "reasoning_effort" in original.payload["params"]
        assert "reasoning_effort" not in actual.payload["params"]
        assert actual.payload["params"]["top_p"] == 1.0
        assert actual.version != original.version
        assert actual.expected["design"]["collection_recipe"] == RECIPE
        assert actual.expected["design_hash"] != original.expected["design_hash"]


def test_unknown_recipe_or_wrong_profile_never_selects_the_study_controls():
    for recipe, profile in [("unknown", STUDY_PROFILE), (RECIPE, "gpt-oss-120b"), (RECIPE, None)]:
        with pytest.raises(ValueError):
            study_fingerprint_items("smoke", 101, {"collection_recipe": recipe}, profile_id=profile)
    with pytest.raises(ValueError):
        study_fingerprint_items("smoke", 101, profile_id=STUDY_PROFILE)


def test_packaged_study_profile_matches_sourced_profile():
    import hashlib
    import json
    from importlib.resources import files
    from pathlib import Path

    from servegauge.profiles import load_profile

    name = "profiles/qwen3-0.6b-study.yaml"
    bundle = files("servegauge").joinpath("resources/workspace-v1")
    manifest = json.loads(bundle.joinpath("manifest.json").read_bytes())
    blob = bundle.joinpath(name).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == manifest["files"][name]
    assert blob == Path(name).read_bytes()
    profile = load_profile(STUDY_PROFILE)
    assert profile.tokenizer["qualified_for_counts"] is False
    assert profile.reference_tiers == []
