"""Owner and synthetic modes cannot silently substitute for one another."""

from __future__ import annotations

import copy

import pytest

from servegauge.comparison.authorization import OwnerReview, authorize
from servegauge.core.hashing import content_hash


def manifest(*, synthetic=False, eligible=True):
    return {
        "selection_hash": "a" * 64,
        "controls_hash": "b" * 64,
        "synthetic_only": synthetic,
        "m15_eligible": eligible,
        "source_eligibility": {"round": {"eligible_source_scope": eligible}},
    }


def review(value):
    return OwnerReview(
        kit_hash=content_hash(value),
        selection_hash=value["selection_hash"],
        controls_hash=value["controls_hash"],
        source_run_ids=["round"],
        approved_utc="2026-10-20T00:00:00Z",
        m15_report_site_completed=True,
    )


def test_pure_owner_gate_requires_actual_parent_environment_and_separate_review():
    value = manifest()
    approved = review(value)
    for environment in [{}, {"SERVEGAUGE_OWNER_RUN": "0"}]:
        with pytest.raises(ValueError):
            authorize(
                value,
                kit_hash=content_hash(value),
                mode="owner",
                review=approved,
                environment=environment,
            )
    # Explicit mappings test the pure gate; the test never sets an owner variable.
    authorize(
        value,
        kit_hash=content_hash(value),
        mode="owner",
        review=approved,
        environment={"SERVEGAUGE_OWNER_RUN": "1"},
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "kit",
        "selection",
        "controls",
        "runs",
        "missing_review",
        "ineligible",
        "mock",
        "no_report_site",
    ],
)
def test_owner_execution_refuses_unreviewed_changed_or_ineligible_inputs(mutation):
    value = manifest()
    approved = review(value)
    if mutation == "kit":
        approved.kit_hash = "0" * 64
    elif mutation == "selection":
        approved.selection_hash = "0" * 64
    elif mutation == "controls":
        approved.controls_hash = "0" * 64
    elif mutation == "runs":
        approved.source_run_ids = ["other-round"]
    elif mutation == "missing_review":
        approved = None
    elif mutation == "ineligible":
        value["m15_eligible"] = False
    elif mutation == "mock":
        value["synthetic_only"] = True
    else:
        approved.m15_report_site_completed = False
    with pytest.raises(ValueError):
        authorize(
            value,
            kit_hash=content_hash(value),
            mode="owner",
            review=approved,
            environment={"SERVEGAUGE_OWNER_RUN": "1"},
        )


def test_synthetic_mode_requires_all_source_scope_mock_and_no_owner_context():
    value = manifest(synthetic=True, eligible=False)
    authorize(value, kit_hash=content_hash(value), mode="synthetic", review=None, environment={})
    for changed in [manifest(), manifest(synthetic=False, eligible=False)]:
        with pytest.raises(ValueError):
            authorize(
                changed,
                kit_hash=content_hash(changed),
                mode="synthetic",
                review=None,
                environment={},
            )
    with pytest.raises(ValueError):
        authorize(
            value,
            kit_hash=content_hash(value),
            mode="synthetic",
            review=None,
            environment={"SERVEGAUGE_OWNER_RUN": "1"},
        )
    with pytest.raises(ValueError):
        authorize(value, kit_hash="0" * 64, mode="synthetic", review=None, environment={})


def test_review_and_manifest_are_not_mutated_to_grant_authority():
    value = manifest()
    approved = review(value)
    before = copy.deepcopy(value), approved.model_dump(mode="json")
    authorize(
        value,
        kit_hash=content_hash(value),
        mode="owner",
        review=approved,
        environment={"SERVEGAUGE_OWNER_RUN": "1"},
    )
    assert (value, approved.model_dump(mode="json")) == before
