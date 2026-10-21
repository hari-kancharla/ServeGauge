"""Known-answer budget, epoch and candidate-cut diagnostics; no alarm admission."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from servegauge.observatory.changes import (
    TransportPolicy,
    first_cut,
    load_transport_policy,
    transport_epoch,
)
from servegauge.stats.change_variants import BatchADWIN


def test_candidate_allocation_covers_resets_streams_epochs_and_intervals():
    policy = load_transport_policy()
    assert policy.max_streams == 100 and policy.max_updates == 120
    assert policy.batch_sizes == {"core": 400, "full": 800}
    assert policy.delta == pytest.approx(1.736111111111111e-8)
    per_epoch = policy.max_streams * policy.max_updates**2 * policy.delta
    assert per_epoch == pytest.approx(0.025)
    assert 2 * per_epoch <= policy.family_alpha
    assert 2 * per_epoch <= 1


@pytest.mark.parametrize(
    "change",
    [
        {"family_alpha": 0},
        {"family_alpha": float("nan")},
        {"max_streams": True},
        {"max_streams": 0},
        {"max_updates": 0},
        {"epoch_days": 31},
        {"epoch_false_event_budget": 0},
        {"batch_sizes": {"core": 0, "full": 800}},
    ],
)
def test_unsupported_policy_refuses(change):
    config = load_transport_policy().model_dump()
    config.update(change)
    with pytest.raises(ValidationError):
        TransportPolicy.model_validate(config)


def test_epoch_conversion_is_aware_fixed_and_adjacent():
    policy = load_transport_policy()
    now = datetime(2026, 10, 20, 3, tzinfo=timezone(timedelta(hours=-7)))
    start, end = transport_epoch(now, policy)
    assert start.tzinfo is UTC and start <= now < end
    assert end - start == timedelta(days=30)
    assert transport_epoch(end, policy)[0] == end
    assert transport_epoch(now.astimezone(UTC), policy) == (start, end)
    with pytest.raises(ValueError, match="aware"):
        transport_epoch(now.replace(tzinfo=None), policy)


def test_first_cut_matches_frozen_batch_detector_and_preserves_direction():
    policy = load_transport_policy()
    batches = [[0] * 400, [1] * 400]
    evidence = first_cut(batches, policy)
    assert evidence is not None
    assert evidence.left_batches == 1
    assert evidence.left_n == evidence.right_n == 400
    assert evidence.difference == 1
    assert evidence.ci_low > 0 and evidence.ci_high == 1
    assert evidence.reference_tier == "MOCK" and evidence.qualified is False
    expected = BatchADWIN(delta=policy.delta)
    assert not expected.update(batches[0])
    assert expected.update(batches[1])
    reverse = first_cut(list(reversed(batches)), policy)
    assert reverse is not None and reverse.ci_high < 0
    assert reverse.difference == -1
    assert reverse.ci_low == -1
    assert first_cut([[0] * 400, [0] * 400], policy) is None


@pytest.mark.parametrize("batches", [[[0, None], [1]], [[0, True], [1]], [[], [1]], [[2], [0]]])
def test_missing_or_invalid_bits_are_not_imputed(batches):
    with pytest.raises(ValueError, match="binary"):
        first_cut(batches, load_transport_policy())
