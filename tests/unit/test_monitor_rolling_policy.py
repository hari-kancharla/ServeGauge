"""Versioned policy and complete batch-boundary known-answer examples."""

import pytest

from servegauge.observatory.rolling import first_rolling_cut, load_rolling_policy


def test_opposite_complete_batches_have_directed_effect_evidence():
    p = load_rolling_policy()
    cut = first_rolling_cut([(1,) * 400, (0,) * 400], p)
    assert cut.left_n == cut.right_n == 400
    assert cut.difference == -1 and cut.ci_high < 0 and not cut.qualified
    assert first_rolling_cut([(1,) * 400] * 3, p) is None


@pytest.mark.parametrize(
    "rows",
    [[(1,) * 399, (0,) * 399], [(True,) * 400], [(1,) * 400, (0,) * 800], [()], [(1,) * 400] * 122],
)
def test_noncanonical_or_unbounded_batches_refuse(rows):
    with pytest.raises(ValueError):
        first_rolling_cut(rows, load_rolling_policy())


def test_missing_or_changed_canonical_contract_refuses(tmp_path):
    from pathlib import Path

    import yaml
    from pydantic import ValidationError

    from servegauge.observatory.rolling import RollingTransportPolicy

    payload = yaml.safe_load(Path("configs/monitor_transport_v2.yaml").read_text())
    for key, value in [
        ("max_updates", True),
        ("max_streams", 101),
        ("horizon_days", 31),
        ("family_alpha", 0.06),
        ("max_history_batches", 122),
    ]:
        with pytest.raises(ValidationError):
            RollingTransportPolicy.model_validate({**payload, key: value})
    bad = tmp_path / "policy.yaml"
    bad.write_text(yaml.safe_dump({**payload, "batch_sizes": {"core": 399, "full": 800}}))
    with pytest.raises(ValueError, match="invalid_or_unavailable"):
        load_rolling_policy(bad)
