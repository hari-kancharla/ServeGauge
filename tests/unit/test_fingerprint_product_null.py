import copy
import json

import numpy as np
import pytest
from typer.testing import CliRunner

from servegauge.cli import app
from servegauge.core.hashing import content_hash
from servegauge.validation.fingerprint_product import configuration
from servegauge.validation.fingerprint_product_null import (
    _cached_chunk,
    _checked_trial,
    family_result,
    null_components,
    null_trial,
)
from servegauge.verdicts.config import load_policy


def small_configuration():
    cfg = copy.deepcopy(configuration())
    cfg["design"].update(fp1_cells=2, fp1_samples=4, fp2_cells=2, fp2_samples=4, fp3_samples=4)
    return cfg


def test_components_are_seeded_complete_and_share_fp3_anchor_observations():
    cfg = small_configuration()
    a, b = null_components(cfg, 101), null_components(cfg, 101)
    assert set(a) == {"FP1", "FP2", "FP3-mean", "FP3-dispersion"}
    for method in a:
        assert all(np.array_equal(x.a, y.a) for x, y in zip(a[method], b[method], strict=True))
        assert all(np.isfinite(c.a).all() for c in a[method])
    mean, dispersion = a["FP3-mean"][0], a["FP3-dispersion"][0]
    for name in ("a", "b", "c", "d"):
        first, second = getattr(mean, name), getattr(dispersion, name)
        assert first.shape == (4, 2) and second.shape == (4, 4)
        assert np.array_equal(second[:, :2], first)
        assert np.array_equal(second[:, 2:], first**2)
    assert not np.array_equal(mean.a, mean.b)
    assert not np.array_equal(mean.a, null_components(cfg, 102)["FP3-mean"][0].a)


def test_bonferroni_subtests_precede_one_three_check_holm_family():
    values = {"FP1": 0.02, "FP2": 0.04, "FP3-mean": 0.001, "FP3-dispersion": 0.7}
    result = family_result(values, alpha=0.05)
    assert result["check_p_values"] == {"FP1": 0.02, "FP2": 0.04, "FP3": 0.002}
    assert result["adjusted_p_values"] == pytest.approx({"FP1": 0.04, "FP2": 0.04, "FP3": 0.006})
    assert result["family_rejected"] is True
    with pytest.raises(ValueError):
        family_result({**values, "FP2": float("nan")}, alpha=0.05)
    with pytest.raises(ValueError):
        family_result({"FP1": 0.2}, alpha=0.05)


def test_constant_complete_null_uses_actual_exact_randomization():
    cfg = small_configuration()
    cfg["healthy_logit_jitter"] = 0
    cfg["probability_grid"] = [[1, 0, 0, 0]]
    row = null_trial(cfg, load_policy(), 103)
    assert set(row["p_values"].values()) == {1.0}
    assert row["family_rejected"] is False
    assert row["n"] == {"FP1": 8, "FP2": 8, "FP3-mean": 4, "FP3-dispersion": 4}


@pytest.mark.parametrize("field", ["seed", "n", "p_values", "adjusted_p_values", "family_rejected"])
def test_seeded_trial_refuses_tampered_inference(field):
    cfg, policy = small_configuration(), load_policy()
    row = null_trial(cfg, policy, 105)
    assert _checked_trial(row, cfg, policy, 105) == row
    changed = copy.deepcopy(row)
    if field == "seed":
        changed[field] += 1
    elif field == "n":
        changed[field]["FP1"] += 1
    elif field == "p_values":
        changed[field]["FP1"] = -0.1
    elif field == "adjusted_p_values":
        changed[field]["FP3"] = -0.1
    else:
        changed[field] = not row[field]
    with pytest.raises(ValueError):
        _checked_trial(changed, cfg, policy, 105)


def test_cached_chunks_bind_source_config_full_trials_and_expected_seeds(tmp_path):
    cfg, policy = small_configuration(), load_policy()
    header = {"source_hash": "source", "configuration_hash": "configuration"}
    path = tmp_path / "chunk.json"
    assert _cached_chunk(path, header, [107], cfg, policy) is None
    rows = [null_trial(cfg, policy, 107)]
    value = {**header, "trials": rows, "trials_hash": content_hash(rows)}
    path.write_text(json.dumps(value))
    assert _cached_chunk(path, header, [107], cfg, policy) == rows
    for changed in [
        {**value, "source_hash": "new-source"},
        {**value, "configuration_hash": "new-config"},
        {**value, "trials_hash": "corrupted"},
    ]:
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            _cached_chunk(path, header, [107], cfg, policy)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        _cached_chunk(path, header, [108], cfg, policy)


@pytest.mark.parametrize("accepted,exit_code", [(True, 0), (False, 1)])
def test_null_cli_reports_only_the_observed_scoped_acceptance(
    monkeypatch, tmp_path, accepted, exit_code
):
    import servegauge.validation.fingerprint_product_null as module

    monkeypatch.setattr(
        module, "null_study", lambda out: {"accepted": accepted, "scope": "iid grid only"}
    )
    result = CliRunner().invoke(
        app, ["validate", "fingerprint-product-null", "--out", str(tmp_path)]
    )
    assert result.exit_code == exit_code
    assert json.loads(result.stdout)["scope"] == "iid grid only"
    assert json.loads(result.stdout)["model_api_calls"] == 0


def test_null_cli_refuses_changed_or_corrupt_evidence_without_echoing_it(monkeypatch, tmp_path):
    import servegauge.validation.fingerprint_product_null as module

    def refuse(out):
        raise ValueError("private-corrupted-input")

    monkeypatch.setattr(module, "null_study", refuse)
    result = CliRunner().invoke(
        app, ["validate", "fingerprint-product-null", "--out", str(tmp_path)]
    )
    assert result.exit_code == 3
    assert "private-corrupted-input" not in result.output


def test_study_resumes_intact_chunks_and_refuses_changed_source_or_summary(monkeypatch, tmp_path):
    import servegauge.validation.fingerprint_product_null as module

    cfg = small_configuration()
    cfg.update(null_replicates=4, workers=1)
    monkeypatch.setattr(module, "configuration", lambda: cfg)
    first = module.null_study(tmp_path)
    assert len(first["trials"]) == 4
    assert len({r["seed"] for r in first["trials"]}) == 4
    assert not list(tmp_path.rglob("*.part"))

    def unexpected(*args):
        raise AssertionError("intact saved trials must be reused")

    monkeypatch.setattr(module, "null_trial", unexpected)
    assert module.null_study(tmp_path) == first
    path = tmp_path / "null.json"
    path.unlink()
    assert module.null_study(tmp_path) == first
    altered = copy.deepcopy(first)
    altered["accepted"] = not first["accepted"]
    altered.pop("result_hash")
    altered["result_hash"] = content_hash(altered)
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="summary"):
        module.null_study(tmp_path)
    path.write_text(json.dumps(first))
    cfg["healthy_logit_jitter"] += 0.1
    with pytest.raises(ValueError, match="configuration"):
        module.null_study(tmp_path)
