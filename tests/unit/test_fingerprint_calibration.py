import copy
import json

import numpy as np
import pytest
from typer.testing import CliRunner

from servegauge.cli import app
from servegauge.scoring.fingerprint_analysis import analyze_evidence
from servegauge.scoring.fingerprint_calibration import fit_signal_calibration
from servegauge.validation.common import source_hash
from servegauge.validation.robustness_study import configuration, generate_evidence
from servegauge.verdicts.config import load_policy


def calibration():
    cfg = configuration()
    banks = [
        {
            "profile": p,
            "batches": [
                {
                    "phase": "calibration",
                    "groups": g,
                    "trials": [
                        {"eligible": True, "results": {"x": {"adjusted_p": float(v)}}}
                        for v in np.linspace(0, 1, cfg["calibration_replicates"])
                    ],
                }
                for g in cfg["effective_group_sizes"]
            ],
        }
        for p in cfg["profiles"]
    ]
    fitted = fit_signal_calibration(banks, cfg, source_hash="known-source")
    return cfg, banks, fitted


def test_cutoff_uses_calibration_only_and_every_model_upper_bound():
    cfg, _, fitted = calibration()
    for record in fitted.by_groups.values():
        assert 0 < record["cutoff"] <= cfg["alpha"]
        assert all(
            r["interval"][1] <= cfg["maximum_family_false_positive_rate"]
            for r in record["controls"].values()
        )


@pytest.mark.parametrize(
    "mismatch", ["none", "source", "scope", "anchors", "size", "groups", "policy"]
)
def test_calibration_contract_never_silently_transfers(mismatch):
    cfg, _, fitted = calibration()
    e = generate_evidence(11, "iid", groups=30, noise=0, cfg=cfg)
    source = "known-source"
    if mismatch == "source":
        source = "changed-source"
    elif mismatch == "scope":
        e.calibration_scope = None
    elif mismatch == "anchors":
        e.anchor_tokens.reverse()
    elif mismatch == "size":
        fitted.observations_per_session = 1
    elif mismatch == "groups":
        fitted.by_groups.pop("30")
    elif mismatch == "policy":
        fitted.permutations = 1
    result = analyze_evidence(
        e,
        policy=load_policy(),
        minimum_groups=30,
        minimum_anchor_tokens=2,
        signal_calibration=fitted,
        current_source_hash=source,
    )
    if mismatch == "none":
        assert result["status"] == "INFO"
        assert result["signal_policy"]["qualification"] == "matched MOCK calibration"
    else:
        assert result["status"] == "INCONCLUSIVE"
        assert all(c["details"].get("distribution_signal") is None for c in result["checks"])


def test_incomplete_calibration_is_rejected_without_touching_input():
    cfg, banks, _ = calibration()
    banks[0]["batches"][0]["trials"].pop()
    before = copy.deepcopy(banks)
    with pytest.raises(ValueError, match="complete_independent"):
        fit_signal_calibration(banks, cfg, source_hash="known-source")
    assert banks == before


def test_cli_calibration_contract_and_input_preservation(tmp_path):
    cfg, _, fitted = calibration()
    fitted.source_hash = source_hash()
    evidence = generate_evidence(811, "iid", groups=30, noise=0, cfg=cfg)
    path, policy, output = (tmp_path / n for n in ("evidence.json", "policy.json", "result.json"))
    path.write_text(evidence.model_dump_json())
    policy.write_text(fitted.model_dump_json())
    runner = CliRunner()
    args = ["fingerprint", "analyze", str(path), "--calibration", str(policy)]
    result = runner.invoke(app, [*args, "--out", str(output)])
    assert result.exit_code == 0, result.output
    assert (
        json.loads(output.read_text())["signal_policy"]["qualification"]
        == "matched MOCK calibration"
    )
    before = {p: p.read_bytes() for p in (path, policy)}
    for destination in (path, policy):
        refused = runner.invoke(app, [*args, "--out", str(destination)])
        assert refused.exit_code == 3
        assert {p: p.read_bytes() for p in (path, policy)} == before
    fitted.source_hash = "wrong-source"
    policy.write_text(fitted.model_dump_json())
    refused = runner.invoke(app, args)
    assert refused.exit_code == 2 and "calibration_source_mismatch" in refused.output
    policy.write_text('{"private-value":"invalid"}')
    invalid = runner.invoke(app, args)
    assert invalid.exit_code == 3 and "private-value" not in invalid.output
