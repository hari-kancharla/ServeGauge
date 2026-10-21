"""Actual reserved fingerprint captures survive a transport-removed public roundtrip."""

import hashlib
import json

import pytest
from integration.test_fingerprint_pack_workflow import capture_run
from typer.testing import CliRunner

from servegauge.cli import app
from servegauge.refs.fingerprint import compare_fingerprint
from servegauge.refs.pack import ReferenceRun, build_pack, load_pack, save_pack
from servegauge.runner.ledger import Ledger
from servegauge.storage.analysis import export_outcomes, reconstruct
from servegauge.storage.reference_export import reference_preflight, source_state, stage_reference
from servegauge.storage.workspace import initialize
from servegauge.verdicts.config import DecisionPolicy


@pytest.mark.slow
async def test_native_fingerprint_public_copy_keeps_comparisons_rights_and_mock_refusal(
    tmp_path, monkeypatch
):
    from pathlib import Path

    monkeypatch.setenv("SERVEGAUGE_TOKENIZER_CACHE", str(Path.cwd() / ".cache/tokenizers"))
    initialize(tmp_path)
    reference = await capture_run(tmp_path, "publish-reference", [1, 2, 3])
    controls = [
        await capture_run(tmp_path, "publish-control-a", [11], "other:control-a"),
        await capture_run(tmp_path, "publish-control-b", [12], "other:control-b"),
    ]
    target = await capture_run(tmp_path, "publish-target", [104])
    monkeypatch.chdir(tmp_path)
    source = ReferenceRun.model_validate(
        export_outcomes(reconstruct(tmp_path / "data/raw", reference))
    )
    control_sources = [
        ReferenceRun.model_validate(export_outcomes(reconstruct(tmp_path / "data/raw", run)))
        for run in controls
    ]
    pack = build_pack(source, "MOCK", seed=8, controls=control_sources)
    original = save_pack(pack, tmp_path / "refs")
    original_bytes = original.read_bytes()
    state = source_state(tmp_path, pack)
    assert state["rights_verified"] and state["blockers"] == ["scripted_mock_not_public"]
    out = tmp_path / "reference-dataset"
    result = stage_reference(tmp_path, pack.pack_id, out)
    assert result["publication_blockers"] == ["scripted_mock_not_public"]
    public_path = out / next(n for n in result["files"] if n.endswith("/pack.json"))
    public = load_pack(str(public_path))
    for key, view in pack.fingerprints["views"].items():
        assert public.fingerprints["views"][key]["metrics"] == view["metrics"]
        for endpoint, samples in view["references"].items():
            for method, sample in samples.items():
                projected = public.fingerprints["views"][key]["references"][endpoint][method]
                assert {k: v for k, v in sample.items() if k != "details"} == {
                    k: v for k, v in projected.items() if k != "details"
                }
                assert (
                    projected["details"]["source_observation_hash"]
                    == sample["details"]["observation_hash"]
                )
    assert public.calibration == pack.calibration and public.outcomes == pack.outcomes
    target_analysis = reconstruct(tmp_path / "data/raw", target)
    declaration = target_analysis.plan.endpoints[0]
    samples = export_outcomes(target_analysis)["fingerprints"][declaration.id]
    from servegauge.scoring.common import Samples

    policy = DecisionPolicy.model_validate(target_analysis.metadata["policy"])
    comparisons = {}
    for method in ["FP1", "FP2", "FP3-mean", "FP3-dispersion"]:
        base = "FP3" if method.startswith("FP3") else method
        sample = Samples.model_validate(samples[base])
        args = dict(
            target=declaration.id, method=method, policy=policy, seed=77, endpoint=declaration
        )
        before = compare_fingerprint(pack, sample, **args)
        after = compare_fingerprint(public, sample, **args)
        assert before == after
        comparisons[method] = before
    # Removing transport data must not remove the known reused-stream refusal.
    reused = declaration.model_copy(deep=True)
    reused.metadata["mock_declaration"]["seed"] = 1
    with pytest.raises(ValueError, match="mock_target_seed_must_be_independent"):
        compare_fingerprint(
            public,
            Samples.model_validate(samples["FP1"]),
            target=reused.id,
            method="FP1",
            policy=policy,
            seed=77,
            endpoint=reused,
        )
    frozen = {n: (out / n).read_bytes() for n in [*result["files"], "local-selection.json"]}
    cli = CliRunner().invoke(app, ["refs", "publish", pack.pack_id, "--out", str(out), "--dry-run"])
    assert cli.exit_code == 0 and json.loads(cli.output) == result
    with pytest.raises(ValueError):
        reference_preflight(tmp_path, pack.pack_id, out)
    assert all((out / n).read_bytes() == blob for n, blob in frozen.items())
    assert original.read_bytes() == original_bytes
    ledger = Ledger(tmp_path)
    ledger.refresh()
    assert ledger.development_spend() == 0 and not ledger.pending
    (tmp_path / "reference-publication-native-acceptance.json").write_text(
        json.dumps(
            {
                "source_pack_id": pack.pack_id,
                "public_pack_id": public.pack_id,
                "source_pack_unchanged": True,
                "source_state": state,
                "comparisons": comparisons,
                "comparisons_exact": True,
                "known_reuse_still_refuses": True,
                "public_file_hashes": {
                    n: hashlib.sha256(blob).hexdigest() for n, blob in frozen.items()
                },
                "model_api_calls": 0,
                "new_spend_usd": 0,
                "pending_reservations": 0,
                "mock_upload_refused": True,
            },
            indent=2,
        )
    )
