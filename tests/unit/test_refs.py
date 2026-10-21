import json
from pathlib import Path

import pytest

from servegauge.refs.pack import (
    ReferenceEndpoint,
    ReferenceRun,
    build_pack,
    load_pack,
    reference_for,
    save_pack,
)


def reference_run() -> ReferenceRun:
    return ReferenceRun(
        run_id="r",
        profile="p",
        configuration={"level": "core", "temperature": 1},
        item_set_hashes={"S2": "abc"},
        baseline_endpoint_id="b",
        endpoints=[
            ReferenceEndpoint(id=key, profile="p", kind="mock", seed=seed, clean=True)
            for seed, key in enumerate("abcd")
        ],
        checks={
            "T02": {
                key: {f"item{i:02}": [float((i + index) % 7 != 0)] * 2 for i in range(50)}
                for index, key in enumerate("abcd")
            }
        },
    )


def test_minimal_tiers_qualifications_and_calibration(tmp_path: Path) -> None:
    source = reference_run()
    for tier in ["MOCK", "BASELINE", "R3"]:
        pack = build_pack(source, tier, seed=8)
        path = save_pack(pack, tmp_path)
        assert load_pack(str(path)) == pack
        assert save_pack(pack, tmp_path) == path
        values, calibration = reference_for(
            pack,
            target="a",
            check="T02",
            item_ids=sorted(source.checks["T02"]["a"]),
            configuration=source.configuration,
        )
        assert len(values) == calibration["n"] == 50
        assert calibration["resamples"] >= 2000
        assert len(calibration["reference_interval"]) == 2
        if tier == "R3":
            assert "a" not in pack.outcomes["a"]["T02"]["peer_ids"]
            assert pack.outcomes["a"]["T02"]["median_absolute_deviation"]
        with pytest.raises(ValueError, match="matched_items"):
            reference_for(
                pack, target="a", check="T02", item_ids=["one"], configuration=source.configuration
            )
        with pytest.raises(ValueError, match="configuration"):
            reference_for(pack, target="a", check="T02", item_ids=[], configuration={})
    source.endpoints[1].seed = 0
    with pytest.raises(ValueError, match="different_seeds"):
        build_pack(source, "MOCK", seed=8)
    source.endpoints = source.endpoints[:3]
    with pytest.raises(ValueError, match="OTHER"):
        build_pack(source, "R3", seed=8)
    source.baseline_endpoint_id = None
    with pytest.raises(ValueError, match="baseline"):
        build_pack(source, "BASELINE", seed=8)


def test_R3_never_leaks_target_and_missing_items_fail() -> None:
    source = reference_run()
    before = build_pack(source, "R3", seed=8)
    for item in source.checks["T02"]["a"]:
        source.checks["T02"]["a"][item] = [0.0, 0.0]
    after = build_pack(source, "R3", seed=8)
    assert before.outcomes["a"] == after.outcomes["a"]
    assert before.calibration["a"] == after.calibration["a"]
    del source.checks["T02"]["b"]["item00"]
    partial = build_pack(source, "R3", seed=8)
    assert "T02" not in partial.outcomes["a"]
    assert "conditional" in partial.manifest["unavailable_views"]["a"]["T02"]
    source.checks["T03"] = source.checks["T02"]
    with pytest.raises(ValueError, match="match_exactly"):
        build_pack(source, "R3", seed=8)


def test_R3_uses_per_check_eligible_peers_and_F1_keeps_independent_repeat():
    source = reference_run()
    source.checks["T01"] = {
        key: {f"item{i:02}": [float(i % 5 != 0)] * 2 for i in range(50)} for key in "abcd"
    }
    del source.checks["T02"]["a"]
    pack = build_pack(source, "R3", seed=8)
    assert pack.outcomes["a"]["T02"]["peer_ids"] == ["b", "c", "d"]
    assert "T02" not in pack.outcomes["b"]  # Only two OTHER supported peers remain.
    values, calibration = reference_for(
        pack,
        target="a",
        check="T01",
        item_ids=sorted(source.checks["T01"]["a"]),
        configuration=source.configuration,
    )
    assert len(values) == calibration["n"] == 50
    assert calibration["f1_anchor"] == calibration["f1_repeat"]
    assert calibration["reference_estimate"] == 1
    for key in source.checks["T01"]:
        for item in source.checks["T01"][key]:
            source.checks["T01"][key][item] = [1.0]
    assert "T01" not in build_pack(source, "R3", seed=8).outcomes["a"]


def test_pack_integrity_and_cli_build(tmp_path: Path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from servegauge.cli import app
    from servegauge.core.ids import safe_filename

    source = reference_run()
    profile_text = Path("profiles/gpt-oss-120b.yaml").read_text()
    source.profile = "gpt-oss-120b"
    for endpoint in source.endpoints:
        endpoint.profile = source.profile
    monkeypatch.chdir(tmp_path)
    Path("profiles").mkdir()
    Path("profiles/gpt-oss-120b.yaml").write_text(profile_text)
    directory = Path("data/runs") / safe_filename(source.run_id)
    directory.mkdir(parents=True)
    (directory / "outcomes.json").write_text(source.model_dump_json())
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["refs", "build", "--profile", source.profile, "--tier", "R3", "--from-run", source.run_id],
    )
    assert result.exit_code == 3, result.output
    assert "sealed raw evidence" in result.output
    pack = build_pack(source, "R3", seed=8)
    path = save_pack(pack)
    damaged = json.loads(path.read_text())
    damaged["outcomes"]["a"]["T02"]["values"][0] += 1
    path.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="content_mismatch"):
        load_pack(str(path))
    assert runner.invoke(app, ["doctor", "--profiles"]).exit_code == 0
    assert (
        runner.invoke(
            app,
            [
                "refs",
                "build",
                "--profile",
                source.profile,
                "--tier",
                "R2",
                "--from-run",
                source.run_id,
            ],
        ).exit_code
        == 3
    )


def test_saved_full_pack_accepts_exact_check_subset_but_not_changed_check() -> None:
    source = reference_run()
    source.configuration = {
        "level": "core",
        "seed": 8,
        "item_set_hash": "whole",
        "checks": {
            "T02": {"item_set_hash": "schema-items"},
            "T01": {"item_set_hash": "trigger-items"},
        },
    }
    pack = build_pack(source, "BASELINE", seed=8)
    subset = {
        **source.configuration,
        "item_set_hash": "subset",
        "checks": {"T02": {"item_set_hash": "schema-items"}},
    }
    values, calibration = reference_for(
        pack,
        target="a",
        check="T02",
        item_ids=sorted(source.checks["T02"]["a"]),
        configuration=subset,
    )
    assert len(values) == calibration["n"] == 50
    subset["checks"]["T02"]["item_set_hash"] = "changed"
    with pytest.raises(ValueError, match="configuration"):
        reference_for(
            pack,
            target="a",
            check="T02",
            item_ids=sorted(source.checks["T02"]["a"]),
            configuration=subset,
        )


def test_accounting_pack_requires_identical_inputs_and_math_retains_repeats():
    source = reference_run()
    source.checks["B01"] = source.checks["T02"]
    source.checks["R01"] = source.checks["T02"]
    source.inputs["B01"] = {
        endpoint: {item: ["same-hash"] for item in values}
        for endpoint, values in source.checks["B01"].items()
    }
    pack = build_pack(source, "R3", seed=8)
    values, band = reference_for(
        pack,
        target="a",
        check="R01",
        item_ids=sorted(source.checks["R01"]["a"]),
        configuration=source.configuration,
    )
    assert len(band["binary_repeats"]) == len(values) == 50
    with pytest.raises(ValueError, match="identical_input"):
        reference_for(
            pack,
            target="a",
            check="B01",
            item_ids=sorted(source.checks["B01"]["a"]),
            configuration=source.configuration,
        )
    reference_for(
        pack,
        target="a",
        check="B01",
        item_ids=sorted(source.checks["B01"]["a"]),
        configuration=source.configuration,
        input_hashes=source.inputs["B01"]["a"],
    )
    source.inputs["B01"]["b"]["item00"] = ["changed"]
    assert "B01" not in build_pack(source, "R3", seed=8).outcomes["a"]
