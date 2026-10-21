"""Prepared execution uses closed synthetic tickets and ordinary reserved inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from unit.test_comparison_kit import kit_inputs as kit_inputs
from unit.test_comparison_kit import prepare

from servegauge.core.hashing import content_hash
from servegauge.runner.comparison import execute_procedure, external_config


@pytest.mark.asyncio
async def test_ordinary_child_uses_original_procedures_and_preserves_underpowered_controls(
    kit_inputs, tmp_path, monkeypatch
):
    manifest = prepare(kit_inputs)
    kit = kit_inputs[-1]
    anchor = content_hash(manifest)
    work = tmp_path / "execution"
    work.mkdir(mode=0o700)
    (work / "execution-ticket.json").write_text(
        json.dumps({"kit_hash": anchor, "mode": "synthetic", "owner_review": None})
    )
    # The fixture uses the repository's pinned public tokenizer cache.
    from servegauge.tokencount.cache import cache_dir

    monkeypatch.setenv("SERVEGAUGE_TOKENIZER_CACHE", str(Path(".cache/tokenizers").absolute()))
    assert cache_dir().is_dir()
    monkeypatch.chdir(work)
    # Default tests prove the execution logic with sockets blocked; the real
    # kernel/container boundary has separate native acceptance.
    monkeypatch.setattr("servegauge.runner.comparison.require_container_boundary", lambda: None)
    result = await execute_procedure(kit, kit_hash=anchor, procedure="ordinary", work=work)
    assert result["owner_comparison_executed"] is False
    assert result["total_setup_seconds"] is None
    assert result["first_valid_result_utc"] is not None
    assert len(result["cases"]) == 4
    assert {row["verdict"] for row in result["cases"]} == {"FAIL", "INCONCLUSIVE"}
    assert all(row["n"] == 4 and row["original_population_n"] == 4 for row in result["cases"])
    assert (work / "procedure-results.json").is_file()


@pytest.mark.asyncio
async def test_missing_mismatched_or_unreviewed_owner_ticket_never_reaches_execution(
    kit_inputs, tmp_path, monkeypatch
):
    manifest = prepare(kit_inputs)
    kit = kit_inputs[-1]
    anchor = content_hash(manifest)

    def forbidden(*args, **kwargs):
        raise AssertionError("must not execute")

    monkeypatch.setattr("servegauge.runner.comparison.run_check", forbidden)
    monkeypatch.setattr("servegauge.runner.comparison.require_container_boundary", lambda: None)
    for problem in ["missing", "changed", "owner_without_review"]:
        work = tmp_path / problem
        work.mkdir()
        if problem != "missing":
            (work / "execution-ticket.json").write_text(
                json.dumps(
                    {
                        "kit_hash": "0" * 64 if problem == "changed" else anchor,
                        "mode": "owner" if problem == "owner_without_review" else "synthetic",
                        "owner_review": None,
                    }
                )
            )
        monkeypatch.chdir(work)
        with pytest.raises(ValueError):
            await execute_procedure(kit, kit_hash=anchor, procedure="ordinary", work=work)
        assert not (work / "procedure-results.json").exists()


async def test_external_config_is_serial_and_excludes_reported_result_inputs(
    kit_inputs,
):
    manifest = prepare(kit_inputs)
    config = external_config(kit_inputs[-1], manifest)
    assert config["sharing"] is False
    assert config["evaluateOptions"]["cache"] is False
    assert config["evaluateOptions"]["maxConcurrency"] == 1
    assert config["prompts"] == ["{{case_id}}"]
    for row in config["tests"]:
        assert row["assert"] == [{"type": "python", "value": "file://assertion.py"}]
        variables = row["vars"]["independent_policy"]
        assert not {"verdict", "violations", "fault_id", "expected_verdict"} & variables.keys()
        assert len(variables["unit_keys"]) == 4
