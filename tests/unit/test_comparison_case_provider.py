"""External callbacks consume a whole finite case without calling any detector."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

import pytest
from unit.test_comparison_sources import hashes
from unit.test_comparison_sources import source as source

from servegauge.comparison.case_provider import call_api, run_raw_case
from servegauge.core.hashing import canonical_json, content_hash
from servegauge.runner.ledger import Ledger
from servegauge.storage.comparison import prepare_sources
from servegauge.storage.comparison_program import compile_program


@pytest.mark.asyncio
async def test_whole_case_preserves_all_attempts_and_raw_units_without_detector(
    source, tmp_path, monkeypatch
):
    root, selection, snapshot = source
    prepare_sources(root, selection, snapshot)
    anchor = content_hash(selection.model_dump(mode="json"))
    program = compile_program(snapshot, selection_hash=anchor)
    before = hashes(snapshot)

    def forbidden(*args, **kwargs):
        raise AssertionError("external verdict must be independent")

    monkeypatch.setattr("servegauge.runner.check.run_check", forbidden)
    monkeypatch.setattr("servegauge.runner.evaluate.collect", forbidden)
    monkeypatch.setattr("servegauge.runner.evaluate.evaluate", forbidden)
    for case in program.cases:
        output = tmp_path / ("raw-" + case.case_id)
        result = await run_raw_case(program, case_id=case.case_id, root=output)
        if os.name != "nt":
            assert output.stat().st_mode & 0o077 == 0
            for name in ["raw-result.json", "finite-audit.json"]:
                assert (output / name).stat().st_mode & 0o077 == 0
        assert set(result["primary"]) == set(case.logical_unit_keys)
        assert len(result["primary"]) == 4 and len(result["confirmations"]) == 1
        assert result["replay_adds_independent_n"] is False
        assert not {"verdict", "violations", "original_check"} & result.keys()
        audit = json.loads((output / "finite-audit.json").read_text())
        assert audit["remaining_slots"] == 0 and not audit["refusals"]
        assert audit["consumed_attempts"] == sum(
            len(row.source.record.attempts) for phase in case.phases for row in phase.exchanges
        )
        assert len(result["support"]) == sum(len(phase.exchanges) for phase in case.phases)
        assert all(row["source_address"]["case_id"] == case.case_id for row in result["support"])
        ledger = Ledger(output)
        ledger.refresh()
        assert not ledger.pending and ledger.development_spend() == 0
        with pytest.raises(ValueError):
            await run_raw_case(program, case_id=case.case_id, root=output)
    assert hashes(snapshot) == before


def test_external_provider_has_no_detector_imports():
    import servegauge.comparison.case_provider as module

    tree = ast.parse(Path(module.__file__).read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(
        name
        and any(part in name.split(".") for part in ["scoring", "evaluate", "check", "verdicts"])
        for name in imports
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "problem", ["unknown_case", "duplicate", "program_changed", "audit_changed", "raw_changed"]
)
async def test_callback_refusal_or_changed_completed_output_poison_the_worker(
    source, tmp_path, monkeypatch, problem
):
    root, selection, snapshot = source
    prepare_sources(root, selection, snapshot)
    program = compile_program(
        snapshot, selection_hash=content_hash(selection.model_dump(mode="json"))
    )
    blob = canonical_json(program.model_dump(mode="json"))
    work = tmp_path / "external-worker"
    work.mkdir(mode=0o700)
    (work / "program.json").write_bytes(blob)
    monkeypatch.setenv("SERVEGAUGE_TOKENIZER_CACHE", str(Path(".cache/tokenizers").absolute()))
    monkeypatch.chdir(work)
    monkeypatch.setattr("servegauge.comparison.case_provider._worker_scope", None)
    monkeypatch.setattr("servegauge.comparison.case_provider._worker_refused", False, raising=False)
    monkeypatch.setattr("servegauge.comparison.case_provider._completed_outputs", {}, raising=False)
    options = {
        "config": {
            "program": "program.json",
            "program_sha256": hashlib.sha256(blob).hexdigest(),
            "basePath": str(work),
        }
    }
    first, second = [case.case_id for case in program.cases]
    if problem == "unknown_case":
        refused = await call_api("missing", options, {"vars": {"case_id": "missing"}})
    else:
        initial = await call_api(first, options, {"vars": {"case_id": first}})
        assert "output" in initial and "error" not in initial
        if problem == "duplicate":
            refused = await call_api(first, options, {"vars": {"case_id": first}})
        else:
            if problem == "program_changed":
                (work / "program.json").write_bytes(blob + b" ")
            else:
                name = "finite-audit.json" if problem == "audit_changed" else "raw-result.json"
                (work / ("case-" + first) / name).write_text("{}")
            refused = await call_api(second, options, {"vars": {"case_id": second}})
    assert "error" in refused and "output" not in refused
    # Fixing the changed input cannot resurrect a refused worker.
    (work / "program.json").write_bytes(blob)
    later = await call_api(second, options, {"vars": {"case_id": second}})
    assert "error" in later and not (work / ("case-" + second)).exists()
