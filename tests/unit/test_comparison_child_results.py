"""Validate actual reserved replay artifacts at the parent process boundary, offline."""

from __future__ import annotations

import asyncio
import copy
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from comparison_workflow import prepare_comparison_fixture

from servegauge.comparison.case_provider import run_raw_case
from servegauge.comparison.independent import get_assert
from servegauge.core.hashing import canonical_json, content_hash
from servegauge.runner import comparison_launcher as launcher
from servegauge.runner.comparison import execute_procedure, external_config
from servegauge.storage.comparison import _files
from servegauge.storage.comparison_controls import ControlSelection
from servegauge.storage.comparison_kit import prepare_kit
from servegauge.storage.comparison_program import ComparisonProgram
from servegauge.storage.comparison_results import normalize_external
from servegauge.validation.async_offline import SocketlessLoop


@pytest.fixture(scope="module")
def child_results(tmp_path_factory):
    with asyncio.Runner(loop_factory=SocketlessLoop) as loop:
        return loop.run(build_child_results(tmp_path_factory))


async def build_child_results(tmp_path_factory):
    base = tmp_path_factory.mktemp("comparison-child-results")
    cache = Path(".cache/tokenizers").absolute()
    fixture = base / "fixture"
    snapshot, failures, _ = await prepare_comparison_fixture(fixture)
    controls = ControlSelection.model_validate_json(
        (fixture / "control-selection.json").read_bytes()
    )
    kit = fixture / "source/.cache/comparisons/kit"
    manifest = prepare_kit(
        fixture / "source",
        snapshot,
        selection_hash=failures.selection_hash,
        controls=controls,
        controls_hash=content_hash(controls.model_dump(mode="json")),
        out=kit,
    )
    anchor = content_hash(manifest)
    work = base / "ordinary-child"
    work.mkdir(mode=0o700)
    (work / "execution-ticket.json").write_bytes(
        canonical_json(
            {
                "kit_hash": anchor,
                "mode": "synthetic",
                "owner_review": None,
            }
        )
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SERVEGAUGE_TOKENIZER_CACHE", str(cache))
        patch.chdir(work)
        # This default test validates evidence with every socket disabled. Actual
        # kernel enforcement and the pinned tool have separate native acceptance.
        patch.setattr("servegauge.runner.comparison.require_container_boundary", lambda: None)
        ordinary = await execute_procedure(kit, kit_hash=anchor, procedure="ordinary", work=work)
        config = external_config(kit, manifest)
        program = ComparisonProgram.model_validate_json(
            (kit / "procedures/program.json").read_bytes()
        )
        external_rows = []
        for case, test in zip(program.cases, config["tests"], strict=True):
            raw = await run_raw_case(
                program, case_id=case.case_id, root=base / ("raw-" + case.case_id)
            )
            assertion = get_assert(raw, {"vars": test["vars"]})
            external_rows.append(
                {"testCase": test, "gradingResult": {"componentResults": [assertion]}}
            )
    external_raw = {"results": {"results": external_rows}}
    external = copy.deepcopy(ordinary)
    external.update(procedure="external", cases=normalize_external(external_raw, manifest["cases"]))
    return kit, manifest, anchor, work, {"ordinary": ordinary, "external": external}, external_raw


def invoke(inputs, tmp_path, monkeypatch, *, procedure, problem=None):
    kit, manifest, anchor, child, results, external_raw = inputs
    output = tmp_path / "output"

    def prepare(**kwargs):
        assert kwargs["inputs"] == kit and kwargs["output"] == output
        output.mkdir(mode=0o700)
        return SimpleNamespace(arguments=["no-process-may-start"], environment={})

    def subprocess_result(arguments, **kwargs):
        assert arguments == ["no-process-may-start"] and kwargs["env"] == {}
        if problem == "timeout":
            raise subprocess.TimeoutExpired(arguments, 1200)
        for source in _files(child):
            relative = source.relative_to(child)
            if relative.as_posix() == "execution-ticket.json":
                continue
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        value = copy.deepcopy(results[procedure])
        if problem == "scope":
            value["kit_hash"] = "0" * 64
        elif problem == "source":
            value["runtime"]["source"]["source_hash"] = "0" * 64
        elif problem == "counts":
            value["cases"][0]["violations"] = 0
        elif problem == "audit":
            path = output / ("case-" + value["cases"][0]["case_id"]) / "finite-audit.json"
            audit = json.loads(path.read_bytes())
            audit["observations"].pop()
            path.write_bytes(canonical_json(audit))
        (output / "procedure-results.json").write_bytes(canonical_json(value))
        if procedure == "external":
            raw = copy.deepcopy(external_raw)
            if problem == "external_reason":
                raw["results"]["results"][0]["gradingResult"]["componentResults"][0]["reason"] = (
                    "{}"
                )
            (output / "promptfoo-results.json").write_bytes(canonical_json(raw))
        if problem == "malformed_json":
            (output / "procedure-results.json").write_text('{"duplicate":1,"duplicate":2}')
        return subprocess.CompletedProcess(arguments, 1 if problem == "exit" else 0, b"", b"")

    monkeypatch.setattr(launcher, "prepare_container", prepare)
    monkeypatch.setattr(
        launcher,
        "subprocess",
        SimpleNamespace(run=subprocess_result, TimeoutExpired=subprocess.TimeoutExpired),
    )
    return launcher._run_procedure(
        kit=kit,
        manifest=manifest,
        kit_hash=anchor,
        mode="synthetic",
        review=None,
        procedure=procedure,
        docker=Path("/missing/docker"),
        config=tmp_path / "anonymous",
        host="unix:///var/run/docker.sock",
        image="sha256:" + "a" * 64,
        output=output,
        user="65534:65534",
    )


@pytest.mark.parametrize("procedure", ["ordinary", "external"])
def test_parent_accepts_exact_scoped_artifacts_and_preserves_inconclusive_controls(
    child_results,
    tmp_path,
    monkeypatch,
    procedure,
):
    result = invoke(child_results, tmp_path, monkeypatch, procedure=procedure)
    assert result["harness_error"] is None and result["execution_seconds"] >= 0
    assert {row["verdict"] for row in result["cases"]} == {"FAIL", "INCONCLUSIVE"}
    assert all(row["n"] == 4 and row["original_population_n"] == 4 for row in result["cases"])
    assert result["new_model_api_calls"] == result["new_spend_usd"] == 0
    for name in ["container.stdout", "container.stderr", "execution-ticket.json"]:
        assert (tmp_path / "output" / name).is_file()


@pytest.mark.parametrize(
    "problem", ["exit", "timeout", "scope", "source", "counts", "audit", "malformed_json"]
)
def test_parent_rejects_failed_or_changed_child_evidence_without_zero_counts(
    child_results,
    tmp_path,
    monkeypatch,
    problem,
):
    result = invoke(child_results, tmp_path, monkeypatch, procedure="ordinary", problem=problem)
    assert result["harness_error"] == "comparison_child_or_evidence_failed"
    assert result["first_valid_result_utc"] is None
    for row in result["cases"]:
        assert row["verdict"] == "ERROR" and row["original_population_n"] == 4
        assert row["n"] is row["violations"] is row["ci_low"] is row["ci_high"] is None


def test_parent_recomputes_external_summary_from_the_retained_independent_reason(
    child_results,
    tmp_path,
    monkeypatch,
):
    result = invoke(
        child_results, tmp_path, monkeypatch, procedure="external", problem="external_reason"
    )
    assert result["harness_error"] == "comparison_child_or_evidence_failed"
    assert all(row["verdict"] == "ERROR" and row["violations"] is None for row in result["cases"])
