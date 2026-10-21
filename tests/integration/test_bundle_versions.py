"""Versioned local artifacts preserve legacy files and actual reserved replay."""

import json
import shlex

import pytest
from integration.test_scientific import prepared as prepared

from servegauge.core.hashing import content_hash
from servegauge.core.ids import safe_filename
from servegauge.diagnose.bundle import read_bundle, write_bundle
from servegauge.diagnose.repro import replay_bundle
from servegauge.diagnose.rules import diagnoses
from servegauge.runner.check import run_check
from servegauge.storage.analysis import reconstruct


async def test_new_drafts_versions_legacy_preservation_and_reserved_id_replay(tmp_path, prepared):
    plan, transports = prepared
    result = await run_check(plan, tests="P01", root=tmp_path, approved=True, transports=transports)
    analysis = reconstruct(tmp_path / "data/raw", result.run_id)
    diagnosis = diagnoses(analysis.checks)[0]
    record_hash = content_hash([r.model_dump(mode="json") for r in analysis.records])
    legacy = tmp_path / "bundles" / safe_filename(diagnosis.diagnosis_id)
    legacy.mkdir(parents=True)
    value = {"version": "bundle-v1", "redact_content": False, "diagnosis": diagnosis.model_dump()}
    blob = json.dumps(value).encode()
    (legacy / "bundle.json").write_bytes(blob)
    legacy_redacted = legacy.with_name(legacy.name + "-redacted")
    legacy_redacted.mkdir()
    (legacy_redacted / "bundle.json").write_text('{"legacy":"preserved"}')

    current = write_bundle(analysis, diagnosis, root=tmp_path)
    assert current != legacy and current.name.endswith("-v2")
    assert (legacy / "bundle.json").read_bytes() == blob
    assert read_bundle(str(legacy), tmp_path)["version"] == "bundle-v1"
    assert read_bundle(diagnosis.diagnosis_id, tmp_path)["version"] == "bundle-v2"
    issue = (current / "issue.md").read_text()
    assert diagnosis.code in issue.replace("\\", "") and "Draft" in issue
    assert (current / "README.md").read_text().find("issue.md") >= 0
    replayed = await replay_bundle(
        diagnosis.diagnosis_id, root=tmp_path, approved=True, transports=transports
    )
    assert replayed["reproduced"] and replayed["budget_scope"] == result.run_id

    redacted = write_bundle(analysis, diagnosis, root=tmp_path, redact_content=True)
    assert redacted != legacy_redacted
    assert (legacy_redacted / "bundle.json").read_text() == '{"legacy":"preserved"}'
    data = json.loads((redacted / "bundle.json").read_text())
    assert data["version"] == "bundle-v2"
    assert data["evidence_hashes_and_structure"]["version"] == "content-summary-v2"
    assert (redacted / "issue.md").is_file()
    assert content_hash([r.model_dump(mode="json") for r in analysis.records]) == record_hash
    with pytest.raises(ValueError, match="redacted"):
        read_bundle(str(redacted), tmp_path)
    (redacted / "unexpected-private.txt").write_text("PRIVATE")
    with pytest.raises(ValueError, match="unexpected_content"):
        write_bundle(analysis, diagnosis, root=tmp_path, redact_content=True)


def test_unknown_bundle_version_refuses_without_echoing_the_payload(tmp_path):
    path = tmp_path / "bundle.json"
    path.write_text('{"version":"bundle-future","payload":"PRIVATE"}')
    with pytest.raises(ValueError, match="unsupported_bundle_version") as error:
        read_bundle(str(path), tmp_path)
    assert "PRIVATE" not in str(error.value)


async def test_proxy_examples_quote_endpoint_values_and_paths_with_spaces(tmp_path, prepared):
    plan, transports = prepared
    result = await run_check(plan, tests="P01", root=tmp_path, approved=True, transports=transports)
    analysis = reconstruct(tmp_path / "data/raw", result.run_id)
    url = "http://127.0.0.1:8011/v1?label='quoted'$(printf untrusted)"
    analysis.plan.endpoints[0].base_url = url
    destination = write_bundle(
        analysis, diagnoses(analysis.checks)[0], root=tmp_path / "project with spaces"
    )
    lines = (destination / "README.md").read_text().splitlines()
    commands = [line for line in lines if line.startswith("uv run servegauge mock serve")]
    assert len(commands) == 2
    arguments = shlex.split(commands[0])
    assert arguments[arguments.index("--backend") + 1] == "proxy:" + url
    assert arguments[arguments.index("--proxy-plan") + 1] == str(destination / "proxy-plan.yaml")
    assert (
        "''quoted''" in commands[1]
        and "'" + str(destination / "proxy-plan.yaml") + "'" in commands[1]
    )
