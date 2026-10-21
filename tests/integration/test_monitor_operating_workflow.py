"""Native sealed history crosses reviewed producer versions and records one local event."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from filelock import FileLock

from servegauge.core.hashing import content_hash
from servegauge.observatory.admission import load_admission, statistical_pins
from servegauge.observatory.evaluation import evaluate_history
from servegauge.observatory.identity import producer_identity
from servegauge.observatory.provenance import implementation_hash
from servegauge.observatory.rolling import load_rolling_policy
from servegauge.storage.transport_events import TransportEventJournal


def contract_receipt(root, producers):
    """Test-only source binding using the frozen real lab, not release qualification."""
    producer = producer_identity()
    lab = json.loads(Path("docs/validation/product-monitor-operating-lab.json").read_text())
    receipt = {
        "version": "monitor-operating-qualification-v1",
        "id": "scripted-transport-operating-v1",
        "scope": "scripted-independent-transport-v1",
        "reference_tier": "MOCK",
        "source_hash": producer["source_hash"],
        "generator": producer["generator"],
        "runtime": producer["runtime"],
        "producer_sources": sorted(producers),
        "policy_hash": content_hash(load_rolling_policy().model_dump(mode="json")),
        "study_plan_sha256": hashlib.sha256(
            Path("configs/monitor_operating_study_v1.json").read_bytes()
        ).hexdigest(),
        "statistical_implementation_pins": statistical_pins(),
        "all_source_gates_passed": True,
        "source_tests_passed": 1,
        "study": lab["study"],
        "native": {
            "canonical_units": 800,
            "full_canonical_units": 800,
            "first_attempt_wire_equal": True,
            "preflight_all_error_abstains": True,
            "compatible_producer_bridge": True,
            "raw_only_equal": True,
            "idempotent": True,
        },
        "artifacts": lab["artifact_sha256"],
        "test_contract_only": True,
    }
    (root / "configs").mkdir(parents=True)
    (root / "docs/validation").mkdir(parents=True)
    # Copy only the explicitly pinned immutable report files; no workspace traversal.
    for name in lab["artifact_sha256"]:
        assert "Starter Files" not in name
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(name), target)
    for name in ["monitor_operating_study_v1.json"]:
        shutil.copyfile(Path("configs") / name, root / "configs" / name)
    relative = "docs/validation/product-monitor-operating-qualification-v1.json"
    payload = json.dumps(receipt, sort_keys=True).encode()
    (root / relative).write_bytes(payload)
    (root / "configs/monitor_transport_admission_v1.json").write_text(
        json.dumps(
            {
                "version": "monitor-admission-registry-v1",
                "qualifications": [
                    {
                        "id": receipt["id"],
                        "receipt": relative,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
        )
    )
    return receipt


def data_hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "data").rglob("*"))
        if p.is_file()
    }


@pytest.mark.slow
def test_native_reviewed_producers_record_one_immutable_event_and_repeat_read_only(tmp_path):
    root = Path.cwd()
    source = implementation_hash()
    copy = tmp_path / "producer-copy"
    shutil.copytree(
        root / "src/servegauge",
        copy / "src/servegauge",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (copy / "src/servegauge/producer_version_marker.py").write_text(
        '"""Unused native inventory marker."""\n'
    )
    for folder, suffixes in [("configs", {".yaml", ".json"}), ("profiles", {".yaml"})]:
        (copy / folder).mkdir(parents=True)
        for path in (root / folder).iterdir():
            if not path.is_symlink() and path.is_file() and path.suffix in suffixes:
                shutil.copyfile(path, copy / folder / path.name)
    for name in ["uv.lock", "pyproject.toml", "README.md"]:
        shutil.copyfile(root / name, copy / name)
    data = tmp_path / "capture"
    (data / "configs").mkdir(parents=True)
    shutil.copyfile(root / "configs/budgets.yaml", data / "configs/budgets.yaml")
    outputs = []
    for index, (cwd, error) in enumerate([(copy, 0.05), (root, 0.60)]):
        out = tmp_path / f"producer-{index}.json"
        environment = {
            **os.environ,
            "PYTHONPATH": str(cwd / "src"),
            "SERVEGAUGE_TOKENIZER_CACHE": str(root / ".cache/tokenizers"),
        }
        process = subprocess.run(
            [
                sys.executable,
                str(root / "tests/support/monitor_producer_probe.py"),
                "--root",
                str(data),
                "--out",
                str(out),
                "--stamp",
                f"2026-03-{index + 1:02}T00:00:00+00:00",
                "--error-rate",
                str(error),
            ],
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0, process.stderr[-2000:]
        outputs.append(json.loads(out.read_text()))
    assert outputs[0]["producer"]["source_hash"] != outputs[1]["producer"]["source_hash"] == source
    assert outputs[0]["producer"]["generator"] == outputs[1]["producer"]["generator"]
    assert outputs[0]["producer"]["runtime"] == outputs[1]["producer"]["runtime"]
    proof = tmp_path / "contract-proof"
    receipt = contract_receipt(proof, {o["producer"]["source_hash"] for o in outputs})
    assert load_admission(proof).admission is not None
    before = data_hashes(data)
    preview = evaluate_history(data, "producer-bridge", proof_root=proof, persist=False)
    assert preview["status"] == "read_only" and len(preview["events"]) == 1
    assert data_hashes(data) == before
    result = evaluate_history(data, "producer-bridge", proof_root=proof)
    assert result["status"] == "recorded" and result["events"] == preview["events"]
    event = result["events"][0]
    assert event["contrast"]["left_n"] == event["contrast"]["right_n"] == 400
    assert event["contrast"]["ci_high"] < 0 and event["quality_verdict"] == "INCONCLUSIVE"
    assert set(event["producer_sources"].values()) == set(receipt["producer_sources"])
    saved = data_hashes(data)
    assert evaluate_history(data, "producer-bridge", proof_root=proof)["status"] == "unchanged"
    assert data_hashes(data) == saved
    journal = TransportEventJournal(data / "data/raw", "producer-bridge")
    assert len(journal.accepted_events()) == 1
    with FileLock(str(journal.directory / ".record.lock"), timeout=0):
        child = subprocess.run(
            [
                sys.executable,
                "-m",
                "servegauge.cli",
                "monitor",
                "evaluate",
                "--observatory-id",
                "producer-bridge",
                "--root",
                str(data),
                "--proof-root",
                str(proof),
            ],
            cwd=root,
            capture_output=True,
            text=True,
        )
    assert child.returncode == 3 and "evaluation refused" in child.stderr
    assert data_hashes(data) == saved
    raw_only = tmp_path / "raw-only"
    shutil.copytree(
        data / "data/raw",
        raw_only / "data/raw",
        ignore=shutil.ignore_patterns("transport-events", "transport-history"),
    )
    rebuilt = evaluate_history(raw_only, "producer-bridge", proof_root=proof, persist=False)
    assert rebuilt == preview
    import polars as pl

    from servegauge.storage.tables import rebuild

    table_counts = rebuild(data / "data/raw", tmp_path / "event-tables")
    assert table_counts["change_events"] == 1
    stored = json.loads(
        pl.read_parquet(tmp_path / "event-tables/change_events.parquet")["record_json"][0]
    )
    assert stored == event
    from servegauge.report.site.builder import build_site, publication_preflight

    run_ids = [o["result"]["jobs"][0]["run_id"] for o in outputs]
    site = tmp_path / "selected-site"
    build_site(data, site, runs=run_ids)
    projected = json.loads((site / "site.json").read_text())["monitor_events"]
    assert len(projected) == 1
    assert projected[0]["contrast"] == event["contrast"]
    assert projected[0]["qualification"] == event["qualification"]
    assert projected[0]["qualification_status"] == "recorded_scope_only"
    assert set(projected[0]["run_ids"]) == set(run_ids)
    assert "MOCK" in (site / "timeline.html").read_text()
    assert publication_preflight(site, root=data)["network_writes"] == 0
    build_site(data, site, runs=run_ids)
    subset = tmp_path / "selected-subset"
    build_site(data, subset, runs=run_ids[-1:])
    assert json.loads((subset / "site.json").read_text())["monitor_events"] == []
    assert data_hashes(data) == saved
    from servegauge.storage.export import publication_preflight as export_preflight
    from servegauge.storage.export import stage_export

    exported = tmp_path / "selected-export"
    export_result = stage_export(data, exported, runs=run_ids)
    assert export_result["publication_blockers"] == ["scripted_mock_not_public"]
    numeric = pl.read_parquet(next(exported.glob("data/*/change_events-*.parquet")))
    assert numeric.height == 1
    for key in [
        "left_successes",
        "right_successes",
        "left_n",
        "right_n",
        "difference",
        "ci_low",
        "ci_high",
    ]:
        assert numeric[key][0] == event["contrast"][key]
    assert numeric["qualification_sha256"][0] == event["qualification"]["sha256"]
    with pytest.raises(ValueError, match="publication"):
        export_preflight(data, exported)
    assert data_hashes(data) == saved
    table_raw = tmp_path / "table-raw-only"
    shutil.copytree(data / "data/raw", table_raw / "data/raw")
    rebuild(table_raw / "data/raw", table_raw / "tables")
    assert pl.read_parquet(tmp_path / "event-tables/change_events.parquet").equals(
        pl.read_parquet(table_raw / "tables/change_events.parquet")
    )
    # A lost prior input must refuse before replacing the existing table files.
    before_table = (table_raw / "tables/change_events.parquet").read_bytes()
    import servegauge.core.ids

    lost = (
        table_raw
        / "data/raw"
        / servegauge.core.ids.safe_filename(outputs[0]["result"]["jobs"][0]["run_id"])
        / "analysis.json"
    )
    lost.unlink()
    with pytest.raises((OSError, ValueError)):
        rebuild(table_raw / "data/raw", table_raw / "tables")
    assert (table_raw / "tables/change_events.parquet").read_bytes() == before_table
    (tmp_path / "operating-events-native-acceptance.json").write_text(
        json.dumps(
            {
                "source_hash": source,
                "result": result,
                "raw_only_equal": True,
                "repeat_is_read_only": True,
                "process_lock_exit": child.returncode,
                "test_contract_only": True,
                "canonical_units": 800,
                "site": {"selected_events": 1, "subset_events": 0, "exact_contrast": True},
                "export": {
                    "numeric_events": 1,
                    "exact_contrast": True,
                    "mock_publication_refused": True,
                },
                "model_api_calls": 0,
                "new_spend_usd": 0,
            },
            indent=2,
        )
        + "\n"
    )
