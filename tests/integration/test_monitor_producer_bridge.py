"""Actual package inventories bridge only an identical recorded mock generator/runtime."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import fields
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from servegauge.core.hashing import content_hash
from servegauge.observatory.identity import generator_manifest
from servegauge.observatory.provenance import implementation_hash
from servegauge.observatory.rolling import load_rolling_policy
from servegauge.observatory.temporal import TemporalFrame, replay_candidates
from servegauge.observatory.transport import TransportBatch


@pytest.mark.slow
def test_actual_producer_inventory_bridge_full_batch_and_all_error_preflight(tmp_path):
    root = Path.cwd()
    source = implementation_hash()
    copy = tmp_path / "producer-copy"
    shutil.copytree(
        root / "src/servegauge",
        copy / "src/servegauge",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (copy / "src/servegauge/producer_version_marker.py").write_text(
        '"""Unused additive inventory marker for native compatibility acceptance."""\n'
    )
    for folder, suffixes in [("configs", {".yaml", ".json"}), ("profiles", {".yaml"})]:
        (copy / folder).mkdir(parents=True)
        for p in (root / folder).iterdir():
            if not p.is_symlink() and p.is_file() and p.suffix in suffixes:
                shutil.copyfile(p, copy / folder / p.name)
    for name in ["uv.lock", "pyproject.toml", "README.md"]:
        shutil.copyfile(root / name, copy / name)
    changed = implementation_hash(copy / "src/servegauge")
    assert changed != source and generator_manifest(copy / "src/servegauge") == generator_manifest()
    (tmp_path / "configs").mkdir()
    shutil.copyfile(root / "configs/budgets.yaml", tmp_path / "configs/budgets.yaml")
    helper = root / "tests/support/monitor_producer_probe.py"
    outputs = []
    for index, (cwd, level, error) in enumerate(
        [(copy, "core", 0), (root, "core", 0), (root, "full", 0), (root, "core", 1)]
    ):
        out = tmp_path / f"producer-{index}.json"
        environment = {**os.environ, "SERVEGAUGE_TOKENIZER_CACHE": str(root / ".cache/tokenizers")}
        environment["PYTHONPATH"] = str(copy / "src") if cwd == copy else str(root / "src")
        command = [
            sys.executable,
            str(helper),
            "--root",
            str(tmp_path),
            "--out",
            str(out),
            "--stamp",
            f"2026-02-{index + 1:02}T00:00:00+00:00",
            "--level",
            level,
            "--error-rate",
            str(error),
        ]
        process = subprocess.run(
            command, cwd=cwd, env=environment, capture_output=True, text=True, check=False
        )
        assert process.returncode == 0, process.stderr[-2000:]
        outputs.append(json.loads(out.read_text()))
    a, b, full, outage = outputs
    assert a["producer"]["source_hash"] == a["binding"]["source_hash"] == changed
    assert b["producer"]["source_hash"] == b["binding"]["source_hash"] == source
    assert a["producer"]["generator"] == b["producer"]["generator"]
    assert a["producer"]["runtime"] == b["producer"]["runtime"]
    assert all(v["binding"]["loaded_source_hash"] == v["binding"]["source_hash"] for v in outputs)
    assert a["batch"]["ready"] and b["batch"]["ready"] and not a["batch"]["qualified"]
    assert (
        full["batch"]["ready"]
        and full["p01_records"]
        == full["p01_sent_records"]
        == full["p01_attempts"]
        == full["p01_first_successes"]
        == 800
    )
    assert (
        not outage["batch"]["ready"] and outage["p01_sent_records"] == outage["p01_attempts"] == 0
    )
    assert outage["p01_records"] == 400 and outage["batch"]["values"] == []
    measurement = content_hash(
        [
            "scripted-independent-transport-v1",
            a["producer"]["generator"]["generator_hash"],
            a["producer"]["runtime"]["runtime_hash"],
        ]
    )
    frames = []
    init = {f.name for f in fields(TransportBatch) if f.init}
    for index, row in enumerate([a, b]):
        batch = TransportBatch(
            **{k: tuple(v) if k == "values" else v for k, v in row["batch"].items() if k in init}
        )
        start = datetime.fromisoformat(row["result"]["start"])
        frames.append(
            TemporalFrame(
                "bridge",
                f"window-{index}",
                "daily",
                start,
                start + timedelta(days=1),
                row["producer"]["source_hash"],
                batch,
                measurement_source_hash=measurement,
                producer_runtime_hash=row["producer"]["runtime"]["runtime_hash"],
            )
        )
    replay = replay_candidates(frames, load_rolling_policy(), expected_source_hash=measurement)
    assert [s.status for s in replay] == ["BASELINE", "NO_CANDIDATE"]
    assert all(not s.qualified for s in replay)
    assert implementation_hash() == source
    (tmp_path / "producer-bridge-acceptance.json").write_text(
        json.dumps(
            {
                "source_hash": source,
                "alternate_actual_producer_source_hash": changed,
                "generator": a["producer"]["generator"],
                "runtime": a["producer"]["runtime"],
                "raw_first_counts": [r["p01_first_successes"] for r in outputs],
                "full_canonical_units": 800,
                "all_error_preflight_abstains": True,
                "compatible_actual_producer_bridge": True,
                "operating_qualification_accepted": False,
                "model_api_calls": 0,
                "new_spend_usd": 0,
            },
            indent=2,
        )
        + "\n"
    )
