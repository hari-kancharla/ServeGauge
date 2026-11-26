"""The real entrypoint separates usage from outcomes; estimation cannot dispatch."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from servegauge import cli
from servegauge.storage.workspace import initialize

ROOT = Path(__file__).resolve().parents[2]
CHECK = [
    "--endpoint",
    "http://127.0.0.1:8123/v1",
    "--model",
    "mock",
    "--profile",
    "gpt-oss-120b",
    "--tests",
    "P01",
    "--level",
    "core",
]


def objects(value: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    result = []
    while value.strip():
        value = value.lstrip()
        row, end = decoder.raw_decode(value)
        result.append(row)
        value = value[end:]
    return result


def files(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def process(
    args: list[str], *, cwd: Path = ROOT, prefix: str = ""
) -> subprocess.CompletedProcess[str]:
    environment = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "OPENROUTER_API_KEY",
            "SERVEGAUGE_OWNER_RUN",
            "TRUEWEIGHTS_OWNER_RUN",
            # CI runners force colored Rich output, which splits option names with
            # escape codes; children print plain text so output checks stay stable.
            "GITHUB_ACTIONS",
            "FORCE_COLOR",
            "PY_COLORS",
        }
    }
    environment["NO_COLOR"] = "1"
    environment["SERVEGAUGE_TOKENIZER_CACHE"] = str(ROOT / ".cache/tokenizers")
    # Every child refuses all sockets; no account or provider preflight can escape.
    code = (
        "import asyncio,socket,sys\n"
        "def blocked(*a,**k): raise RuntimeError('network_refused')\n"
        "class NoSockets(socket.socket):\n"
        "    def __new__(cls,*a,**k): return blocked()\n"
        "socket.socket=NoSockets; socket.getaddrinfo=blocked\n"
        "from servegauge.validation.async_offline import SocketlessLoop\n"
        "class OfflinePolicy(asyncio.DefaultEventLoopPolicy):\n"
        "    def new_event_loop(self): return SocketlessLoop()\n"
        "asyncio.set_event_loop_policy(OfflinePolicy())\n"
        "import servegauge.cli as cli\n"
        + prefix
        + "\nsys.argv=['servegauge',*sys.argv[1:]]\ncli.main()\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
    )


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["not-a-command"],
        ["--definitely-invalid-option"],
        ["estimate", "--plan"],
        ["check", "--max-usd", "PRIVATE_PARSER_VALUE"],
        ["check", "--max-usd", "https://unused.invalid/?credential=PRIVATE_PARSER_VALUE"],
    ],
)
def test_real_entrypoint_usage_is_three_without_echoing_values(args: list[str]) -> None:
    result = process(args)
    assert result.returncode == 3
    assert "PRIVATE_PARSER_VALUE" not in result.stdout + result.stderr
    assert "unused.invalid" not in result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    assert "usage" in result.stderr.lower()


@pytest.mark.parametrize("code", [0, 1, 2, 3, 4, 5])
def test_entrypoint_preserves_explicit_command_outcomes(code: int) -> None:
    prefix = (
        "import typer\n"
        "probe=typer.Typer()\n"
        "@probe.command()\n"
        "def outcome(): raise typer.Exit(" + str(code) + ")\n"
        "cli.app=probe\n"
    )
    assert process([], prefix=prefix).returncode == code


def test_real_help_budget_and_harness_outcomes_remain_distinct(tmp_path: Path) -> None:
    assert process(["--help"]).returncode == 0
    assert process(["run", "--plan", "configs/plans/transport_smoke.yaml"]).returncode == 4
    bad = tmp_path / "raw"
    bad.mkdir()
    (bad / "bad.jsonl").write_text("not-json\n")
    assert (
        process(["rebuild", "--raw", str(bad), "--out", str(tmp_path / "tables")]).returncode == 5
    )
    result = process(
        ["estimate", "--plan", "configs/plans/transport_smoke.yaml"],
        prefix=(
            "def fail(*a,**k): raise RuntimeError('PRIVATE_HARNESS_VALUE')\n"
            "cli.estimate_plan=fail\n"
        ),
    )
    assert result.returncode == 5 and "PRIVATE_HARNESS_VALUE" not in result.stdout + result.stderr


@pytest.mark.parametrize("baseline", [False, True])
def test_check_argument_estimate_matches_actual_dry_run_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, baseline: bool
) -> None:
    initialize(tmp_path)
    monkeypatch.chdir(tmp_path)
    before = files(tmp_path)
    args = [*CHECK]
    if baseline:
        args += ["--baseline", "http://127.0.0.1:8124/v1", "--baseline-model", "mock"]
    ordinary = process(["check", *args, "--dry-run"], cwd=tmp_path)
    assert ordinary.returncode == 0, ordinary.stdout + ordinary.stderr
    estimated = process(["estimate", "--check-args", *args], cwd=tmp_path)
    assert estimated.returncode == 0, estimated.stdout + estimated.stderr
    actual = objects(estimated.stdout)
    assert len(actual) == 1 and actual[0] == objects(ordinary.stdout)[0]
    assert actual[0]["total_reserved_cost_usd"] == "0"
    assert len(actual[0]["cost_table"]) == (2 if baseline else 1)
    assert files(tmp_path) == before


def test_estimate_expands_the_same_requirements_and_rejects_excluded_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize(tmp_path)
    requirement = tmp_path / "requires.yaml"
    requirement.write_text("app: example\nrequires:\n  - id: cap\n    tests: [P01]\n")
    monkeypatch.chdir(tmp_path)
    args = CHECK[:6] + ["--requires", str(requirement), "--level", "core"]
    original = process(["check", *args, "--dry-run"], cwd=tmp_path)
    estimated = process(["estimate", "--check-args", *args], cwd=tmp_path)
    assert original.returncode == estimated.returncode == 0, original.stderr + estimated.stderr
    assert objects(estimated.stdout)[0] == objects(original.stdout)[0]
    before = files(tmp_path)
    rejected = process(["estimate", "--check-args", *args, "--tests", "P02"], cwd=tmp_path)
    assert rejected.returncode == 3 and files(tmp_path) == before


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--check-args"],
        ["--endpoint", "http://127.0.0.1:8123/v1"],
        ["--plan", "plan.yaml", "--check-args", *CHECK],
        ["--plan", "plan.yaml", "--level", "smoke"],
        ["--plan", "plan.yaml", "--tests", "P01"],
        ["--plan", "plan.yaml", "--api-key-env", "UNUSED_KEY_ENV"],
        ["--plan", "plan.yaml", "--max-usd", "0"],
        ["--check-args", *CHECK, "--yes"],
        ["--check-args", *CHECK, "--dry-run"],
        ["--check-args", *CHECK, "--out", "foreign-output"],
        ["--check-args", *CHECK, "--max-usd", "nan"],
        ["--check-args", *CHECK, "--max-usd", "inf"],
        ["--check-args", *CHECK, "--max-usd", "-1"],
    ],
)
def test_estimate_modes_refuse_ambiguous_or_incomplete_inputs_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    initialize(tmp_path)
    source = ROOT / "configs/plans/transport_smoke.yaml"
    (tmp_path / "plan.yaml").write_bytes(source.read_bytes())
    monkeypatch.chdir(tmp_path)
    before = files(tmp_path)
    result = process(["estimate", *args], cwd=tmp_path)
    assert result.returncode == 3 and files(tmp_path) == before


def test_estimate_plan_mode_and_help_remain_available() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.app, ["estimate", "--plan", "configs/plans/transport_smoke.yaml"])
    assert result.exit_code == 0
    assert objects(result.stdout)[0]["total_reserved_cost_usd"] == "0"
    help_result = process(["estimate", "--help"])
    assert help_result.returncode == 0 and "--check-args" in help_result.stdout


@pytest.mark.parametrize("tier", ["MOCK", "BASELINE", "R3"])
def test_selected_reference_pack_retains_its_tier_in_both_estimates(
    tmp_path: Path, tier: str
) -> None:
    from unit.test_refs import reference_run

    from servegauge.refs.pack import build_pack, save_pack

    initialize(tmp_path)
    source = reference_run()
    source.profile = "gpt-oss-120b"
    for endpoint in source.endpoints:
        endpoint.profile = source.profile
    # Synthetic pack metadata exercises binding only; it cannot qualify a provider.
    pack = build_pack(source, tier, seed=8)
    saved = save_pack(pack, tmp_path / "refs")
    before = files(tmp_path)
    args = [*CHECK, "--baseline", str(saved)]
    ordinary = process(["check", *args, "--dry-run"], cwd=tmp_path)
    estimate = process(["estimate", "--check-args", *args], cwd=tmp_path)
    assert ordinary.returncode == estimate.returncode == 0, ordinary.stderr + estimate.stderr
    planned = objects(estimate.stdout)[0]
    assert planned == objects(ordinary.stdout)[0]
    assert {row["reference_tier"] for row in planned["mde_table"]} == {pack.tier}
    assert files(tmp_path) == before
