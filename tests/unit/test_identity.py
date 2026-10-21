"""Packaging and compatibility checks for the ServeGauge rename."""

import importlib.metadata
import subprocess
import sys
from pathlib import Path

import pytest

from servegauge import __version__
from servegauge.cli import main
from servegauge.runner.budget import BudgetManager, BudgetPolicy
from servegauge.tokencount.cache import cache_dir


def test_distribution_and_legacy_replay_entrypoint() -> None:
    from trueweights import __version__ as legacy_version
    from trueweights.cli import main as legacy_main

    assert importlib.metadata.version("servegauge") == __version__ == legacy_version
    assert legacy_main is main
    distribution = importlib.metadata.distribution("servegauge")
    commands = {entry.name: entry.value for entry in distribution.entry_points}
    assert commands["servegauge"] == commands["trueweights"] == "servegauge.cli:main"
    for module in ("servegauge.cli", "trueweights.cli"):
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Commands" in result.stdout


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({}, False),
        ({"SERVEGAUGE_OWNER_RUN": "1"}, True),
        ({"TRUEWEIGHTS_OWNER_RUN": "1"}, True),
        ({"SERVEGAUGE_OWNER_RUN": "0", "TRUEWEIGHTS_OWNER_RUN": "1"}, False),
    ],
)
def test_owner_flag_alias_preserves_caps_and_new_flag_precedence(
    tmp_path: Path, environment: dict[str, str], expected: bool
) -> None:
    policy = BudgetPolicy(
        monthly_usd=50,
        per_run_usd=25,
        dev_live_usd=5,
        selfhost_usd_per_session=20,
        stop_at_fraction="0.9",
    )
    budget = BudgetManager(tmp_path, "identity-check", policy, environment=environment)
    assert budget.owner is expected
    assert budget.run_limit == (25 if expected else 5)


def test_cache_environment_alias_and_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SERVEGAUGE_TOKENIZER_CACHE", raising=False)
    monkeypatch.setenv("TRUEWEIGHTS_TOKENIZER_CACHE", str(tmp_path / "legacy"))
    assert cache_dir() == tmp_path / "legacy"
    monkeypatch.setenv("SERVEGAUGE_TOKENIZER_CACHE", str(tmp_path / "current"))
    assert cache_dir() == tmp_path / "current"
