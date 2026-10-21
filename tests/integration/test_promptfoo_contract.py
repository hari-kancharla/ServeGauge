"""Optional real external runtime; default tests never execute it or open sockets."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest
from promptfoo_workflow import exercise_promptfoo_contract

from servegauge.comparison.isolation import ToolRuntime
from servegauge.storage.comparison import _guard


@pytest.mark.slow
@pytest.mark.enable_socket
def test_actual_promptfoo_python_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if sys.platform != "darwin":
        pytest.skip("native macOS boundary; equivalent other-platform isolation remains pending")
    tool = Path(
        os.environ.get("SERVEGAUGE_PROMPTFOO_TOOL_ROOT", ".cache/tools/promptfoo-0.124.1")
    ).absolute()
    _guard(tool)
    if not tool.is_dir() or not shutil.which("node"):
        pytest.skip("optional pinned tool/runtime not installed")
    monkeypatch.setenv("SYNTHETIC_ACCOUNT_CANARY", "synthetic-parent-only")
    value = exercise_promptfoo_contract(
        tmp_path / "contract",
        ToolRuntime(
            node=Path(shutil.which("node") or "node").absolute(),
            tool=tool,
            python=Path(sys.executable),
            package_root=Path(__file__).resolve().parents[2] / "src",
        ),
    )
    assert value["cache_did_not_reuse_identical_prompts"]
