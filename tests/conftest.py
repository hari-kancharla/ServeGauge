"""All default tests are offline; even the async loop opens no Unix sockets."""

import asyncio
import json
import platform
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

import pytest

from servegauge.tokencount.cache import configure_offline
from servegauge.validation.async_offline import EmptySafeSelector as EmptySafeSelector
from servegauge.validation.async_offline import SocketlessLoop

configure_offline()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    registry = Path(__file__).resolve().parents[1] / "configs/fingerprint_qualification.json"
    required = json.loads(registry.read_text(encoding="utf-8"))["runtime"]
    observed = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
    }
    if observed != required:
        refusal = pytest.mark.skip(
            reason="Empirical fingerprint admission requires its pinned runtime"
        )
        for item in items:
            if item.get_closest_marker("qualified_runtime") is not None:
                item.add_marker(refusal)


def pytest_asyncio_loop_factories(
    config: pytest.Config, item: pytest.Item
) -> dict[str, Callable[[], asyncio.AbstractEventLoop]]:
    return {"socketless": SocketlessLoop}
