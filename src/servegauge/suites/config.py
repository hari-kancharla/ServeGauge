"""Versioned suite designs; measurement thresholds stay in configs/margins.yaml."""

from pathlib import Path
from typing import Any

import yaml


def suite_config(path: Path = Path("configs/suites.yaml")) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value.get("version"):
        raise ValueError("versioned_suite_config_required")
    return value
