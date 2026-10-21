"""Validated local policy loading without echoing configuration contents."""

from pathlib import Path

import yaml

from servegauge.runner.budget import BudgetPolicy


def load_budget(root: Path) -> BudgetPolicy:
    try:
        return BudgetPolicy.model_validate(
            yaml.safe_load((root / "configs/budgets.yaml").read_text(encoding="utf-8"))
        )
    except (OSError, ValueError, yaml.YAMLError):
        raise ValueError("Invalid or missing configs/budgets.yaml") from None
