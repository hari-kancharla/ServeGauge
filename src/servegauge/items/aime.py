"""Opt-in, local-only benchmark intake. Text is never an exportable artifact."""

import json
from pathlib import Path
from typing import Any

from servegauge.core.hashing import content_hash
from servegauge.core.models import ValueModel


class LocalAime(ValueModel):
    source: str = "https://huggingface.co/datasets/MathArena/aime_2025"
    license: str = "CC-BY-NC-SA-4.0"
    content_hash: str
    rows: list[dict[str, Any]]
    samples: int = 8

    def export(self) -> dict[str, Any]:
        raise ValueError("AIME_text_export_forbidden_local_only")


def load_local_aime(path: Path) -> LocalAime:
    if path.suffix != ".jsonl":
        raise ValueError("local_AIME_JSONL_required_no_download")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or any(
        not isinstance(row, dict)
        or not isinstance(row.get("problem"), str)
        or not str(row.get("answer", "")).isdigit()
        for row in rows
    ):
        raise ValueError("AIME_problem_and_integer_answer_required")
    return LocalAime(content_hash=content_hash(rows), rows=rows)
