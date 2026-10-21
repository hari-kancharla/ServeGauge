"""Load a finite replay book only from externally anchored, verified private sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from servegauge.core.hashing import content_hash
from servegauge.core.ids import safe_filename
from servegauge.core.models import ValueModel
from servegauge.mock.sealed_replay import FiniteReplay, ReplayObservation
from servegauge.storage.comparison import (
    ComparisonSelection,
    _case,
    _load,
    _private_json,
    verify_sources,
)


class ReplayBook(ValueModel):
    selection_hash: str
    cases: list[dict[str, Any]]
    observations: list[ReplayObservation]

    def tape(self) -> FiniteReplay:
        """Create one finite execution; a new tape is a disclosed reuse, never new n."""
        return FiniteReplay(self.observations)


def load_replay_book(out: Path, *, selection_hash: str) -> ReplayBook:
    """Read and reverify without serving any source record or changing its n."""
    verify_sources(out, selection_hash=selection_hash)
    selection = ComparisonSelection.model_validate(
        _private_json((out / "selection.json").read_bytes())
    )
    if content_hash(selection.model_dump(mode="json")) != selection_hash:
        raise ValueError("comparison_replay_selection_changed")
    # Re-read pins validate bytes used for this in-memory book, not merely a prior
    # check of files that might have changed between verification and loading.
    sources = {
        pin.run_id: _load(out / "sources" / safe_filename(pin.run_id), pin)
        for pin in selection.sources
    }
    cases: list[dict[str, Any]] = []
    observations: list[ReplayObservation] = []
    for selection_case in selection.cases:
        source = sources[selection_case.run_id]
        case = _case(source, selection_case)
        cases.append(case)
        for row in case["observations"]:
            observations.append(
                ReplayObservation(
                    case_id=selection_case.case_id,
                    role=row["role"],
                    source_position=row["source_position"],
                    record_hash=row["record_hash"],
                    request_hash=row["request_hash"],
                    record=source.records[row["exchange_id"]].model_copy(deep=True),
                )
            )
    return ReplayBook(selection_hash=selection_hash, cases=cases, observations=observations)
