"""Explicit selected-run monitoring and owner-assertion projections; no new inference."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from servegauge.core.hashing import content_hash
from servegauge.core.redaction import Redactor
from servegauge.observatory.notice_workflow import publication_status
from servegauge.observatory.notices import aware, failure_subject
from servegauge.storage.analysis import Analysis
from servegauge.storage.notices import notice_snapshots
from servegauge.storage.observatory_rows import event_inputs, observatory_rows


def observatory_context(root: Path, analyses: list[Analysis], *, as_of: datetime) -> dict[str, Any]:
    stamp = aware(as_of).isoformat()
    if not analyses:
        return {"monitor_events": [], "reply_notes": [], "publication_as_of": stamp}
    selected = {str(a.manifest["run_id"]) for a in analyses}
    raw = root / "data/raw"
    records = [r for a in analyses for r in a.records]
    rows = observatory_rows(raw, records, selected_runs=selected)
    events = []
    for row in rows["change_events"]:
        original = json.loads(row["record_json"])
        links = event_inputs(raw, original)
        contrast = original["contrast"]
        events.append(
            {
                **original,
                "endpoint_id": links[0][1],
                "run_ids": [r for r, _ in links],
                "difference_text": f"{contrast['difference'] * 100:+.4g} pp",
                "interval_text": (
                    f"[{contrast['ci_low'] * 100:.4g}, {contrast['ci_high'] * 100:.4g}] pp"
                ),
                "qualification_status": "recorded_scope_only",
            }
        )
    snapshots = notice_snapshots(raw)
    notes = []
    for analysis in analyses:
        for check in analysis.checks:
            if check.verdict != "FAIL":
                continue
            subject = failure_subject(analysis, check.endpoint_id, check.check_id)
            related = [
                s
                for s in snapshots
                if all(
                    s["spec"]["subject"][k] == getattr(subject, k)
                    for k in ["run_id", "endpoint_id", "check_id"]
                )
            ]
            matching = [
                s for s in related if s["spec"]["subject"] == subject.model_dump(mode="json")
            ]
            assertions = [
                {
                    "type": e["type"],
                    "at": e["at"],
                    "owner_asserted": True,
                    "reference_sha256": e["reference_sha256"],
                }
                for s in matching
                for e in s["events"]
            ]
            notes.append(
                {
                    "run_id": check.run_id,
                    "endpoint_id": check.endpoint_id,
                    "check_id": check.check_id,
                    "subject_id": subject.subject_id,
                    "synthetic_mock": subject.synthetic_mock,
                    "state": publication_status(root, subject, now=as_of),
                    "assertions": assertions,
                    "earlier_revisions": len(related) - len(matching),
                    "notice_snapshot_hash": content_hash(matching),
                    "delivery_verified": False,
                }
            )
    return dict(
        Redactor.from_environment().redact(
            {
                "monitor_events": sorted(
                    events, key=lambda e: (e["contrast"]["observation_end"], e["event_id"])
                ),
                "reply_notes": notes,
                "publication_as_of": stamp,
            }
        )
    )
