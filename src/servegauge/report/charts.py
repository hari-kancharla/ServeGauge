"""Fixed presentation palette and exact outcome counts, never a quality score."""

from collections import Counter
from typing import Any

from servegauge.core.models import CheckResult

LEGEND = (
    ("PASS", "Pass", "#186354"),
    ("FAIL", "Fail", "#b3483e"),
    ("ERROR", "Error", "#955029"),
    ("INCONCLUSIVE", "Inconclusive", "#a48535"),
    ("INFO", "Info", "#637785"),
    ("NOT_SUPPORTED", "Unsupported", "#8e8b99"),
    ("NOT_APPLICABLE", "Not applicable", "#a7afa5"),
)


def chart_payload(checks: list[CheckResult], endpoints: list[dict[str, str]]) -> dict[str, Any]:
    labels = {e["id"]: e["label"] for e in endpoints}
    order = {verdict: index for index, (verdict, _, _) in enumerate(LEGEND)}
    counts = Counter((c.endpoint_id, c.verdict) for c in checks)
    rows = [
        {
            "endpoint": endpoint,
            "label": labels[endpoint],
            "verdict": verdict,
            "count": count,
            "order": order[verdict],
        }
        for (endpoint, verdict), count in sorted(counts.items())
    ]
    return {
        "endpoints": endpoints,
        "legend": [
            {"verdict": key, "label": label, "color": color} for key, label, color in LEGEND
        ],
        "counts": rows,
    }
