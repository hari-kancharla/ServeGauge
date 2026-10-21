"""One content identity and one Windows-safe path conversion, used everywhere."""

from __future__ import annotations

import hashlib
import re

from servegauge.core.hashing import content_hash


def exchange_id(run_id: str, endpoint_id: str, suite_id: str, item_id: str, sample_idx: int) -> str:
    return content_hash([run_id, endpoint_id, suite_id, item_id, sample_idx])


def safe_filename(value: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9._-]", "_", value)[:100]
    suffix = hashlib.sha256(value.encode()).hexdigest()[:8]
    return f"{prefix}-{suffix}"
