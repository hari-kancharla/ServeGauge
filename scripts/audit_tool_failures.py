"""Audit a named retained run offline; no credentials, dispatch or raw-store writes."""

import argparse
import hashlib
from pathlib import Path

from servegauge.core.hashing import canonical_json, content_hash
from servegauge.diagnose.tool_audit import audit_tools
from servegauge.storage.analysis import reconstruct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spend = Path("budget/dev_spend.json")
    before = hashlib.sha256(spend.read_bytes()).hexdigest()
    records = reconstruct(Path("data/raw"), args.run).records
    result = {
        "run_id": args.run,
        "source_records": len(records),
        "record_hash": content_hash([r.model_dump(mode="json") for r in records]),
        "audit": audit_tools(records),
        "spend_hash_before": before,
        "spend_hash_after": hashlib.sha256(spend.read_bytes()).hexdigest(),
        "model_calls": 0,
        "raw_store_writes": 0,
    }
    assert result["spend_hash_after"] == before
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(canonical_json(result) + b"\n")
    for endpoint, summary in result["audit"]["endpoints"].items():
        print(endpoint, summary["counts"])


if __name__ == "__main__":
    main()
