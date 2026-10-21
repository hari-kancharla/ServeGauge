"""Isolated, socket-blocked scheduled producer used only by native acceptance."""

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from servegauge.tokencount.cache import configure_offline
from servegauge.validation.async_offline import SocketlessLoop
from servegauge.validation.common import offline_only

configure_offline()
parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--stamp", required=True)
parser.add_argument("--level", choices=["core", "full"], default="core")
parser.add_argument("--error-rate", type=float, default=0)
args = parser.parse_args()
assert not args.out.exists()
with offline_only():
    from servegauge.mock.faults import FaultConfig
    from servegauge.observatory.identity import producer_identity
    from servegauge.observatory.rolling import load_rolling_policy
    from servegauge.observatory.schedule import ObservatoryConfig
    from servegauge.observatory.tick import tick
    from servegauge.observatory.transport import transport_batch
    from servegauge.storage.analysis import reconstruct
    from servegauge.storage.monitor import MonitorJournal

    config = ObservatoryConfig.model_validate(
        {
            "id": "producer-bridge",
            "enabled": True,
            "requests_per_minute": 600000,
            "models": [
                {
                    "id": "one",
                    "profile": "gpt-oss-120b",
                    "mock_context_length": 512,
                    "faults": {"error_rate": args.error_rate},
                    "endpoints": [
                        {
                            "id": "mock:bridge",
                            "kind": "mock",
                            "base_url": "http://127.0.0.1:8011/v1",
                            "model": "mock",
                            "claimed_profile": "gpt-oss-120b",
                        }
                    ],
                }
            ],
        }
    )
    config.cadences["daily"].tests = ["P01"]
    config.cadences["daily"].fingerprint_level = None
    config.cadences["daily"].level = args.level
    with asyncio.Runner(loop_factory=SocketlessLoop) as runner:
        result = runner.run(
            tick(config, "daily", root=args.root, now=datetime.fromisoformat(args.stamp))
        )
    journal = MonitorJournal(args.root / "data/raw", result["window_id"])
    spec = journal.spec()
    run = result["jobs"][0]["run_id"]
    analysis = reconstruct(args.root / "data/raw", run)
    batch = transport_batch(
        analysis,
        "mock:bridge",
        producer_source_hash=spec["binding"]["source_hash"],
        resumed=False,
        faults=FaultConfig(error_rate=args.error_rate),
        policy=load_rolling_policy(),
    )
    records = [r for r in analysis.records if r.item_id.startswith("P01-")]
    value = {
        "producer": producer_identity(),
        "binding": spec["binding"],
        "result": result,
        "batch": asdict(batch),
        "p01_records": len(records),
        "p01_sent_records": sum(bool(r.attempts) for r in records),
        "p01_attempts": sum(len(r.attempts) for r in records),
        "p01_first_successes": sum(
            bool(r.attempts) and r.attempts[0].status == 200 for r in records
        ),
        "operating_qualification_accepted": False,
        "model_api_calls": 0,
        "new_spend_usd": 0,
    }
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
