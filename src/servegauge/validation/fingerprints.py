"""Scripted fingerprint stimuli and independent controls, separate from their scorers."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from servegauge.core.models import CheckResult, Exchange, Item
from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.refs.calibrate import calibrate
from servegauge.scoring.common import content
from servegauge.scoring.fingerprints import Cells
from servegauge.scoring.fingerprints import fingerprint_checks as compare
from servegauge.scoring.longout_scoring import reasoning_samples
from servegauge.suites.s1_protocol import probe
from servegauge.suites.s5_longout import reasoning_items
from servegauge.validation.capture import captures, endpoint
from servegauge.validation.common import named_seed, settings
from servegauge.verdicts.config import load_policy
from servegauge.verdicts.engine import Measurement


def fingerprint_items() -> list[Item]:
    cfg = settings()
    items = []
    languages = ("Choose one", "Elige uno", "Choisissez un", "Wähle eins")
    families = (
        ("red", "blue", "green", "gold"),
        ("heads", "tails", "edge", "again"),
        ("1", "2", "3", "4"),
        ("cat", "dog", "fox", "owl"),
        ("east", "west", "north", "south"),
        ("spring", "summer", "autumn", "winter"),
        ("tea", "water", "coffee", "juice"),
        ("oak", "pine", "elm", "ash"),
        ("A", "B", "C", "D"),
        ("circle", "square", "triangle", "star"),
    )
    for index in range(cfg["fingerprint"]["cells"]):
        choices = list(families[(index // 4) % len(families)])
        item = probe(
            "FP1",
            index,
            "cell",
            languages[index % 4] + ": " + ", ".join(choices) + ". One answer only.",
            {"type": "sample", "choices": choices},
            params={"temperature": 1.0, "max_tokens": 128},
            samples=cfg["fingerprint"]["samples"],
        )
        item.suite_id, item.version = "S6", "fingerprint-lab-v1"
        items.append(item)
    items.extend(i for i in reasoning_items("core", cfg["seed"]) if i.expected["check_id"] == "R01")
    return items


def fingerprint_checks(
    items: list[Item], rows: dict[str, list[Exchange]], seed: int
) -> list[CheckResult]:
    def cells(records: list[Exchange]) -> Cells:
        by_item: dict[str, list[Exchange]] = defaultdict(list)
        for record in records:
            by_item[record.item_id].append(record)
        return [
            [content(r) for r in sorted(by_item[i.item_id], key=lambda r: r.sample_idx)]
            for i in items
            if i.suite_id == "S6"
        ]

    cfg, policy = settings(), load_policy()
    controls = []
    for i, name in enumerate(cfg["fingerprint"]["different_model_controls"]):
        control_seed = named_seed(seed, "control", i)
        controls.append(
            cells(
                captures(
                    [i for i in items if i.suite_id == "S6"],
                    ScriptedBackend(control_seed, FaultConfig(), model_control=name),
                    endpoint("mock:control-" + str(i), control_seed, clean=False),
                    "validation-control-" + str(seed),
                )
            )
        )
    math = {key: reasoning_samples(items, records, key)["R01"] for key, records in rows.items()}
    keys = sorted(math["mock:target"].values)
    reference = np.mean(
        [[math[f"mock:reference-{i}"].values[k] for k in keys] for i in range(3)], axis=0
    ).tolist()
    reference_repeats = math["mock:reference-0"].repeats
    calibration = calibrate(reference, seed=seed)
    extra = Measurement(
        check_id="R01",
        suite_id="S5",
        endpoint_id="mock:target",
        run_id="validation-" + str(seed),
        check_class="comparative",
        outcomes=[math["mock:target"].values[k] for k in keys],
        reference=reference,
        reference_tier="MOCK",
        binary=False,
        noise_band=float(str(calibration["noise_band"])),
        seed=seed,
        evidence=math["mock:target"].evidence,
        details={
            "binary_repeats": [math["mock:target"].repeats[k] for k in keys],
            "reference_binary_repeats": [reference_repeats[k] for k in keys],
        },
    )
    return compare(
        cells(rows["mock:target"]),
        [cells(rows[f"mock:reference-{i}"]) for i in range(3)],
        controls,
        seed=seed,
        length=cfg["fingerprint"]["fixed_length"],
        policy=policy,
        extra=[extra],
        evidence=[r.exchange_id for r in rows["mock:target"] if r.suite_id == "S6"],
    )
