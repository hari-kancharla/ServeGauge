import random

import pytest
from tests_support import exchange

from servegauge.items.aime import load_local_aime
from servegauge.mock.backends import ScriptedBackend
from servegauge.mock.faults import FaultConfig
from servegauge.runner.endpoints import resolve_endpoint
from servegauge.scoring.degeneration import _longest_repeat
from servegauge.scoring.longout_scoring import reasoning_samples
from servegauge.scoring.math_scoring import exact_math_answer
from servegauge.suites.registry import prepare_item
from servegauge.suites.s5_longout import reasoning_items
from servegauge.verdicts.config import load_policy
from servegauge.verdicts.engine import Measurement, decide_family


def test_exact_longest_repeat_against_brute_force():
    rng = random.Random(42)
    for _ in range(100):
        text = "".join(rng.choice("abcdef") for _ in range(50))
        expected = max(
            (
                size
                for size in range(1, len(text))
                if len({text[i : i + size] for i in range(len(text) - size + 1)})
                < len(text) - size + 1
            ),
            default=0,
        )
        assert _longest_repeat(text) == expected


def test_math_exact_match_keeps_integer_format():
    assert exact_math_answer(" 2\n", 2) and not exact_math_answer("02", 2)
    assert not exact_math_answer("The answer is 2", 2)
    with pytest.raises(ValueError, match="integer_math_gold"):
        exact_math_answer("True", True)


def test_s5_clustered_repeats_and_output_fault():
    endpoint = resolve_endpoint("http://127.0.0.1:8011/v1", "mock", "gpt-oss-120b")
    endpoint.metadata["max_completion_tokens"] = 65536
    items = reasoning_items(
        "core", 42, {"items": 12, "samples": 2, "lengths": [4000, 8000], "long_items": 2}
    )
    engine = ScriptedBackend(1, FaultConfig(truncate_output_after=6000))
    records = []
    for item in items:
        body = prepare_item(item, endpoint).body
        for index in range(item.expected["samples"]):
            response = engine.respond(body)
            records.append(
                exchange(
                    endpoint_id=endpoint.id,
                    item_id=item.item_id,
                    sample_idx=index,
                    request=body,
                    response=response,
                    usage=response["usage"],
                )
            )
    scores = reasoning_samples(items, records, endpoint.id)
    assert len(scores["R01"].values) == 12 and all(
        len(row) == 2 for row in scores["R01"].repeats.values()
    )
    assert all(scores["R03.4000"].values.values())
    assert not any(scores["R03.8000"].values.values())
    result = decide_family(
        [
            Measurement(
                check_id="R01",
                suite_id="S5",
                endpoint_id="e",
                run_id="r",
                check_class="comparative",
                outcomes=[1.0] * 12,
                reference=[1.0] * 12,
                binary=False,
                reference_tier="BASELINE",
                noise_band=0,
                seed=42,
                details={
                    "binary_repeats": [[1.0, 1.0]] * 12,
                    "reference_binary_repeats": [[1.0, 1.0]] * 12,
                },
            )
        ],
        load_policy(),
    )[0]
    assert result.verdict == "INCONCLUSIVE" and result.delta_ci_low < -0.03


def test_aime_loader_local_only_and_export_refusal(tmp_path):
    path = tmp_path / "own-synthetic.jsonl"
    path.write_text('{"problem":"Own synthetic example: 1+1?","answer":2}\n', encoding="utf-8")
    loaded = load_local_aime(path)
    assert loaded.samples == 8 and loaded.license == "CC-BY-NC-SA-4.0"
    with pytest.raises(ValueError, match="export_forbidden"):
        loaded.export()
