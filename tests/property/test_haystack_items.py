import re

import pytest

from servegauge.items.haystacks import context_case
from servegauge.tokencount.counters import decode, encode


def test_seeded_context_lengths_records_and_boundary_canaries():
    for seed in range(42):
        length = 4000 if seed % 2 else 16000
        case = context_case(
            seed,
            length=length,
            depth=[0, 10, 25, 50, 75, 90, 100][seed % 7],
            task_type=seed % 2,
            aggregate_records=16,
            padding_tokens=64,
            maximum_excess=128,
            iterations=4,
        )
        assert length <= case.visible_tokens <= length + 128
        tokens = encode(case.text)
        assert case.expected["head"] in decode(tokens[:200])
        assert case.expected["tail"] in decode(tokens[-200:])
        keys = case.mock_task["keys"]
        for field in ("single", "multi1", "multi2", "head", "tail"):
            found = re.findall(rf"(?m)^{keys[field]}=(\S+)", case.text)
            assert found == [case.expected[field]]
        assert re.findall(rf"(?m)^{keys['track']}=(\S+)", case.text)[-1] == case.expected["track"]
        assert (
            sum(
                int(value)
                for value in re.findall(
                    rf"(?m)^{case.mock_task['aggregate_prefix']}\d+=(\d+)", case.text
                )
            )
            == case.expected["total"]
        )
        assert case.mock_task["type"] == "context" and "total" not in case.mock_task
    with pytest.raises(ValueError):
        context_case(
            1,
            length=4,
            depth=50,
            task_type=0,
            aggregate_records=16,
            padding_tokens=64,
            maximum_excess=128,
            iterations=4,
        )
