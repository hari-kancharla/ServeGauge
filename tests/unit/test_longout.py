import json

import pytest

from servegauge.items.longout import long_output
from servegauge.scoring.degeneration import degeneration
from servegauge.tokencount.counters import text_tokens


@pytest.mark.parametrize("kind", ["json", "sequence"])
@pytest.mark.parametrize("target", [4000, 8000, 16000, 32000])
def test_generated_output_checker_and_length(kind, target):
    case = long_output(42, target, kind)
    assert target <= text_tokens(case.text) < target + 128
    assert case.matches(case.text) and not case.matches(case.text[:-10])
    if kind == "json":
        value = json.loads(case.text)
        value[-1]["square"] += 1
        assert not case.matches(json.dumps(value))


def test_degeneracy_known_answers():
    scores = degeneration("abcabcabc", ngram=3)
    assert scores["longest_repeat"] == 6 / 9
    assert scores["repeated_ngram"] == 4 / 7
    assert degeneration("abcdef", ngram=3)["longest_repeat"] == 0
