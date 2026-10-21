import json

import pytest
from jsonschema import Draft202012Validator

from servegauge.items.schemas import LADDER, schema_case


def test_1000_seeded_schema_cases_validate_and_are_reproducible():
    for seed in range(1000):
        category = LADDER[seed % len(LADDER)]
        case = schema_case(seed, category)
        Draft202012Validator.check_schema(case.schema)
        Draft202012Validator(case.schema).validate(case.value)
        assert schema_case(seed, category) == case
        json.dumps(case.value, ensure_ascii=False, allow_nan=False)
    assert len(schema_case(0, "enum200").schema["properties"]["category"]["enum"]) == 200
    with pytest.raises(ValueError):
        schema_case(0, "unknown")
