import json

from jsonschema import Draft202012Validator

from servegauge.items.toolcalls import DOMAINS, domain_case, toolcall_item


def test_ten_thousand_tool_corpus_seeds_are_internally_consistent() -> None:
    for domain in DOMAINS:
        _, schema, _ = domain_case(domain, 1)
        Draft202012Validator.check_schema(schema)
    domains, call_counts, histories = set(), set(), set()
    for seed in range(10000):
        item = toolcall_item(seed, max_tools=30, samples=2, numeric_tolerance=1e-9)
        tools = item.payload["tools"]
        offered = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
        assert 1 <= len(tools) <= 30 and len(offered) == len(tools)
        assert bool(item.expected["calls"]) == item.expected["should_call"]
        for call in item.expected["calls"]:
            assert call["name"] in offered
            Draft202012Validator(offered[call["name"]]).validate(call["arguments"])
        assert item == toolcall_item(seed, max_tools=30, samples=2, numeric_tolerance=1e-9)
        assert item.license == "CC-BY-4.0" and item.source == "generated"
        json.dumps(item.model_dump(mode="json"), allow_nan=False)
        domains.add(item.tags[0])
        call_counts.add(len(item.expected["calls"]))
        histories.add(item.expected["history"])
    assert domains == set(DOMAINS) and call_counts == {0, 1, 2} and histories == {False, True}
