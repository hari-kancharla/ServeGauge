from hypothesis import given, settings
from hypothesis import strategies as st

from servegauge.transport.sse import SseParser

WIRE = (
    '\ufeff: keepalive\r\nevent: message\r\nid: 7\r\ndata: {"text":"café 雨"}\r\n'
    "data: second line\r\n\r\n: another\n\ndata: [DONE]\n\n"
).encode()


def parsed(chunks: list[bytes]) -> list[dict[str, object]]:
    parser = SseParser()
    events = []
    for chunk in chunks:
        events.extend(parser.feed(chunk, 4.0))
    events.extend(parser.finish(4.0))
    assert parser.terminal
    return [event.model_dump() for event in events]


@given(st.lists(st.integers(0, len(WIRE)), max_size=100))
@settings(max_examples=250, derandomize=True)
def test_arbitrary_byte_chunking(cuts: list[int]) -> None:
    positions = sorted({0, len(WIRE), *cuts})
    chunks = [WIRE[a:b] for a, b in zip(positions, positions[1:], strict=False)]
    assert parsed(chunks) == parsed([WIRE])


def test_comments_ids_retry_and_incomplete_eof() -> None:
    parser = SseParser()
    events = parser.feed(b": comment\rretry: 12\rid: hello\rdata: x\r\rdata: partial", 1.0)
    assert events[0].is_comment
    assert events[1].data == "x"
    assert events[1].event_id == "hello"
    assert events[1].retry_ms == 12
    assert parser.finish(2.0) == []
    assert parser.incomplete
    assert not parser.terminal


def test_empty_data_unknown_fields_and_bad_utf8() -> None:
    parser = SseParser()
    events = parser.feed(b"unknown: x\nid: bad\x00id\nretry: nope\ndata:\n\n", 0)
    assert len(events) == 1
    assert events[0].data == ""
    assert events[0].event_id is None
    parser.feed(b"data: \xff\n\n", 0)
    assert parser.invalid_utf8
