"""WHATWG line framing plus lossless comment/field capture and arrival timestamps."""

from __future__ import annotations

import codecs

from servegauge.core.models import SseEvent


class SseParser:
    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8-sig")("replace")
        self._buffer = ""
        self._skip_lf = False
        self._data: list[str] = []
        self._lines: list[str] = []
        self._event: str | None = None
        self._id: str | None = None
        self._retry: int | None = None
        self.terminal = False
        self.incomplete = False
        self.invalid_utf8 = False

    def feed(self, chunk: bytes, t_ms: float) -> list[SseEvent]:
        text = self._decoder.decode(chunk)
        self.invalid_utf8 |= "\ufffd" in text
        result: list[SseEvent] = []
        for char in text:
            if self._skip_lf:
                self._skip_lf = False
                if char == "\n":
                    continue
            if char in "\r\n":
                result.extend(self._line(self._buffer, t_ms))
                self._buffer = ""
                self._skip_lf = char == "\r"
            else:
                self._buffer += char
        return result

    def _line(self, line: str, t_ms: float) -> list[SseEvent]:
        if line == "":
            result = []
            if self._data:
                data = "\n".join(self._data)
                result.append(
                    SseEvent(
                        t_ms=t_ms,
                        event=self._event,
                        data=data,
                        event_id=self._id,
                        retry_ms=self._retry,
                        raw_lines=self._lines.copy(),
                    )
                )
                self.terminal |= data == "[DONE]"
            self._data, self._lines, self._event = [], [], None
            return result
        if line.startswith(":"):
            return [
                SseEvent(
                    t_ms=t_ms, data=line[1:].removeprefix(" "), is_comment=True, raw_lines=[line]
                )
            ]
        self._lines.append(line)
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "data":
            self._data.append(value)
        elif field == "event":
            self._event = value or None
        elif field == "id" and "\0" not in value:
            self._id = value
        elif field == "retry" and value.isascii() and value.isdigit():
            self._retry = int(value)
        return []

    def finish(self, t_ms: float) -> list[SseEvent]:
        tail = self._decoder.decode(b"", final=True)
        self.invalid_utf8 |= "\ufffd" in tail
        self._buffer += tail
        # EOF without an empty-line dispatch is incomplete, not a valid event.
        self.incomplete = bool(self._buffer or self._data)
        return []
