"""Errors exposed to users contain only a controlled code, never upstream exception text."""


class SafeError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code
