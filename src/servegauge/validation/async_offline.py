"""Single-thread in-process ASGI validation without event-loop wakeup sockets."""

import asyncio
import selectors


class EmptySafeSelector(selectors.SelectSelector):
    def select(self, timeout: float | None = None) -> list[tuple[selectors.SelectorKey, int]]:
        # Windows rejects select([], [], [], timeout); there is no I/O to await.
        if not self.get_map():
            return []
        return super().select(timeout)


class SocketlessLoop(asyncio.SelectorEventLoop):
    """Validation only. Production transports retain the standard event loop."""

    def __init__(self) -> None:
        super().__init__(selector=EmptySafeSelector())

    def _make_self_pipe(self) -> None:
        pass

    def _close_self_pipe(self) -> None:
        pass

    def _write_to_self(self) -> None:
        pass
