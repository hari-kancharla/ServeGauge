"""Local preview of a completed inventory; no directory listings or request logs."""

from __future__ import annotations

import hashlib
import mimetypes
import re
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from servegauge.report.site.builder import verify_site


def site_file(out: Path, url: str, inventory: dict[str, str]) -> Path | None:
    """Resolve only exact manifest paths, with no normalization of dot segments."""
    try:
        if out.is_symlink():
            return None
        parts = urlsplit(url)
        path = unquote(parts.path, errors="strict")
        if parts.scheme or parts.netloc or not path.startswith("/"):
            return None
        name = "index.html" if path == "/" else path[1:]
        if name not in inventory or any(
            not re.fullmatch(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*", part)
            for part in name.split("/")
        ):
            return None
        target = out
        for part in name.split("/"):
            target /= part
            if target.is_symlink():
                return None
        return target if target.is_file() else None
    except (OSError, ValueError, UnicodeError):
        return None


def serve_site(out: Path, port: int, *, on_ready: Callable[[int], None]) -> None:
    """Bind loopback only; stop with Ctrl-C. This is not production hosting."""
    manifest = verify_site(out)
    inventory = dict(manifest["files"])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _respond(self, *, head: bool = False) -> None:
            target = site_file(out, self.path, inventory)
            if target is None:
                self.send_error(404, "Not in the verified site inventory")
                return
            try:
                payload = target.read_bytes()
            except OSError:
                self.send_error(404, "Site file unavailable")
                return
            name = target.relative_to(out).as_posix()
            if hashlib.sha256(payload).hexdigest() != inventory[name]:
                self.send_error(409, "Site changed; rebuild into a fresh directory")
                return
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(name)[0] or "text/plain")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            if not head:
                self.wfile.write(payload)

        def do_GET(self) -> None:
            self._respond()

        def do_HEAD(self) -> None:
            self._respond(head=True)

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        on_ready(server.server_port)
        server.serve_forever()
