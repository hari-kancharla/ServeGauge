"""Append independent zstd frames under a process-safe lock; deduplicate identities."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import zstandard as zstd
from filelock import FileLock
from pydantic import ValidationError

from servegauge.core.hashing import canonical_json
from servegauge.core.ids import safe_filename
from servegauge.core.models import Exchange
from servegauge.core.redaction import Redactor


def raw_files(root: Path) -> list[Path]:
    return sorted({*root.rglob("*.jsonl.zst"), *root.rglob("*.jsonl")})


def _chunks(path: Path) -> Iterator[bytes]:
    with path.open("rb") as source:
        if path.suffix != ".zst":
            yield from iter(lambda: source.read(65536), b"")
            return
        decoder = zstd.ZstdDecompressor().decompressobj()
        started = False
        for chunk in iter(lambda: source.read(65536), b""):
            if decoder.eof:
                decoder = zstd.ZstdDecompressor().decompressobj()
            while chunk:
                started = True
                yield decoder.decompress(chunk)
                chunk = decoder.unused_data
                if chunk:
                    decoder = zstd.ZstdDecompressor().decompressobj()
        if started and not decoder.eof:
            raise ValueError("Incomplete compressed frame")


def read_file(path: Path) -> Iterator[Exchange]:
    try:
        pending = b""
        for chunk in _chunks(path):
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                if line.strip():
                    yield Exchange.model_validate_json(line)
        if pending:
            raise ValueError("Incomplete raw line")
    except (ValidationError, ValueError, UnicodeError, zstd.ZstdError):
        raise ValueError("Invalid raw record; rebuild refused to discard evidence") from None


class RawStore:
    def __init__(self, root: Path = Path("data/raw"), *, redactor: Redactor | None = None) -> None:
        self.root = root
        self.redactor = redactor or Redactor.from_environment()
        self._sizes: dict[Path, int] = {}
        self._seen: set[str] = set()

    def exchanges(self, *, run_id: str | None = None) -> Iterator[Exchange]:
        seen: set[str] = set()
        directory = self.root if run_id is None else self.root / safe_filename(run_id)
        for path in raw_files(directory):
            for record in read_file(path):
                if record.exchange_id not in seen:
                    seen.add(record.exchange_id)
                    yield record

    def _refresh_index(self) -> None:
        for path in raw_files(self.root):
            size = path.stat().st_size
            if self._sizes.get(path) != size:
                self._seen.update(e.exchange_id for e in read_file(path))
                self._sizes[path] = size

    def contains(self, exchange_id: str) -> bool:
        self._refresh_index()
        return exchange_id in self._seen

    def append(self, record: Exchange) -> bool:
        clean = Exchange.model_validate(self.redactor.redact(record.model_dump(mode="json")))
        directory = self.root / safe_filename(clean.run_id) / safe_filename(clean.endpoint_id)
        directory.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.root / ".append.lock")):
            self._refresh_index()
            if clean.exchange_id in self._seen:
                return False
            path = directory / f"{safe_filename(clean.suite_id)}.jsonl.zst"
            frame = zstd.ZstdCompressor().compress(
                canonical_json(clean.model_dump(mode="json")) + b"\n"
            )
            with path.open("ab") as target:
                target.write(frame)
                target.flush()
                os.fsync(target.fileno())
            self._seen.add(clean.exchange_id)
            self._sizes[path] = path.stat().st_size
        return True
