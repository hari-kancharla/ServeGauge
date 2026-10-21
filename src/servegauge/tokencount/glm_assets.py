"""Explicit public-data setup and offline, checksum-pinned GLM tokenizer assets.

No model weights, remote modules, account credentials or implicit downloads.
Original license notices stay with the private cache.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from functools import lru_cache
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx

from servegauge.core.errors import SafeError

MODEL = "zai-org/GLM-5.3"
REVISION = "ba32f8c19c8738e2c57c71798c601d477a7bf844"
FILES = {
    "tokenizer.json": "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
    "tokenizer_config.json": "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc",
    "chat_template.jinja": "0e2b6268ac54f04d2bc1157e8cfb896cb66f25c0b5e319ea12e56014c3914f24",
    "LICENSE": "96e1622099fc9d6b70c9760f007d99e66d7497eec636b63c60fe208401e9170c",
}
MAX_BYTES = 32 * 1024 * 1024
_CACHE_ERROR = "glm_tokenizer_cache_missing_or_invalid; run setup-cache --profile glm-5.3"
type Identity = tuple[int, int, int, int, int]


def _guard(path: Path) -> None:
    # Lexical refusal precedes every filesystem operation, including symlink checks.
    if ".." in path.parts or any(p.casefold() == "starter files" for p in path.parts):
        raise SafeError("glm_tokenizer_path_refused")
    if any(p.is_symlink() for p in [*reversed(path.absolute().parents), path]):
        raise SafeError("glm_tokenizer_path_refused")


def cache_directory() -> Path:
    path = (
        Path(
            os.environ.get(
                "SERVEGAUGE_TOKENIZER_CACHE",
                os.environ.get("TRUEWEIGHTS_TOKENIZER_CACHE", ".cache/tokenizers"),
            )
        )
        / "hf-glm-5.3"
    )
    _guard(path)
    return path.absolute()


def _identity(path: Path) -> Identity:
    value = path.stat()
    if not stat.S_ISREG(value.st_mode) or value.st_size > MAX_BYTES:
        raise SafeError(_CACHE_ERROR)
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


@lru_cache(maxsize=16)
def _read_checked(path: Path, checksum: str, identity: Identity) -> bytes:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != checksum or _identity(path) != identity:
        raise SafeError(_CACHE_ERROR)
    return raw


def _asset(directory: Path, name: str) -> bytes:
    path = directory / name
    _guard(path)
    try:
        return _read_checked(path, FILES[name], _identity(path))
    except OSError:
        raise SafeError(_CACHE_ERROR) from None


def verified_assets() -> dict[str, bytes]:
    directory = cache_directory()
    return {name: _asset(directory, name) for name in FILES}


def asset_url(name: str) -> str:
    if name not in FILES:
        raise SafeError("glm_tokenizer_unknown_asset")
    return f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"


def _public_url(url: str) -> None:
    parts = urlsplit(url)
    host = parts.hostname or ""
    if (
        parts.scheme != "https"
        or parts.username is not None
        or parts.password is not None
        or parts.port not in {None, 443}
        or parts.fragment
        or not any(
            host == domain or host.endswith("." + domain) for domain in ("huggingface.co", "hf.co")
        )
    ):
        raise SafeError("glm_tokenizer_public_download_refused")


def _download(client: httpx.Client, name: str) -> bytes:
    url = asset_url(name)
    for _ in range(6):
        _public_url(url)
        with client.stream("GET", url) as response:
            if response.is_redirect:
                url = urljoin(url, response.headers.get("location", ""))
                continue
            response.raise_for_status()
            length = response.headers.get("content-length")
            if length is not None and (not length.isdecimal() or int(length) > MAX_BYTES):
                raise SafeError("glm_tokenizer_public_download_refused")
            data = bytearray()
            for block in response.iter_bytes():
                data.extend(block)
                if len(data) > MAX_BYTES:
                    raise SafeError("glm_tokenizer_public_download_refused")
            raw = bytes(data)
            if hashlib.sha256(raw).hexdigest() != FILES[name]:
                raise SafeError("glm_tokenizer_public_checksum_mismatch")
            return raw
    raise SafeError("glm_tokenizer_public_redirect_limit")


def setup_glm_cache(*, source: Path | None = None) -> Path:
    """Explicit bounded public fetch, or verified local copy; neither invokes inference."""
    if source is not None:
        _guard(source)
    directory = cache_directory()
    try:
        verified_assets()
        return directory
    except SafeError:
        pass
    # Validate the complete source, including notices, before creating any output.
    if source is not None:
        payloads = {name: _asset(source, name) for name in FILES}
    else:
        try:
            with httpx.Client(trust_env=False, follow_redirects=False, timeout=60) as client:
                payloads = {name: _download(client, name) for name in FILES}
        except (httpx.HTTPError, OSError, ValueError):
            raise SafeError("glm_tokenizer_public_download_failed") from None
    _guard(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name, raw in payloads.items():
        target = directory / name
        _guard(target)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=directory, prefix=".pending-", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    verified_assets()
    return directory
