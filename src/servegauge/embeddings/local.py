"""Content-addressed local SentenceTransformer assets and CPU-only inference.

Only declared, verified data files enter the private model snapshot. No model ID
is handed to a downloader, custom Python modules are refused, and text that would
be truncated cannot become an embedding observation.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import shutil
import tempfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, cast

import numpy as np
from pydantic import Field, field_validator, model_validator

from servegauge.core.errors import SafeError
from servegauge.core.hashing import content_hash
from servegauge.core.models import ValueModel
from servegauge.stats.common import FloatArray


def _relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or str(path) != name
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or ":" in name
        or any(p.lower() == "starter files" for p in path.parts)
        or any(p.endswith((".", " ")) for p in path.parts)
    ):
        raise ValueError("canonical_relative_model_asset_required")
    return path


class LocalEmbeddingSpec(ValueModel):
    version: Literal["local-embedding-v1"] = "local-embedding-v1"
    model_id: str = Field(min_length=1, max_length=256)
    revision: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    dimensions: int = Field(strict=True, ge=1)
    max_tokens: int = Field(strict=True, ge=2)
    normalize_embeddings: bool = Field(strict=True)
    license_file: str
    files: dict[str, str]

    @field_validator("files")
    @classmethod
    def checked_files(cls, files: dict[str, str]) -> dict[str, str]:
        if not files or len({name.casefold() for name in files}) != len(files):
            raise ValueError("distinct_pinned_model_assets_required")
        for name, digest in files.items():
            path = _relative(name)
            if path.suffix not in {
                ".json",
                ".safetensors",
                ".txt",
                ".model",
                ".md",
            } and path.name not in {"LICENSE", "NOTICE"}:
                raise ValueError("model_data_files_only_no_pickle_or_code")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("sha256_model_asset_pins_required")
        return files

    @model_validator(mode="after")
    def complete(self) -> LocalEmbeddingSpec:
        _relative(self.license_file)
        if (
            self.license_file not in self.files
            or "modules.json" not in self.files
            or not any(name.endswith(".safetensors") for name in self.files)
        ):
            raise ValueError("pinned_modules_license_and_safetensors_required")
        return self

    @property
    def artifact_hash(self) -> str:
        return content_hash(self.model_dump(mode="json"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _graph(spec: LocalEmbeddingSpec, root: Path) -> None:
    modules = json.loads((root / "modules.json").read_text(encoding="utf-8"))
    if not isinstance(modules, list) or not modules:
        raise SafeError("known_local_embedding_modules_required")
    kinds = []
    aliases = {
        "sentence_transformers.models.Transformer": "Transformer",
        "sentence_transformers.base.modules.Transformer": "Transformer",
        "sentence_transformers.models.Pooling": "Pooling",
        "sentence_transformers.sentence_transformer.modules.Pooling": "Pooling",
        "sentence_transformers.models.Normalize": "Normalize",
        "sentence_transformers.sentence_transformer.modules.Normalize": "Normalize",
    }
    for index, module in enumerate(modules):
        if (
            not isinstance(module, dict)
            or module.get("idx") != index
            or module.get("type") not in aliases
        ):
            raise SafeError("known_local_embedding_modules_required")
        kind = aliases[module["type"]]
        kinds.append(kind)
        path = module.get("path")
        if not isinstance(path, str):
            raise SafeError("known_local_embedding_module_path_required")
        if path:
            _relative(path)
        config = str(PurePosixPath(path) / "config.json")
        if kind in {"Transformer", "Pooling"} and config not in spec.files:
            raise SafeError("pinned_embedding_module_configuration_required")
    if kinds not in [["Transformer", "Pooling"], ["Transformer", "Pooling", "Normalize"]]:
        raise SafeError("known_local_embedding_modules_required")
    for name in spec.files:
        if name.endswith(".json"):
            value = json.loads((root / name).read_text(encoding="utf-8"))
            if isinstance(value, dict) and any(
                k in value for k in ("auto_map", "custom_pipelines", "peft_type")
            ):
                raise SafeError("custom_embedding_code_or_adapters_not_supported")


def _inventory(root: Path) -> set[str]:
    result = set()

    def visit(directory: Path) -> None:
        for path in directory.iterdir():
            relative = path.relative_to(root).as_posix()
            _relative(relative)
            if path.is_symlink():
                raise SafeError("materialized_model_must_not_contain_links")
            if path.is_dir():
                visit(path)
            elif path.is_file():
                result.add(relative)
            else:
                raise SafeError("materialized_model_data_files_required")

    visit(root)
    return result


def _verify(spec: LocalEmbeddingSpec, root: Path) -> None:
    if _inventory(root) != set(spec.files) or any(
        _sha(root / name) != digest for name, digest in spec.files.items()
    ):
        raise SafeError("pinned_local_model_snapshot_mismatch")
    _graph(spec, root)


def materialize_model(spec: LocalEmbeddingSpec, source: Path, cache: Path) -> Path:
    # Revalidate nested mutable dictionaries as well as newly loaded manifests.
    spec = LocalEmbeddingSpec.model_validate(spec.model_dump(mode="json"))
    if any(p.lower() == "starter files" for p in [*source.parts, *cache.parts]):
        raise SafeError("dedicated_local_model_directory_required")
    destination = cache / spec.artifact_hash
    if destination.exists():
        _verify(spec, destination)
        return destination
    cache.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="model-", dir=cache))
    try:
        for name, digest in spec.files.items():
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with (source / name).open("rb") as first, target.open("wb") as second:
                while block := first.read(1024 * 1024):
                    h.update(block)
                    second.write(block)
            if h.hexdigest() != digest:
                raise SafeError("pinned_local_model_asset_mismatch")
        _verify(spec, temporary)
        try:
            temporary.rename(destination)
        except OSError:
            if not destination.is_dir():
                raise
            _verify(spec, destination)
        return destination
    except (OSError, ValueError, TypeError):
        raise SafeError("local_embedding_assets_unavailable_or_invalid") from None
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


class _Model(Protocol):
    max_seq_length: int | None
    device: Any
    dtype: Any

    def eval(self) -> Any: ...
    def tokenizer(self, texts: list[str], **kwargs: Any) -> Any: ...
    def encode(self, texts: list[str], **kwargs: Any) -> Any: ...


def _sentence_model(path: Path, spec: LocalEmbeddingSpec) -> _Model:
    try:
        package = importlib.import_module("sentence_transformers")
        torch = importlib.import_module("torch")
    except ImportError:
        raise SafeError("optional_embed_runtime_unavailable_install_locked_extra") from None
    torch.set_num_threads(1)
    return cast(
        _Model,
        package.SentenceTransformer(
            str(path.resolve()),
            device="cpu",
            backend="torch",
            local_files_only=True,
            trust_remote_code=False,
            token=False,
            model_kwargs={
                "use_safetensors": True,
                "torch_dtype": torch.float32,
                "attn_implementation": "eager",
            },
        ),
    )


def _runtime() -> dict[str, str]:
    result = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "architecture": platform.machine(),
    }
    for package in ("sentence-transformers", "torch", "transformers", "numpy"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unavailable"
    return result


@dataclass
class LocalEncoder:
    spec: LocalEmbeddingSpec
    _model: _Model
    identity: dict[str, Any]

    @classmethod
    def load(cls, spec: LocalEmbeddingSpec, source: Path, cache: Path) -> LocalEncoder:
        spec = LocalEmbeddingSpec.model_validate(spec.model_dump(mode="json"))
        path = materialize_model(spec, source, cache)
        try:
            model = _sentence_model(path, spec)
            maximum = model.max_seq_length
            if (
                type(maximum) is not int
                or maximum < spec.max_tokens
                or str(model.device).split(":")[0] != "cpu"
                or str(model.dtype) != "torch.float32"
            ):
                raise SafeError("pinned_cpu_embedding_configuration_unavailable")
            model.max_seq_length = spec.max_tokens
            model.eval()
            identity = {
                "artifact_hash": spec.artifact_hash,
                "model_id": spec.model_id,
                "revision": spec.revision,
                "dimensions": spec.dimensions,
                "max_tokens": spec.max_tokens,
                "normalize_embeddings": spec.normalize_embeddings,
                "device": "cpu",
                "dtype": "float32",
                "prompt": "",
                "runtime": _runtime(),
            }
            return cls(spec.model_copy(deep=True), model, identity)
        except (OSError, ValueError, TypeError, AttributeError, RuntimeError):
            raise SafeError("local_embedding_model_cannot_be_loaded") from None

    def encode(self, texts: list[str]) -> FloatArray:
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise SafeError("complete_nonempty_embedding_texts_required")
        try:
            tokenized = self._model.tokenizer(
                texts, add_special_tokens=True, padding=False, truncation=False
            )
            ids = tokenized["input_ids"]
            if (
                not isinstance(ids, list)
                or len(ids) != len(texts)
                or any(
                    not isinstance(row, list) or not row or len(row) > self.spec.max_tokens
                    for row in ids
                )
            ):
                raise SafeError("embedding_input_would_be_truncated_or_is_invalid")
            values = np.asarray(
                self._model.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    device="cpu",
                    precision="float32",
                    convert_to_numpy=True,
                    convert_to_tensor=False,
                    normalize_embeddings=self.spec.normalize_embeddings,
                    prompt="",
                ),
                dtype=np.float64,
            )
            if values.shape != (len(texts), self.spec.dimensions) or not np.all(
                np.isfinite(values)
            ):
                raise SafeError("complete_finite_embedding_vectors_required")
            if self.spec.normalize_embeddings and not np.allclose(
                np.linalg.norm(values, axis=1),
                1,
                rtol=0,
                atol=float(100 * np.finfo(np.float32).eps),
            ):
                raise SafeError("nonzero_normalized_embedding_vectors_required")
            return values
        except (ValueError, TypeError, LookupError, AttributeError, RuntimeError):
            raise SafeError("local_embedding_observation_unavailable") from None
