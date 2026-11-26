"""Recorded runtime and conservative static generator identity; no admission."""

from __future__ import annotations

import _random
import ast
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import os
import platform
import random
import sys
import sysconfig
from pathlib import Path
from typing import Any

from servegauge.core.hashing import content_hash
from servegauge.observatory.provenance import implementation_hash

_IMPORTED_SOURCE_HASH = implementation_hash()

GENERATOR_ROOTS = (
    "servegauge.mock.server",
    "servegauge.observatory.tick",
    "servegauge.runner.check",
    "servegauge.runner.execute",
    "servegauge.runner.scheduler",
    "servegauge.suites.registry",
)
RUNTIME_DEPENDENCIES = (
    "servegauge",
    "httpx",
    "pydantic",
    "fastapi",
    "starlette",
    "numpy",
    "scipy",
    "PyYAML",
    "tiktoken",
    "tokenizers",
    "openai-harmony",
)


def _sha(path: Path | None) -> str | None:
    if path is None or any(p.lower() == "starter files" for p in path.parts):
        return None
    if path.is_symlink() or not path.is_file():
        return None
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _local(value: str) -> bool:
    return value == "servegauge" or value.startswith("servegauge.")


def _imports(
    tree: ast.AST, module: str, is_package: bool, known: set[str]
) -> tuple[set[str], list[str]]:
    names: set[str] = set()
    unresolved = []
    import_aliases = {"__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "importlib":
            import_aliases.update(
                a.asname or a.name for a in node.names if a.name == "import_module"
            )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names if _local(a.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parent = module if is_package else module.rpartition(".")[0]
                try:
                    base = importlib.util.resolve_name(
                        "." * node.level + (node.module or ""), parent
                    )
                except (ImportError, ValueError):
                    unresolved.append(f"{module}:{node.lineno}:relative_import")
                    continue
            else:
                base = node.module or ""
            if _local(base):
                names.add(base)
                names.update(
                    base + "." + a.name for a in node.names if base + "." + a.name in known
                )
        elif isinstance(node, ast.Call) and (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            or isinstance(node.func, ast.Name)
            and node.func.id in import_aliases
        ):
            if (
                not node.args
                or not isinstance(node.args[0], ast.Constant)
                or not isinstance(node.args[0].value, str)
            ):
                unresolved.append(f"{module}:{node.lineno}:dynamic_import")
                continue
            target = node.args[0].value
            if target.startswith("."):
                if (
                    len(node.args) < 2
                    or not isinstance(node.args[1], ast.Constant)
                    or not isinstance(node.args[1].value, str)
                ):
                    unresolved.append(f"{module}:{node.lineno}:dynamic_relative_import")
                    continue
                try:
                    target = importlib.util.resolve_name(target, node.args[1].value)
                except (ImportError, ValueError):
                    unresolved.append(f"{module}:{node.lineno}:dynamic_relative_import")
                    continue
            if _local(target):
                names.add(target)
    return names, unresolved


def generator_manifest(
    package: Path | None = None, *, roots: tuple[str, ...] = GENERATOR_ROOTS
) -> dict[str, Any]:
    base = package if package is not None else Path(__file__).resolve().parents[1]
    if (
        any(p.lower() == "starter files" for p in base.parts)
        or base.is_symlink()
        or not (base / "__init__.py").is_file()
    ):
        raise ValueError("generator_package_inventory_required")
    files = sorted(base.rglob("*.py"))
    if any(p.is_symlink() for p in files):
        raise ValueError("generator_package_symlink")
    modules: dict[str, Path] = {}
    for file in files:
        parts = list(file.relative_to(base).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        name = ".".join(["servegauge", *parts])
        if name in modules:
            raise ValueError("ambiguous_generator_module")
        modules[name] = file
    if not roots or any(name not in modules for name in roots):
        raise ValueError("generator_root_module_missing")
    pending = set(roots)
    visited = set()
    pins = {}
    unresolved = []
    while pending:
        name = min(pending)
        pending.remove(name)
        if name in visited:
            continue
        visited.add(name)
        path = modules.get(name)
        if path is None:
            unresolved.append(name + ":module_missing")
            continue
        for index in range(1, len(name.split("."))):
            parent = ".".join(name.split(".")[:index])
            if parent in modules and parent not in visited:
                pending.add(parent)
        try:
            source = path.read_bytes()
            tree = ast.parse(source)
        except (OSError, SyntaxError, ValueError):
            raise ValueError("generator_module_unreadable") from None
        pins["src/servegauge/" + path.relative_to(base).as_posix()] = hashlib.sha256(
            source
        ).hexdigest()
        names, unknown = _imports(tree, name, path.name == "__init__.py", set(modules))
        pending.update(names - visited)
        unresolved.extend(unknown)
    value = {
        "version": "monitor-generator-v1",
        "roots": sorted(set(roots)),
        "pins": dict(sorted(pins.items())),
        "unresolved": sorted(set(unresolved)),
        "complete": not unresolved,
    }
    return {**value, "generator_hash": content_hash(value)}


class _DlInfo(ctypes.Structure):
    dli_fname: bytes | None
    dli_fbase: int | None
    dli_sname: bytes | None
    dli_saddr: int | None
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


def _linux_python_library() -> Path | None:
    """Find the loaded CPython object through its dynamically resolved API symbol."""
    if sys.platform != "linux":
        return None
    try:
        address = ctypes.cast(ctypes.pythonapi.Py_GetVersion, ctypes.c_void_p).value
        if type(address) is not int or not 0 < address < 1 << (ctypes.sizeof(ctypes.c_void_p) * 8):
            return None
        try:
            library = ctypes.CDLL(None)
            lookup = library.dladdr
        except AttributeError:
            library = ctypes.CDLL("libdl.so.2")
            lookup = library.dladdr
        lookup.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
        lookup.restype = ctypes.c_int
        info = _DlInfo()
        if lookup(ctypes.c_void_p(address), ctypes.byref(info)) == 0:
            return None
        filename = info.dli_fname
        if not filename:
            return None
        path = Path(os.fsdecode(filename))
        if not path.is_absolute() or any(p.lower() == "starter files" for p in path.parts):
            return None
        # A statically linked interpreter reports its own (often venv-symlinked) path.
        # Resolve to the real file that was loaded, as the executable hash does.
        return path.resolve()
    except (OSError, AttributeError, ValueError, TypeError, ctypes.ArgumentError):
        return None


def _windows_python_library() -> Path | None:
    """Resolve the loaded CPython DLL; absent or truncated evidence stays unknown."""
    if sys.platform != "win32":
        return None
    handle = getattr(sys, "dllhandle", None)
    if type(handle) is not int or not 0 < handle < 1 << (ctypes.sizeof(ctypes.c_void_p) * 8):
        return None
    try:
        library = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        filename = library.GetModuleFileNameW
        filename.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_wchar), ctypes.c_uint32]
        filename.restype = ctypes.c_uint32
        buffer = ctypes.create_unicode_buffer(32768)
        length = filename(ctypes.c_void_p(handle), buffer, len(buffer))
        if not 0 < length < len(buffer):
            return None
        path = Path(buffer.value)
        return path if path.is_absolute() else None
    except (OSError, ValueError, AttributeError):
        return None


def runtime_envelope(*, lock_path: Path = Path("uv.lock")) -> dict[str, Any]:
    """Record actual local inputs; this alone never qualifies their behavior."""
    dependencies: dict[str, str | None] = {}
    for name in RUNTIME_DEPENDENCIES:
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    random_path = getattr(random, "__file__", None)
    extension_path = getattr(_random, "__file__", None)
    origin = getattr(_random.__spec__, "origin", None)
    executable = Path(sys.executable)
    if any(p.lower() == "starter files" for p in executable.parts):
        raise ValueError("runtime_executable_unavailable")
    executable_hash = _sha(executable.resolve())
    if sys.platform == "win32":
        shared = 1
        library_hash = _sha(_windows_python_library())
    else:
        shared = sysconfig.get_config_var("Py_ENABLE_SHARED")
        if sys.platform == "linux":
            library_hash = _sha(_linux_python_library()) if shared == 1 else None
        else:
            library_dir = sysconfig.get_config_var("LIBDIR")
            library_name = sysconfig.get_config_var("LDLIBRARY")
            library_hash = (
                _sha(Path(library_dir) / library_name)
                if isinstance(library_dir, str) and isinstance(library_name, str)
                else None
            )
    rng = {
        "origin": "built-in"
        if origin == "built-in"
        else "extension"
        if extension_path
        else "unknown",
        "interpreter_sha256": executable_hash,
        "shared_library_required": shared == 1,
        "shared_library_sha256": library_hash,
        "extension_sha256": _sha(Path(extension_path) if extension_path else None),
    }
    rng_complete = bool(
        executable_hash
        and shared in {0, 1}
        and (shared == 0 or library_hash)
        and (origin == "built-in" or rng["extension_sha256"])
    )
    value: dict[str, Any] = {
        "version": "monitor-runtime-v1",
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "build": list(platform.python_build()),
            "compiler": platform.python_compiler(),
        },
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "dependencies": dependencies,
        "lock_sha256": _sha(lock_path),
        "stdlib_random_sha256": _sha(Path(random_path) if random_path else None),
        "rng_implementation": rng,
        "rng_implementation_sha256": content_hash(rng) if rng_complete else None,
    }
    value["complete"] = bool(
        value["lock_sha256"]
        and value["stdlib_random_sha256"]
        and value["rng_implementation_sha256"]
        and all(dependencies.values())
    )
    return {**value, "runtime_hash": content_hash(value)}


def producer_identity() -> dict[str, Any]:
    """Record code/runtime separately; changed-after-import code cannot be admitted."""
    source = implementation_hash()
    value = {
        "source_hash": source,
        "loaded_source_hash": _IMPORTED_SOURCE_HASH,
        "runtime": runtime_envelope(),
        "generator": generator_manifest(),
    }
    if implementation_hash() != source:
        raise ValueError("producer_source_changed_during_binding")
    return value
