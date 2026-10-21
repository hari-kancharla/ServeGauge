"""Build real artifacts from a used miniature checkout; private probes must stay out."""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest


@pytest.mark.slow
@pytest.mark.parametrize("ignored_parent", [False, True])
def test_distributions_exclude_private_files_without_losing_public_inputs(tmp_path, ignored_parent):
    root = Path(__file__).resolve().parents[2]
    project = tmp_path / ".cache" / "project" if ignored_parent else tmp_path / "project"
    project.mkdir(parents=True)
    public = {}
    for name in [
        ".gitattributes",
        ".gitignore",
        ".python-version",
        "CITATION.cff",
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "uv.lock",
    ]:
        public[name] = (root / name).read_bytes()
    for name in [
        ".github/workflows/ci.yml",
        "configs/public.json",
        "docs/public.md",
        "profiles/public.yaml",
        "scripts/public.py",
        "tests/public.py",
        "src/servegauge/__init__.py",
        "src/trueweights/__init__.py",
    ]:
        public[name] = b"# public distribution fixture\n"
    for name, data in public.items():
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    private = [
        "AGENTS.md",
        "README_FIRST.md",
        "KICKOFF_PROMPT.md",
        "docs/IMPLEMENT.md",
        "docs/SPEC.md",
        "docs/PLAN.md",
        "docs/STATUS.md",
        "docs/RESEARCH.md",
        "docs/plans/private.md",
        "budget/dev_spend.json",
        ".cache/tokenizers/private.txt",
        ".hypothesis/examples/private.txt",
        ".env",
        "data/raw/private.json",
        "data/ledger/private.json",
        "reports/private.json",
        "bundles/private.json",
        "dist/private.txt",
        "configs/.env.local",
        "configs/.ENV.local",
        "src/servegauge/.env.local",
        "src/servegauge/__pycache__/private.pyc",
        "owner-private.json",
        "budget/untracked-owner.json",
    ]
    marker = b"private-distribution-probe"
    for name in private:
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(marker)
    uv = shutil.which("uv")
    assert uv is not None, "uv is required for the offline distribution acceptance"
    environment = dict(os.environ)
    environment["UV_OFFLINE"] = "1"
    environment["UV_PYTHON_DOWNLOADS"] = "never"
    out = tmp_path / "artifacts"
    process = subprocess.run(
        [uv, "build", "--offline", "--out-dir", str(out)],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert process.returncode == 0, process.stderr
    with tarfile.open(next(out.glob("*.tar.gz"))) as archive:
        prefix = "servegauge-0.1.0.dev0/"
        members = {member.name.removeprefix(prefix): member for member in archive.getmembers()}
        assert set(members) == set(public) | {"PKG-INFO"}
        for name, expected in public.items():
            stream = archive.extractfile(members[name])
            assert stream is not None and stream.read() == expected
        for member in members.values():
            stream = archive.extractfile(member)
            assert stream is not None and marker not in stream.read()
    with zipfile.ZipFile(next(out.glob("*.whl"))) as archive:
        assert {
            name for name in archive.namelist() if name.startswith(("servegauge/", "trueweights/"))
        } == {
            "servegauge/__init__.py",
            "trueweights/__init__.py",
        }
        assert all(marker not in archive.read(name) for name in archive.namelist())
    assert all((project / name).read_bytes() == marker for name in private)
