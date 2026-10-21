"""Source bindings must survive wheel installation without an empty inventory."""

import pytest

from servegauge.core.hashing import content_hash
from servegauge.observatory.provenance import implementation_hash
from servegauge.validation.common import source_hash


def test_actual_package_matches_repository_hash_without_development_root_dependency(
    tmp_path, monkeypatch
):
    expected = source_hash()
    assert implementation_hash() == expected
    monkeypatch.setattr("servegauge.validation.common.ROOT", tmp_path)
    assert source_hash() == content_hash({})
    assert implementation_hash() == expected


def test_relocated_package_uses_actual_files_and_canonical_source_names(tmp_path):
    package = tmp_path / "site-packages/servegauge"
    (package / "observatory").mkdir(parents=True)
    (package / "__init__.py").write_text('"""Sample package."""\n')
    (package / "observatory/tick.py").write_text("version = 1\n")
    expected = content_hash(
        {
            "src/servegauge/__init__.py": content_hash('"""Sample package."""\n'),
            "src/servegauge/observatory/tick.py": content_hash("version = 1\n"),
        }
    )
    assert implementation_hash(package) == expected
    (package / "observatory/tick.py").write_text("version = 2\n")
    assert implementation_hash(package) != expected


def test_missing_or_empty_source_inventory_refuses(tmp_path):
    with pytest.raises(ValueError, match="package"):
        implementation_hash(tmp_path)
    (tmp_path / "__init__.py").write_text("")
    with pytest.raises(ValueError, match="package"):
        implementation_hash(tmp_path)
