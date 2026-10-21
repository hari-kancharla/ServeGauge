import json
from pathlib import Path

import duckdb
import polars as pl
from tests_support import exchange
from typer.testing import CliRunner

from servegauge.cli import app
from servegauge.storage.raw_store import RawStore
from servegauge.storage.tables import TABLE_NAMES, rebuild


def test_append_dedup_restart_and_rebuild(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    store = RawStore(root)
    assert store.append(exchange())
    assert not store.append(exchange())
    assert RawStore(root).append(exchange(1))
    assert len(list(store.exchanges())) == 2
    counts = rebuild(root, tmp_path / "tables")
    assert counts["exchanges"] == 2
    assert counts["runs"] == 1
    assert counts["endpoints"] == 1
    assert set(counts) == set(TABLE_NAMES)
    with duckdb.connect(str(tmp_path / "tables/catalog.duckdb"), read_only=True) as conn:
        assert conn.sql("select count(*) from exchanges").fetchone() == (2,)
    before = pl.read_parquet(tmp_path / "tables/exchanges.parquet").to_dicts()
    rebuild(root, tmp_path / "tables")
    assert before == pl.read_parquet(tmp_path / "tables/exchanges.parquet").to_dicts()


def test_fixture_rebuild_matches_independent_golden(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/raw_sample")
    result = CliRunner().invoke(app, ["rebuild", "--raw", str(fixture), "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    actual = pl.read_parquet(tmp_path / "exchanges.parquet")
    golden = json.loads(Path("tests/fixtures/golden_tables.json").read_text())
    assert actual.select(list(golden[0])).to_dicts() == golden


def test_raw_store_rejects_corruption_without_silent_loss(tmp_path: Path) -> None:
    import pytest

    (tmp_path / "bad.jsonl").write_text("not-json\n")
    with pytest.raises(ValueError, match="Invalid raw record"):
        list(RawStore(tmp_path).exchanges())


def test_truncated_frame_cannot_disappear_during_rebuild(tmp_path: Path) -> None:
    import pytest

    store = RawStore(tmp_path)
    store.append(exchange())
    path = next(tmp_path.rglob("*.jsonl.zst"))
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="Invalid raw record"):
        list(store.exchanges())


def test_concurrent_store_instances_cannot_duplicate(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    def append(sample: int) -> bool:
        return RawStore(tmp_path).append(exchange(sample % 10))

    with ThreadPoolExecutor(max_workers=4) as pool:
        appended = list(pool.map(append, range(40)))
    assert sum(appended) == 10
    assert len(list(RawStore(tmp_path).exchanges())) == 10
