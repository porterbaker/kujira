from datetime import datetime, timezone

import pytest

from cfb.ingest import raw_store
from cfb.ingest.manifest import Manifest, ManifestRow


def _row(i, status=200):
    return ManifestRow(
        run_id="r1", endpoint="plays", path="/plays", params=f'{{"week": {i}}}',
        requested_at=datetime(2026, 10, 13, tzinfo=timezone.utc).isoformat(),
        http_status=status, attempts=1, row_count=10, bytes=100,
        file_path=f"data/raw/plays/week={i}", truncated=False, elapsed_ms=5,
    )


def test_journal_then_compact(settings):
    m = Manifest(settings.paths)
    m.append(_row(1))
    m.append(_row(2, status=429))
    assert m.journal.is_file() and not m.parquet.is_file()
    assert m.compact() == 2
    assert not m.journal.is_file()
    m.append(_row(3))
    df = m.read()  # includes the pending row without compacting
    assert df.height == 3 and m.journal.is_file()
    assert df["http_status"].to_list() == [200, 429, 200]
    assert df["attempts"].sum() == 3


def test_read_tolerates_a_partially_written_line(settings):
    m = Manifest(settings.paths)
    m.append(_row(1))
    with m.journal.open("a") as fh:
        fh.write('{"run_id": "r1", "endpo')  # another process mid-write
    assert m.read().height == 1


def test_write_response_is_atomic_and_never_overwrites(settings):
    path = raw_store.response_path(settings.paths.raw, "roster", (("season", "2026"),))
    raw_store.write_response(path, b'[{"id": "1"}]')
    assert raw_store.read_json(path) == [{"id": "1"}]
    assert not list(path.parent.glob("*.tmp"))
    with pytest.raises(FileExistsError):
        raw_store.write_response(path, b"[]")


def test_refresh_archives_previous_pull(settings):
    path = raw_store.response_path(settings.paths.raw, "roster", (("season", "2026"),))
    raw_store.write_response(path, b"[1]")
    raw_store.write_response(path, b"[2]", refresh=True, pulled_at=datetime(2026, 10, 13, 12, tzinfo=timezone.utc))
    assert raw_store.read_json(path) == [2]
    archived = path.with_name("response.20261013T120000Z.json.gz")
    assert raw_store.read_json(archived) == [1]
