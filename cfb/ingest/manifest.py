"""Request manifest: one row per API request, including how many HTTP calls it took.

Rows are appended to a JSONL journal as requests finish, so a crash loses nothing.
``compact()`` folds the journal into ``data/manifest.parquet``; only the ingesting process
calls it. ``read()`` never modifies files, so it is safe while an ingest is running.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from cfb.config import Paths

SCHEMA = {
    "run_id": pl.String,
    "endpoint": pl.String,
    "path": pl.String,
    "params": pl.String,
    "requested_at": pl.String,  # ISO-8601 UTC
    "http_status": pl.Int32,
    "attempts": pl.Int32,
    "row_count": pl.Int64,
    "bytes": pl.Int64,
    "file_path": pl.String,
    "truncated": pl.Boolean,
    "elapsed_ms": pl.Int64,
    "error": pl.String,
}


@dataclass(frozen=True)
class ManifestRow:
    run_id: str
    endpoint: str
    path: str
    params: str  # JSON, sorted keys
    requested_at: str
    http_status: int | None
    attempts: int
    row_count: int | None
    bytes: int | None
    file_path: str | None
    truncated: bool
    elapsed_ms: int | None
    error: str | None = None


class Manifest:
    def __init__(self, paths: Paths) -> None:
        self.journal: Path = paths.manifest_journal
        self.parquet: Path = paths.manifest

    def append(self, row: ManifestRow) -> None:
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        with self.journal.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(row)) + "\n")
            fh.flush()

    def _journal_rows(self) -> list[dict]:
        if not self.journal.is_file():
            return []
        rows = []
        for line in self.journal.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # blank, or a line another process is still writing
        return rows

    def compact(self) -> int:
        """Fold journal rows into the parquet file. Returns the number of rows added. Writer only."""
        if not self.journal.is_file():
            return 0
        rows = self._journal_rows()
        if rows:
            new = pl.DataFrame(rows, schema=SCHEMA, orient="row")
            frames = [pl.read_parquet(self.parquet), new] if self.parquet.is_file() else [new]
            tmp = self.parquet.with_name(self.parquet.name + ".tmp")
            pl.concat(frames, how="vertical").write_parquet(tmp)
            os.replace(tmp, self.parquet)
        self.journal.unlink()
        return len(rows)

    def read(self) -> pl.DataFrame:
        """Compacted rows plus any pending journal rows. Never modifies files."""
        frames = [pl.read_parquet(self.parquet)] if self.parquet.is_file() else []
        pending = self._journal_rows()
        if pending:
            frames.append(pl.DataFrame(pending, schema=SCHEMA, orient="row"))
        return pl.concat(frames, how="vertical") if frames else pl.DataFrame(schema=SCHEMA)
