"""Raw landing paths: ``data/raw/<endpoint>/<k=v>/.../response.json.gz``.

Raw is the source of truth and is never edited in place. A partition counts as landed
once its ``response.json.gz`` exists. Phase 1 writes it via temp file + atomic rename.
"""

from __future__ import annotations

import gzip
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

Partition = tuple[tuple[str, str], ...]

RESPONSE_FILE = "response.json.gz"


def season_part(season: int | str) -> tuple[str, str]:
    return ("season", str(season))


def week_parts(season_type: str, week: int) -> Partition:
    return (("type", season_type), ("week", f"{week:02d}"))


def partition_dir(raw_root: Path, slug: str, partition: Partition) -> Path:
    path = raw_root / slug
    for key, value in partition:
        path = path / f"{key}={value}"
    return path


def response_path(raw_root: Path, slug: str, partition: Partition) -> Path:
    return partition_dir(raw_root, slug, partition) / RESPONSE_FILE


def is_landed(raw_root: Path, slug: str, partition: Partition) -> bool:
    return response_path(raw_root, slug, partition).is_file()


def read_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def write_response(path: Path, body: bytes, *, refresh: bool = False, pulled_at: datetime | None = None) -> Path:
    """Land a response body verbatim (gzipped) via temp file + atomic rename.

    An existing file is never overwritten. With ``refresh`` it is first renamed to
    ``response.<pulled_at>.json.gz`` and kept.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not refresh:
            raise FileExistsError(f"{path} already landed; raw is never overwritten")
        stamp = (pulled_at or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
        path.rename(path.with_name(f"response.{stamp}.json.gz"))
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wb") as fh:
        fh.write(body)
    os.replace(tmp, path)
    return path
