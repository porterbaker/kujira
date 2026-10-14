"""Shared helpers for the clean layer: raw readers, typed frames, names, parquet output."""

from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from cfb.config import Settings
from cfb.ingest import raw_store


def iter_raw(settings: Settings, slug: str) -> Iterator[tuple[dict[str, str], Any]]:
    """Yield (partition, payload) for every landed file of an endpoint, in path order."""
    root = settings.paths.raw / slug
    for path in sorted(root.glob(f"**/{raw_store.RESPONSE_FILE}")):
        part = dict(seg.split("=", 1) for seg in path.parent.relative_to(root).parts)
        yield part, raw_store.read_json(path)


def frame(rows: list[dict], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=schema, orient="row") if rows else pl.DataFrame(schema=schema)


def to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def to_float(value: Any) -> float | None:
    if value is None or value == "" or value == "--":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def normalize_name(name: str | None) -> str:
    """Lowercase ASCII, no punctuation or generational suffixes: "D'Andre Swift Jr." -> "dandre swift"."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"['’`.]", "", s)
    s = re.sub(r"[^a-z ]+", " ", s)
    s = _SUFFIX.sub(" ", s)
    return " ".join(s.split())


# Coarse position groups used for matching and for coverage reporting.
POSITION_GROUPS = {
    "QB": "QB", "RB": "RB", "FB": "RB", "TB": "RB", "WR": "WR", "TE": "TE",
    "OL": "OL", "OT": "OL", "OG": "OL", "C": "OL", "IOL": "OL", "T": "OL", "G": "OL",
    "DL": "DL", "DT": "DL", "DE": "DL", "NT": "DL", "EDGE": "DL",
    "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
    "DB": "DB", "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "SAF": "DB",
    "K": "ST", "P": "ST", "PK": "ST", "LS": "ST", "ATH": "ATH",
}


def position_group(position: str | None) -> str | None:
    if not position:
        return None
    return POSITION_GROUPS.get(position.strip().upper(), "OTHER")


def write_table(df: pl.DataFrame, settings: Settings, name: str) -> Path:
    path = settings.paths.clean / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)
    return path


def parts_dir(settings: Settings, name: str) -> Path:
    return settings.paths.clean / "_parts" / name


def write_part(df: pl.DataFrame, settings: Settings, name: str, season: int) -> None:
    d = parts_dir(settings, name)
    d.mkdir(parents=True, exist_ok=True)
    df.write_parquet(d / f"season={season}.parquet")


def combine_parts(settings: Settings, name: str) -> Path:
    """Stream per-season parts into one data/clean/<name>.parquet without loading it all."""
    d = parts_dir(settings, name)
    path = settings.paths.clean / f"{name}.parquet"
    tmp = path.with_name(path.name + ".tmp")
    pl.scan_parquet(sorted(d.glob("season=*.parquet"))).sink_parquet(tmp)
    os.replace(tmp, path)
    return path


def read_table(settings: Settings, name: str) -> pl.DataFrame:
    return pl.read_parquet(settings.paths.clean / f"{name}.parquet")


def scan_table(settings: Settings, name: str) -> pl.LazyFrame:
    return pl.scan_parquet(settings.paths.clean / f"{name}.parquet")
