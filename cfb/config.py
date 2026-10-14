"""Project configuration: seasons, paths, API settings, thresholds."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

API_KEY_ENV_VAR = "CFB_APIKEY"

# CFBD play-by-play starts in 2001; nothing earlier is plannable.
EARLIEST_SEASON = 2001


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def clean(self) -> Path:
        return self.data / "clean"

    @property
    def manifest(self) -> Path:
        return self.data / "manifest.parquet"

    @property
    def manifest_journal(self) -> Path:
        return self.data / "manifest_journal.jsonl"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def env_file(self) -> Path:
        return self.root / ".env"


@dataclass(frozen=True)
class ApiSettings:
    base_url: str = "https://api.collegefootballdata.com"
    timeout_s: float = 60.0
    min_interval_s: float = 0.5  # polite throttle between calls
    max_attempts: int = 6  # every attempt counts against the call budget
    backoff_min_s: float = 2.0
    backoff_max_s: float = 60.0
    budget_buffer: int = 250  # abort if planned > remaining - buffer


@dataclass(frozen=True)
class IngestSettings:
    # Season range chosen by the project owner (2026-10-13); --seasons overrides it.
    first_season: int = 2014
    last_season: int = 2026
    current_season: int = 2026  # the only season re-pullable with --refresh
    include_fcs: bool = False  # full FCS ingestion (FCS-only games)
    include_weather: bool = False  # /games/weather (Patreon feature)
    plays_stats_cap: int = 2000  # /plays/stats row cap; a full page may be truncated
    recruit_lookback: int = 5  # recruiting classes pulled before the first season

    # Dry-run estimates, used only until /calendar and /games are on disk.
    est_regular_weeks: int = 16
    est_postseason_weeks: int = 1
    est_fbs_games_regular: int = 840  # games with >= 1 FBS team
    est_fbs_games_postseason: int = 42
    est_fcs_only_games_regular: int = 600
    est_fcs_only_games_postseason: int = 24


@dataclass(frozen=True)
class Thresholds:
    # Fuzzy transfer matches at or above this confidence are accepted; below it they go to
    # reports/transfer_review.csv. Exact unique name matches are always accepted. Set by the
    # project owner 2026-10-13 (None would accept exact matches only).
    transfer_match_threshold: float | None = 85.0
    # Garbage time = |pre-snap margin| > threshold in that quarter (CFBD's definition). Confirmed by
    # the project owner 2026-10-13. Stored as a flag; no play is dropped for it.
    garbage_time: dict[int, int] | None = field(default_factory=lambda: {2: 38, 3: 28, 4: 22})
    # Enriched-endpoint statuses treated as unreliable (Phase 1a probe). Enriched data is 2025+ only.
    unreliable_parse_statuses: tuple[str, ...] = ("invalid",)
    unreliable_attribution_statuses: tuple[str, ...] = (
        "unmatched", "ambiguous", "conflict", "unlinked", "multi_carrier")


@dataclass(frozen=True)
class ModelSettings:
    # 2014 has no earlier season of player data, so it is fitted but never evaluated
    # (project owner, 2026-10-13). Backtests and reports start at eval_first_season.
    burn_in_seasons: tuple[int, ...] = (2014,)
    eval_first_season: int = 2015


@dataclass(frozen=True)
class Settings:
    paths: Paths = field(default_factory=Paths)
    api: ApiSettings = field(default_factory=ApiSettings)
    ingest: IngestSettings = field(default_factory=IngestSettings)
    thresholds: Thresholds = field(default_factory=Thresholds)
    model: ModelSettings = field(default_factory=ModelSettings)

    @property
    def seasons(self) -> tuple[int, ...]:
        return tuple(range(self.ingest.first_season, self.ingest.last_season + 1))


def parse_seasons(spec: str, current_season: int = IngestSettings.current_season) -> tuple[int, ...]:
    """Parse ``"2024"``, ``"2021-2025"`` or ``"2019,2021-2023"`` into sorted unique seasons."""
    seasons: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        try:
            start, end = int(lo), int(hi) if sep else int(lo)
        except ValueError:
            raise ValueError(f"bad season spec {part!r}; use e.g. 2024 or 2021-2025") from None
        if start > end:
            raise ValueError(f"bad season range {part!r}: start after end")
        seasons.update(range(start, end + 1))
    if not seasons:
        raise ValueError("no seasons given")
    out = tuple(sorted(seasons))
    if out[0] < EARLIEST_SEASON or out[-1] > current_season:
        raise ValueError(f"seasons must be within {EARLIEST_SEASON}-{current_season}, got {spec!r}")
    return out


def load_api_key(paths: Paths | None = None) -> str:
    """Read the CFBD key from the environment or ``.env``. Never log the return value."""
    from dotenv import load_dotenv

    load_dotenv((paths or Paths()).env_file, override=False)
    key = os.environ.get(API_KEY_ENV_VAR, "").strip()
    if not key:
        raise RuntimeError(f"{API_KEY_ENV_VAR} is not set; copy .env.example to .env and add your key")
    return key
