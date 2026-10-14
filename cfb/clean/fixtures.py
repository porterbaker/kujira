"""Trim landed raw responses into tests/fixtures/clean/ for an offline end-to-end clean-build test.

Three real games (2014, 2024, 2025) with their plays, credits, enriched rows, box score, and the
teams, rosters, recruits, portal entries, coaches, lines, ratings, and context for those teams.
Each fixture is stored with the raw partition it belongs to, so a test can rebuild a mini raw tree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cfb.clean.common import iter_raw
from cfb.config import Settings
from cfb.ingest import raw_store

GAMES = {2014: (400547743, "05"), 2024: (401628322, "01"), 2025: (401752665, "01")}
TEAMS = {2014: ("Clemson", "North Carolina"), 2024: ("Florida", "Miami"), 2025: ("Florida State", "Alabama")}


def _read(settings: Settings, slug: str, *partition: tuple[str, str]) -> Any:
    return raw_store.read_json(raw_store.response_path(settings.paths.raw, slug, tuple(partition)))


def write_clean_fixtures(settings: Settings, out_dir: Path) -> list[Path]:
    files: list[tuple[str, list[tuple[str, str]], Any]] = []  # (slug, partition, payload)
    all_teams = {t for pair in TEAMS.values() for t in pair}
    athlete_ids: set[int] = set()
    for season, (gid, week) in GAMES.items():
        wk = [("season", str(season)), ("type", "regular"), ("week", week)]
        names = TEAMS[season]
        files.append(("plays", wk, [p for p in _read(settings, "plays", *wk) if p["gameId"] == gid]))
        stats = _read(settings, "plays_stats", ("season", str(season)), ("game", str(gid)))
        files.append(("plays_stats", [("season", str(season)), ("game", str(gid))], stats))
        games = _read(settings, "games", ("season", str(season)), ("type", "regular"))
        files.append(("games", [("season", str(season)), ("type", "regular")],
                      [g for g in games if g["homeTeam"] in names or g["awayTeam"] in names]))
        files.append(("teams", [("season", str(season))],
                      [t for t in _read(settings, "teams", ("season", str(season))) if t["school"] in all_teams]))
        roster = [r for r in _read(settings, "roster", ("season", str(season))) if r["team"] in names]
        files.append(("roster", [("season", str(season))], roster))
        athlete_ids |= {int(r["id"]) for r in roster}
        if season >= 2025:
            for slug in ("passing_plays", "rushing_plays"):
                files.append((slug, wk, [r for r in _read(settings, slug, *wk) if r["gameId"] == gid]))
    # Roster for the season before 2025 so portal entries have an origin roster to match against.
    r24 = [r for r in _read(settings, "roster", ("season", "2024")) if r["team"] in TEAMS[2025]]
    files.append(("roster", [("season", "2024"), ("extra", "origin")], r24))
    t24 = [t for t in _read(settings, "teams", ("season", "2024")) if t["school"] in TEAMS[2025]]
    files.append(("teams", [("season", "2024"), ("extra", "origin")], t24))
    athlete_ids |= {int(r["id"]) for r in r24}

    recruits = []
    for _, payload in iter_raw(settings, "recruiting_players"):
        recruits += [r for r in payload if r.get("athleteId") and int(r["athleteId"]) in athlete_ids]
    files.append(("recruiting_players", [("season", "all"), ("recruit_type", "fixture")], recruits))
    portal = [p for p in _read(settings, "player_portal", ("season", "2025")) if p["origin"] in TEAMS[2025]]
    files.append(("player_portal", [("season", "2025")], portal))
    coaches = []
    for season in (2023, 2024):
        coaches += [c for c in _read(settings, "coaches", ("season", str(season)))
                    if any(s["school"] in TEAMS[2024] for s in c["seasons"])]
    files.append(("coaches", [("season", "2023-2024")], coaches))
    files.append(("lines", [("season", "2024"), ("type", "regular")],
                  [g for g in _read(settings, "lines", ("season", "2024"), ("type", "regular"))
                   if g["homeTeam"] in TEAMS[2024] or g["awayTeam"] in TEAMS[2024]]))
    files.append(("games_players", [("season", "2024"), ("type", "regular"), ("week", "01")],
                  [g for g in _read(settings, "games_players", ("season", "2024"), ("type", "regular"), ("week", "01"))
                   if g["id"] == GAMES[2024][0]]))
    files.append(("ratings_sp", [("season", "2024")],
                  [r for r in _read(settings, "ratings_sp", ("season", "2024"))
                   if r["team"] in TEAMS[2024] or r["team"] == "nationalAverages"]))
    files.append(("talent", [("season", "2024")],
                  [r for r in _read(settings, "talent", ("season", "2024")) if r["team"] in TEAMS[2024]]))
    files.append(("player_returning", [("season", "2024")],
                  [r for r in _read(settings, "player_returning", ("season", "2024")) if r["team"] in TEAMS[2024]]))
    files.append(("draft_picks", [("season", "2025")],
                  [d for d in _read(settings, "draft_picks", ("season", "2025")) if d["collegeTeam"] in TEAMS[2024]]))
    venue_ids = {g.get("venueId") for slug, _, payload in files if slug == "games" for g in payload}
    files.append(("venues", [("season", "all")], [v for v in _read(settings, "venues", ("season", "all"))
                                                   if v["id"] in venue_ids]))
    for slug in ("plays_types", "plays_stats_types"):
        files.append((slug, [("season", "all")], _read(settings, slug, ("season", "all"))))

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, (slug, partition, payload) in enumerate(files):
        path = out_dir / f"{i:02d}_{slug}.json"
        path.write_text(json.dumps({"slug": slug, "partition": partition, "payload": payload}, separators=(",", ":")))
        written.append(path)
    return written


def install_fixtures(fixture_dir: Path, settings: Settings) -> None:
    """Recreate the raw tree described by the fixtures under settings.paths.raw (for tests)."""
    import gzip

    for path in sorted(fixture_dir.glob("*.json")):
        spec = json.loads(path.read_text())
        target = raw_store.response_path(settings.paths.raw, spec["slug"], tuple(map(tuple, spec["partition"])))
        target.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(target, "wt", encoding="utf-8") as fh:
            json.dump(spec["payload"], fh)
