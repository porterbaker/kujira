"""Player box scores (/games/players) in long form: one row per (game, athlete, category, stat).

Paired values such as passing "C/ATT" = "15/22" or kicking "FG" = "1/2" become two rows
("C" and "ATT"; "FG_MADE" and "FG_ATT"). Box scores carry the only per-defender tackle, TFL,
pass-deflection and hurry counts in the data (play-level credits have none).
"""

from __future__ import annotations

import polars as pl

from cfb.clean.common import frame, iter_raw, to_float, to_int
from cfb.config import Settings

BOX_SCHEMA = {
    "game_id": pl.Int64, "season": pl.Int16, "week": pl.Int8, "season_type": pl.String, "team": pl.String,
    "team_id": pl.Int64, "home_away": pl.String, "athlete_id": pl.Int64, "athlete_name": pl.String,
    "category": pl.String, "stat_name": pl.String, "value_raw": pl.String, "value": pl.Float64,
}


def split_stat(stat_name: str, raw: str | None) -> list[tuple[str, float | None]]:
    if raw is not None and "/" in raw:
        made, _, att = raw.partition("/")
        names = stat_name.split("/") if "/" in stat_name else [f"{stat_name}_MADE", f"{stat_name}_ATT"]
        return [(names[0], to_float(made)), (names[1], to_float(att))]
    return [(stat_name, to_float(raw))]


def build_box_season(settings: Settings, season: int, games: pl.DataFrame) -> pl.DataFrame:
    teams_by_game = {gid: {ht: hid, at: aid} for gid, ht, hid, at, aid in
                     games.select("game_id", "home_team", "home_id", "away_team", "away_id").iter_rows()}
    rows = []
    for part, payload in iter_raw(settings, "games_players"):
        if int(part["season"]) != season:
            continue
        for g in payload:
            ids = teams_by_game.get(g["id"], {})
            for t in g.get("teams") or []:
                for cat in t.get("categories") or []:
                    for typ in cat.get("types") or []:
                        for a in typ.get("athletes") or []:
                            for stat_name, value in split_stat(typ["name"], a.get("stat")):
                                rows.append({
                                    "game_id": g["id"], "season": season, "week": int(part["week"]),
                                    "season_type": part["type"], "team": t.get("team"), "team_id": ids.get(t.get("team")),
                                    "home_away": t.get("homeAway"), "athlete_id": to_int(a.get("id")),
                                    "athlete_name": a.get("name"), "category": cat["name"], "stat_name": stat_name,
                                    "value_raw": a.get("stat"), "value": value,
                                })
    return frame(rows, BOX_SCHEMA)
