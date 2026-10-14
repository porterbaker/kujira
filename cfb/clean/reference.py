"""Reference tables: venues, teams, games, play and stat type lookups."""

from __future__ import annotations

import polars as pl

from cfb.clean.common import frame, iter_raw, parse_ts, to_float, to_int
from cfb.config import Settings

VENUE_SCHEMA = {
    "venue_id": pl.Int64, "name": pl.String, "city": pl.String, "state": pl.String, "zip": pl.String,
    "country_code": pl.String, "timezone": pl.String, "latitude": pl.Float64, "longitude": pl.Float64,
    "elevation_m": pl.Float64, "capacity": pl.Int64, "construction_year": pl.Int64, "grass": pl.Boolean,
    "dome": pl.Boolean,
}


def _venue_row(v: dict) -> dict:
    return {
        "venue_id": to_int(v.get("id")), "name": v.get("name"), "city": v.get("city"), "state": v.get("state"),
        "zip": v.get("zip"), "country_code": v.get("countryCode"), "timezone": v.get("timezone"),
        "latitude": to_float(v.get("latitude")), "longitude": to_float(v.get("longitude")),
        "elevation_m": to_float(v.get("elevation")), "capacity": to_int(v.get("capacity")),
        "construction_year": to_int(v.get("constructionYear")), "grass": v.get("grass"), "dome": v.get("dome"),
    }


def build_venues(settings: Settings) -> pl.DataFrame:
    rows = [_venue_row(v) for _, payload in iter_raw(settings, "venues") for v in payload]
    return frame(rows, VENUE_SCHEMA).unique("venue_id", keep="first").sort("venue_id")


TEAM_SCHEMA = {
    "team_id": pl.Int64, "season": pl.Int16, "school": pl.String, "mascot": pl.String,
    "abbreviation": pl.String, "conference": pl.String, "division": pl.String, "classification": pl.String,
    "alternate_names": pl.List(pl.String), "home_venue_id": pl.Int64, "home_venue": pl.String,
    "home_timezone": pl.String, "home_latitude": pl.Float64, "home_longitude": pl.Float64,
    "home_elevation_m": pl.Float64, "home_dome": pl.Boolean, "home_grass": pl.Boolean,
}


def build_teams(settings: Settings) -> pl.DataFrame:
    rows = []
    for part, payload in iter_raw(settings, "teams"):
        season = int(part["season"])
        for t in payload:
            loc = t.get("location") or {}
            rows.append({
                "team_id": to_int(t["id"]), "season": season, "school": t.get("school"), "mascot": t.get("mascot"),
                "abbreviation": t.get("abbreviation"), "conference": t.get("conference"),
                "division": t.get("division"), "classification": t.get("classification"),
                "alternate_names": t.get("alternateNames") or [], "home_venue_id": to_int(loc.get("id")),
                "home_venue": loc.get("name"), "home_timezone": loc.get("timezone"),
                "home_latitude": to_float(loc.get("latitude")), "home_longitude": to_float(loc.get("longitude")),
                "home_elevation_m": to_float(loc.get("elevation")), "home_dome": loc.get("dome"),
                "home_grass": loc.get("grass"),
            })
    return frame(rows, TEAM_SCHEMA).sort("season", "team_id")


GAME_SCHEMA = {
    "game_id": pl.Int64, "season": pl.Int16, "week": pl.Int8, "season_type": pl.String,
    "start_date": pl.Datetime("us", "UTC"), "start_time_tbd": pl.Boolean, "neutral_site": pl.Boolean,
    "conference_game": pl.Boolean, "venue_id": pl.Int64, "venue": pl.String,
    "home_id": pl.Int64, "home_team": pl.String, "home_conference": pl.String, "home_classification": pl.String,
    "home_pregame_elo": pl.Int32,
    "away_id": pl.Int64, "away_team": pl.String, "away_conference": pl.String, "away_classification": pl.String,
    "away_pregame_elo": pl.Int32,
    # Postgame fields (cfb.leakage.GAME_POSTGAME_FIELDS): never features for this game.
    "completed": pl.Boolean, "home_points": pl.Int16, "away_points": pl.Int16, "home_postgame_elo": pl.Int32,
    "away_postgame_elo": pl.Int32, "home_postgame_wp": pl.Float64, "away_postgame_wp": pl.Float64,
    "attendance": pl.Int64, "excitement_index": pl.Float64, "notes": pl.String,
    "is_fbs_game": pl.Boolean,
}


def build_games(settings: Settings) -> pl.DataFrame:
    rows = []
    for _, payload in iter_raw(settings, "games"):
        for g in payload:
            rows.append({
                "game_id": g["id"], "season": g["season"], "week": g["week"], "season_type": g.get("seasonType"),
                "start_date": parse_ts(g.get("startDate")), "start_time_tbd": g.get("startTimeTBD"),
                "neutral_site": g.get("neutralSite"), "conference_game": g.get("conferenceGame"),
                "venue_id": to_int(g.get("venueId")), "venue": g.get("venue"),
                "home_id": g.get("homeId"), "home_team": g.get("homeTeam"), "home_conference": g.get("homeConference"),
                "home_classification": g.get("homeClassification"), "home_pregame_elo": g.get("homePregameElo"),
                "away_id": g.get("awayId"), "away_team": g.get("awayTeam"), "away_conference": g.get("awayConference"),
                "away_classification": g.get("awayClassification"), "away_pregame_elo": g.get("awayPregameElo"),
                "completed": g.get("completed"), "home_points": g.get("homePoints"), "away_points": g.get("awayPoints"),
                "home_postgame_elo": g.get("homePostgameElo"), "away_postgame_elo": g.get("awayPostgameElo"),
                "home_postgame_wp": to_float(g.get("homePostgameWinProbability")),
                "away_postgame_wp": to_float(g.get("awayPostgameWinProbability")),
                "attendance": to_int(g.get("attendance")), "excitement_index": to_float(g.get("excitementIndex")),
                "notes": g.get("notes"),
                "is_fbs_game": "fbs" in (g.get("homeClassification"), g.get("awayClassification")),
            })
    return frame(rows, GAME_SCHEMA).unique("game_id", keep="last").sort("start_date", "game_id")


def build_play_types(settings: Settings) -> pl.DataFrame:
    rows = [{"play_type_id": t["id"], "play_type": t["text"], "abbreviation": t.get("abbreviation")}
            for _, payload in iter_raw(settings, "plays_types") for t in payload]
    return frame(rows, {"play_type_id": pl.Int16, "play_type": pl.String, "abbreviation": pl.String})


def build_stat_types(settings: Settings) -> pl.DataFrame:
    rows = [{"stat_type_id": t["id"], "stat_type": t["name"]}
            for _, payload in iter_raw(settings, "plays_stats_types") for t in payload]
    return frame(rows, {"stat_type_id": pl.Int16, "stat_type": pl.String})


class TeamIndex:
    """Exact team-name lookup: (season, name) from school and alternate names, falling back to a
    name that maps to one team id across all seasons (teams absent from a season's /teams, such
    as UConn and Old Dominion in 2020). Ambiguous names resolve to None."""

    def __init__(self, teams: pl.DataFrame) -> None:
        by_season: dict[tuple[int, str], set[int]] = {}
        any_season: dict[str, set[int]] = {}
        for season, team_id, school, alternates in teams.select(
                "season", "team_id", "school", "alternate_names").iter_rows():
            for name in {school, *(alternates or [])}:
                if name:
                    by_season.setdefault((season, name), set()).add(team_id)
                    any_season.setdefault(name, set()).add(team_id)
        self._season = {k: next(iter(v)) for k, v in by_season.items() if len(v) == 1}
        self._any = {k: next(iter(v)) for k, v in any_season.items() if len(v) == 1}

    def get(self, season: int | None, name: str | None) -> int | None:
        if not name:
            return None
        return self._season.get((season, name)) or self._any.get(name)


def team_name_index(teams: pl.DataFrame) -> TeamIndex:
    return TeamIndex(teams)
