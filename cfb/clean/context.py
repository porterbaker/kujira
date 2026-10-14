"""Game and team context: lines, coaches, draft, benchmarks, team context (talent, returning production)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import polars as pl

from cfb.clean.common import frame, iter_raw, parse_ts, to_float, to_int
from cfb.clean.reference import team_name_index
from cfb.config import Settings

LINE_SCHEMA = {
    "game_id": pl.Int64, "season": pl.Int16, "week": pl.Int8, "season_type": pl.String,
    "start_date": pl.Datetime("us", "UTC"), "home_id": pl.Int64, "home_team": pl.String, "away_id": pl.Int64,
    "away_team": pl.String, "provider": pl.String, "spread": pl.Float64, "spread_open": pl.Float64,
    "over_under": pl.Float64, "over_under_open": pl.Float64, "home_moneyline": pl.Float64,
    "away_moneyline": pl.Float64, "formatted_spread": pl.String,
    "home_score": pl.Int16, "away_score": pl.Int16,  # postgame
}


def build_lines(settings: Settings) -> pl.DataFrame:
    """One row per (game, provider). Spread is home-perspective: negative = home favored."""
    rows = []
    for _, payload in iter_raw(settings, "lines"):
        for g in payload:
            for line in g.get("lines") or []:
                rows.append({
                    "game_id": g["id"], "season": g["season"], "week": g["week"], "season_type": g.get("seasonType"),
                    "start_date": parse_ts(g.get("startDate")), "home_id": g.get("homeTeamId"),
                    "home_team": g.get("homeTeam"), "away_id": g.get("awayTeamId"), "away_team": g.get("awayTeam"),
                    "provider": line.get("provider"), "spread": to_float(line.get("spread")),
                    "spread_open": to_float(line.get("spreadOpen")), "over_under": to_float(line.get("overUnder")),
                    "over_under_open": to_float(line.get("overUnderOpen")),
                    "home_moneyline": to_float(line.get("homeMoneyline")),
                    "away_moneyline": to_float(line.get("awayMoneyline")),
                    "formatted_spread": line.get("formattedSpread"),
                    "home_score": g.get("homeScore"), "away_score": g.get("awayScore"),
                })
    return frame(rows, LINE_SCHEMA).unique(["game_id", "provider"], keep="last").sort("start_date", "game_id", "provider")


COACH_SCHEMA = {
    "coach_id": pl.Int64, "first_name": pl.String, "last_name": pl.String, "hire_date": pl.Datetime("us", "UTC"),
    "season": pl.Int16, "team_id": pl.Int64, "school": pl.String, "conference": pl.String, "games": pl.Int16,
    "wins": pl.Int16, "losses": pl.Int16, "ties": pl.Int16, "preseason_rank": pl.Int16,
    "postseason_rank": pl.Int16, "srs": pl.Float64, "sp_overall": pl.Float64, "sp_offense": pl.Float64,
    "sp_defense": pl.Float64,
}


def build_coaches(settings: Settings) -> pl.DataFrame:
    rows = []
    for _, payload in iter_raw(settings, "coaches"):
        for c in payload:
            for s in c.get("seasons") or []:
                rows.append({
                    "coach_id": c["id"], "first_name": c.get("firstName"), "last_name": c.get("lastName"),
                    "hire_date": parse_ts(c.get("hireDate")), "season": s.get("year"), "team_id": s.get("teamId"),
                    "school": s.get("school"), "conference": s.get("conference"), "games": s.get("games"),
                    "wins": s.get("wins"), "losses": s.get("losses"), "ties": s.get("ties"),
                    "preseason_rank": s.get("preseasonRank"), "postseason_rank": s.get("postseasonRank"),
                    "srs": to_float(s.get("srs")), "sp_overall": to_float(s.get("spOverall")),
                    "sp_offense": to_float(s.get("spOffense")), "sp_defense": to_float(s.get("spDefense")),
                })
    return frame(rows, COACH_SCHEMA).unique(["coach_id", "season", "team_id"], keep="last").sort("season", "team_id")


def build_team_coaching(coaches: pl.DataFrame) -> pl.DataFrame:
    """Per team-season: the coach who started the season and whether that is a new coach.

    The preseason coach is a coach who also coached the team the season before, if any (a mid-season
    firing then leaves him as the starter); otherwise the earliest hire date (an interim hired in
    October loses to the coach hired in the off-season). ``coaching_change`` uses only that, so it
    is known before the season. ``in_season_change`` is not: it is season-end information and must
    never be a same-season feature.
    """
    c = coaches.drop_nulls("team_id")
    prev = c.select("team_id", (pl.col("season") + 1).alias("season"), pl.col("coach_id").alias("prev_coach")) \
        .group_by("team_id", "season").agg(pl.col("prev_coach").unique())
    cur = c.join(prev, on=["team_id", "season"], how="left").with_columns(
        continuing=pl.col("coach_id").is_in(pl.col("prev_coach").fill_null([])),
        has_prev=pl.col("prev_coach").is_not_null(),
    )
    starter = (cur.sort(["continuing", "hire_date", "games"], descending=[True, False, True], nulls_last=True)
               .group_by("team_id", "season").agg(
                   pl.col("coach_id").first().alias("preseason_coach_id"),
                   pl.col("continuing").first().alias("_continuing"),
                   pl.col("has_prev").first().alias("_has_prev"),
                   pl.col("coach_id").n_unique().alias("n_head_coaches")))
    return starter.with_columns(
        coaching_change=pl.when(pl.col("_has_prev")).then(~pl.col("_continuing")),
        in_season_change=pl.col("n_head_coaches") > 1,
    ).drop("_continuing", "_has_prev").sort("season", "team_id")


DRAFT_SCHEMA = {
    "year": pl.Int16, "overall": pl.Int16, "round": pl.Int8, "pick": pl.Int16, "college_athlete_id": pl.Int64,
    "nfl_athlete_id": pl.Int64, "college_id": pl.Int64, "college_team": pl.String, "college_conference": pl.String,
    "nfl_team": pl.String, "name": pl.String, "position": pl.String, "height": pl.Float64, "weight": pl.Float64,
    "pre_draft_ranking": pl.Int32, "pre_draft_position_ranking": pl.Int32, "pre_draft_grade": pl.Int32,
}


def build_draft(settings: Settings) -> pl.DataFrame:
    rows = [{
        "year": d["year"], "overall": d["overall"], "round": d["round"], "pick": d["pick"],
        "college_athlete_id": to_int(d.get("collegeAthleteId")), "nfl_athlete_id": to_int(d.get("nflAthleteId")),
        "college_id": to_int(d.get("collegeId")), "college_team": d.get("collegeTeam"),
        "college_conference": d.get("collegeConference"), "nfl_team": d.get("nflTeam"), "name": d.get("name"),
        "position": d.get("position"), "height": to_float(d.get("height")), "weight": to_float(d.get("weight")),
        "pre_draft_ranking": to_int(d.get("preDraftRanking")),
        "pre_draft_position_ranking": to_int(d.get("preDraftPositionRanking")),
        "pre_draft_grade": to_int(d.get("preDraftGrade")),
    } for _, payload in iter_raw(settings, "draft_picks") for d in payload]
    return frame(rows, DRAFT_SCHEMA).unique(["year", "overall"], keep="first").sort("year", "overall")


# ---------- benchmarks ----------

BENCH_SCHEMA = {
    "source": pl.String, "season": pl.Int16, "week": pl.Int8, "season_type": pl.String, "entity_type": pl.String,
    "entity_id": pl.Int64, "entity_name": pl.String, "team_id": pl.Int64, "team": pl.String,
    "position": pl.String, "metric": pl.String, "value": pl.Float64,
}


def _numeric_leaves(d: Any, prefix: str = "") -> Iterator[tuple[str, float]]:
    """Flatten nested dicts to ("offense.rating", 31.2) pairs, keeping numeric leaves only."""
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _numeric_leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        yield prefix, float(d)


TEAM_RATING_SOURCES = {"ratings_sp": ("sp", ("year", "team", "conference", "division")),
                       "ratings_srs": ("srs", ("year", "team", "conference", "division")),
                       "ratings_elo": ("elo", ("year", "team", "conference")),
                       "ratings_fpi": ("fpi", ("year", "team", "conference"))}


def build_benchmarks(settings: Settings, teams: pl.DataFrame) -> pl.DataFrame:
    """Benchmarks and validation metrics in long form. Never features (see cfb.leakage)."""
    index = team_name_index(teams)
    rows: list[dict] = []

    def add(source, season, entity_type, entity_id, metrics, *, week=None, season_type=None, name=None,
            team=None, position=None):
        team_id = index.get(season, team) if team else None
        for metric, value in metrics:
            if value is not None:
                rows.append({"source": source, "season": season, "week": week, "season_type": season_type,
                             "entity_type": entity_type, "entity_id": entity_id, "entity_name": name,
                             "team_id": team_id, "team": team, "position": position, "metric": metric,
                             "value": float(value)})

    for _, payload in iter_raw(settings, "ppa_players_games"):
        for r in payload:
            add("ppa_player_game", r["season"], "player", to_int(r["id"]),
                [(f"avg_ppa_{k}", v) for k, v in (r.get("averagePPA") or {}).items()], week=r["week"],
                season_type=r.get("seasonType"), name=r.get("name"), team=r.get("team"), position=r.get("position"))
    for _, payload in iter_raw(settings, "ppa_players_season"):
        for r in payload:
            metrics = [(f"avg_ppa_{k}", v) for k, v in _numeric_leaves(r.get("averagePPA") or {})]
            metrics += [(f"total_ppa_{k}", v) for k, v in _numeric_leaves(r.get("totalPPA") or {})]
            add("ppa_player_season", r["season"], "player", to_int(r["id"]), metrics, name=r.get("name"),
                team=r.get("team"), position=r.get("position"))
    for slug, source in (("wepa_players_passing", "wepa_passing"), ("wepa_players_rushing", "wepa_rushing")):
        for _, payload in iter_raw(settings, slug):
            for r in payload:
                add(source, r["year"], "player", to_int(r.get("athleteId")), [("wepa", r.get("wepa")), ("plays", r.get("plays"))],
                    name=r.get("athleteName"), team=r.get("team"), position=r.get("position"))
    for _, payload in iter_raw(settings, "wepa_players_kicking"):
        for r in payload:
            add("paar_kicking", r["year"], "player", to_int(r.get("athleteId")),
                [("paar", r.get("paar")), ("attempts", r.get("attempts"))], name=r.get("athleteName"), team=r.get("team"))
    for slug, (source, skip) in TEAM_RATING_SOURCES.items():
        for _, payload in iter_raw(settings, slug):
            for r in payload:
                if r.get("team") == "nationalAverages":
                    continue
                metrics = [(k, v) for k, v in _numeric_leaves({k: v for k, v in r.items() if k not in skip})]
                add(source, r["year"], "team", index.get(r["year"], r.get("team")), metrics, name=r.get("team"),
                    team=r.get("team"))
    for _, payload in iter_raw(settings, "metrics_wp_pregame"):
        for r in payload:
            add("wp_pregame", r["season"], "game", r.get("gameId"),
                [("spread", r.get("spread")), ("home_win_probability", r.get("homeWinProbability"))],
                week=r.get("week"), season_type=r.get("seasonType"))
    return frame(rows, BENCH_SCHEMA)


def build_team_context(settings: Settings, teams: pl.DataFrame) -> pl.DataFrame:
    """Preseason team priors: 247 talent composite and returning production (allowed as priors)."""
    index = team_name_index(teams)
    talent = frame([{"season": int(r["year"]), "team": r["team"], "talent": to_float(r.get("talent"))}
                    for _, payload in iter_raw(settings, "talent") for r in payload],
                   {"season": pl.Int16, "team": pl.String, "talent": pl.Float64})
    ret_rows = []
    for _, payload in iter_raw(settings, "player_returning"):
        for r in payload:
            row = {"season": r["season"], "team": r["team"]}
            row.update({f"returning_{k}": to_float(v) for k, v in r.items() if k not in ("season", "team", "conference")})
            ret_rows.append(row)
    returning = pl.DataFrame(ret_rows).with_columns(pl.col("season").cast(pl.Int16)) if ret_rows else \
        pl.DataFrame(schema={"season": pl.Int16, "team": pl.String})
    df = talent.join(returning, on=["season", "team"], how="full", coalesce=True)
    return df.with_columns(
        team_id=pl.Series([index.get(s, t) for s, t in zip(df["season"].to_list(), df["team"].to_list())], dtype=pl.Int64)
    ).sort("season", "team")
