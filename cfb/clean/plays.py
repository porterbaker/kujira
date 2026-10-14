"""Play-level tables, built one season at a time.

- ``plays``: every /plays row with pre-snap game state, a play category, kneel/spike flags,
  garbage time, and the reason (if any) it is excluded from the player model.
- ``play_participants``: every /plays/stats credit, tagged offense or defense.
- ``pass_plays`` / ``rush_plays``: one row per dropback / rush with the credited players.
  Attribution comes from /plays/stats for every season (the only source before 2025), then the
  enriched endpoints (2025+), then an exact roster-name match from the play text.

/plays reports ``offenseScore``/``defenseScore`` *after* the play (the score already includes a
touchdown scored on it), so the pre-snap score is taken from the previous play in the game.
"""

from __future__ import annotations

from collections import defaultdict

import polars as pl

from cfb.clean.common import frame, iter_raw, normalize_name, to_int
from cfb.config import Settings

PASS_TYPES = ("Pass Reception", "Pass Incompletion", "Passing Touchdown", "Pass Interception Return",
              "Interception Return Touchdown", "Interception", "Pass", "Pass Completion", "Pass Interception")
RUSH_TYPES = ("Rush", "Rushing Touchdown")
SACK_TYPES = ("Sack",)
# Typed by outcome, not by play: resolved from the /plays/stats credits.
AMBIGUOUS_TYPES = ("Fumble Recovery (Own)", "Fumble Recovery (Opponent)", "Fumble Return Touchdown", "Safety", "Fumble")
COMPLETION_TYPES = ("Pass Reception", "Passing Touchdown", "Pass Completion")
INTERCEPTION_TYPES = ("Pass Interception Return", "Interception Return Touchdown", "Interception", "Pass Interception")

PASSER_STATS = ("Completion", "Incompletion", "Interception Thrown", "Sack Taken")
KNEEL_TEXT = r"(?i)kneel|takes? a knee"
TEAM_RUN_FOR_LOSS = r"^\s*TEAM (?:run|rush)"  # pre-2022 kneel-downs: "TEAM run for a loss of 1 yard"
SPIKE_TEXT = r"(?i)\bspik(?:e|ed|es)\b"
TARGET_TEXT = (r"(?i)\b(?:in)?complete to ([A-Za-z][A-Za-z.'\- ]*?)"
               r"(?:\s+for\b|,|\s+\(|\s+broken\b|\s+to the\b|\s+at the\b|\s+out of\b|\s+in the\b"
               r"|\s+caught\b|\s+deflected\b|\s+dropped\b|\s*\.?\s*$)")

PLAY_RAW_SCHEMA = {
    "play_id": pl.Int64, "game_id": pl.Int64, "drive_id": pl.Int64, "season": pl.Int16, "week": pl.Int8,
    "season_type": pl.String, "drive_number": pl.Int16, "play_number": pl.Int16, "period": pl.Int8,
    "clock_seconds": pl.Int16, "offense": pl.String, "defense": pl.String, "home": pl.String,
    "offense_conference": pl.String, "defense_conference": pl.String,
    "offense_score_after": pl.Int16, "defense_score_after": pl.Int16,
    "offense_timeouts": pl.Int8, "defense_timeouts": pl.Int8, "yardline": pl.Int16, "yards_to_goal": pl.Int16,
    "down": pl.Int8, "distance": pl.Int16, "yards_gained": pl.Int16, "scoring": pl.Boolean,
    "play_type": pl.String, "play_text": pl.String, "ppa": pl.Float64, "wallclock": pl.String,
}

STAT_RAW_SCHEMA = {
    "play_id": pl.Int64, "game_id": pl.Int64, "athlete_id": pl.Int64, "athlete_name": pl.String,
    "team": pl.String, "stat_type": pl.String, "stat": pl.Float64,
}


# ---------- raw loading ----------

def load_raw_plays(settings: Settings, season: int) -> pl.DataFrame:
    rows = []
    for part, payload in iter_raw(settings, "plays"):
        if int(part["season"]) != season:
            continue
        for p in payload:
            clock = p.get("clock") or {}
            minutes, seconds = clock.get("minutes"), clock.get("seconds")
            rows.append({
                "play_id": int(p["id"]), "game_id": p["gameId"], "drive_id": to_int(p.get("driveId")),
                "season": season, "week": int(part["week"]), "season_type": part["type"],
                "drive_number": p.get("driveNumber"), "play_number": p.get("playNumber"), "period": p.get("period"),
                "clock_seconds": None if minutes is None or seconds is None else minutes * 60 + seconds,
                "offense": p.get("offense"), "defense": p.get("defense"), "home": p.get("home"),
                "offense_conference": p.get("offenseConference"), "defense_conference": p.get("defenseConference"),
                "offense_score_after": p.get("offenseScore"), "defense_score_after": p.get("defenseScore"),
                "offense_timeouts": p.get("offenseTimeouts"), "defense_timeouts": p.get("defenseTimeouts"),
                "yardline": p.get("yardline"), "yards_to_goal": p.get("yardsToGoal"), "down": p.get("down"),
                "distance": p.get("distance"), "yards_gained": p.get("yardsGained"), "scoring": p.get("scoring"),
                "play_type": p.get("playType"), "play_text": p.get("playText"), "ppa": p.get("ppa"),
                "wallclock": p.get("wallclock"),
            })
    return frame(rows, PLAY_RAW_SCHEMA).unique("play_id", keep="first")


def load_raw_stats(settings: Settings, season: int) -> pl.DataFrame:
    """Per-game /plays/stats; a game re-pulled by stat type uses its split files, not the capped page."""
    by_game: dict[str, dict[str, list]] = defaultdict(lambda: {"page": [], "splits": []})
    for part, payload in iter_raw(settings, "plays_stats"):
        if part.get("season") != str(season) or "game" not in part:
            continue
        by_game[part["game"]]["splits" if "stat_type" in part else "page"].extend(payload)
    rows = []
    for files in by_game.values():
        for s in files["splits"] or files["page"]:
            rows.append({
                "play_id": to_int(s.get("playId")), "game_id": to_int(s.get("gameId")),
                "athlete_id": to_int(s.get("athleteId")), "athlete_name": s.get("athleteName"),
                "team": s.get("team"), "stat_type": s.get("statType"), "stat": s.get("stat"),
            })
    return frame(rows, STAT_RAW_SCHEMA).unique(["play_id", "athlete_id", "stat_type"], keep="first")


def load_enriched(settings: Settings, season: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    passing, rushing = [], []
    for slug, out in (("passing_plays", passing), ("rushing_plays", rushing)):
        for part, payload in iter_raw(settings, slug):
            if int(part["season"]) == season:
                out.extend(payload)
    pass_df = frame([{
        "play_id": int(r["playId"]), "enriched_passer_id": to_int(r.get("passerId")),
        "enriched_target_id": to_int(r.get("targetId")), "enriched_outcome": r.get("outcome"),
        "air_yards": r.get("airYards"), "pass_depth": r.get("passDepth"), "pass_direction": r.get("passDirection"),
        "pass_location": r.get("passLocation"), "total_yards": r.get("totalYards"),
        "yards_after_catch": r.get("yardsAfterCatch"), "is_spike_enriched": r.get("isSpike"),
        "enriched_parse_status": r.get("parseStatus"),
    } for r in passing], {
        "play_id": pl.Int64, "enriched_passer_id": pl.Int64, "enriched_target_id": pl.Int64,
        "enriched_outcome": pl.String, "air_yards": pl.Int16, "pass_depth": pl.String,
        "pass_direction": pl.String, "pass_location": pl.String, "total_yards": pl.Int16,
        "yards_after_catch": pl.Int16, "is_spike_enriched": pl.Boolean, "enriched_parse_status": pl.String,
    }).unique("play_id")
    rush_df = frame([{
        "play_id": int(r["playId"]), "enriched_rusher_id": to_int(r.get("rusherId")),
        "rush_direction": r.get("rushDirection"), "is_sack_enriched": r.get("isSack"),
        "is_kneel_enriched": r.get("isKneel"), "is_team_rush": r.get("isTeamRush"),
        "enriched_attribution_status": r.get("attributionStatus"), "enriched_parse_status": r.get("parseStatus"),
    } for r in rushing], {
        "play_id": pl.Int64, "enriched_rusher_id": pl.Int64, "rush_direction": pl.String,
        "is_sack_enriched": pl.Boolean, "is_kneel_enriched": pl.Boolean, "is_team_rush": pl.Boolean,
        "enriched_attribution_status": pl.String, "enriched_parse_status": pl.String,
    }).unique("play_id")
    return pass_df, rush_df


# ---------- plays ----------

def garbage_time_expr(thresholds: dict[int, int] | None) -> pl.Expr:
    """|pre-snap margin| above a per-quarter threshold. None (undecided) -> all null."""
    if not thresholds:
        return pl.lit(None, dtype=pl.Boolean)
    margin = pl.col("score_diff").abs()
    expr = pl.lit(False)
    for period, limit in sorted(thresholds.items()):
        expr = pl.when((pl.col("period") == period) & (margin > limit)).then(True).otherwise(expr)
    return expr


def build_plays_frame(raw: pl.DataFrame, games: pl.DataFrame, play_types: pl.DataFrame,
                      garbage_thresholds: dict[int, int] | None) -> pl.DataFrame:
    g = games.select("game_id", "home_id", "away_id", "home_team", "away_team", "neutral_site")
    df = raw.join(g, on="game_id", how="left").sort("game_id", "play_id")
    home_off = pl.col("offense") == pl.col("home_team")
    away_off = pl.col("offense") == pl.col("away_team")
    df = df.with_columns(
        is_home_offense=pl.when(home_off).then(True).when(away_off).then(False).otherwise(None),
        offense_id=pl.when(home_off).then(pl.col("home_id")).when(away_off).then(pl.col("away_id")),
        defense_id=pl.when(home_off).then(pl.col("away_id")).when(away_off).then(pl.col("home_id")),
    ).with_columns(
        home_score_after=pl.when(pl.col("is_home_offense")).then(pl.col("offense_score_after"))
        .otherwise(pl.col("defense_score_after")),
        away_score_after=pl.when(pl.col("is_home_offense")).then(pl.col("defense_score_after"))
        .otherwise(pl.col("offense_score_after")),
    ).with_columns(
        home_score=pl.col("home_score_after").shift(1).over("game_id").fill_null(0),
        away_score=pl.col("away_score_after").shift(1).over("game_id").fill_null(0),
    ).with_columns(
        offense_score=pl.when(pl.col("is_home_offense")).then(pl.col("home_score")).otherwise(pl.col("away_score")),
        defense_score=pl.when(pl.col("is_home_offense")).then(pl.col("away_score")).otherwise(pl.col("home_score")),
    ).with_columns(
        score_diff=(pl.col("offense_score") - pl.col("defense_score")).cast(pl.Int16),
        game_seconds_remaining=pl.when(pl.col("period") <= 4)
        .then((4 - pl.col("period").cast(pl.Int32)) * 900 + pl.col("clock_seconds").cast(pl.Int32)),
        category=pl.when(pl.col("play_type").is_in(PASS_TYPES)).then(pl.lit("pass"))
        .when(pl.col("play_type").is_in(SACK_TYPES)).then(pl.lit("sack"))
        .when(pl.col("play_type").is_in(RUSH_TYPES)).then(pl.lit("rush"))
        .when(pl.col("play_type").is_in(AMBIGUOUS_TYPES)).then(pl.lit("ambiguous"))
        .otherwise(pl.lit("other")),
    ).with_columns(garbage_time=garbage_time_expr(garbage_thresholds))
    df = df.join(play_types.select("play_type", "play_type_id"), on="play_type", how="left")
    return df.drop("home_id", "away_id", "home_team", "away_team", "home_score_after", "away_score_after",
                   "home", "home_score", "away_score")


def finalize_plays(df: pl.DataFrame, credits: pl.DataFrame, pass_enriched: pl.DataFrame,
                   rush_enriched: pl.DataFrame, unreliable_parse: tuple[str, ...] = ("invalid",)) -> pl.DataFrame:
    """Resolve ambiguous categories from credits, set kneel/spike flags and the exclusion reason."""
    df = df.join(credits.select("play_id", "has_passer", "has_sack_taken", "has_rusher"), on="play_id", how="left")
    df = df.join(pass_enriched.select("play_id", "is_spike_enriched", pl.col("enriched_parse_status").alias("_pps")),
                 on="play_id", how="left")
    df = df.join(rush_enriched.select("play_id", "is_kneel_enriched", pl.col("enriched_parse_status").alias("_rps")),
                 on="play_id", how="left")
    text = pl.col("play_text").fill_null("")
    df = df.with_columns(
        category=pl.when(pl.col("category") != "ambiguous").then(pl.col("category"))
        .when(pl.col("has_sack_taken").fill_null(False)).then(pl.lit("sack"))
        .when(pl.col("has_passer").fill_null(False)).then(pl.lit("pass"))
        .when(pl.col("has_rusher").fill_null(False)).then(pl.lit("rush"))
        .otherwise(pl.lit("other")),
    ).with_columns(
        is_kneel=(pl.col("category") == "rush") & (
            text.str.contains(KNEEL_TEXT)
            | (text.str.contains(TEAM_RUN_FOR_LOSS) & (pl.col("yards_gained").fill_null(0) <= 0))
            | pl.col("is_kneel_enriched").fill_null(False)),
        is_spike=(pl.col("category") == "pass") & (text.str.contains(SPIKE_TEXT) | pl.col("is_spike_enriched").fill_null(False)),
        invalid_parse=pl.col("_pps").is_in(unreliable_parse) | pl.col("_rps").is_in(unreliable_parse),
    ).with_columns(
        exclude_reason=pl.when(~pl.col("category").is_in(["pass", "sack", "rush"])).then(pl.lit("non_scrimmage"))
        .when(pl.col("is_kneel")).then(pl.lit("kneel"))
        .when(pl.col("is_spike")).then(pl.lit("spike"))
        .when(pl.col("ppa").is_null()).then(pl.lit("no_ppa"))
        .when(pl.col("invalid_parse").fill_null(False)).then(pl.lit("invalid_parse"))
        .otherwise(None),
    ).with_columns(model_eligible=pl.col("exclude_reason").is_null())
    return df.drop("has_passer", "has_sack_taken", "has_rusher", "is_spike_enriched", "is_kneel_enriched",
                   "_pps", "_rps", "invalid_parse")


# ---------- participants and attribution ----------

def build_participants(stats: pl.DataFrame, plays: pl.DataFrame, games: pl.DataFrame,
                       stat_types: pl.DataFrame) -> pl.DataFrame:
    p = plays.select("play_id", "season", "week", "offense", "defense", "offense_id", "defense_id")
    g = games.select("game_id", "home_team", "home_id", "away_team", "away_id")
    df = stats.join(p, on="play_id", how="left").join(g, on="game_id", how="left")
    df = df.with_columns(
        team_id=pl.when(pl.col("team") == pl.col("offense")).then(pl.col("offense_id"))
        .when(pl.col("team") == pl.col("defense")).then(pl.col("defense_id"))
        .when(pl.col("team") == pl.col("home_team")).then(pl.col("home_id"))
        .when(pl.col("team") == pl.col("away_team")).then(pl.col("away_id")),
    ).with_columns(
        side=pl.when(pl.col("team_id") == pl.col("offense_id")).then(pl.lit("offense"))
        .when(pl.col("team_id") == pl.col("defense_id")).then(pl.lit("defense")),
        in_plays=pl.col("offense").is_not_null(),
    ).join(stat_types, on="stat_type", how="left")
    return df.select("play_id", "game_id", "season", "week", "athlete_id", "athlete_name", "team", "team_id",
                     "side", "stat_type", "stat_type_id", "stat", "in_plays")


def credits_per_play(participants: pl.DataFrame) -> pl.DataFrame:
    off = pl.col("side") == "offense"
    dfn = pl.col("side") == "defense"

    def ids(cond: pl.Expr, name: str) -> pl.Expr:
        return pl.col("athlete_id").filter(cond).unique().sort().alias(name)

    st = pl.col("stat_type")
    return participants.group_by("play_id").agg(
        ids(off & st.is_in(PASSER_STATS), "passers"),
        ids(off & (st == "Reception"), "receivers"),
        ids(off & (st == "Target"), "targets"),
        ids(off & (st == "Rush"), "rushers"),
        ids(dfn & (st == "Sack"), "sacked_by"),
        ids(dfn & (st == "Interception"), "intercepted_by"),
        (off & st.is_in(PASSER_STATS)).any().alias("has_passer"),
        (off & (st == "Sack Taken")).any().alias("has_sack_taken"),
        (off & (st == "Interception Thrown")).any().alias("has_int_thrown"),
        (off & (st == "Completion")).any().alias("has_completion"),
        (off & (st == "Incompletion")).any().alias("has_incompletion"),
        (off & (st == "Rush")).any().alias("has_rusher"),
    )


def _single(col: str) -> pl.Expr:
    return pl.when(pl.col(col).list.len() == 1).then(pl.col(col).list.first())


def _attribution(col: str) -> pl.Expr:
    n = pl.col(col).list.len().fill_null(0)
    return pl.when(n == 1).then(pl.lit("individual")).when(n == 0).then(pl.lit("missing")).otherwise(pl.lit("multiple"))


def roster_name_index(player_seasons: pl.DataFrame, season: int) -> dict[tuple[int, str], int]:
    """(team_id, normalized full name) -> athlete_id, only where the name is unique on that roster."""
    seen: dict[tuple[int, str], set[int]] = defaultdict(set)
    rows = player_seasons.filter(pl.col("season") == season).select("team_id", "name_norm", "athlete_id")
    for team_id, name, athlete_id in rows.iter_rows():
        if team_id is not None and name:
            seen[(team_id, name)].add(athlete_id)
    return {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}


def match_text_targets(df: pl.DataFrame, index: dict[tuple[int, str], int]) -> pl.DataFrame:
    """Exact normalized-name match of the play-text target against the offense's roster."""
    names = df.select(pl.col("play_text").str.extract(TARGET_TEXT, 1).alias("target_name_text"))["target_name_text"]
    norm = {n: normalize_name(n) for n in names.unique().drop_nulls().to_list()}
    matched = [index.get((team, norm.get(name, ""))) if name else None
               for team, name in zip(df["offense_id"].to_list(), names.to_list())]
    return df.with_columns(names.alias("target_name_text"), pl.Series("text_target_id", matched, dtype=pl.Int64))


def build_pass_plays(plays: pl.DataFrame, credits: pl.DataFrame, pass_enriched: pl.DataFrame,
                     rush_enriched: pl.DataFrame, name_index: dict[tuple[int, str], int]) -> pl.DataFrame:
    base = plays.filter(pl.col("category").is_in(["pass", "sack"])).select(
        "play_id", "game_id", "season", "week", "season_type", "offense_id", "defense_id", "period",
        "down", "distance", "yards_to_goal", "score_diff", "yards_gained", "play_type", "play_text", "ppa",
        "garbage_time", "model_eligible", "exclude_reason", "category")
    df = base.join(credits, on="play_id", how="left").join(pass_enriched, on="play_id", how="left")
    df = df.join(rush_enriched.filter(pl.col("is_sack_enriched").fill_null(False))
                 .select("play_id", pl.col("enriched_rusher_id").alias("enriched_sack_qb_id")), on="play_id", how="left")
    df = df.with_columns(
        is_sack=(pl.col("category") == "sack") | pl.col("has_sack_taken").fill_null(False),
    ).with_columns(
        outcome=pl.when(pl.col("is_sack")).then(pl.lit("sack"))
        .when(pl.col("has_int_thrown").fill_null(False) | pl.col("play_type").is_in(INTERCEPTION_TYPES)).then(pl.lit("interception"))
        .when(pl.col("has_completion").fill_null(False) | pl.col("play_type").is_in(COMPLETION_TYPES)).then(pl.lit("completion"))
        .when(pl.col("has_incompletion").fill_null(False) | (pl.col("play_type") == "Pass Incompletion")).then(pl.lit("incompletion"))
        .otherwise(pl.lit("unknown")),
        passer_id=_single("passers"),
        passer_attribution=_attribution("passers"),
        enriched_passer_id=pl.coalesce("enriched_passer_id", "enriched_sack_qb_id"),
    )
    df = match_text_targets(df, name_index)
    stats_target = pl.when(pl.col("outcome") == "completion").then(_single("receivers")).otherwise(_single("targets"))
    df = df.with_columns(stats_target_id=stats_target).with_columns(
        target_id=pl.when(pl.col("is_sack")).then(None)
        .otherwise(pl.coalesce("stats_target_id", "enriched_target_id", "text_target_id")),
        target_source=pl.when(pl.col("is_sack")).then(None)
        .when(pl.col("stats_target_id").is_not_null()).then(pl.lit("stats"))
        .when(pl.col("enriched_target_id").is_not_null()).then(pl.lit("enriched"))
        .when(pl.col("text_target_id").is_not_null()).then(pl.lit("text")),
        sacked_by=pl.col("sacked_by").fill_null([]),
        intercepted_by_id=_single("intercepted_by"),
        passer_agrees_enriched=pl.when(pl.col("enriched_passer_id").is_not_null() & pl.col("passer_id").is_not_null())
        .then(pl.col("enriched_passer_id") == pl.col("passer_id")),
    )
    return df.select(
        "play_id", "game_id", "season", "week", "season_type", "offense_id", "defense_id", "period", "down",
        "distance", "yards_to_goal", "score_diff", "outcome", "is_sack", "passer_id", "passer_attribution",
        "target_id", "target_source", "target_name_text", "sacked_by", "intercepted_by_id", "yards_gained", "ppa",
        "garbage_time", "model_eligible", "exclude_reason",
        "air_yards", "pass_depth", "pass_direction", "pass_location", "yards_after_catch",
        "enriched_passer_id", "enriched_target_id", "enriched_parse_status", "passer_agrees_enriched",
    )


def build_rush_plays(plays: pl.DataFrame, credits: pl.DataFrame, rush_enriched: pl.DataFrame,
                     unreliable_attribution: tuple[str, ...] = ()) -> pl.DataFrame:
    base = plays.filter(pl.col("category") == "rush").select(
        "play_id", "game_id", "season", "week", "season_type", "offense_id", "defense_id", "period", "down",
        "distance", "yards_to_goal", "score_diff", "yards_gained", "play_text", "ppa", "is_kneel", "garbage_time",
        "model_eligible", "exclude_reason")
    df = base.join(credits.select("play_id", "rushers"), on="play_id", how="left")
    df = df.join(rush_enriched.drop("is_sack_enriched", "is_kneel_enriched"), on="play_id", how="left")
    df = df.with_columns(
        rusher_id=_single("rushers"),
        rusher_attribution=_attribution("rushers"),
        is_team_rush=pl.col("is_team_rush").fill_null(pl.col("play_text").fill_null("").str.contains(TEAM_RUN_FOR_LOSS)),
    ).with_columns(
        rusher_agrees_enriched=pl.when(pl.col("enriched_rusher_id").is_not_null() & pl.col("rusher_id").is_not_null())
        .then(pl.col("enriched_rusher_id") == pl.col("rusher_id")),
    )
    df = df.with_columns(enriched_attribution_reliable=pl.when(pl.col("enriched_attribution_status").is_not_null())
                         .then(~pl.col("enriched_attribution_status").is_in(unreliable_attribution)))
    return df.select(
        "play_id", "game_id", "season", "week", "season_type", "offense_id", "defense_id", "period", "down",
        "distance", "yards_to_goal", "score_diff", "rusher_id", "rusher_attribution", "is_kneel", "is_team_rush",
        "yards_gained", "ppa", "garbage_time", "model_eligible", "exclude_reason", "rush_direction",
        "enriched_rusher_id", "enriched_attribution_status", "enriched_attribution_reliable",
        "enriched_parse_status", "rusher_agrees_enriched",
    )


def build_season(settings: Settings, season: int, games: pl.DataFrame, play_types: pl.DataFrame,
                 stat_types: pl.DataFrame, player_seasons: pl.DataFrame) -> dict[str, pl.DataFrame]:
    th = settings.thresholds
    raw = load_raw_plays(settings, season)
    plays = build_plays_frame(raw, games, play_types, th.garbage_time)
    participants = build_participants(load_raw_stats(settings, season), plays, games, stat_types)
    credits = credits_per_play(participants)
    pass_enriched, rush_enriched = load_enriched(settings, season)
    plays = finalize_plays(plays, credits, pass_enriched, rush_enriched, th.unreliable_parse_statuses)
    index = roster_name_index(player_seasons, season)
    return {
        "plays": plays,
        "play_participants": participants,
        "pass_plays": build_pass_plays(plays, credits, pass_enriched, rush_enriched, index),
        "rush_plays": build_rush_plays(plays, credits, rush_enriched, th.unreliable_attribution_statuses),
    }
