"""Player identity: rosters by season, recruits, recruit links, placeholder aliases, players.

Identity is ``athlete_id``. Every athlete credited in /plays/stats appears on that season's
roster (probe: 100% for 2014-2026). Rosters for 2014-2019 also hold negative ids: placeholders
for players who never recorded a stat. They are aliased to a real id only through an exact
shared recruit id, never by name.
"""

from __future__ import annotations

import polars as pl

from cfb.clean.common import frame, iter_raw, normalize_name, position_group, to_float, to_int
from cfb.clean.reference import team_name_index
from cfb.config import Settings

PLAYER_SEASON_SCHEMA = {
    "athlete_id": pl.Int64, "season": pl.Int16, "team": pl.String, "team_id": pl.Int64,
    "first_name": pl.String, "last_name": pl.String, "name_norm": pl.String, "position": pl.String,
    "position_group": pl.String, "jersey": pl.Int16, "class_year": pl.Int8, "class_year_raw": pl.Int16,
    "class_year_suspect": pl.Boolean, "height": pl.Float64, "weight": pl.Float64, "home_city": pl.String,
    "home_state": pl.String, "home_country": pl.String, "home_latitude": pl.Float64,
    "home_longitude": pl.Float64, "home_county_fips": pl.String, "recruit_ids": pl.List(pl.Int64),
    "placeholder_id": pl.Boolean,
}


def build_player_seasons(settings: Settings, teams: pl.DataFrame) -> pl.DataFrame:
    index = team_name_index(teams)
    rows = []
    for part, payload in iter_raw(settings, "roster"):
        season = int(part["season"])
        for r in payload:
            raw_year = to_int(r.get("year"))
            valid = raw_year is not None and 1 <= raw_year <= 5
            athlete_id = int(r["id"])
            first, last = r.get("firstName"), r.get("lastName")
            rows.append({
                "athlete_id": athlete_id, "season": season, "team": r.get("team"),
                "team_id": index.get(season, r.get("team")), "first_name": first, "last_name": last,
                "name_norm": normalize_name(f"{first or ''} {last or ''}"), "position": r.get("position"),
                "position_group": position_group(r.get("position")), "jersey": to_int(r.get("jersey")),
                "class_year": raw_year if valid else None, "class_year_raw": raw_year,
                "class_year_suspect": not valid, "height": to_float(r.get("height")),
                "weight": to_float(r.get("weight")), "home_city": r.get("homeCity"),
                "home_state": r.get("homeState"), "home_country": r.get("homeCountry"),
                "home_latitude": to_float(r.get("homeLatitude")), "home_longitude": to_float(r.get("homeLongitude")),
                "home_county_fips": r.get("homeCountyFIPS"),
                "recruit_ids": [i for i in (to_int(x) for x in (r.get("recruitIds") or [])) if i is not None],
                "placeholder_id": athlete_id < 0,
            })
    return frame(rows, PLAYER_SEASON_SCHEMA).unique(["athlete_id", "season", "team"], keep="first") \
        .sort("season", "team_id", "athlete_id")


RECRUIT_SCHEMA = {
    "recruit_id": pl.Int64, "athlete_id": pl.Int64, "recruit_type": pl.String, "class_year": pl.Int16,
    "ranking": pl.Int32, "name": pl.String, "name_norm": pl.String, "school": pl.String,
    "committed_to": pl.String, "committed_to_id": pl.Int64, "position": pl.String, "position_group": pl.String,
    "height": pl.Float64, "weight": pl.Float64, "stars": pl.Int8, "rating": pl.Float64, "city": pl.String,
    "state_province": pl.String, "country": pl.String, "latitude": pl.Float64, "longitude": pl.Float64,
}


def build_recruits(settings: Settings, teams: pl.DataFrame) -> pl.DataFrame:
    index = team_name_index(teams)
    first_team_season = int(teams["season"].min())
    rows = []
    for _, payload in iter_raw(settings, "recruiting_players"):
        for r in payload:
            year = to_int(r.get("year"))
            home = r.get("hometownInfo") or {}
            committed = r.get("committedTo")
            rows.append({
                "recruit_id": int(r["id"]), "athlete_id": to_int(r.get("athleteId")),
                "recruit_type": r.get("recruitType"), "class_year": year, "ranking": to_int(r.get("ranking")),
                "name": r.get("name"), "name_norm": normalize_name(r.get("name")), "school": r.get("school"),
                "committed_to": committed,
                "committed_to_id": index.get(max(year or first_team_season, first_team_season), committed),
                "position": r.get("position"), "position_group": position_group(r.get("position")),
                "height": to_float(r.get("height")), "weight": to_float(r.get("weight")),
                "stars": to_int(r.get("stars")), "rating": to_float(r.get("rating")), "city": r.get("city"),
                "state_province": r.get("stateProvince"), "country": r.get("country"),
                "latitude": to_float(home.get("latitude")), "longitude": to_float(home.get("longitude")),
            })
    return frame(rows, RECRUIT_SCHEMA).unique("recruit_id", keep="first").sort("class_year", "recruit_id")


def build_recruit_links(player_seasons: pl.DataFrame, recruits: pl.DataFrame) -> pl.DataFrame:
    """(athlete_id, recruit_id) from roster.recruitIds and from recruiting.athleteId, with the path."""
    via_roster = (player_seasons.select("athlete_id", "recruit_ids").explode("recruit_ids")
                  .drop_nulls("recruit_ids").rename({"recruit_ids": "recruit_id"}).unique()
                  .with_columns(via_roster=pl.lit(True)))
    via_recruiting = (recruits.drop_nulls("athlete_id").select("athlete_id", "recruit_id").unique()
                      .with_columns(via_recruiting=pl.lit(True)))
    links = via_roster.join(via_recruiting, on=["athlete_id", "recruit_id"], how="full", coalesce=True)
    links = links.with_columns(pl.col("via_roster").fill_null(False), pl.col("via_recruiting").fill_null(False))
    athletes_per_recruit = links.group_by("recruit_id").agg(pl.col("athlete_id").n_unique().alias("_n"))
    return (links.join(athletes_per_recruit, on="recruit_id")
            .with_columns(
                path=pl.when(pl.col("via_roster") & pl.col("via_recruiting")).then(pl.lit("both"))
                .when(pl.col("via_roster")).then(pl.lit("roster")).otherwise(pl.lit("recruiting")),
                recruit_in_table=pl.col("recruit_id").is_in(recruits["recruit_id"].implode()),
                recruit_shared=pl.col("_n") > 1)
            .drop("_n").sort("athlete_id", "recruit_id"))


def build_athlete_aliases(links: pl.DataFrame, player_seasons: pl.DataFrame) -> pl.DataFrame:
    """Placeholder (negative) ids -> a real id, by exact keys only, never fuzzy names.

    1. ``shared_recruit_id``: the single real id that shares one of the placeholder's recruit ids.
    2. ``exact_name_team``: the same normalized name on the same team in non-overlapping seasons.
       Each placeholder must map to one real id; several placeholders may map to the same real id
       (one person re-keyed across seasons) only if their seasons don't overlap each other.
    """
    neg = links.filter(pl.col("athlete_id") < 0).select(pl.col("athlete_id").alias("placeholder_id"), "recruit_id")
    pos = links.filter(pl.col("athlete_id") > 0).select("athlete_id", "recruit_id")
    by_recruit = (neg.join(pos, on="recruit_id").group_by("placeholder_id").agg(pl.col("athlete_id").unique())
                  .filter(pl.col("athlete_id").list.len() == 1)
                  .with_columns(pl.col("athlete_id").list.first(), method=pl.lit("shared_recruit_id")))

    spans = (player_seasons.filter(pl.col("team_id").is_not_null() & (pl.col("name_norm") != ""))
             .group_by("athlete_id", "name_norm", "team_id")
             .agg(pl.col("season").min().alias("first"), pl.col("season").max().alias("last")))
    ph = spans.filter(pl.col("athlete_id") < 0).rename({"athlete_id": "placeholder_id", "first": "pf", "last": "pl"})
    real = spans.filter(pl.col("athlete_id") > 0)
    pairs = ph.join(real, on=["name_norm", "team_id"]).filter((pl.col("pl") < pl.col("first")) | (pl.col("pf") > pl.col("last")))
    pairs = pairs.filter(pl.col("athlete_id").n_unique().over("placeholder_id") == 1).sort("athlete_id", "pf")
    disjoint = pairs.with_columns(
        ok=(pl.col("pl").shift(1).over("athlete_id").is_null()) | (pl.col("pf") > pl.col("pl").shift(1).over("athlete_id"))
    ).filter(pl.col("ok").all().over("athlete_id"))
    by_name = (disjoint.select("placeholder_id", "athlete_id").unique()
               .join(by_recruit.select("placeholder_id"), on="placeholder_id", how="anti")
               .with_columns(method=pl.lit("exact_name_team")))
    return pl.concat([by_recruit, by_name.select(by_recruit.columns)]).sort("placeholder_id")


def build_players(player_seasons: pl.DataFrame, participants: pl.LazyFrame, box: pl.LazyFrame,
                  aliases: pl.DataFrame) -> pl.DataFrame:
    """One row per athlete_id seen on a roster, in /plays/stats, or in a box score."""
    ps = player_seasons.sort("season")
    pos_mode = (ps.drop_nulls("position").group_by("athlete_id", "position")
                .agg(pl.len().alias("n"), pl.col("season").max().alias("last"))
                .sort("n", "last", descending=True).unique("athlete_id", keep="first")
                .select("athlete_id", pl.col("position").alias("primary_position")))
    roster = ps.group_by("athlete_id").agg(
        pl.col("first_name").last(), pl.col("last_name").last(), pl.col("name_norm").last(),
        pl.col("height").drop_nulls().last(), pl.col("weight").drop_nulls().last(),
        pl.col("home_state").drop_nulls().last(), pl.col("home_country").drop_nulls().last(),
        pl.col("season").min().alias("first_roster_season"), pl.col("season").max().alias("last_roster_season"),
        pl.col("season").n_unique().alias("n_roster_seasons"),
    ).join(pos_mode, on="athlete_id", how="left")
    seen = pl.concat([
        participants.select("athlete_id", pl.col("athlete_name").alias("name"), "season"),
        box.select("athlete_id", pl.col("athlete_name").alias("name"), "season"),
    ]).drop_nulls("athlete_id").group_by("athlete_id").agg(
        pl.col("name").last().alias("seen_name"), pl.col("season").min().alias("first_seen_season"),
        pl.col("season").max().alias("last_seen_season"),
    ).collect()
    players = roster.join(seen, on="athlete_id", how="full", coalesce=True)
    players = players.with_columns(
        in_roster=pl.col("first_roster_season").is_not_null(),
        name_norm=pl.coalesce("name_norm", pl.col("seen_name").map_elements(normalize_name, return_dtype=pl.String)),
        position_group=pl.col("primary_position").map_elements(position_group, return_dtype=pl.String),
        placeholder_id=pl.col("athlete_id") < 0,
    ).join(aliases.select(pl.col("placeholder_id").alias("athlete_id"), pl.col("athlete_id").alias("alias_of")),
           on="athlete_id", how="left")
    return players.with_columns(canonical_id=pl.coalesce("alias_of", "athlete_id")).drop("alias_of").sort("athlete_id")
