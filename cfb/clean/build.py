"""`cfb clean build`: raw JSON -> typed parquet tables in data/clean/.

Large play-level tables are built one season at a time and streamed into a single parquet file,
keeping memory near 1 GB.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable

import polars as pl

from cfb.clean import box, context, people, plays, reference, transfers
from cfb.clean.common import combine_parts, parts_dir, read_table, scan_table, write_part, write_table
from cfb.config import Settings

SEASON_TABLES = ("plays", "play_participants", "pass_plays", "rush_plays")


def build_all(settings: Settings, seasons: tuple[int, ...] | None = None, log: Callable[[str], None] = print) -> None:
    seasons = seasons or settings.seasons
    started = time.monotonic()

    def step(name: str, df: pl.DataFrame) -> pl.DataFrame:
        write_table(df, settings, name)
        log(f"  {name}: {df.height:,} rows x {df.width} cols ({time.monotonic() - started:.0f}s)")
        return df

    log("reference")
    step("venues", reference.build_venues(settings))
    teams = step("teams", reference.build_teams(settings))
    games = step("games", reference.build_games(settings))
    play_types = step("play_types", reference.build_play_types(settings))
    stat_types = step("stat_types", reference.build_stat_types(settings))

    log("people")
    player_seasons = step("player_seasons", people.build_player_seasons(settings, teams))
    recruits = step("recruits", people.build_recruits(settings, teams))

    log("plays, by season")
    for name in (*SEASON_TABLES, "player_game_box"):
        shutil.rmtree(parts_dir(settings, name), ignore_errors=True)
    for season in seasons:
        out = plays.build_season(settings, season, games, play_types, stat_types, player_seasons)
        for name, df in out.items():
            write_part(df, settings, name, season)
        write_part(box.build_box_season(settings, season, games), settings, "player_game_box", season)
        log(f"  {season}: {out['plays'].height:,} plays, {out['play_participants'].height:,} credits "
            f"({time.monotonic() - started:.0f}s)")
        del out
    for name in (*SEASON_TABLES, "player_game_box"):
        combine_parts(settings, name)
    shutil.rmtree(settings.paths.clean / "_parts", ignore_errors=True)

    log("identity")
    links = step("recruit_links", people.build_recruit_links(player_seasons, recruits))
    aliases = step("athlete_aliases", people.build_athlete_aliases(links, player_seasons))
    step("players", people.build_players(player_seasons, scan_table(settings, "play_participants"),
                                         scan_table(settings, "player_game_box"), aliases))

    log("transfers")
    tr = step("transfers", transfers.build_transfers(settings, teams))
    matches = step("transfer_matches", transfers.match_transfers(
        tr, player_seasons, settings.thresholds.transfer_match_threshold))
    review = transfers.review_rows(tr, matches)
    settings.paths.reports.mkdir(parents=True, exist_ok=True)
    review.write_csv(settings.paths.reports / "transfer_review.csv")
    log(f"  transfer_review.csv: {review.height:,} fuzzy candidates awaiting review")

    log("context")
    step("lines", context.build_lines(settings))
    coaches = step("coaches", context.build_coaches(settings))
    step("team_coaching", context.build_team_coaching(coaches))
    step("draft", context.build_draft(settings))
    step("benchmarks", context.build_benchmarks(settings, teams))
    step("team_context", context.build_team_context(settings, teams))
    log(f"done in {time.monotonic() - started:.0f}s")


def table_inventory(settings: Settings) -> list[tuple[str, int, int]]:
    out = []
    for path in sorted(settings.paths.clean.glob("*.parquet")):
        lf = pl.scan_parquet(path)
        out.append((path.stem, lf.select(pl.len()).collect().item(), len(lf.collect_schema())))
    return out


__all__ = ["build_all", "table_inventory", "read_table"]
