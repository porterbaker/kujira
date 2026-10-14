"""Modeling dataset for Stage 1: one row per modeled play plus fixed entity indices.

Indices are global (every season), so every as-of fit has the same parameter shapes and can be
warm-started from the previous one. A fit through (season S, week W) uses plays from earlier
seasons plus season S weeks <= W; later plays never enter the likelihood.

Missing attribution is never imputed:
- unknown passer / rusher (1-10% of plays): no player term, plus a season-level indicator;
- unknown target (incompletions, mostly 2021-2024): no receiver term. Known-target plays get a
  season-level intercept, which absorbs the fact that known targets skew toward completions in
  the seasons where incompletion targets are missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from cfb.clean.common import read_table, scan_table
from cfb.config import Settings

ROLES = ("pass", "rush", "recv")
ROLE_ID_COL = {"pass": "passer_id", "rush": "rusher_id", "recv": "target_id"}
DIST_BINS = [2, 5, 9, 10, 15]          # <=2, 3-5, 6-9, 10, 11-15, 16+
YTG_BINS = [5, 10, 20, 40, 60, 80]     # <=5, 6-10, 11-20, 21-40, 41-60, 61-80, 81+
FCS_GROUP = "FCS (pooled)"


@dataclass
class Dataset:
    plays: pl.DataFrame
    team_seasons: pl.DataFrame
    conf_groups: pl.DataFrame
    players: dict[str, pl.DataFrame]        # per role: rp_idx, athlete_id, covariates
    player_seasons: dict[str, pl.DataFrame]  # per role: ps_idx, rp_idx, season, pos_idx, level, parent, gap
    player_games: dict[str, pl.DataFrame]    # per role: pg_idx, rp_idx, game_id, season
    positions: dict[str, list[str]]          # per role: position groups (alpha index order)
    seasons: list[int]
    context_columns: list[str]
    transfers: pl.DataFrame                  # athlete_id, first accepted transfer_date

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.plays.write_parquet(directory / "plays.parquet")
        self.team_seasons.write_parquet(directory / "team_seasons.parquet")
        self.conf_groups.write_parquet(directory / "conf_groups.parquet")
        self.transfers.write_parquet(directory / "transfers.parquet")
        for role in ROLES:
            self.players[role].write_parquet(directory / f"players_{role}.parquet")
            self.player_seasons[role].write_parquet(directory / f"player_seasons_{role}.parquet")
            self.player_games[role].write_parquet(directory / f"player_games_{role}.parquet")
        pl.DataFrame({"role": [r for r in ROLES for _ in self.positions[r]],
                      "position_group": [p for r in ROLES for p in self.positions[r]]}).write_parquet(directory / "positions.parquet")
        pl.DataFrame({"season": self.seasons}).write_parquet(directory / "seasons.parquet")
        pl.DataFrame({"column": self.context_columns}).write_parquet(directory / "context_columns.parquet")

    @classmethod
    def load(cls, directory: Path) -> Dataset:
        pos = pl.read_parquet(directory / "positions.parquet")
        return cls(
            plays=pl.read_parquet(directory / "plays.parquet"),
            team_seasons=pl.read_parquet(directory / "team_seasons.parquet"),
            conf_groups=pl.read_parquet(directory / "conf_groups.parquet"),
            players={r: pl.read_parquet(directory / f"players_{r}.parquet") for r in ROLES},
            player_seasons={r: pl.read_parquet(directory / f"player_seasons_{r}.parquet") for r in ROLES},
            player_games={r: pl.read_parquet(directory / f"player_games_{r}.parquet") for r in ROLES},
            positions={r: pos.filter(pl.col("role") == r)["position_group"].to_list() for r in ROLES},
            seasons=pl.read_parquet(directory / "seasons.parquet")["season"].to_list(),
            context_columns=pl.read_parquet(directory / "context_columns.parquet")["column"].to_list(),
            transfers=pl.read_parquet(directory / "transfers.parquet"),
        )


def model_dir(settings: Settings) -> Path:
    return settings.paths.data / "model"


def _week_ordinal(games: pl.DataFrame) -> pl.DataFrame:
    """Regular weeks keep their number; all postseason games become last regular week + 1."""
    last_regular = games.filter(pl.col("season_type") == "regular").group_by("season").agg(pl.col("week").max().alias("_last"))
    return games.join(last_regular, on="season", how="left").select(
        "game_id", "start_date",
        week_ord=pl.when(pl.col("season_type") == "regular").then(pl.col("week").cast(pl.Int16))
        .otherwise(pl.col("_last").cast(pl.Int16) + 1))


def _bucket(x: pl.Expr, edges: list[int]) -> pl.Expr:
    """Right-closed buckets: 0 for x <= edges[0], ..., len(edges) for x > edges[-1]."""
    return pl.sum_horizontal(*[(x > e).cast(pl.Int8) for e in edges])


def _context(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    home = pl.when(pl.col("neutral_site").fill_null(False)).then(pl.lit("neutral")) \
        .when(pl.col("is_home_offense")).then(pl.lit("home")).otherwise(pl.lit("away"))
    df = df.with_columns(
        down_c=pl.col("down").clip(1, 4),
        dist_b=_bucket(pl.col("distance").clip(0, None), DIST_BINS),
        ytg_b=_bucket(pl.col("yards_to_goal").clip(1, 99), YTG_BINS),
        venue_c=home,
    )
    cols, exprs = [], []
    for d in (2, 3, 4):
        cols.append(f"down_{d}")
        exprs.append((pl.col("down_c") == d).cast(pl.Float32).alias(f"down_{d}"))
    for b in range(1, len(DIST_BINS) + 1):
        cols.append(f"dist_{b}")
        exprs.append((pl.col("dist_b") == b).cast(pl.Float32).alias(f"dist_{b}"))
    for b in range(1, len(YTG_BINS) + 1):
        cols.append(f"ytg_{b}")
        exprs.append((pl.col("ytg_b") == b).cast(pl.Float32).alias(f"ytg_{b}"))
    for v in ("away", "neutral"):
        cols.append(f"venue_{v}")
        exprs.append((pl.col("venue_c") == v).cast(pl.Float32).alias(f"venue_{v}"))
    return df.with_columns(exprs).drop("down_c", "dist_b", "ytg_b", "venue_c"), cols


def _levels(df: pl.DataFrame, entity: str, season: str = "season") -> pl.DataFrame:
    """Order each entity's seasons: level (0 = first), parent row index, and gap in seasons."""
    df = df.sort(entity, season).with_row_index("_row")
    return df.with_columns(
        level=pl.col(season).rank("ordinal").over(entity).cast(pl.Int32) - 1,
        parent=pl.col("_row").shift(1).over(entity).cast(pl.Int64).fill_null(-1),
        gap=(pl.col(season) - pl.col(season).shift(1).over(entity)).fill_null(0).cast(pl.Int32),
    ).drop("_row")


def build_dataset(settings: Settings, exclude_garbage_time: bool = True) -> Dataset:
    games = read_table(settings, "games")
    weeks = _week_ordinal(games)
    state = scan_table(settings, "plays").select("play_id", "is_home_offense", "neutral_site").collect()
    keep = pl.col("model_eligible") & (~pl.col("garbage_time") if exclude_garbage_time else pl.lit(True))
    base_cols = ["play_id", "game_id", "season", "offense_id", "defense_id", "down", "distance", "yards_to_goal", "ppa"]
    pas = scan_table(settings, "pass_plays").filter(keep).select(
        *base_cols, "is_sack", "passer_id", "target_id", "outcome").collect()
    rus = scan_table(settings, "rush_plays").filter(keep).select(*base_cols, "rusher_id").collect()
    plays = pl.concat([
        pas.with_columns(play_kind=pl.lit(0, pl.Int8), rusher_id=pl.lit(None, pl.Int64)),
        rus.with_columns(play_kind=pl.lit(1, pl.Int8), is_sack=pl.lit(False), passer_id=pl.lit(None, pl.Int64),
                         target_id=pl.lit(None, pl.Int64), outcome=pl.lit(None, pl.String)),
    ], how="diagonal").join(state, on="play_id", how="left").join(weeks, on="game_id", how="left")
    plays = plays.filter(pl.col("offense_id").is_not_null() & pl.col("defense_id").is_not_null())
    plays, ctx_cols = _context(plays)
    seasons = sorted(plays["season"].unique().to_list())
    plays = plays.with_columns(season_idx=pl.col("season").replace_strict(seasons, list(range(len(seasons))), return_dtype=pl.Int32))

    # Team-seasons and conference groups (FCS teams pooled into one group per season).
    teams = read_table(settings, "teams").select("team_id", "season", "conference", "classification")
    ts = pl.concat([plays.select(pl.col("offense_id").alias("team_id"), "season"),
                    plays.select(pl.col("defense_id").alias("team_id"), "season")]).unique()
    ts = ts.join(teams, on=["team_id", "season"], how="left").with_columns(
        is_fcs=pl.col("classification").fill_null("unknown") != "fbs",
    ).with_columns(conf_group=pl.when(pl.col("is_fcs")).then(pl.lit(FCS_GROUP))
                   .otherwise(pl.col("conference").fill_null("FBS Independents")))
    ts = _levels(ts, "team_id").with_row_index("ts_idx")
    cg = ts.select("conf_group", "season").unique().sort("season", "conf_group").with_row_index("cg_idx")
    ts = ts.join(cg, on=["conf_group", "season"], how="left").sort("ts_idx")
    ts_map = ts.select("team_id", "season", "ts_idx")
    plays = plays.join(ts_map.rename({"team_id": "offense_id", "ts_idx": "off_ts"}), on=["offense_id", "season"]) \
        .join(ts_map.rename({"team_id": "defense_id", "ts_idx": "def_ts"}), on=["defense_id", "season"])

    # Players per role.
    roster = read_table(settings, "player_seasons").select("athlete_id", "season", "position_group").unique(["athlete_id", "season"])
    plist = read_table(settings, "players").select("athlete_id", pl.col("position_group").alias("career_group"))
    covariates = _covariates(settings)
    players, pseasons, pgames, positions = {}, {}, {}, {}
    for role in ROLES:
        col = ROLE_ID_COL[role]
        obs = plays.filter(pl.col(col).is_not_null()).select(pl.col(col).alias("athlete_id"), "season", "game_id")
        rp = obs.select("athlete_id").unique().sort("athlete_id").with_row_index("rp_idx")
        rp = rp.join(covariates, on="athlete_id", how="left").sort("rp_idx")
        ps = obs.select("athlete_id", "season").unique().join(roster, on=["athlete_id", "season"], how="left") \
            .join(plist, on="athlete_id", how="left") \
            .with_columns(position_group=pl.coalesce("position_group", "career_group").fill_null("unknown")) \
            .drop("career_group").join(rp.select("athlete_id", "rp_idx"), on="athlete_id")
        ps = _levels(ps, "athlete_id").with_row_index("ps_idx")
        positions[role] = sorted(ps["position_group"].unique().to_list())
        ps = ps.with_columns(pos_idx=pl.col("position_group").replace_strict(positions[role], list(range(len(positions[role]))),
                                                                             return_dtype=pl.Int32))
        pg = obs.select("athlete_id", "game_id", "season").unique().sort("athlete_id", "season", "game_id") \
            .join(rp.select("athlete_id", "rp_idx"), on="athlete_id").with_row_index("pg_idx")
        plays = plays.join(rp.select(pl.col("athlete_id").alias(col), pl.col("rp_idx").alias(f"{role}_rp")), on=col, how="left") \
            .join(ps.select(pl.col("athlete_id").alias(col), "season", pl.col("ps_idx").alias(f"{role}_ps"),
                            pl.col("pos_idx").alias(f"{role}_pos")), on=[col, "season"], how="left") \
            .join(pg.select(pl.col("athlete_id").alias(col), "game_id", pl.col("pg_idx").alias(f"{role}_pg")),
                  on=[col, "game_id"], how="left")
        players[role], pseasons[role], pgames[role] = rp, ps, pg

    plays = plays.with_columns(
        has_target_term=(pl.col("play_kind") == 0) & ~pl.col("is_sack"),
        target_known=pl.col("recv_rp").is_not_null(),
        passer_known=pl.col("pass_rp").is_not_null(),
        rusher_known=pl.col("rush_rp").is_not_null(),
    ).sort("season", "week_ord", "game_id", "play_id")
    transfers = read_table(settings, "transfer_matches").filter(pl.col("accepted")).join(
        read_table(settings, "transfers").select("transfer_key", "transfer_date", "season"), on="transfer_key") \
        .group_by("athlete_id").agg(pl.col("transfer_date").min().alias("first_transfer_date"))
    return Dataset(plays, ts, cg, players, pseasons, pgames, positions, seasons, ctx_cols, transfers)


COVARIATES = ("recruit_rating_z", "recruit_stars_z", "recruit_missing", "height_z", "weight_z")


def _covariates(settings: Settings) -> pl.DataFrame:
    """Pre-college and physical covariates for the career-effect prior. Missing -> 0 with a flag."""
    rec = read_table(settings, "recruits").select("recruit_id", "rating", "stars")
    links = read_table(settings, "recruit_links").select("athlete_id", "recruit_id")
    best = links.join(rec, on="recruit_id").group_by("athlete_id").agg(pl.col("rating").max(), pl.col("stars").max())
    r_mu, r_sd = rec["rating"].mean(), rec["rating"].std()
    s_mu, s_sd = rec["stars"].cast(pl.Float64).mean(), rec["stars"].cast(pl.Float64).std()
    ps = read_table(settings, "player_seasons").sort("season")
    body = ps.group_by("athlete_id").agg(pl.col("height").drop_nulls().first(), pl.col("weight").drop_nulls().first(),
                                         pl.col("position_group").drop_nulls().mode().first().alias("pos"))
    body = body.with_columns(
        height_z=((pl.col("height") - pl.col("height").mean().over("pos")) / pl.col("height").std().over("pos")),
        weight_z=((pl.col("weight") - pl.col("weight").mean().over("pos")) / pl.col("weight").std().over("pos")),
    )
    out = body.join(best, on="athlete_id", how="full", coalesce=True).with_columns(
        recruit_missing=pl.col("rating").is_null().cast(pl.Float32),
        recruit_rating_z=((pl.col("rating") - r_mu) / r_sd).fill_null(0.0).cast(pl.Float32),
        recruit_stars_z=((pl.col("stars").cast(pl.Float64) - s_mu) / s_sd).fill_null(0.0).cast(pl.Float32),
        height_z=pl.col("height_z").fill_null(0.0).fill_nan(0.0).clip(-4, 4).cast(pl.Float32),
        weight_z=pl.col("weight_z").fill_null(0.0).fill_nan(0.0).clip(-4, 4).cast(pl.Float32),
    )
    return out.select("athlete_id", *COVARIATES)


def cutoff_date(games: pl.DataFrame, season: int, through_week: int) -> datetime:
    """When week through_week+1 of `season` starts: the as-of date for transfer covariates."""
    w = _week_ordinal(games)
    later = w.join(games.select("game_id", "season"), on="game_id").filter(
        (pl.col("season") == season) & (pl.col("week_ord") > through_week))
    if later.height:
        return later["start_date"].min()
    return w.join(games.select("game_id", "season"), on="game_id").filter(pl.col("season") == season)["start_date"].max()


def transfer_indicator(ds: Dataset, role: str, as_of: datetime | None) -> np.ndarray:
    """1 if the player had an accepted transfer before `as_of` (None = no cutoff)."""
    t = ds.players[role].select("rp_idx", "athlete_id").join(ds.transfers, on="athlete_id", how="left")
    flag = t["first_transfer_date"].is_not_null()
    if as_of is not None:
        flag = flag & (t["first_transfer_date"] < as_of).fill_null(False)
    return flag.cast(pl.Float32).to_numpy()
