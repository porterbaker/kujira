"""Stage 1 validation against CFBD benchmarks, the NFL draft, and held-out plays.

All comparisons use as-of outputs. "End-of-season" values for season S come from the preseason fit of
S+1 (data through S's postseason), restricted to players whose latest season is S. Draft data is used
only here, never as a model input.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from cfb.clean.common import read_table
from cfb.config import Settings
from cfb.player_model.asof import asof_dir
from cfb.player_model.data import ROLE_ID_COL, Dataset, model_dir

CFBD = {  # role -> (CFBD avg metric, total metric, WEPA source, CFBD positions compared)
    "pass": ("avg_ppa_pass", "total_ppa_pass", "wepa_passing", ("QB",)),
    "rush": ("avg_ppa_rush", "total_ppa_rush", "wepa_rushing", ("RB", "QB")),
    "recv": ("avg_ppa_pass", "total_ppa_pass", None, ("WR", "TE", "RB")),
}
MIN_PLAYS = {"pass": 150, "rush": 75, "recv": 40, "defense": 300}
DRAFT_ROLE = {"Quarterback": "pass", "Running Back": "rush", "Wide Receiver": "recv", "Tight End": "recv",
              **{p: "defense" for p in ("Cornerback", "Defensive Tackle", "Safety", "Defensive End", "Outside Linebacker",
                                        "Inside Linebacker", "Defensive Edge", "Linebacker")}}


def load_values(settings: Settings) -> pl.DataFrame:
    root = asof_dir(settings)
    parts = sorted((root / "player_values").glob("*.parquet")) + sorted((root / "defense_values").glob("*.parquet"))
    cols = ["athlete_id", "season", "through_week", "role", "position_group", "last_season", "plays_season",
            "plays_career", "value_mean", "value_sd", "c_mean", "c_sd", "is_projection"]
    return pl.concat([pl.read_parquet(p).select(cols) for p in parts], how="vertical_relaxed")


def end_of_season(values: pl.DataFrame, ds: Dataset) -> pl.DataFrame:
    eos = values.filter((pl.col("through_week") == 0) & (pl.col("last_season") == pl.col("season") - 1)) \
        .with_columns(season=pl.col("last_season")).drop("through_week", "plays_season", "is_projection")
    counts = pl.concat([
        ds.plays.filter(pl.col(c).is_not_null()).group_by(pl.col(c).alias("athlete_id"), "season")
        .agg(pl.len().alias("plays_in_season")).with_columns(role=pl.lit(r)) for r, c in ROLE_ID_COL.items()])
    return eos.join(counts.with_columns(pl.col("season").cast(pl.Int16)), on=["athlete_id", "season", "role"], how="left")


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = pl.Series(a).rank().to_numpy(), pl.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def cfbd_comparison(settings: Settings, eos: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    bench = read_table(settings, "benchmarks")
    rows, worst = [], {}
    for role, (avg, total, wepa_src, positions) in CFBD.items():
        ppa = bench.filter((pl.col("source") == "ppa_player_season") & pl.col("metric").is_in([avg, total])) \
            .pivot(on="metric", index=["entity_id", "season", "position"], values="value", aggregate_function="first") \
            .rename({"entity_id": "athlete_id", avg: "cfbd_avg", total: "cfbd_total"}) \
            .with_columns(cfbd_plays=(pl.col("cfbd_total") / pl.col("cfbd_avg")).round())
        mine = eos.filter((pl.col("role") == role) & (pl.col("plays_in_season") >= MIN_PLAYS[role]))
        j = mine.join(ppa, on=["athlete_id", "season"], how="inner")
        if wepa_src:
            w = bench.filter((pl.col("source") == wepa_src) & (pl.col("metric") == "wepa")) \
                .select(pl.col("entity_id").alias("athlete_id"), "season", pl.col("value").alias("wepa"))
            j = j.join(w, on=["athlete_id", "season"], how="left")
        for pos in positions:
            g = j.filter((pl.col("position") == pos) & pl.col("cfbd_avg").is_not_null())
            if g.height < 20:
                continue
            row = {"role": role, "position": pos, "n": g.height,
                   "pearson_ppa": float(np.corrcoef(g["value_mean"], g["cfbd_avg"])[0, 1]),
                   "spearman_ppa": _rank_corr(g["value_mean"].to_numpy(), g["cfbd_avg"].to_numpy())}
            if wepa_src:
                gw = g.filter(pl.col("wepa").is_not_null())
                row |= {"n_wepa": gw.height,
                        "pearson_wepa": float(np.corrcoef(gw["value_mean"], gw["wepa"])[0, 1]) if gw.height > 20 else None,
                        "spearman_wepa": _rank_corr(gw["value_mean"].to_numpy(), gw["wepa"].to_numpy()) if gw.height > 20 else None}
            rows.append(row)
            if pos == positions[0]:
                ranked = g.with_columns(
                    our_pct=pl.col("value_mean").rank().over("season") / pl.len().over("season"),
                    cfbd_pct=pl.col("cfbd_avg").rank().over("season") / pl.len().over("season"),
                ).with_columns(pct_gap=pl.col("our_pct") - pl.col("cfbd_pct"))
                worst[role] = ranked.sort(pl.col("pct_gap").abs(), descending=True).head(10)
    return pl.DataFrame(rows), worst


def disagreement_context(settings: Settings, ds: Dataset, worst: pl.DataFrame, role: str) -> pl.DataFrame:
    """Data columns that explain a gap: garbage-time share CFBD includes and we exclude, sack share,
    and the average defensive unit faced (our opponent adjustment)."""
    col = ROLE_ID_COL[role]
    keys = worst.select("athlete_id", "season")
    src = "pass_plays" if role in ("pass", "recv") else "rush_plays"
    from cfb.clean.common import scan_table
    allp = scan_table(settings, src).filter(pl.col("model_eligible")).select(
        pl.col(col).alias("athlete_id"), "season", "garbage_time", "defense_id",
        *(["is_sack"] if role == "pass" else [])).collect()
    allp = allp.join(keys, on=["athlete_id", "season"])
    units = pl.concat([pl.read_parquet(p) for p in sorted((asof_dir(settings) / "team_units").glob("*week=00.parquet"))])
    agg = allp.group_by("athlete_id", "season").agg(
        pl.col("garbage_time").mean().alias("garbage_share"),
        *([pl.col("is_sack").mean().alias("sack_share")] if role == "pass" else []))
    # Opponent defensive unit at the end of that season (from the next preseason fit).
    opp = allp.join(units.select(pl.col("team_id").alias("defense_id"), (pl.col("season") - 1).cast(pl.Int16).alias("season"),
                                 "D_mean"), on=["defense_id", "season"], how="left") \
        .group_by("athlete_id", "season").agg(pl.col("D_mean").mean().alias("avg_opp_D"))
    return worst.join(agg, on=["athlete_id", "season"], how="left").join(opp, on=["athlete_id", "season"], how="left")


def draft_check(settings: Settings, eos: pl.DataFrame) -> pl.DataFrame:
    """Within draft position: Spearman(career effect, -overall pick) among drafted players, and AUC of the
    career effect for drafted vs undrafted final-season players in the same role."""
    draft = read_table(settings, "draft").filter(pl.col("college_athlete_id").is_not_null()).select(
        pl.col("college_athlete_id").alias("athlete_id"), (pl.col("year") - 1).cast(pl.Int16).alias("season"),
        "overall", "position")
    players = read_table(settings, "players").select("athlete_id", "last_roster_season")
    rows = []
    for pos, role in DRAFT_ROLE.items():
        e = eos.filter((pl.col("role") == role) & (pl.col("plays_career") >= MIN_PLAYS[role]))
        drafted = e.join(draft.filter(pl.col("position") == pos), on=["athlete_id", "season"], how="inner")
        if drafted.height < 15:
            continue
        rho = _rank_corr(drafted["c_mean"].to_numpy(), -drafted["overall"].to_numpy())
        groups = set(drafted["position_group"].drop_nulls().unique().to_list())
        pool = e.join(players, on="athlete_id").filter(
            (pl.col("last_roster_season") == pl.col("season")) & pl.col("position_group").is_in(list(groups))
            & (pl.col("season") <= 2024))
        pool = pool.join(draft.select("athlete_id", "season", pl.lit(True).alias("drafted")), on=["athlete_id", "season"],
                         how="left").with_columns(pl.col("drafted").fill_null(False))
        auc = None
        if pool["drafted"].sum() >= 10 and (~pool["drafted"]).sum() >= 10:
            r = pool["c_mean"].rank().to_numpy()
            n1, n0 = int(pool["drafted"].sum()), int((~pool["drafted"]).sum())
            auc = float((r[pool["drafted"].to_numpy()].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        rows.append({"draft_position": pos, "role": role, "drafted_n": drafted.height,
                     "spearman_c_vs_pick": rho, "pool_n": pool.height, "auc_drafted": auc})
    return pl.DataFrame(rows)


def heldout_summary(settings: Settings, first_season: int | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """By season (burn-in seasons flagged) and by week bucket (seasons from `first_season` only)."""
    first = settings.model.eval_first_season if first_season is None else first_season
    fits = pl.concat([pl.read_parquet(p) for p in sorted((asof_dir(settings) / "fits").glob("*.parquet"))]) \
        .filter(pl.col("heldout_plays") > 0)
    w = pl.col("heldout_plays")
    by_season = fits.group_by("season").agg(
        w.sum().alias("plays"), ((pl.col("lpd_model") * w).sum() / w.sum()).alias("lpd_model"),
        ((pl.col("lpd_context") * w).sum() / w.sum()).alias("lpd_context"),
        (pl.col("lpd_model") > pl.col("lpd_context")).mean().alias("weeks_model_better"),
    ).with_columns(gain=pl.col("lpd_model") - pl.col("lpd_context"), burn_in=pl.col("season") < first).sort("season")
    bucket = pl.when(pl.col("through_week") + 1 <= 3).then(pl.lit("weeks 1-3")) \
        .when(pl.col("through_week") + 1 <= 8).then(pl.lit("weeks 4-8")).otherwise(pl.lit("weeks 9+"))
    by_bucket = fits.filter(pl.col("season") >= first).with_columns(bucket=bucket).group_by("bucket").agg(
        w.sum().alias("plays"), ((pl.col("lpd_model") * w).sum() / w.sum()).alias("lpd_model"),
        ((pl.col("lpd_context") * w).sum() / w.sum()).alias("lpd_context"),
    ).with_columns(gain=pl.col("lpd_model") - pl.col("lpd_context")).sort("bucket")
    return by_season, by_bucket


def heldout_by_play_group(settings: Settings, ds: Dataset, first_season: int | None = None) -> pl.DataFrame:
    """Held-out gain per season split by play group: rushes, passes with a receiver term, and passes without
    one. The target-known indicator (kappa) is in the model but not the context baseline, and whether a pass
    has a recorded target depends on its outcome (sacks, throwaways and untargeted incompletions have none;
    2021-2024 incompletions mostly lack targets), so the pass groups carry outcome information."""
    from cfb.player_model.model import make_arrays

    first = settings.model.eval_first_season if first_season is None else first_season
    a = make_arrays(ds, ds.plays, compact=False)
    known = (a.target_term * a.roles["recv"].play_known) > 0
    group = np.where(a.kind == 1, "rush", np.where(known, "pass_target_known", "pass_no_target"))
    attrs = pl.DataFrame({"play_idx": np.arange(len(a.y)), "group": group})
    held = pl.concat([pl.read_parquet(p) for p in sorted((asof_dir(settings) / "heldout").glob("*.parquet"))])
    held = held.filter(pl.col("season") >= first).join(attrs, on="play_idx")
    gain = held.group_by("season", "group").agg((pl.col("lpd_model") - pl.col("lpd_context")).mean().alias("gain"))
    share = held.group_by("season").agg((pl.col("group") == "pass_no_target").mean().alias("share_pass_no_target"))
    return gain.pivot(on="group", index="season", values="gain").join(share, on="season").select(
        "season", "rush", "pass_target_known", "pass_no_target", "share_pass_no_target").sort("season")
