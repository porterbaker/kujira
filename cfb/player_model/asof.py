"""As-of weekly fits: the values used to predict a week-w game come from data through week w-1.

Fit (S, W) includes every play from seasons before S plus season S plays with week_ord <= W.
W = 0 is the preseason fit (all earlier seasons, including their postseasons); postseason games
use the fit through the last regular week. Every fit shares one parameter layout, so each is
warm-started from the previous one. The within-season hierarchical scales are held at the mean of single-season
NUTS fits on the three seasons before S (`cfb player hyper`); scales_for refuses a fit that would see
scales from its own season or later (except the burn-in season, which is never evaluated). Outputs, appended per fit (the run is resumable):

- player_values: (athlete_id, season, through_week, role) posterior mean/sd of c + s, the player's
  EPA per play relative to the position average; players whose latest season is S-1 are projected
  forward with the season random-walk variance added;
- team_units: O and D for every team on season S's schedule (known before the season);
- heldout: log predictive density of the next week's plays vs a context-only baseline.
"""

from __future__ import annotations

import pickle
import time
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from cfb.clean.common import read_table
from cfb.config import Settings
from cfb.player_model.data import ROLES, Dataset, cutoff_date, model_dir, transfer_indicator
from cfb.player_model.fit import effects_from_samples, heldout_loglik, run_svi, sample_guide, subset_plays
from cfb.player_model.model import context_model, make_arrays, to_device
from cfb.player_model.validate import load_hyper, scales_for, step_hyper_dir

EFFECT_SITES = ["O", "D", *(f"{r}_{x}" for r in ROLES for x in ("c", "s", "alpha", "tau_s"))]

VALUE_SCHEMA = {
    "athlete_id": pl.Int64, "position_group": pl.String, "last_season": pl.Int16, "plays_season": pl.UInt32,
    "plays_career": pl.UInt32, "season": pl.Int16, "through_week": pl.Int16, "role": pl.String,
    "value_mean": pl.Float64, "value_sd": pl.Float64, "c_mean": pl.Float64, "c_sd": pl.Float64,
    "is_projection": pl.Boolean,
}


def asof_dir(settings: Settings) -> Path:
    return model_dir(settings) / "asof"


def schedule(ds: Dataset, games: pl.DataFrame, seasons: tuple[int, ...], cadence: int = 1) -> list[tuple[int, int]]:
    """(season, through_week) fits: preseason (0), every `cadence` weeks, and always the last regular week
    played so far (postseason games, week_ord = last regular + 1, are predicted from that fit)."""
    scheduled = dict(games.filter(pl.col("season_type") == "regular").group_by("season")
                     .agg(pl.col("week").max()).iter_rows())
    played = dict(ds.plays.group_by("season").agg(pl.col("week_ord").max()).iter_rows())
    out = []
    for season in seasons:
        if season not in played:
            continue
        last = min(int(scheduled[season]), int(played[season]))
        weeks = sorted({0, last, *range(cadence, last + 1, cadence)})
        out += [(season, w) for w in weeks if not (season == min(ds.seasons) and w == 0)]
    return out


def included_mask(ds: Dataset, season: int, through_week: int) -> np.ndarray:
    p = ds.plays
    return ((p["season"] < season) | ((p["season"] == season) & (p["week_ord"] <= through_week))).to_numpy()


def next_week_idx(ds: Dataset, season: int, through_week: int) -> np.ndarray:
    p = ds.plays
    return np.flatnonzero(((p["season"] == season) & (p["week_ord"] == through_week + 1)).to_numpy())


def player_values(ds: Dataset, a, eff: dict[str, np.ndarray], mask: np.ndarray, season: int,
                  through_week: int) -> pl.DataFrame:
    """Value draws c[p] + s[p, latest season <= S] for players active in S or S-1 through the cutoff."""
    frames = []
    inc = ds.plays.filter(pl.Series(mask))
    for role in ROLES:
        col = f"{role}_ps"
        used = inc.filter(pl.col(col).is_not_null())
        counts = used.group_by(col).agg(pl.len().alias("n")).rename({col: "ps_idx"})
        ps = ds.player_seasons[role].join(counts, on="ps_idx", how="inner")
        latest = ps.sort("season").group_by("athlete_id").agg(
            pl.col("ps_idx").last(), pl.col("rp_idx").last(), pl.col("season").last().alias("last_season"),
            pl.col("position_group").last(), pl.col("n").filter(pl.col("season") == season).sum().alias("plays_season"),
            pl.col("n").sum().alias("plays_career"))
        latest = latest.filter(pl.col("last_season") >= season - 1)
        if latest.is_empty():
            continue
        ps_i, rp_i = latest["ps_idx"].to_numpy(), latest["rp_idx"].to_numpy()
        career = eff[f"{role}_c"][:, rp_i]
        draws = career + eff[f"{role}_s"][:, ps_i]
        gap = (season - latest["last_season"].to_numpy()).clip(0)
        tau_s = eff[f"{role}_tau_s"].mean()
        sd = np.sqrt(draws.var(0) + (tau_s ** 2) * gap)
        frames.append(latest.select("athlete_id", "position_group", "last_season", "plays_season", "plays_career")
                      .with_columns(season=pl.lit(season, pl.Int16), through_week=pl.lit(through_week, pl.Int16),
                                    role=pl.lit(role), value_mean=draws.mean(0), value_sd=sd,
                                    c_mean=career.mean(0), c_sd=career.std(0), is_projection=pl.Series(gap > 0)))
    if not frames:
        return pl.DataFrame(schema=VALUE_SCHEMA)
    return pl.concat(frames).cast({k: v for k, v in VALUE_SCHEMA.items()})


def team_units(a, eff: dict[str, np.ndarray], season: int, through_week: int) -> pl.DataFrame:
    keys = a.keys["team_seasons"]
    rows = keys.with_columns(O_mean=eff["O"].mean(0), O_sd=eff["O"].std(0), D_mean=eff["D"].mean(0), D_sd=eff["D"].std(0))
    return rows.filter(pl.col("season") == season).select(
        "team_id", "season", "conf_group", "is_fcs", "O_mean", "O_sd", "D_mean", "D_sd").with_columns(
        through_week=pl.lit(through_week, pl.Int16))


def run_asof(settings: Settings, seasons: tuple[int, ...] | None = None, *, cadence: int = 1, cold_steps: int = 10000,
             warm_steps: int = 600, lr: float = 0.005, draws: int = 100, guide_kind: str = "normal",
             log: Callable[[str], None] = print) -> None:
    out = asof_dir(settings)
    for sub in ("player_values", "team_units", "heldout", "fits"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    ds = Dataset.load(model_dir(settings))
    games = read_table(settings, "games")
    seasons = seasons or tuple(ds.seasons)
    plan = schedule(ds, games, seasons, cadence)
    a = make_arrays(ds, ds.plays, compact=False)
    st, d = to_device(a)
    hyper = load_hyper(settings)
    step_hyper = load_hyper(settings, step_hyper_dir(settings))
    burn_in = settings.model.burn_in_seasons
    state_path = out / "last_params.pkl"
    params = pickle.loads(state_path.read_bytes()) if state_path.is_file() else None
    ctx_params = None
    log(f"as-of plan: {len(plan)} fits, {st.n_plays:,} plays in the layout; scale fits for {sorted(hyper)}, "
        f"step scale fits for {sorted(step_hyper)}")
    for season, week in plan:
        _, scales = scales_for(hyper, season, burn_in)
        _, steps = scales_for(step_hyper, season, burn_in, first=min(burn_in) + 1)
        d["fixed"] = {k: jnp.float32(v) for k, v in {**scales, **steps}.items()}
        tag = f"season={season}_week={week:02d}"
        if (out / "fits" / f"{tag}.parquet").is_file():
            continue
        started = time.monotonic()
        mask = included_mask(ds, season, week)
        as_of = cutoff_date(games, season, week)
        d = {**d, "mask": jnp.asarray(mask),
             "roles": {r: {**v, "X": v["X"].at[:, -1].set(jnp.asarray(transfer_indicator(ds, r, as_of)))}
                       for r, v in d["roles"].items()}}
        steps = cold_steps if params is None else warm_steps
        guide, params, trace = run_svi(st, d, guide_kind, steps, lr=lr, init_params=params, log_every=steps)
        batches = sample_guide(guide, params, n=draws)
        eff = effects_from_samples(st, d, batches, EFFECT_SITES)
        player_values(ds, a, eff, mask, season, week).write_parquet(out / "player_values" / f"{tag}.parquet")
        team_units(a, eff, season, week).write_parquet(out / "team_units" / f"{tag}.parquet")

        # Held-out: next week's plays under this fit vs a context-only fit on the same cutoff.
        test = next_week_idx(ds, season, week)
        lpd = base = np.array([])
        if len(test):
            st_t, d_t = subset_plays(st, d, test)
            lpd = heldout_loglik(st_t, d_t, batches)
            cguide, ctx_params, _ = run_svi(st, d, "normal", 400 if ctx_params is not None else 2000, lr=0.01,
                                            init_params=ctx_params, model_fn=context_model, log_every=10**9)
            base = heldout_loglik(st_t, d_t, sample_guide(cguide, ctx_params, n=draws), model_fn=context_model)
            pl.DataFrame({"play_idx": test, "lpd_model": lpd, "lpd_context": base}).with_columns(
                season=pl.lit(season), through_week=pl.lit(week)).write_parquet(out / "heldout" / f"{tag}.parquet")
        del batches, eff
        state_path.write_bytes(pickle.dumps(jax.device_get(params)))
        secs = time.monotonic() - started
        pl.DataFrame({"season": [season], "through_week": [week], "plays_used": [int(mask.sum())], "steps": [steps],
                      "final_loss": [trace[-1][1]], "seconds": [secs], "heldout_plays": [len(test)],
                      "lpd_model": [float(lpd.mean()) if len(lpd) else None],
                      "lpd_context": [float(base.mean()) if len(base) else None]},
                     schema_overrides={"lpd_model": pl.Float64, "lpd_context": pl.Float64, "final_loss": pl.Float64,
                                       "seconds": pl.Float64}).write_parquet(out / "fits" / f"{tag}.parquet")
        log(f"  {season} wk {week:>2}: {int(mask.sum()):>9,} plays, {steps} steps, {secs:.0f}s"
            + (f", next-week lpd {lpd.mean():.4f} vs context {base.mean():.4f} ({len(test):,} plays)" if len(test) else ""))
