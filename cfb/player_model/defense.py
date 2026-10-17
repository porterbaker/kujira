"""Defender model from box scores (2016+; 2014-15 box scores have no defensive category).

Per defender-game counts of disruptive events, with exposure = the opponent's scrimmage plays:
    count[e] ~ Poisson(exposure * exp(alpha_e[position] + c_e[player] + s_e[player, season]))
for e in sacks, non-sack TFL (box TFL includes sacks: TFL >= SACKS in 99.9% of rows), passes defended
(PD excludes interceptions: PD = 0 in 72% of interception games), interceptions, QB hurries.
c ~ N(X gamma, tau_c) with the offensive model's recruit/size covariates is the level entering the first
season; s is the random-walk change since then (s[first] = 0, the same identification as the offense).

Value per team defensive play = -sum_e (rate_e - position-average rate_e) * delta_e, where delta_e is
the measured EPA of that event relative to an average play of its kind (as of the fit). Hurries have
no play-level record, so they get a rate but no EPA value. Tackles are not modeled as value (they are
recorded regardless of outcome); Stage 2 uses them only as a playing-time signal.

tau_c (per event) is held at the mean of single-season NUTS fits on the three seasons before S, the
same rule as the offensive scales; 2016, the first box-score season, is the burn-in and uses its own.

Only defender-games that appear in a box score are observed (a defender appears once he records any
stat), so rates are per game played-with-a-stat; the hierarchical prior shrinks thin samples.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
from numpyro.infer import Predictive

from cfb.clean.common import read_table, scan_table
from cfb.config import Settings
from cfb.player_model.data import COVARIATES, _covariates, _levels, _week_ordinal, cutoff_date, model_dir
from cfb.player_model.fit import run_svi, sample_guide
from cfb.player_model.model import Levels, _chain, _hyper, _levels as _level_groups, _levels_data

EVENTS = ("sacks", "tfl_nonsack", "pd", "ints", "qb_hurries")
VALUED = ("sacks", "tfl_nonsack", "pd", "ints")
FIRST_SEASON = 2016


@dataclass
class DefenseData:
    rows: pl.DataFrame          # one row per defender-game
    defenders: pl.DataFrame     # dp_idx, athlete_id, covariates
    seasons: pl.DataFrame       # ds_idx, dp_idx, athlete_id, season, position_group, pos_idx, parent, gap
    positions: list[str]


def build_defense_data(settings: Settings) -> DefenseData:
    games = read_table(settings, "games")
    weeks = _week_ordinal(games)
    fbs = read_table(settings, "teams").filter(pl.col("classification") == "fbs").select("team_id", "season")
    box = scan_table(settings, "player_game_box").filter(
        (pl.col("season") >= FIRST_SEASON) & pl.col("athlete_id").is_not_null() & (pl.col("athlete_id") > 0)
        & (((pl.col("category") == "defensive") & pl.col("stat_name").is_in(["SACKS", "TFL", "PD", "QB HUR", "TOT"]))
           | ((pl.col("category") == "interceptions") & (pl.col("stat_name") == "INT")))
    ).collect()
    wide = box.group_by("game_id", "athlete_id", "team_id", "season").agg(
        *[pl.col("value").filter(pl.col("stat_name") == k).sum().alias(v) for k, v in
          (("SACKS", "sacks"), ("TFL", "tfl"), ("PD", "pd"), ("QB HUR", "qb_hurries"), ("TOT", "tackles"))],
        pl.col("value").filter(pl.col("category") == "interceptions").sum().alias("ints"),
    ).with_columns(tfl_nonsack=(pl.col("tfl") - pl.col("sacks")).clip(0, None)).join(fbs, on=["team_id", "season"])
    exposure = scan_table(settings, "plays").filter(pl.col("category").is_in(["pass", "sack", "rush"])) \
        .group_by("game_id", "defense_id").agg(pl.len().alias("exposure")).collect() \
        .rename({"defense_id": "team_id"})
    rows = wide.join(exposure, on=["game_id", "team_id"]).join(weeks, on="game_id").filter(pl.col("exposure") > 0)

    roster = read_table(settings, "player_seasons").select("athlete_id", "season", "position_group").unique(["athlete_id", "season"])
    career = read_table(settings, "players").select("athlete_id", pl.col("position_group").alias("career_group"))
    dp = rows.select("athlete_id").unique().sort("athlete_id").with_row_index("dp_idx") \
        .join(_covariates(settings), on="athlete_id", how="left") \
        .with_columns([pl.col(c).fill_null(0.0) for c in COVARIATES]).sort("dp_idx")
    ds = rows.select("athlete_id", "season").unique().join(roster, on=["athlete_id", "season"], how="left") \
        .join(career, on="athlete_id", how="left") \
        .with_columns(position_group=pl.coalesce("position_group", "career_group").fill_null("unknown")).drop("career_group") \
        .join(dp.select("athlete_id", "dp_idx"), on="athlete_id")
    ds = _levels(ds, "athlete_id").with_row_index("ds_idx")
    positions = sorted(ds["position_group"].unique().to_list())
    ds = ds.with_columns(pos_idx=pl.col("position_group").replace_strict(positions, list(range(len(positions))),
                                                                         return_dtype=pl.Int32))
    rows = rows.join(dp.select("athlete_id", "dp_idx"), on="athlete_id") \
        .join(ds.select("athlete_id", "season", "ds_idx", "pos_idx"), on=["athlete_id", "season"]) \
        .sort("season", "week_ord", "game_id", "athlete_id")
    return DefenseData(rows, dp, ds, positions)


def event_values(settings: Settings, through: pl.Expr | None = None) -> dict[str, float]:
    """EPA of each event relative to an average play of its kind, from plays matching `through`."""
    p = scan_table(settings, "plays").filter(pl.col("model_eligible"))
    if through is not None:
        p = p.filter(through)
    p = p.select("category", "play_type", "yards_gained", "ppa").collect()
    drop = p.filter(pl.col("category").is_in(["pass", "sack"]))
    att = p.filter(pl.col("category") == "pass")
    rush = p.filter(pl.col("category") == "rush")
    ints = att.filter(pl.col("play_type").is_in(["Pass Interception Return", "Interception Return Touchdown",
                                                 "Interception", "Pass Interception"]))
    inc = att.filter(pl.col("play_type") == "Pass Incompletion")
    tfl = rush.filter(pl.col("yards_gained") < 0)
    return {
        "sacks": float(drop.filter(pl.col("category") == "sack")["ppa"].mean() - drop["ppa"].mean()),
        "ints": float(ints["ppa"].mean() - att["ppa"].mean()),
        "pd": float(inc["ppa"].mean() - att["ppa"].mean()),
        "tfl_nonsack": float(tfl["ppa"].mean() - rush["ppa"].mean()),
    }


@dataclass(frozen=True)
class DefStatic:
    n_rows: int
    n_dp: int
    n_ds: int
    n_pos: int
    n_cov: int
    n_levels: int = 2  # season-chain depth; 1 = every defender-season is a first season (s = 0)


def to_device(dd: DefenseData, mask: np.ndarray) -> tuple[DefStatic, dict, Levels]:
    parent = dd.seasons["parent"].to_numpy()
    levels = _level_groups(parent, dd.seasons["gap"].to_numpy(), np.argsort(dd.seasons["season"].to_numpy(), kind="stable"))
    st = DefStatic(dd.rows.height, dd.defenders.height, dd.seasons.height, len(dd.positions), len(COVARIATES),
                   len(levels.idx))
    d = {
        "counts": jnp.asarray(dd.rows.select(EVENTS).to_numpy().astype(np.float32)),
        "exposure": jnp.asarray(dd.rows["exposure"].to_numpy().astype(np.float32)),
        "dp": jnp.asarray(dd.rows["dp_idx"].to_numpy()), "ds": jnp.asarray(dd.rows["ds_idx"].to_numpy()),
        "pos": jnp.asarray(dd.rows["pos_idx"].to_numpy()),
        "X": jnp.asarray(dd.defenders.select(COVARIATES).to_numpy().astype(np.float32)),
        "ds_dp": jnp.asarray(dd.seasons["dp_idx"].to_numpy()),
        "mask": jnp.asarray(mask), "levels": _levels_data(levels),
    }
    return st, d, levels


def defense_model(st: DefStatic, d: dict) -> None:
    E = len(EVENTS)
    alpha = numpyro.sample("alpha", dist.Normal(-4.0, 2.0).expand([E, st.n_pos]).to_event(2))
    gamma = numpyro.sample("gamma", dist.Normal(0, 0.3).expand([E, st.n_cov]).to_event(2))
    tau_c = _hyper("tau_c", dist.HalfNormal(0.5).expand([E]).to_event(1), d)
    tau_s = _hyper("tau_s", dist.HalfNormal(0.3).expand([E]).to_event(1), d)
    c_z = numpyro.sample("c_z", dist.Normal(0, 1).expand([E, st.n_dp]).to_event(2))
    c = numpyro.deterministic("c", d["X"] @ gamma.T + (tau_c[:, None] * c_z).T)          # (n_dp, E)
    if st.n_levels > 1:
        s_z = numpyro.sample("s_z", dist.Normal(0, 1).expand([E, st.n_ds]).to_event(2))
        s = numpyro.deterministic("s", jnp.stack([_chain(s_z[e], d["levels"], 0.0, tau_s[e]) for e in range(E)], 1))
    else:
        s = numpyro.deterministic("s", jnp.zeros((st.n_ds, E)))
    log_rate = alpha.T[d["pos"]] + c[d["dp"]] + s[d["ds"]]                                 # (n_rows, E)
    # Poisson kernel y*log(lambda) - lambda: box scores credit half-sacks (y = 0.5), which a Poisson pmf
    # rejects; the dropped term (log y!) does not depend on lambda, so the fit is unaffected.
    log_lam = jnp.log(d["exposure"])[:, None] + log_rate
    ll = (d["counts"] * log_lam - jnp.exp(log_lam)).sum(-1)
    numpyro.factor("y", jnp.where(d["mask"], ll, 0.0).sum())


def defender_values(dd: DefenseData, draws: dict[str, np.ndarray], deltas: dict[str, float], mask: np.ndarray,
                    season: int, through_week: int, tau_s: np.ndarray) -> pl.DataFrame:
    inc = dd.rows.filter(pl.Series(mask))
    counts = inc.group_by("ds_idx").agg(pl.len().alias("games"), pl.col("exposure").sum().alias("exposure"))
    ds = dd.seasons.join(counts, on="ds_idx")
    latest = ds.sort("season").group_by("athlete_id").agg(
        pl.col("ds_idx").last(), pl.col("dp_idx").last(), pl.col("pos_idx").last(), pl.col("position_group").last(),
        pl.col("season").last().alias("last_season"),
        pl.col("exposure").filter(pl.col("season") == season).sum().alias("plays_season"),
        pl.col("exposure").sum().alias("plays_career")).filter(pl.col("last_season") >= season - 1)
    from cfb.player_model.asof import VALUE_SCHEMA
    schema = {**VALUE_SCHEMA, **{f"rate_{e}": pl.Float64 for e in EVENTS}}
    if latest.is_empty():
        return pl.DataFrame(schema=schema)
    dp_i, ds_i, pos_i = latest["dp_idx"].to_numpy(), latest["ds_idx"].to_numpy(), latest["pos_idx"].to_numpy()
    alpha = draws["alpha"]                                   # (draws, E, n_pos)
    eff = draws["c"][:, dp_i, :] + draws["s"][:, ds_i, :]    # (draws, n, E)
    gap = (season - latest["last_season"].to_numpy()).clip(0)
    if gap.any():  # project to season S: add the season random-walk innovation, per draw
        rng = np.random.default_rng(season * 100 + through_week)
        eff = eff + rng.standard_normal(eff.shape) * tau_s[:, None, :] * np.sqrt(gap)[None, :, None]
    base = np.transpose(alpha[:, :, pos_i], (0, 2, 1))       # (draws, n, E)
    rate, avg = np.exp(base + eff), np.exp(base)
    w = np.array([-deltas.get(e, 0.0) if e in VALUED else 0.0 for e in EVENTS])
    value = ((rate - avg) * w).sum(-1)                        # EPA per team defensive play vs position average
    career = ((np.exp(base + draws["c"][:, dp_i, :]) - avg) * w).sum(-1)  # career effect alone (s = 0)
    out = latest.select("athlete_id", "position_group", "last_season", "plays_season", "plays_career").with_columns(
        season=pl.lit(season, pl.Int16), through_week=pl.lit(through_week, pl.Int16), role=pl.lit("defense"),
        value_mean=value.mean(0), value_sd=value.std(0), c_mean=career.mean(0), c_sd=career.std(0),
        is_projection=pl.Series(gap > 0))
    rates = {f"rate_{e}": rate[:, :, i].mean(0) for i, e in enumerate(EVENTS)}
    return out.with_columns(**{k: pl.Series(v) for k, v in rates.items()}).cast(schema)


def run_defense_asof(settings: Settings, plan: list[tuple[int, int]], *, cold_steps: int = 6000, warm_steps: int = 500,
                     lr: float = 0.01, draws: int = 100, log: Callable[[str], None] = print) -> None:
    out = model_dir(settings) / "asof" / "defense_values"
    out.mkdir(parents=True, exist_ok=True)
    dd = build_defense_data(settings)
    games = read_table(settings, "games")
    weeks = _week_ordinal(games).join(games.select("game_id", "season"), on="game_id")
    from cfb.player_model.validate import load_hyper, scales_for

    hyper = load_hyper(settings, hyper_dir(settings))
    params = None
    for season, week in [(s, w) for s, w in plan if s >= FIRST_SEASON]:
        path = out / f"season={season}_week={week:02d}.parquet"
        if path.is_file():
            continue
        _, scales = scales_for(hyper, season, BURN_IN)
        fixed = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in scales.items()}
        started = time.monotonic()
        r = dd.rows
        mask = ((r["season"] < season) | ((r["season"] == season) & (r["week_ord"] <= week))).to_numpy()
        if not mask.any():
            continue
        st, d, _ = to_device(dd, mask)
        d["fixed"] = fixed
        through = (pl.col("season") < season) | ((pl.col("season") == season) & pl.col("game_id").is_in(
            weeks.filter((pl.col("season") == season) & (pl.col("week_ord") <= week))["game_id"].implode()))
        deltas = event_values(settings, through)
        steps = cold_steps if params is None else warm_steps
        guide, params, trace = run_svi(st, d, "normal", steps, lr=lr, init_params=params,
                                       model_fn=defense_model, log_every=steps)
        batches = sample_guide(guide, params, n=draws)
        sampled = {k: [] for k in ("alpha", "c", "s", "tau_s")}
        for i, b in enumerate(batches):
            res = Predictive(functools.partial(defense_model, st), posterior_samples=b, batch_ndims=1,
                             return_sites=["c", "s", "tau_s"])(jax.random.PRNGKey(i), d)
            sampled["c"].append(np.asarray(res["c"]))
            sampled["s"].append(np.asarray(res["s"]))
            sampled["alpha"].append(np.asarray(b["alpha"]))
            sampled["tau_s"].append(np.broadcast_to(np.asarray(res["tau_s"]), (len(b["alpha"]), len(EVENTS))))
        dr = {k: np.concatenate(v, 0) for k, v in sampled.items()}
        vals = defender_values(dd, dr, deltas, mask, season, week, dr["tau_s"])
        vals.with_columns(**{f"delta_{k}": pl.lit(v) for k, v in deltas.items()}).write_parquet(path)
        log(f"  defense {season} wk {week:>2}: {int(mask.sum()):,} defender-games, {steps} steps, "
            f"{time.monotonic() - started:.0f}s, loss {trace[-1][1]:,.0f}")
        jax.clear_caches()  # as in run_asof: compiled fits otherwise accumulate in JAX's cache


# ---------- validation and scales (NUTS vs SVI, as for the offensive model) ----------

BURN_IN = (FIRST_SEASON,)  # the first box-score season uses its own scales, so its fits are never evaluated


def hyper_dir(settings: Settings) -> Path:
    return model_dir(settings) / "hyper_defense"


def _nuts(st: DefStatic, d: dict, chains: int, warmup: int, samples: int, seed: int = 0) -> tuple[dict, dict]:
    """NUTS on the defender model: flat draws of alpha, gamma, tau_c, c and a diagnostics dict."""
    from numpyro.diagnostics import summary

    from cfb.player_model.fit import run_chains

    started = time.monotonic()
    # Deterministic sites (c, s: one value per defender and event) are kept out of the stored draws to bound
    # memory; c is rebuilt from gamma, tau_c and c_z below.
    by_chain, extra = run_chains(functools.partial(defense_model, st), d, chains, warmup, samples,
                                 {"alpha", "gamma", "tau_c", "c_z"}, seed)
    X = np.asarray(d["X"])
    by_chain["c"] = (np.einsum("pk,abek->abpe", X, by_chain["gamma"])
                     + np.swapaxes(by_chain["tau_c"][..., None] * by_chain.pop("c_z"), -1, -2))  # (chain, draw, n_dp, E)
    diag = {"seconds": time.monotonic() - started, "chains": chains, "warmup": warmup, "samples": samples,
            "divergences": int(extra["diverging"].sum()),
            "mean_leapfrog_steps": float(extra["num_steps"].mean()), "sites": {}}
    for site, s in summary(by_chain, group_by_chain=True).items():
        rhat, ess = np.atleast_1d(s["r_hat"]).ravel(), np.atleast_1d(s["n_eff"]).ravel()
        diag["sites"][site] = {"size": int(rhat.size), "max_rhat": float(np.nanmax(rhat)),
                               "share_rhat_gt_1.01": float(np.mean(rhat > 1.01)), "min_ess": float(np.nanmin(ess))}
    flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in by_chain.items()}
    diag["tau_c"] = dict(zip(EVENTS, flat["tau_c"].mean(0).tolist()))
    return flat, diag


def _write_scales(settings: Settings, season: int, diag: dict) -> Path:
    import json

    out = hyper_dir(settings)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"season={season}.json"
    path.write_text(json.dumps({"season": season, "scales": {"tau_c": [diag["tau_c"][e] for e in EVENTS]},
                                "nuts": diag}, indent=1))
    return path


def fit_defense_scales(settings: Settings, seasons: tuple[int, ...], chains: int = 4, warmup: int = 500,
                       samples: int = 500, log: Callable[[str], None] = print) -> None:
    """Single-season defender NUTS for each of `seasons` (2016+) without a scale file yet."""
    todo = [s for s in seasons if s >= FIRST_SEASON and not (hyper_dir(settings) / f"season={s}.json").is_file()]
    if not todo:
        return
    full = build_defense_data(settings)
    for season in todo:
        dd = subset_seasons(full, (season,))
        st, d, _ = to_device(dd, np.ones(dd.rows.height, dtype=bool))
        flat, diag = _nuts(st, d, chains, warmup, samples)
        del flat
        log(f"  defender scales {season}: {dd.rows.height:,} defender-games, {diag['seconds'] / 60:.1f} min, "
            f"divergences {diag['divergences']}, worst R-hat {max(v['max_rhat'] for v in diag['sites'].values()):.3f}")
        _write_scales(settings, season, diag)


def subset_seasons(dd: DefenseData, seasons: tuple[int, ...]) -> DefenseData:
    """Rows and entities of `seasons` only, re-indexed; season chains are cut at the first included season."""
    from cfb.player_model.model import _remap

    rows = dd.rows.filter(pl.col("season").is_in(seasons))
    ds = dd.seasons.filter(pl.col("season").is_in(seasons))
    dp_old = np.unique(rows["dp_idx"].to_numpy())
    dp_map, ds_map = _remap(dp_old, dd.defenders.height), _remap(ds["ds_idx"].to_numpy(), dd.seasons.height)
    parent = ds["parent"].to_numpy()
    parent = np.where(parent >= 0, ds_map[np.maximum(parent, 0)], -1)
    ds = ds.with_columns(ds_idx=pl.Series(np.arange(ds.height), dtype=pl.UInt32),
                         dp_idx=pl.Series(dp_map[ds["dp_idx"].to_numpy()]), parent=pl.Series(parent),
                         gap=pl.Series(np.where(parent >= 0, ds["gap"].to_numpy(), 0)))
    defenders = dd.defenders[dp_old.tolist()].with_columns(dp_idx=pl.Series(np.arange(len(dp_old)), dtype=pl.UInt32))
    rows = rows.with_columns(dp_idx=pl.Series(dp_map[rows["dp_idx"].to_numpy()]),
                             ds_idx=pl.Series(ds_map[rows["ds_idx"].to_numpy()]))
    return DefenseData(rows, defenders, ds, dd.positions)


def _c_table(draws_c: np.ndarray, method: str) -> pl.DataFrame:
    """Posterior mean/sd of c per (defender, event) from (draws, n_dp, E)."""
    n = draws_c.shape[1]
    return pl.concat([pl.DataFrame({"dp_idx": np.arange(n), "event": e, "mean": draws_c[:, :, i].mean(0),
                                    "sd": draws_c[:, :, i].std(0)}) for i, e in enumerate(EVENTS)]
                     ).with_columns(method=pl.lit(method))


def validate_defense(settings: Settings, season: int = FIRST_SEASON, chains: int = 4, warmup: int = 500,
                     samples: int = 500, svi_steps: int = 10000, log: Callable[[str], None] = print) -> Path:
    """NUTS on one season of box scores, then SVI with tau_c free and held at the NUTS mean. Also
    writes that season's defender scale file."""
    import json

    out = model_dir(settings) / "validation"
    out.mkdir(parents=True, exist_ok=True)
    dd = subset_seasons(build_defense_data(settings), (season,))
    st, d, _ = to_device(dd, np.ones(dd.rows.height, dtype=bool))
    log(f"defender validation {season}: {dd.rows.height:,} defender-games, {st.n_dp:,} defenders")
    flat, diag = _nuts(st, d, chains, warmup, samples)
    diag["season"] = season
    tau_c = flat["tau_c"].mean(0)
    log(f"  NUTS done in {diag['seconds'] / 60:.1f} min; divergences {diag['divergences']}; "
        f"worst R-hat {max(v['max_rhat'] for v in diag['sites'].values()):.3f}")
    _write_scales(settings, season, diag)
    tables = [_c_table(flat["c"], "nuts")]
    del flat

    for method, fixed in (("svi_normal", {}), ("svi_fixed", {"tau_c": tau_c})):
        dv = {**d, "fixed": {k: jnp.asarray(v, dtype=jnp.float32) for k, v in fixed.items()}}
        guide, params, trace = run_svi(st, dv, "normal", svi_steps, lr=0.01, model_fn=defense_model,
                                       log=log, log_every=svi_steps // 2)
        batches = sample_guide(guide, params, n=300)
        cs = [np.asarray(Predictive(functools.partial(defense_model, st), posterior_samples=b, batch_ndims=1,
                                    return_sites=["c"])(jax.random.PRNGKey(i), dv)["c"]) for i, b in enumerate(batches)]
        tables.append(_c_table(np.concatenate(cs, 0), method))
        if not fixed:
            diag["svi_normal_tau_c"] = dict(zip(EVENTS, np.concatenate([np.asarray(b["tau_c"]) for b in batches]).mean(0).tolist()))
        diag[method] = {"trace": trace}
    eff = pl.concat(tables)
    nuts = eff.filter(pl.col("method") == "nuts").select("dp_idx", "event", pl.col("mean").alias("m0"), pl.col("sd").alias("s0"))
    rows = []
    for (method, event), g in eff.filter(pl.col("method") != "nuts").join(nuts, on=["dp_idx", "event"]).group_by(
            "method", "event", maintain_order=True):
        rows.append({"method": method, "event": event, "corr_mean": float(np.corrcoef(g["mean"], g["m0"])[0, 1]),
                     "spread_ratio": float(g["mean"].std() / g["m0"].std()), "sd_ratio": float((g["sd"] / g["s0"]).median())})
    diag["compare"] = rows
    path = out / f"defense_{season}.json"
    path.write_text(json.dumps(diag, indent=1))
    log(f"wrote {path}")
    return path
