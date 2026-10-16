"""Validation seasons: NUTS diagnostics, then SVI (AutoNormal) against NUTS, with the hierarchical
scales free or held at the NUTS estimates. The 2019 run also tried AutoLowRankMultivariateNormal; it
matched AutoNormal and fixed nothing, so it is no longer run by default.

Compared quantity: each player-season's value in each role, c[player] + s[player, season]
(EPA per play relative to the position average), and the team-season units O and D.
"""

from __future__ import annotations

import functools
import json
import time
from collections.abc import Callable
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import polars as pl
from numpyro.diagnostics import summary

from cfb.config import Settings
from cfb.player_model.data import ROLES, Dataset, model_dir
from cfb.player_model.fit import effects_from_samples, run_chains, run_svi, sample_guide
from cfb.player_model.model import (SEASON_STEP_SCALES, WITHIN_SEASON_SCALES, ModelArrays, make_arrays, model,
                                    to_device)

HYPER = ("nu_minus_1", "sigma", "O_tau_conf", "O_tau_team0", "O_tau_team", "O_rho", "O_fcs_scale",
         "D_tau_conf", "D_tau_team0", "D_tau_team", "D_rho", "D_fcs_scale",
         *(f"{r}_{p}" for r in ROLES for p in ("tau_c", "tau_s", "tau_g", "alpha_free", "gamma")),
         "intercept", "beta", "kappa_target_known", "iota_passer_unknown", "iota_rusher_unknown")
EFFECT_SITES = ["O", "D", *(f"{r}_{x}" for r in ROLES for x in ("c", "s", "alpha"))]


def validation_dir(settings: Settings) -> Path:
    return model_dir(settings) / "validation"


def effect_table(a: ModelArrays, draws: dict[str, np.ndarray], method: str) -> pl.DataFrame:
    """Posterior mean/sd of every player-season value and team unit from (draws, ...) arrays."""
    frames = []
    for role in ROLES:
        r = a.roles[role]
        value = draws[f"{role}_c"][:, r.ps_rp] + draws[f"{role}_s"]
        keys = a.keys[f"player_seasons_{role}"]
        frames.append(keys.select("athlete_id", "season", "position_group").with_columns(
            role=pl.lit(role), entity=pl.lit("player"), mean=value.mean(0), sd=value.std(0)))
    for unit in ("O", "D"):
        keys = a.keys["team_seasons"]
        frames.append(keys.select(pl.col("team_id").alias("athlete_id"), "season",
                                  pl.lit(None, pl.String).alias("position_group")).with_columns(
            role=pl.lit(unit), entity=pl.lit("team"), mean=draws[unit].mean(0), sd=draws[unit].std(0)))
    return pl.concat(frames).with_columns(method=pl.lit(method))


def run_nuts(a: ModelArrays, chains: int, warmup: int, samples: int, seed: int = 0) -> tuple[dict, dict]:
    st, d = to_device(a)
    started = time.monotonic()
    keep, extra = run_chains(functools.partial(model, st), d, chains, warmup, samples,
                             set(HYPER) | set(EFFECT_SITES), seed)
    diag = {
        "seconds": time.monotonic() - started, "chains": chains, "warmup": warmup, "samples": samples,
        "divergences": int(extra["diverging"].sum()),
        "mean_leapfrog_steps": float(extra["num_steps"].mean()),
        "sites": {},
    }
    stats = summary(keep, group_by_chain=True)
    for site, s in stats.items():
        rhat, ess = np.atleast_1d(s["r_hat"]), np.atleast_1d(s["n_eff"])
        if np.isnan(rhat).all():  # constant draws, e.g. s = 0 when every player-season is a first season
            diag["sites"][site] = {"size": int(rhat.size), "constant": True}
            continue
        diag["sites"][site] = {"size": int(rhat.size), "max_rhat": float(np.nanmax(rhat)),
                               "share_rhat_gt_1.01": float(np.mean(rhat > 1.01)), "min_ess": float(np.nanmin(ess)),
                               "median_ess": float(np.nanmedian(ess))}
    # Per chain, to tell a stuck chain (one that never adapted) from slow mixing.
    diag["per_chain"] = {"seed": seed, "divergences": extra["diverging"].sum(1).tolist(),
                         "mean_leapfrog_steps": extra["num_steps"].mean(1).tolist(),
                         **{k: v.mean(1).tolist() for k, v in keep.items() if k in HYPER and v.ndim == 2}}
    flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in keep.items()}
    diag["hyper"] = {k: {"mean": float(v.mean()), "sd": float(v.std()), "median": float(np.median(v))}
                     for k, v in flat.items() if k in HYPER and v.ndim == 1}
    return flat, diag


def worst_scale_rhat(diag: dict, names: tuple[str, ...]) -> float:
    return max(diag["sites"][k]["max_rhat"] for k in names if k in diag["sites"] and "max_rhat" in diag["sites"][k])


def season_label(seasons: tuple[int, ...]) -> str:
    return str(seasons[0]) if len(seasons) == 1 else f"{seasons[0]}-{seasons[-1]}"


def nuts_scales(diag: dict, names: tuple[str, ...]) -> dict[str, float]:
    """Posterior means of the hierarchical scales a NUTS run estimated (single-season runs leave the
    season-step scales at their priors, so callers pass only the names that run identifies)."""
    return {k: diag["hyper"][k]["mean"] for k in names if k in diag["hyper"]}


def run_validation(settings: Settings, seasons: tuple[int, ...], chains: int = 4, warmup: int = 500,
                   samples: int = 500, svi_steps: int = 30000, seed: int = 0,
                   log: Callable[[str], None] = print) -> Path:
    """NUTS on `seasons`, then SVI three ways: every scale free (svi_normal), every scale held at the
    NUTS posterior mean (svi_fixed), and, with two or more seasons, only the within-season scales held
    (svi_partial, the as-of configuration)."""
    out = validation_dir(settings)
    out.mkdir(parents=True, exist_ok=True)
    label = season_label(seasons)
    ds = Dataset.load(model_dir(settings))
    a = make_arrays(ds, ds.plays.filter(pl.col("season").is_in(seasons)), compact=True)
    del ds
    log(f"validation seasons {label}: {len(a.y):,} plays")

    log(f"NUTS: {chains} chains x ({warmup} warmup + {samples} draws), sequential")
    draws, diag = run_nuts(a, chains, warmup, samples, seed=seed)
    tables = [effect_table(a, draws, "nuts")]
    worst = worst_scale_rhat(diag, WITHIN_SEASON_SCALES + SEASON_STEP_SCALES)
    log(f"  done in {diag['seconds'] / 60:.1f} min; divergences {diag['divergences']}; worst scale R-hat {worst:.3f}; "
        f"leapfrog steps by chain {[round(x) for x in diag['per_chain']['mean_leapfrog_steps']]}")
    del draws
    if worst > MAX_SCALE_RHAT:  # a stuck chain makes NUTS a meaningless reference for SVI
        (out / f"diagnostics_{label}_rejected_seed={seed}.json").write_text(json.dumps(diag, indent=1))
        raise RuntimeError(f"NUTS did not converge (worst scale R-hat {worst:.3f}); rerun with another --seed")

    variants = {"svi_normal": {}, "svi_fixed": nuts_scales(diag, WITHIN_SEASON_SCALES + SEASON_STEP_SCALES)}
    if len(seasons) > 1:
        variants["svi_partial"] = nuts_scales(diag, WITHIN_SEASON_SCALES)
    st, d = to_device(a)
    for method, fixed in variants.items():
        log(f"{method}: {svi_steps} steps, {len(fixed)} scales held at NUTS means")
        started = time.monotonic()
        dv = {**d, "fixed": {k: jnp.float32(v) for k, v in fixed.items()}}
        guide, params, trace = run_svi(st, dv, "normal", svi_steps, lr=0.005, log=log, log_every=5000)
        batches = sample_guide(guide, params, n=500)
        eff = effects_from_samples(st, dv, batches, EFFECT_SITES)
        tables.append(effect_table(a, eff, method))
        scales = {k: float(np.concatenate([np.asarray(b[k]) for b in batches]).mean())
                  for k in WITHIN_SEASON_SCALES + SEASON_STEP_SCALES if k in batches[0]}
        diag[method] = {"seconds": time.monotonic() - started, "trace": trace, "fixed": fixed, "svi_scales": scales}
    pl.concat(tables).write_parquet(out / f"effects_{label}.parquet")
    (out / f"diagnostics_{label}.json").write_text(json.dumps(diag, indent=1))
    log(f"wrote {out}")
    return out


def compare(effects: pl.DataFrame) -> pl.DataFrame:
    """Per SVI method and role, against NUTS: correlation and RMSE of posterior means, spread of the
    means (sd of SVI means / sd of NUTS means), and median posterior-sd ratio."""
    nuts = effects.filter(pl.col("method") == "nuts").select("role", "athlete_id", "season",
                                                              pl.col("mean").alias("m0"), pl.col("sd").alias("s0"))
    rows = []
    for method in effects.filter(pl.col("method") != "nuts")["method"].unique(maintain_order=True):
        j = effects.filter(pl.col("method") == method).join(nuts, on=["role", "athlete_id", "season"])
        for role, g in j.group_by("role", maintain_order=True):
            rows.append({"method": method, "role": role[0], "n": g.height,
                         "corr_mean": float(np.corrcoef(g["mean"], g["m0"])[0, 1]),
                         "rmse_mean": float(np.sqrt(((g["mean"] - g["m0"]) ** 2).mean())),
                         "spread_ratio": float(g["mean"].std() / g["m0"].std()),
                         "sd_ratio": float((g["sd"] / g["s0"]).median())})
    return pl.DataFrame(rows)


HYPER_WINDOW = 3  # seasons averaged for the scales of the next season's fits
MAX_SCALE_RHAT = 1.1  # a scale fit worse than this is retried with a new seed, then rejected
SCALE_FIT_ATTEMPTS = 2


def hyper_dir(settings: Settings) -> Path:
    return model_dir(settings) / "hyper"


def step_hyper_dir(settings: Settings) -> Path:
    return model_dir(settings) / "hyper_step"


def fit_hyper(settings: Settings, seasons: tuple[int, ...], chains: int = 4, warmup: int = 500, samples: int = 500,
              seed: int = 0, step: bool = False, log: Callable[[str], None] = print) -> None:
    """Single-season NUTS for each of `seasons` not yet fitted; writes hyper/season=YYYY.json with the
    within-season scales. With `step`, two-season NUTS on (S-1, S) instead, written to hyper_step/season=S.json
    with the season-step scales, which SVI shrinks as it does the within-season ones (2014-15 validation).
    The as-of fits for season S average the HYPER_WINDOW fits before S."""
    out = step_hyper_dir(settings) if step else hyper_dir(settings)
    names = SEASON_STEP_SCALES if step else WITHIN_SEASON_SCALES
    out.mkdir(parents=True, exist_ok=True)
    todo = [s for s in seasons if not (out / f"season={s}.json").is_file()]
    if not todo:
        log("every requested season already has a scale fit")
        return
    ds = Dataset.load(model_dir(settings))
    for season in todo:
        span = (season - 1, season) if step else (season,)
        a = make_arrays(ds, ds.plays.filter(pl.col("season").is_in(span)), compact=True)
        for attempt in range(seed, seed + SCALE_FIT_ATTEMPTS):
            log(f"scale fit {season_label(span)}: {len(a.y):,} plays; NUTS {chains} x ({warmup} + {samples}), seed {attempt}")
            draws, diag = run_nuts(a, chains, warmup, samples, seed=attempt)
            del draws
            scales = nuts_scales(diag, names)
            # Judged on the scales the file stores: a step fit's slow within-season scales (rush_tau_g) are not
            # used from it, and its 2017-18 run was rejected on rush_tau_g alone with every step scale <= 1.023.
            worst = worst_scale_rhat(diag, tuple(scales))
            log(f"  done in {diag['seconds'] / 60:.1f} min; divergences {diag['divergences']}; worst scale R-hat {worst:.3f}; "
                f"leapfrog steps by chain {[round(x) for x in diag['per_chain']['mean_leapfrog_steps']]}")
            if worst <= MAX_SCALE_RHAT:
                break
            # Kept for inspection under a name load_hyper's season=*.json glob does not match.
            (out / f"rejected_season={season}_seed={attempt}.json").write_text(
                json.dumps({"season": season, "scales": scales, "nuts": diag}, indent=1))
            log(f"  rejected: worst scale R-hat {worst:.3f} > {MAX_SCALE_RHAT}")
        else:
            log(f"  no scale file for {season} after {SCALE_FIT_ATTEMPTS} attempts")
            continue
        del a
        path = out / f"season={season}.json"
        path.write_text(json.dumps({"season": season, "scales": scales, "nuts": diag}, indent=1))
        log(f"wrote {path}")


def load_hyper(settings: Settings, directory: Path | None = None) -> dict[int, dict]:
    """Season -> within-season scales (NUTS posterior means) for every season fitted so far."""
    out = {}
    for path in sorted((directory or hyper_dir(settings)).glob("season=*.json")):
        h = json.loads(path.read_text())
        out[int(h["season"])] = h["scales"]
    return out


def scales_for(hyper: dict[int, dict[str, float]], season: int, burn_in: tuple[int, ...],
               window: int = HYPER_WINDOW, first: int | None = None) -> tuple[tuple[int, ...], dict[str, float]]:
    """Scales for season S's as-of fits: the mean of the scale fits from the `window` seasons before S.
    A burn-in season with no earlier fit uses its own (its fits are never evaluated). `first` is the first
    season a fit can exist for (default the first burn-in season; step fits start a season later); a season
    before it has no scales, which are then left free."""
    first = min(burn_in) if first is None else first
    if season < first:
        return (), {}
    used = tuple(y for y in range(season - window, season) if y >= first)
    if not used and season in burn_in:
        used = (season,)
    missing = [y for y in used if y not in hyper]
    if not used or missing:
        raise ValueError(f"no scale fit available for {missing or season} (needed by the as-of fits of {season}); "
                         "run `cfb player hyper`")
    check_hyper_asof(used, season, burn_in)
    names = set.intersection(*(set(hyper[s]) for s in used))
    means = {k: np.mean([np.asarray(hyper[s][k], dtype=float) for s in used], axis=0) for k in sorted(names)}
    return used, {k: float(v) if v.ndim == 0 else v.tolist() for k, v in means.items()}
def check_hyper_asof(hyper_seasons: tuple[int, ...], season: int, burn_in: tuple[int, ...]) -> None:
    """Leakage guard: a fit for an evaluated season may only use scales estimated from earlier seasons."""
    if season not in burn_in and max(hyper_seasons) >= season:
        raise ValueError(f"scales estimated on {hyper_seasons} would leak into the as-of fit for {season}")
