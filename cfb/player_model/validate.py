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

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS

from cfb.config import Settings
from cfb.player_model.data import ROLES, Dataset, model_dir
from cfb.player_model.fit import effects_from_samples, run_svi, sample_guide
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
    kernel = NUTS(functools.partial(model, st), target_accept_prob=0.85, max_tree_depth=8)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=chains,
                chain_method="sequential", progress_bar=False)
    started = time.monotonic()
    mcmc.run(jax.random.PRNGKey(seed), d, extra_fields=("diverging", "num_steps"))
    extra = mcmc.get_extra_fields()
    by_chain = mcmc.get_samples(group_by_chain=True)
    keep = {k: np.asarray(v) for k, v in by_chain.items() if k in HYPER or k in EFFECT_SITES}
    diag = {
        "seconds": time.monotonic() - started, "chains": chains, "warmup": warmup, "samples": samples,
        "divergences": int(np.asarray(extra["diverging"]).sum()),
        "mean_leapfrog_steps": float(np.asarray(extra["num_steps"]).mean()),
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
    flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in keep.items()}
    diag["hyper"] = {k: {"mean": float(v.mean()), "sd": float(v.std()), "median": float(np.median(v))}
                     for k, v in flat.items() if k in HYPER and v.ndim == 1}
    return flat, diag


def season_label(seasons: tuple[int, ...]) -> str:
    return str(seasons[0]) if len(seasons) == 1 else f"{seasons[0]}-{seasons[-1]}"


def nuts_scales(diag: dict, names: tuple[str, ...]) -> dict[str, float]:
    """Posterior means of the hierarchical scales a NUTS run estimated (single-season runs leave the
    season-step scales at their priors, so callers pass only the names that run identifies)."""
    return {k: diag["hyper"][k]["mean"] for k in names if k in diag["hyper"]}


def run_validation(settings: Settings, seasons: tuple[int, ...], chains: int = 4, warmup: int = 500,
                   samples: int = 500, svi_steps: int = 30000, log: Callable[[str], None] = print) -> Path:
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
    draws, diag = run_nuts(a, chains, warmup, samples)
    tables = [effect_table(a, draws, "nuts")]
    log(f"  done in {diag['seconds'] / 60:.1f} min; divergences {diag['divergences']}")
    del draws

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


def hyper_dir(settings: Settings) -> Path:
    return model_dir(settings) / "hyper"


def fit_hyper(settings: Settings, seasons: tuple[int, ...], chains: int = 4, warmup: int = 500, samples: int = 500,
              log: Callable[[str], None] = print) -> None:
    """Single-season NUTS for each of `seasons` not yet fitted; writes hyper/season=YYYY.json with the
    within-season scales. The as-of fits for season S average the HYPER_WINDOW seasons before S."""
    out = hyper_dir(settings)
    out.mkdir(parents=True, exist_ok=True)
    todo = [s for s in seasons if not (out / f"season={s}.json").is_file()]
    if not todo:
        log("every requested season already has a scale fit")
        return
    ds = Dataset.load(model_dir(settings))
    for season in todo:
        a = make_arrays(ds, ds.plays.filter(pl.col("season") == season), compact=True)
        log(f"scale fit {season}: {len(a.y):,} plays; NUTS {chains} x ({warmup} + {samples})")
        draws, diag = run_nuts(a, chains, warmup, samples)
        del draws, a
        scales = nuts_scales(diag, WITHIN_SEASON_SCALES)
        worst = max(diag["sites"][k]["max_rhat"] for k in scales)
        log(f"  done in {diag['seconds'] / 60:.1f} min; divergences {diag['divergences']}; worst scale R-hat {worst:.3f}")
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
               window: int = HYPER_WINDOW) -> tuple[tuple[int, ...], dict[str, float]]:
    """Scales for season S's as-of fits: the mean of the scale fits from the `window` seasons before S.
    A burn-in season with no earlier fit uses its own (its fits are never evaluated)."""
    used = tuple(y for y in range(season - window, season) if y >= min(burn_in))
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
