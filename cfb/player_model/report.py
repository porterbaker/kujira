"""Consolidate Stage 1 outputs and write reports/phase3_player_model.md."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from cfb.clean.common import read_table
from cfb.clean.quality import md_table
from cfb.config import Settings
from cfb.player_model import benchmarks as B
from cfb.player_model.asof import asof_dir
from cfb.player_model.data import Dataset, model_dir
from cfb.player_model.validate import HYPER_WINDOW, compare, hyper_dir, validation_dir


def consolidate(settings: Settings) -> tuple[Path, Path]:
    """player_values (offense + defense) and team_units as single tables in data/model/."""
    values = B.load_values(settings)
    v_path = model_dir(settings) / "player_values.parquet"
    values.sort("season", "through_week", "role", "athlete_id").write_parquet(v_path)
    units = pl.concat([pl.read_parquet(p) for p in sorted((asof_dir(settings) / "team_units").glob("*.parquet"))])
    u_path = model_dir(settings) / "team_units.parquet"
    units.sort("season", "through_week", "team_id").write_parquet(u_path)
    return v_path, u_path


def _fmt(df: pl.DataFrame, digits: int = 3) -> pl.DataFrame:
    return df.with_columns([pl.col(c).round(digits) for c, t in df.schema.items() if t in (pl.Float32, pl.Float64)])


def _validation_section(settings: Settings, label: str) -> list[str]:
    vd = validation_dir(settings)
    diag_path, eff_path = vd / f"diagnostics_{label}.json", vd / f"effects_{label}.parquet"
    if not diag_path.is_file():
        return [f"Validation outputs for {label} not found; run `cfb player validate --seasons {label}`."]
    diag = json.loads(diag_path.read_text())
    sites = pl.DataFrame([{"site": k, **v} for k, v in diag["sites"].items()]).sort(
        "max_rhat", descending=True, nulls_last=True)
    out = [
        f"NUTS on {label}: {diag['chains']} chains x ({diag['warmup']} warmup + {diag['samples']} draws), "
        f"{diag['seconds'] / 60:.0f} minutes. Divergences: **{diag['divergences']}**. Mean leapfrog steps per "
        f"iteration: {diag['mean_leapfrog_steps']:.0f} (tree depth cap 8 = 255 steps).", "",
        "R-hat and effective sample size by site (vector sites summarized over their elements):", "",
        md_table(_fmt(sites)), "",
    ]
    cmp = compare(pl.read_parquet(eff_path))
    out += ["SVI (AutoNormal unless noted) against NUTS on the same data, for player-season values c + s and "
            "team units. *spread_ratio* = sd of SVI means / sd of NUTS means; *sd_ratio* = median ratio of "
            "posterior sds.", "", md_table(_fmt(cmp)), ""]
    for method, k in diag.items():
        if method.startswith("svi_"):
            fixed = k.get("fixed")
            out.append(f"- `{method}`: {k['seconds'] / 60:.1f} minutes"
                       + (f"; {k['note']}" if "note" in k else "")
                       + (f"; {len(fixed)} scales held at NUTS means" if fixed and "note" not in k else "")
                       + "; ELBO (step: mean loss) " + ", ".join(f"{s}: {l:,.0f}" for s, l in k["trace"]))
    return out


def _scales_section(settings: Settings) -> list[str]:
    fits = {int(p.stem.split("=")[1]): json.loads(p.read_text()) for p in sorted(hyper_dir(settings).glob("season=*.json"))}
    if not fits:
        return ["Scale fits not found; run `cfb player hyper`."]
    names = list(next(iter(fits.values()))["scales"])
    table = pl.DataFrame([{"scale": k, **{str(y): f["scales"][k] for y, f in fits.items()}} for k in names])
    health = pl.DataFrame([{"season": y, "divergences": f["nuts"]["divergences"],
                            "worst_scale_rhat": max(f["nuts"]["sites"][k]["max_rhat"] for k in names),
                            "worst_scale": max(names, key=lambda k: f["nuts"]["sites"][k]["max_rhat"]),
                            "min_scale_ess": min(f["nuts"]["sites"][k]["min_ess"] for k in names),
                            "minutes": f["nuts"]["seconds"] / 60} for y, f in fits.items()])
    return [
        "Single-season NUTS posterior means of the within-season scales (4 chains x 500 draws each). The as-of "
        f"fits for season S hold each scale at the mean of the fits for the {HYPER_WINDOW} seasons before S; "
        "the burn-in season uses its own fit. No evaluated fit sees a scale estimated from its own season or "
        "later (`scales_for`, tested). The season-step scales (tau_s, tau_team, rho) are estimated by each "
        "as-of fit.", "", md_table(_fmt(table, 4)), "", md_table(_fmt(health, 3)), ""]


def build_report(settings: Settings, validation_season: int = 2019) -> str:
    ds = Dataset.load(model_dir(settings))
    values = B.load_values(settings)
    eos = B.end_of_season(values, ds)
    fits = pl.concat([pl.read_parquet(p) for p in sorted((asof_dir(settings) / "fits").glob("*.parquet"))])
    by_season, by_bucket = B.heldout_summary(settings)
    cmp, worst = B.cfbd_comparison(settings, eos)
    draft = B.draft_check(settings, eos)
    defense = [pl.read_parquet(p) for p in sorted((asof_dir(settings) / "defense_values").glob("*.parquet"))]
    deltas = defense[-1].select(pl.selectors.starts_with("delta_")).head(1) if defense else None

    out = [
        "# Phase 3: Stage 1 player value model", "",
        "Generated by `cfb player report`. All values are as of a fit date: the fit through week w-1 is the "
        "one used for week w games.", "",
        "## 1. Model", "",
        "```",
        "ppa ~ StudentT(nu, mu, sigma[pass|rush])",
        "pass: mu = a[pass,season] + ctx.b_pass + O[off] - D[def] + theta_pass[passer] + 1{target known}(k[season] + theta_recv[target])",
        "rush: mu = a[rush,season] + ctx.b_rush + O[off] - D[def] + theta_rush[rusher]",
        "theta_role[p,s,g] = alpha_role[position] + c[p] + s[p,s] + g[p,s,g]",
        "c ~ N(X.gamma, tau_c)   X = recruit rating, stars, missing recruit, height, weight (within position), transferred (as of fit date)",
        "s[first] = 0; s[next] = s[prev] + N(0, tau_s*sqrt(gap));  g ~ N(0, tau_g)",
        "alpha coded relative to a reference position per role (QB passing, RB rushing, WR receiving)",
        "within-season scales (tau_c, tau_g, unit tau_conf / tau_team0 / FCS scale) held at NUTS fits on the 3 prior seasons",
        "O, D = conference-season mean + AR(1) team deviation across seasons (FCS teams one pooled group, tighter spread)",
        "```", "",
        f"- Data: {ds.plays.height:,} plays (model-eligible, garbage time excluded), {len(ds.seasons)} seasons. "
        "Context = down, distance bucket, yards-to-goal bucket, home/away/neutral.",
        "- Missing attribution is not imputed: unknown passers and rushers get a season-level indicator, and "
        "unknown targets (2021-2024 incompletions) get no receiver term. Receiver values in 2021-2024 rest "
        "mostly on completions and are less reliable.",
        "- Defense: the play model has only the team-season unit D. Individual defenders are valued by the "
        "box-score model (section 7). Kickers, punters and offensive linemen are out of scope.",
        "- *Value* = c + s: EPA per play relative to the position average (alpha) in that role.", "",
        "## 2. Inference check: NUTS vs SVI", "",
        "Mean-field SVI biases hierarchical scales, so the as-of fits hold the within-season scales at NUTS "
        "estimates (empirical Bayes) and SVI fits the effects. The checks below justify that choice.", "",
        f"### 2a. One season ({validation_season})", "", *_validation_section(settings, str(validation_season)), "",
        "### 2b. Two seasons (2014-2015): the season random walk", "",
        *_validation_section(settings, "2014-2015"), "",
        "### 2c. Scales used by the as-of fits", "", *_scales_section(settings), "",
        "## 3. As-of fitting", "",
        f"- {fits.height} fits: each season's preseason (week 0) and every week through the last regular week. "
        f"Warm-started SVI (AutoNormal, scales from section 2c); {int(fits['steps'].sum()):,} steps in total, "
        f"{fits['seconds'].sum() / 3600:.1f} hours.",
        f"- Output: `data/model/player_values.parquet` ({values.height:,} rows) keyed by (athlete_id, season, "
        "through_week, role), and `data/model/team_units.parquet`.", "",
        "## 4. Held-out plays: next-week log predictive density vs a context-only baseline", "",
        "Each fit predicts the following week's plays, which it has not seen. Higher is better; *gain* is "
        "nats per play.", "", md_table(_fmt(by_season, 4)), "", md_table(_fmt(by_bucket, 4)), "",
        "## 5. Agreement with CFBD player PPA and WEPA", "",
        "End-of-season values against CFBD's season averages (`/ppa/players/season`, which includes garbage time) "
        f"and opponent-adjusted WEPA. Minimum plays in season: {B.MIN_PLAYS}.", "", md_table(_fmt(cmp)), "",
    ]
    for role, w in worst.items():
        ctx = B.disagreement_context(settings, ds, w, role)
        names = read_table(settings, "players").select("athlete_id", "first_name", "last_name")
        ctx = ctx.join(names, on="athlete_id", how="left").select(
            pl.format("{} {}", "first_name", "last_name").alias("player"), "season", "plays_in_season", "cfbd_plays",
            "value_mean", "value_sd", "cfbd_avg", "our_pct", "cfbd_pct",
            *[c for c in ("garbage_share", "sack_share", "avg_opp_D") if c in ctx.columns])
        out += [f"Largest rank disagreements, role `{role}`:", "", md_table(_fmt(ctx)), ""]
    out += ["## 6. Draft check (validation only)", "",
            "Career effect c at the end of the final college season vs NFL draft. Spearman > 0 means higher c, "
            "earlier pick. AUC = probability a drafted player's c exceeds an undrafted final-season player's in "
            "the same role and position group.", "", md_table(_fmt(draft)), ""]
    out += ["## 7. Defender model (box scores, 2016+)", ""]
    if deltas is not None:
        out += ["EPA of each event vs an average play of its kind (as of the latest fit):", "", md_table(_fmt(deltas)), ""]
    return "\n".join(out)


def write_report(settings: Settings) -> Path:
    consolidate(settings)
    path = settings.paths.reports / "phase3_player_model.md"
    path.write_text(build_report(settings))
    return path
