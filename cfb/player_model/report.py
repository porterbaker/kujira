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
from cfb.player_model.defense import EVENTS, hyper_dir as defense_hyper_dir
from cfb.player_model.model import SEASON_STEP_SCALES, WITHIN_SEASON_SCALES
from cfb.player_model.validate import (HYPER_WINDOW, MAX_SCALE_RHAT, compare, hyper_dir, step_hyper_dir,
                                       validation_dir)


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
    # As strings: md_table prints every float with one decimal, which would show an R-hat of 1.14 as 1.1.
    return df.with_columns([pl.col(c).round(digits).cast(pl.String) for c, t in df.schema.items()
                            if t in (pl.Float32, pl.Float64)])


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


def _load_fits(directory: Path) -> dict[int, dict]:
    return {int(p.stem.split("=")[1]): json.loads(p.read_text()) for p in sorted(directory.glob("season=*.json"))}


def _fit_tables(fits: dict[int, dict], names: tuple[str, ...], label=str) -> list[str]:
    """Scale values by season, then per-fit health (divergences, worst R-hat and ESS over `names`)."""
    table = pl.DataFrame([{"scale": k, **{label(y): f["scales"][k] for y, f in fits.items()}} for k in names])
    health = pl.DataFrame([{
        "fit": label(y), "draws": f"{f['nuts']['chains']} x {f['nuts']['samples']}",
        "divergences": f["nuts"]["divergences"],
        "worst_scale_rhat": max(f["nuts"]["sites"][k]["max_rhat"] for k in names),
        "worst_scale": max(names, key=lambda k: f["nuts"]["sites"][k]["max_rhat"]),
        "min_scale_ess": min(f["nuts"]["sites"][k]["min_ess"] for k in names),
        "minutes": f["nuts"]["seconds"] / 60} for y, f in fits.items()])
    return [md_table(_fmt(table, 4)), "", md_table(_fmt(health, 3)), ""]


def _rejected(settings: Settings) -> list[str]:
    rows = []
    for directory in (hyper_dir(settings), step_hyper_dir(settings)):
        for path in sorted(directory.glob("rejected_*.json")):
            f = json.loads(path.read_text())
            sites = {k: v for k, v in f["nuts"]["sites"].items() if k in f["scales"] and "max_rhat" in v}
            worst = max(sites, key=lambda k: sites[k]["max_rhat"])
            final = directory / f"season={f['season']}.json"
            outcome = ("pending" if not final.is_file()
                       else "accepted later (see notes)" if "note" in json.loads(final.read_text()) else "replaced")
            rows.append({"file": f"{directory.name}/{path.name}", "worst_stored_scale": worst,
                         "rhat": sites[worst]["max_rhat"], "ess": sites[worst]["min_ess"], "outcome": outcome})
    return [md_table(_fmt(pl.DataFrame(rows), 3)), ""] if rows else ["None.", ""]


def _scales_section(settings: Settings) -> list[str]:
    within, step = _load_fits(hyper_dir(settings)), _load_fits(step_hyper_dir(settings))
    if not within or not step:
        return ["Scale fits not found; run `cfb player hyper` and `cfb player hyper --step`."]
    burn_in = settings.model.burn_in_seasons
    out = [
        "Every hierarchical scale is held at NUTS posterior means (empirical Bayes); SVI fits only the effects. "
        f"The as-of fits for season S use the mean of the fits from the {HYPER_WINDOW} seasons before S. Burn-in "
        f"seasons ({', '.join(map(str, burn_in))}) with no earlier fit use their own and are never evaluated. No "
        "evaluated fit sees a scale estimated from its own season or later (`scales_for`, tested). A fit whose "
        f"worst stored scale has R-hat above {MAX_SCALE_RHAT} is rejected and retried with the next seed.", "",
        "**Within-season scales**: single-season NUTS (tau_c, tau_g, unit conference / team / FCS scales).", "",
        *_fit_tables(within, WITHIN_SEASON_SCALES, str),
        "**Season-step scales**: two-season NUTS on (S-1, S), labeled by the pair (tau_s, tau_team, rho). "
        "Single-season fits cannot identify them, and SVI shrinks them 3-10x (section 2b).", "",
        *_fit_tables(step, SEASON_STEP_SCALES, lambda y: f"{y - 1}-{y}"),
    ]
    notes = [f"- `{y}`: {f['note']}" for d in (within, step) for y, f in d.items() if "note" in f]
    notes += [f"- `{y - 1}-{y}`: from the section 2b validation run" for y, f in step.items() if "source" in f]
    if notes:
        out += ["Notes:", "", *notes, ""]
    out += ["Rejected fits (kept for inspection):", "", *_rejected(settings)]
    return out


def _defense_scales_section(settings: Settings) -> list[str]:
    fits = _load_fits(defense_hyper_dir(settings))
    out = []
    val_path = validation_dir(settings) / "defense_2016.json"
    if val_path.is_file():
        v = json.loads(val_path.read_text())
        out += [f"NUTS vs SVI on {v['season']} box scores ({v['divergences']} divergences, worst R-hat "
                f"{max(s['max_rhat'] for s in v['sites'].values()):.3f}). tau_c per event, NUTS / SVI free: "
                + ", ".join(f"{e} {v['tau_c'][e]:.3f} / {v['svi_normal_tau_c'][e]:.3f}" for e in EVENTS)
                + ". SVI is about 6% low, so the defender scales are held at NUTS too.", "",
                md_table(_fmt(pl.DataFrame(v["compare"]))), ""]
    if fits:
        table = pl.DataFrame([{"event": e, **{str(y): f["scales"]["tau_c"][i] for y, f in fits.items()}}
                              for i, e in enumerate(EVENTS)])
        health = pl.DataFrame([{"season": y, "divergences": f["nuts"]["divergences"],
                                "worst_rhat": max(s["max_rhat"] for s in f["nuts"]["sites"].values()),
                                "minutes": f["nuts"]["seconds"] / 60} for y, f in fits.items()])
        out += ["Defender tau_c by season (single-season NUTS):", "", md_table(_fmt(table, 4)), "",
                md_table(_fmt(health, 3)), ""]
    return out


def build_report(settings: Settings, validation_season: int = 2019) -> str:
    ds = Dataset.load(model_dir(settings))
    values = B.load_values(settings)
    eos = B.end_of_season(values, ds)
    fits = pl.concat([pl.read_parquet(p) for p in sorted((asof_dir(settings) / "fits").glob("*.parquet"))])
    first = settings.model.eval_first_season
    by_season, by_bucket = B.heldout_summary(settings, first)
    # End-of-season S values come from season S+1's preseason fit; only evaluated fits count.
    eos_eval = eos.filter(pl.col("season") + 1 >= first)
    cmp, worst = B.cfbd_comparison(settings, eos_eval)
    draft = B.draft_check(settings, eos_eval)
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
        "within-season scales (tau_c, tau_g, unit tau_conf / tau_team0 / FCS scale) held at single-season NUTS fits on the 3 prior seasons",
        "season-step scales (tau_s, unit tau_team / rho) held at two-season NUTS fits on the 3 prior season pairs",
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
        "Mean-field SVI biases hierarchical scales, so the as-of fits hold every scale at NUTS estimates "
        "(empirical Bayes) and SVI fits the effects. The checks below justify that choice.", "",
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
        f"nats per play. Burn-in seasons are shown but flagged; the week buckets cover {first} onward.", "",
        md_table(_fmt(by_season, 4)), "", md_table(_fmt(by_bucket, 4)), "",
        "**Most of this gain is not player value.** The model has a target-known indicator (kappa) that the "
        "context baseline lacks, and a pass has no recorded target exactly when it fails (sack, throwaway, "
        "incompletion without a target), so the model is told part of the outcome. In 2021-2024 most "
        "incompletions also lack targets, so a recorded target nearly means a completion. Gain by play group "
        f"({first} onward):", "",
        md_table(_fmt(B.heldout_by_play_group(settings, ds, first), 4)), "",
        "Rushes, which carry no such indicator, show what players and units add over context: a few "
        "thousandths of a nat per play. The indicator could in principle absorb part of a passer's completion "
        "skill where incompletion targets are missing (2021-2024), but the spread of end-of-season passer values "
        "there (sd 0.044-0.062) is no lower than in 2016-2018, so there is no sign of it. A baseline with the "
        "same indicators would measure the player gain on passes directly; that is an open question for the "
        "owner (see PLAN.md).", "",
        "## 5. Agreement with CFBD player PPA and WEPA", "",
        "End-of-season values against CFBD's season averages (`/ppa/players/season`, which includes garbage time) "
        f"and opponent-adjusted WEPA, seasons {first - 1} onward (from evaluated fits). Minimum plays in season: "
        f"{B.MIN_PLAYS}.", "", md_table(_fmt(cmp)), "",
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
    out += ["## 7. Defender model (box scores, 2016+)", "", *_defense_scales_section(settings)]
    if deltas is not None:
        out += ["EPA of each event vs an average play of its kind (as of the latest fit):", "", md_table(_fmt(deltas)), ""]
    return "\n".join(out)


def write_report(settings: Settings) -> Path:
    consolidate(settings)
    path = settings.paths.reports / "phase3_player_model.md"
    path.write_text(build_report(settings))
    return path
