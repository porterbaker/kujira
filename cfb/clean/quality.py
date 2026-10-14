"""Phase 2 data-quality report: reconciliation, attribution, recruit coverage, transfers, identity.

Offline: reads data/clean (and the 2025+ enriched raw files for the kneel/spike rule check).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from cfb.clean.build import table_inventory
from cfb.clean.common import read_table, scan_table
from cfb.clean.plays import KNEEL_TEXT, SPIKE_TEXT, TEAM_RUN_FOR_LOSS, load_enriched
from cfb.config import Settings

SCRIMMAGE = ["pass", "sack", "rush"]


def pct(n: float, d: float) -> str:
    return f"{n / d:.1%}" if d else "–"


def _cell(col: str, v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v) if col in ("season", "year") else f"{v:,}"
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


def md_table(df: pl.DataFrame) -> str:
    cols = df.columns
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.iter_rows():
        out.append("| " + " | ".join(_cell(c, v) for c, v in zip(cols, row)) + " |")
    return "\n".join(out)


def reconciliation(settings: Settings) -> pl.DataFrame:
    plays = scan_table(settings, "plays")
    part = scan_table(settings, "play_participants")
    games = scan_table(settings, "games").filter(pl.col("is_fbs_game") & pl.col("completed"))
    g = games.group_by("season").agg(pl.len().alias("fbs_games"))
    gp = plays.group_by("season").agg(pl.col("game_id").n_unique().alias("games_with_plays"))
    gs = part.group_by("season").agg(pl.col("game_id").n_unique().alias("games_with_credits"),
                                      pl.len().alias("credits"), pl.col("in_plays").sum().alias("credits_in_plays"))
    credited = part.filter(pl.col("side") == "offense").select("play_id").unique().with_columns(c=pl.lit(True))
    sc = plays.filter(pl.col("category").is_in(SCRIMMAGE)).join(credited, on="play_id", how="left") \
        .group_by("season").agg(pl.len().alias("scrimmage"), pl.col("c").fill_null(False).sum().alias("with_credit"))
    df = g.join(gp, on="season", how="left").join(gs, on="season", how="left").join(sc, on="season", how="left") \
        .sort("season").collect()
    return df.select(
        "season", "fbs_games",
        pl.format("{} ({})", "games_with_plays", pl.struct("games_with_plays", "fbs_games").map_elements(
            lambda r: pct(r["games_with_plays"], r["fbs_games"]), return_dtype=pl.String)).alias("games with plays"),
        pl.format("{} ({})", "games_with_credits", pl.struct("games_with_credits", "fbs_games").map_elements(
            lambda r: pct(r["games_with_credits"], r["fbs_games"]), return_dtype=pl.String)).alias("games with credits"),
        pl.struct("credits_in_plays", "credits").map_elements(
            lambda r: pct(r["credits_in_plays"], r["credits"]), return_dtype=pl.String).alias("credit play_ids in /plays"),
        pl.struct("with_credit", "scrimmage").map_elements(
            lambda r: pct(r["with_credit"], r["scrimmage"]), return_dtype=pl.String).alias("scrimmage plays with a credit"),
    )


def play_quality(settings: Settings) -> pl.DataFrame:
    plays = scan_table(settings, "plays")
    s = plays.filter(pl.col("category").is_in(SCRIMMAGE))
    state_missing = pl.any_horizontal(pl.col(c).is_null() for c in ("down", "distance", "yards_to_goal", "clock_seconds"))
    q = s.group_by("season").agg(
        pl.len().alias("scrimmage"), pl.col("ppa").is_not_null().mean().alias("ppa"),
        state_missing.mean().alias("state_missing"), (pl.col("exclude_reason") == "invalid_parse").sum().alias("invalid_parse"),
        pl.col("is_kneel").sum().alias("kneels"), pl.col("is_spike").sum().alias("spikes"),
        pl.col("garbage_time").mean().alias("garbage"), pl.col("model_eligible").sum().alias("model_eligible"),
    )
    amb = plays.filter(pl.col("play_type").is_in(["Fumble Recovery (Own)", "Fumble Recovery (Opponent)",
                                                  "Fumble Return Touchdown", "Safety", "Fumble"]))
    amb = amb.group_by("season").agg((pl.col("category") == "other").sum().alias("ambiguous_unresolved"))
    df = q.join(amb, on="season", how="left").sort("season").collect()
    return df.select("season", "scrimmage",
                     pl.col("ppa").map_elements(lambda v: f"{v:.2%}", return_dtype=pl.String).alias("with ppa"),
                     pl.col("state_missing").map_elements(lambda v: f"{v:.2%}", return_dtype=pl.String).alias("state missing"),
                     "ambiguous_unresolved", "invalid_parse", "kneels", "spikes",
                     pl.col("garbage").map_elements(lambda v: f"{v:.1%}", return_dtype=pl.String).alias("garbage time"),
                     "model_eligible")


def attribution_rates(settings: Settings) -> pl.DataFrame:
    """Numeric attribution shares per season (formatted for the report by ``attribution``)."""
    pp = scan_table(settings, "pass_plays")
    rp = scan_table(settings, "rush_plays")
    p = pp.group_by("season").agg(
        (pl.col("passer_attribution") == "individual").filter(~pl.col("is_sack")).mean().alias("passer"),
        pl.col("target_id").is_not_null().filter(pl.col("outcome") == "completion").mean().alias("receiver_on_comp"),
        (pl.col("target_source") == "stats").fill_null(False).filter(pl.col("outcome") == "incompletion").mean().alias("inc_stats"),
        pl.col("target_id").is_not_null().filter(pl.col("outcome") == "incompletion").mean().alias("inc_any"),
        (pl.col("passer_attribution") == "individual").filter(pl.col("is_sack")).mean().alias("qb_on_sack"),
        (pl.col("sacked_by").list.len() > 0).filter(pl.col("is_sack")).mean().alias("defender_on_sack"),
        pl.col("intercepted_by_id").is_not_null().filter(pl.col("outcome") == "interception").mean().alias("interceptor"),
    )
    r = rp.filter(~pl.col("is_kneel")).group_by("season").agg(
        (pl.col("rusher_attribution") == "individual").mean().alias("rusher"),
        pl.col("is_team_rush").mean().alias("team_rush"))
    return p.join(r, on="season").sort("season").collect()


def attribution(settings: Settings) -> pl.DataFrame:
    df = attribution_rates(settings)
    fmt = {c: pl.col(c).map_elements(lambda v: "–" if v is None else f"{v:.1%}", return_dtype=pl.String)
           for c in df.columns if c != "season"}
    return df.select("season", fmt["passer"].alias("passer (pass)"), fmt["receiver_on_comp"].alias("receiver (comp)"),
                     fmt["inc_stats"].alias("target (inc, stats)"), fmt["inc_any"].alias("target (inc, any source)"),
                     fmt["rusher"].alias("rusher"), fmt["team_rush"].alias("team rush"), fmt["qb_on_sack"].alias("QB (sack)"),
                     fmt["defender_on_sack"].alias("defender (sack)"), fmt["interceptor"].alias("interceptor (INT)"))


def enriched_checks(settings: Settings, seasons: tuple[int, ...]) -> list[str]:
    lines = []
    plays = scan_table(settings, "plays")
    for season in (s for s in seasons if s >= 2025):
        pe, re_ = load_enriched(settings, season)
        if pe.is_empty():
            continue
        ps = plays.filter(pl.col("season") == season).select("play_id", "category", "play_text", "yards_gained").collect()
        ids = set(ps["play_id"].to_list())
        text = pl.col("play_text").fill_null("")
        rule = ps.with_columns(
            kneel_rule=(pl.col("category") == "rush") & (text.str.contains(KNEEL_TEXT) | (
                text.str.contains(TEAM_RUN_FOR_LOSS) & (pl.col("yards_gained").fill_null(0) <= 0))),
            spike_rule=(pl.col("category") == "pass") & text.str.contains(SPIKE_TEXT),
        )
        k = rule.join(re_.select("play_id", "is_kneel_enriched"), on="play_id", how="inner")
        sp = rule.join(pe.select("play_id", "is_spike_enriched"), on="play_id", how="inner")
        kt = k.filter(pl.col("is_kneel_enriched")).height
        kr = k.filter(pl.col("kneel_rule")).height
        kb = k.filter(pl.col("kneel_rule") & pl.col("is_kneel_enriched")).height
        st, sr = sp.filter(pl.col("is_spike_enriched")).height, sp.filter(pl.col("spike_rule")).height
        sb = sp.filter(pl.col("spike_rule") & pl.col("is_spike_enriched")).height
        pp = scan_table(settings, "pass_plays").filter(pl.col("season") == season).collect()
        rp = scan_table(settings, "rush_plays").filter(pl.col("season") == season).collect()
        lines += [
            f"- **{season}**: enriched play ids found in `/plays`: passing {pct(sum(i in ids for i in pe['play_id']), pe.height)}, "
            f"rushing {pct(sum(i in ids for i in re_['play_id']), re_.height)}. "
            f"Passer agrees with enriched: {pp['passer_agrees_enriched'].sum():,}/{pp['passer_agrees_enriched'].count():,}; "
            f"rusher: {rp['rusher_agrees_enriched'].sum():,}/{rp['rusher_agrees_enriched'].count():,}.",
            f"  Kneel rule vs CFBD `isKneel`: {kb}/{kt} CFBD kneels caught, {kr - kb} extra flagged by the rule. "
            f"Spike rule vs `isSpike`: {sb}/{st} caught, {sr - sb} extra.",
        ]
    return lines


def recruit_coverage(settings: Settings) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, int]]:
    """Coverage among players with at least one credited play for an FBS team that season."""
    part = scan_table(settings, "play_participants").select("athlete_id", "season", "team_id").unique()
    fbs = scan_table(settings, "teams").filter(pl.col("classification") == "fbs").select("team_id", "season")
    involved = part.join(fbs, on=["team_id", "season"]).select("athlete_id", "season").unique()
    ps = scan_table(settings, "player_seasons").select("athlete_id", "season", "position_group").unique(["athlete_id", "season"])
    links = scan_table(settings, "recruit_links")
    by_path = links.group_by("athlete_id").agg(pl.col("via_roster").any().alias("roster"),
                                               pl.col("via_recruiting").any().alias("recruiting"))
    df = involved.join(ps, on=["athlete_id", "season"], how="left").join(by_path, on="athlete_id", how="left") \
        .with_columns(pl.col("roster").fill_null(False), pl.col("recruiting").fill_null(False),
                      pl.col("position_group").fill_null("unknown")) \
        .with_columns(any=pl.col("roster") | pl.col("recruiting")).collect()

    def summarize(by: str) -> pl.DataFrame:
        out = df.group_by(by).agg(pl.len().alias("players"), pl.col("any").mean(), pl.col("roster").mean(),
                                  pl.col("recruiting").mean()).sort(by)
        return out.with_columns(*(pl.col(c).map_elements(lambda v: f"{v:.1%}", return_dtype=pl.String)
                                  for c in ("any", "roster", "recruiting")))

    totals = {"players": df.height, "roster": int(df["roster"].sum()), "recruiting": int(df["recruiting"].sum()),
              "any": int(df["any"].sum())}
    return summarize("season"), summarize("position_group"), totals


def transfer_diagnostics(settings: Settings) -> list[str]:
    tr = read_table(settings, "transfers")
    m = read_table(settings, "transfer_matches").join(tr.select("transfer_key", "season", "origin_season", "origin_id"),
                                                      on="transfer_key")
    by = m.group_by("season").agg(
        pl.len().alias("entries"), pl.col("origin_id").is_not_null().sum().alias("origin team known"),
        (pl.col("method") == "exact").sum().alias("exact"), (pl.col("method") == "fuzzy").sum().alias("fuzzy"),
        (pl.col("method") == "none").sum().alias("no candidate"), pl.col("accepted").sum().alias("accepted"),
    ).sort("season")
    exact = m.filter(pl.col("method") == "exact")
    offset = (exact["candidate_roster_season"] - exact["origin_season"]).value_counts().sort("candidate_roster_season")
    fz = m.filter(pl.col("method") == "fuzzy").with_columns(
        bin=pl.when(pl.col("confidence") >= 95).then(pl.lit("1: 95-100")).when(pl.col("confidence") >= 90).then(pl.lit("2: 90-95"))
        .when(pl.col("confidence") >= 85).then(pl.lit("3: 85-90")).when(pl.col("confidence") >= 75).then(pl.lit("4: 75-85"))
        .when(pl.col("confidence") >= 60).then(pl.lit("5: 60-75")).otherwise(pl.lit("6: <60")))
    hist = fz.group_by("bin").agg(pl.len().alias("fuzzy matches"),
                                  pl.col("last_name_match").mean().map_elements(lambda v: f"{v:.0%}", return_dtype=pl.String)
                                  .alias("last name matches"),
                                  pl.col("first_name_compatible").mean().map_elements(lambda v: f"{v:.0%}", return_dtype=pl.String)
                                  .alias("first name compatible"),
                                  pl.col("destination_confirmed").mean().map_elements(lambda v: f"{v:.0%}", return_dtype=pl.String)
                                  .alias("destination roster confirms"),
                                  pl.col("position_match").mean().map_elements(lambda v: f"{v:.0%}", return_dtype=pl.String)
                                  .alias("position group matches")).sort("bin")
    ex = fz.join(tr.select("transfer_key", "first_name", "last_name", "position", "origin", "destination"), on="transfer_key")
    examples = pl.concat([ex.filter(pl.col("confidence").is_between(lo, hi, closed="left")).sample(min(4, n), seed=1)
                          for lo, hi, n in ((92, 101, 10**9), (85, 92, 10**9), (75, 85, 10**9), (60, 75, 10**9))
                          if ex.filter(pl.col("confidence").is_between(lo, hi, closed="left")).height]) \
        .sort("confidence", descending=True).select(
        pl.format("{} {}", "first_name", "last_name").alias("portal name"), "position", "origin", "candidate_name",
        "candidate_position", "last_name_match", "first_name_compatible", "shared_last_name",
        "destination_confirmed", "confidence")
    with_dest = exact.join(tr.filter(pl.col("destination_id").is_not_null()).select("transfer_key"), on="transfer_key")
    exact_dest = with_dest["destination_confirmed"].mean() if with_dest.height else None
    return [
        md_table(by), "",
        f"Exact matches: the roster season that matched vs the origin season derived from the transfer date: "
        + ", ".join(f"{o:+d}: {n:,}" for o, n in offset.iter_rows()) + ". "
        + (f"Where the destination is a known team, its next-season roster confirms {exact_dest:.0%} of exact matches."
           if exact_dest is not None else ""), "",
        "Fuzzy-match confidence (name score adjusted for position group, destination roster, season distance, "
        "and a close second candidate). None are accepted until a threshold is set:", "", md_table(hist), "",
        "Random examples from each band (my read of the samples: 92+ all correct; 85-92 nearly all correct, mostly "
        "nicknames like Zachary/Zach; 75-85 mixed, including brothers; below 75 mostly wrong):", "", md_table(examples),
    ]


def identity(settings: Settings) -> list[str]:
    ps = read_table(settings, "player_seasons")
    players = read_table(settings, "players")
    aliases = read_table(settings, "athlete_aliases")
    suspect = ps.group_by("season").agg(pl.col("class_year_suspect").mean()).sort("season")
    return [
        f"- `players`: {players.height:,} athlete ids; {players['in_roster'].sum():,} on a roster, "
        f"{players.filter(~pl.col('in_roster')).height:,} seen only in stats or box scores.",
        f"- Placeholder (negative) ids, 2014–2019 rosters only: {ps.filter(pl.col('placeholder_id'))['athlete_id'].n_unique():,}. "
        f"None appear in `/plays/stats`. Aliased to a real id: {aliases.height:,} "
        f"({', '.join(f'{k}: {v}' for k, v in aliases.group_by('method').len().iter_rows())}).",
        f"- Roster rows whose team doesn't map to a team id: {ps['team_id'].null_count():,} (NAIA and other non-NCAA schools).",
        "- Class year missing (raw value is the season, or 0): "
        + ", ".join(f"{s}: {v:.0%}" for s, v in suspect.iter_rows()) + ".",
    ]


def lookahead(settings: Settings) -> list[str]:
    plays = scan_table(settings, "plays")
    changed = plays.filter((pl.col("offense_score_after") != pl.col("offense_score"))
                           | (pl.col("defense_score_after") != pl.col("defense_score"))).select(pl.len()).collect().item()
    total = plays.select(pl.len()).collect().item()
    return [
        f"- `/plays` scores are recorded after the play. {changed:,} of {total:,} plays ({pct(changed, total)}) have a "
        "post-play score different from the pre-snap score. The clean `offense_score`, `defense_score` and `score_diff` "
        "are pre-snap (previous play in the game); the post-play values keep an `_after` suffix and must not be used as state.",
        "- `garbage_time` uses the pre-snap margin only.",
        "- `games`: postgame columns (points, postgame Elo/WP, attendance, excitement, notes, completed) are listed in "
        "`cfb.leakage.GAME_POSTGAME_FIELDS`. `lines` carries `home_score`/`away_score` (postgame) for evaluation only.",
        "- `team_coaching.coaching_change` uses only the coach who started the season; `in_season_change` is "
        "season-end information and never a same-season feature.",
        "- `transfers.transfer_date` dates each portal entry, so transfer status can be applied as of a date.",
        "- `player_seasons` is a roster snapshot pulled after (or, for 2026, during) the season: use it for static "
        "attributes (position, size, hometown), not for who was available on a given date.",
    ]


def recommendation(settings: Settings) -> list[str]:
    a = attribution_rates(settings)
    ppa = scan_table(settings, "plays").filter(pl.col("category").is_in(SCRIMMAGE)).group_by("season") \
        .agg(pl.col("ppa").is_not_null().mean().alias("ppa")).collect()
    a = a.join(ppa, on="season")

    def seasons(cond: pl.Expr) -> str:
        got = sorted(a.filter(cond)["season"].to_list())
        return ", ".join(map(str, got)) or "none"

    offense = (pl.col("passer") >= 0.92) & (pl.col("receiver_on_comp") >= 0.95) & (pl.col("rusher") >= 0.9) & (pl.col("ppa") >= 0.99)
    return [
        f"- Offensive inputs complete (ppa ≥ 99%, passer ≥ 92%, receiver on completions ≥ 95%, rusher ≥ 90%): "
        f"{seasons(offense)}. Not met: {seasons(~offense)}.",
        f"- Incompletion targets ≥ 70% (any source): {seasons(pl.col('inc_any') >= 0.7)}. "
        f"Below 60%: {seasons(pl.col('inc_any') < 0.6)}.",
        f"- Defender credited on < 80% of sacks: {seasons(pl.col('defender_on_sack') < 0.8)}. "
        f"Interceptor credited on < 80% of interceptions: {seasons(pl.col('interceptor') < 0.8)}.",
        "",
        "**Recommendation: keep 2014 as the start season.** Every input the offensive play model needs is complete "
        "from 2014 on, and the earliest seasons give the career and season effects more history. 2014 has no prior "
        "season of player data, so treat it as a burn-in season and evaluate the game model from 2015. Two gaps are "
        "season-specific rather than reasons to start later, and Phase 3 has to handle them explicitly: incompletion "
        "targets in 2021–2024, and defender credits in the seasons listed above.",
    ]


def build_report(settings: Settings) -> str:
    seasons = settings.seasons
    inv = pl.DataFrame(table_inventory(settings), schema=["table", "rows", "columns"], orient="row")
    season_cov, pos_cov, totals = recruit_coverage(settings)
    th = settings.thresholds.garbage_time or {}
    out = [
        "# Phase 2 data-quality report", "",
        "Generated by `cfb clean report` from `data/clean`. Rebuild the tables with `cfb clean build`.", "",
        "## 1. Tables", "", md_table(inv), "",
        "## 2. Play reconciliation", "",
        "FBS games = completed games with at least one FBS team. Credits = `/plays/stats` rows.", "",
        md_table(reconciliation(settings)), "", *enriched_checks(settings, seasons), "",
        "## 3. Play quality", "",
        "Scrimmage = pass, sack and rush plays (fumble and safety plays resolved from their credits). "
        "*State missing* = down, distance, yards to goal or clock missing. *Ambiguous unresolved* = fumble/safety plays "
        "with no credit to say whether they were passes or rushes. *Invalid parse* = 2025+ enriched `parseStatus` "
        "`invalid`. Garbage time uses the **provisional** definition "
        + ", ".join(f"Q{q} margin > {m}" for q, m in sorted(th.items())) + " (flag only).", "",
        md_table(play_quality(settings)), "",
        "## 4. Player attribution", "",
        "Shares of the relevant plays with exactly one credited player. Incompletion targets come from `/plays/stats` "
        "`Target` credits, then (2025+) the enriched `targetId`, then an exact roster-name match from the play text.", "",
        md_table(attribution(settings)), "",
        "## 5. Recruit linking", "",
        f"Among {totals['players']:,} player-seasons with at least one credited play for an FBS team: "
        f"linked through `roster.recruitIds` {pct(totals['roster'], totals['players'])}, through "
        f"`recruiting.athleteId` {pct(totals['recruiting'], totals['players'])}, either {pct(totals['any'], totals['players'])}.", "",
        md_table(season_cov), "", md_table(pos_cov), "",
        "## 6. Transfers", "", *transfer_diagnostics(settings), "",
        "## 7. Identity", "", *identity(settings), "",
        "## 8. Lookahead safeguards in the clean layer", "", *lookahead(settings), "",
        "## 9. Start season", "", *recommendation(settings), "",
    ]
    return "\n".join(out)


def write_report(settings: Settings) -> Path:
    path = settings.paths.reports / "phase2_data_quality.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(settings))
    return path
