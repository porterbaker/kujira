"""Phase 1a probe analysis: reproducible from raw data, offline.

``write_report`` renders ``reports/phase1a_probe.md``. ``write_fixtures`` trims probe responses
into ``tests/fixtures/``. The analysis functions are pure and are tested against those fixtures.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import polars as pl

from cfb.config import Settings
from cfb.ingest import raw_store
from cfb.ingest.manifest import Manifest
from cfb.ingest.probe import (
    COVERAGE_SEASON,
    ENRICHED_SEASON,
    PROBE_SEASON,
    load_infos,
    load_landed,
)
from cfb.ingest.schema_check import MODEL_BY_SLUG, compare

PASS_TYPES = frozenset({"Pass Reception", "Pass Incompletion", "Passing Touchdown", "Pass Interception Return",
                        "Interception Return Touchdown", "Interception"})
RUSH_TYPES = frozenset({"Rush", "Rushing Touchdown"})
SCRIMMAGE_TYPES = PASS_TYPES | RUSH_TYPES | {"Sack"}
DEFENSIVE_STAT_TYPES = frozenset({"Tackle", "Pass Breakup", "QB Hurry", "Fumble Forced", "Sack", "Interception"})


# ---------- analysis (pure) ----------

def ratio(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.1%})" if d else "0/0"


def credits_by_play(stats: list[dict]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for s in stats:
        out[s["playId"]].add(s["statType"])
    return out


def attribution_coverage(plays: list[dict], stats: list[dict]) -> dict[str, tuple[int, int]]:
    """Share of one game's scrimmage plays that /plays/stats credits to a player, by role."""
    credits = credits_by_play(stats)
    game_ids = {s["gameId"] for s in stats}
    plays = [p for p in plays if p["gameId"] in game_ids]

    def cov(types: set[str] | frozenset[str], need: set[str]) -> tuple[int, int]:
        rows = [p for p in plays if p["playType"] in types]
        return sum(bool(credits[p["id"]] & need) for p in rows), len(rows)

    return {
        "passer on pass plays": cov(PASS_TYPES, {"Completion", "Incompletion", "Interception Thrown"}),
        "receiver on completions": cov({"Pass Reception", "Passing Touchdown"}, {"Reception"}),
        "target on incompletions": cov({"Pass Incompletion"}, {"Target"}),
        "rusher on rushes": cov(RUSH_TYPES, {"Rush"}),
        "QB on sacks": cov({"Sack"}, {"Sack Taken"}),
        "defender on sacks": cov({"Sack"}, {"Sack"}),
        "defender on interceptions": cov({"Pass Interception Return", "Interception Return Touchdown", "Interception"},
                                         {"Interception"}),
    }


def join_rate(child_ids: list[str] | set[str], parent_ids: set[str]) -> tuple[int, int]:
    child = set(child_ids)
    return len(child & parent_ids), len(child)


def enriched_coverage(plays: list[dict], enriched_ids: set[str], team_class: dict[str, str]) -> dict[str, tuple[int, int]]:
    """Scrimmage plays found in /passing/plays or /rushing/plays, by offense classification."""
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for p in plays:
        if p["playType"] in SCRIMMAGE_TYPES:
            cls = team_class.get(p["offense"], "unknown")
            out[cls][0] += p["id"] in enriched_ids
            out[cls][1] += 1
    return {k: (v[0], v[1]) for k, v in sorted(out.items())}


def sack_rusher_is_passer(passing: list[dict], rushing: list[dict]) -> tuple[int, int, int]:
    """(sack rusherIds that are a passer for the same game+offense, sacks with a rusherId, sacks)."""
    passers: dict[tuple, set[str]] = defaultdict(set)
    for r in passing:
        if r.get("passerId"):
            passers[(r["gameId"], r["offense"])].add(r["passerId"])
    sacks = [r for r in rushing if r.get("isSack")]
    with_id = [r for r in sacks if r.get("rusherId")]
    return sum(r["rusherId"] in passers[(r["gameId"], r["offense"])] for r in with_id), len(with_id), len(sacks)


def ppa_agreement(plays: list[dict], enriched: list[dict]) -> tuple[int, float]:
    ppa = {p["id"]: p["ppa"] for p in plays}
    diffs = [abs(r["ppa"] - ppa[r["playId"]]) for r in enriched
             if r.get("ppa") is not None and ppa.get(r["playId"]) is not None]
    return len(diffs), max(diffs, default=0.0)


def ppa_nulls(plays: list[dict]) -> dict[str, tuple[int, int]]:
    scrim = [p for p in plays if p["playType"] in SCRIMMAGE_TYPES]
    other = [p for p in plays if p["playType"] not in SCRIMMAGE_TYPES]
    return {
        "scrimmage plays with ppa": (sum(p["ppa"] is not None for p in scrim), len(scrim)),
        "other plays with ppa": (sum(p["ppa"] is not None for p in other), len(other)),
    }


def elo_chain(games: list[dict]) -> dict[str, Any]:
    """Is pregame Elo the team's previous postgame Elo? A leak would correlate the gap with this game's margin."""
    last: dict[str, int] = {}
    match = differ = first = missing = 0
    gaps, margins = [], []
    for g in sorted(games, key=lambda g: (g["startDate"], g["id"])):
        margin = (g.get("homePoints") or 0) - (g.get("awayPoints") or 0)
        for side, sign in (("home", 1), ("away", -1)):
            team, pre, post = g[f"{side}Team"], g[f"{side}PregameElo"], g[f"{side}PostgameElo"]
            if pre is None:
                missing += 1
            elif team not in last:
                first += 1
            elif last[team] == pre:
                match += 1
            else:
                differ += 1
                gaps.append(pre - last[team])
                margins.append(sign * margin)
            if post is not None:
                last[team] = post
    corr = statistics.correlation(gaps, margins) if len(gaps) > 2 else None
    return {"match": match, "differ": differ, "first": first, "missing": missing, "gap_margin_corr": corr}


def spread_convention(lines: list[dict]) -> tuple[int, int]:
    """Negative spread rows whose formattedSpread names the home team as favorite."""
    rows = [(g, line) for g in lines for line in g["lines"] if line.get("spread") is not None and line["spread"] < 0]
    return sum(line["formattedSpread"].startswith(g["homeTeam"]) for g, line in rows), len(rows)


def wp_spread_source(wp: list[dict], lines: list[dict]) -> dict[str, tuple[int, int]]:
    by_game = {w["gameId"]: w["spread"] for w in wp}
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for g in lines:
        for line in g["lines"]:
            if g["id"] in by_game and line.get("spread") is not None:
                out[line["provider"]][0] += abs(by_game[g["id"]] - line["spread"]) < 1e-9
                out[line["provider"]][1] += 1
    return {k: (v[0], v[1]) for k, v in sorted(out.items())}


def team_classes(games: list[dict]) -> dict[str, str]:
    out = {}
    for g in games:
        out[g["homeTeam"]] = g["homeClassification"]
        out[g["awayTeam"]] = g["awayClassification"]
    return out


# ---------- report ----------

def _table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _schema_rows(settings: Settings) -> list[list[Any]]:
    rows = []
    for path in sorted(settings.paths.raw.glob(f"**/{raw_store.RESPONSE_FILE}")):
        rel = path.relative_to(settings.paths.raw).as_posix()
        slug = rel.split("/")[1] if rel.startswith("_probe/") else rel.split("/")[0]
        payload = raw_store.read_json(path)
        if not payload or slug not in MODEL_BY_SLUG:
            continue
        d = compare(payload, MODEL_BY_SLUG[slug])
        issues = "none" if d.clean else "; ".join(
            f"{k}: {v}" for k, v in (("missing", d.missing), ("extra", d.extra),
                                     ("nulls", d.null_violations), ("types", d.type_mismatches)) if v)
        rows.append([f"`{rel.removesuffix('/' + raw_store.RESPONSE_FILE)}`", d.rows, issues])
    return rows


def build_report(settings: Settings) -> str:
    ctx, _ = load_landed(settings)
    manifest = Manifest(settings.paths).read()
    info_before, info_after = load_infos(settings)
    s, cs, es = PROBE_SEASON, COVERAGE_SEASON, ENRICHED_SEASON
    out = ["# Phase 1a probe report", "",
           "Generated by `cfb ingest probe --report` from `data/raw` and `data/manifest.parquet`.", ""]

    # Calls
    counted = int(manifest.filter(pl.col("endpoint") != "info")["attempts"].sum())
    info_calls = int(manifest.filter(pl.col("endpoint") == "info")["attempts"].sum())
    out += ["## 1. Calls and budget", ""]
    if info_before:
        out.append(f"Account: {info_before.get('tierName')}, monthly limit {info_before.get('monthlyLimit')}, "
                   f"resets {info_before.get('resetAt')}. Remaining after the latest run: "
                   f"{(info_after or {}).get('remainingCalls')}.")
    out += ["", f"HTTP requests recorded in the manifest: {counted} data calls + {info_calls} `/info` calls, "
            f"{int(manifest['attempts'].sum()) - manifest.height} retries. "
            "Comparing `/info` before and after each run shows `/info` itself is not counted against the quota.", ""]

    # Enriched endpoint availability
    enr = manifest.filter(pl.col("endpoint").is_in(["passing_plays", "rushing_plays"]) & (pl.col("http_status") == 200))
    out += ["## 2. `/passing/plays` and `/rushing/plays` availability", "",
            _table(["endpoint", "params", "rows"], [[r["endpoint"], f"`{r['params']}`", r["row_count"]]
                                                   for r in enr.sort("requested_at").iter_rows(named=True)]), "",
            "Empty for every 2014 and 2024 request regardless of filter; populated for 2025 and 2026. "
            "The registry sets `first_season=2025` for both, so bulk ingestion skips earlier seasons.", ""]

    # Schema
    out += ["## 3. Response shapes vs vendored models", "", _table(["partition", "rows", "differences"], _schema_rows(settings)), ""]

    # Joins (2025, all four sources)
    plays25, pas25, rus25 = ctx.get(f"plays_{es}", []), ctx.get(f"passing_{es}", []), ctx.get(f"rushing_{es}", [])
    stats25 = ctx.get(f"stats_{es}_power", [])
    ids25 = {p["id"] for p in plays25}
    out += [f"## 4. `play_id` joins ({es} week 1)", "", _table(["source", "ids found in /plays"], [
        ["/passing/plays", ratio(*join_rate([r["playId"] for r in pas25], ids25))],
        ["/rushing/plays", ratio(*join_rate([r["playId"] for r in rus25], ids25))],
        ["/plays/stats (one game)", ratio(*join_rate([r["playId"] for r in stats25], ids25))],
    ]), ""]
    for season, key in ((s, f"stats_{s}_power"), (s, f"stats_{s}_g5"), (s, f"stats_{s}_fcs"), (cs, f"stats_{cs}_power")):
        ids = {p["id"] for p in ctx.get(f"plays_{season}", [])}
        out.append(f"- {key}: {ratio(*join_rate([r['playId'] for r in ctx.get(key, [])], ids))} of play ids found in /plays")
    out.append("")

    classes = team_classes(ctx.get(f"games_{s}", []))
    cov = enriched_coverage(plays25, {r["playId"] for r in pas25} | {r["playId"] for r in rus25}, classes)
    out += ["Scrimmage plays in `/plays` that appear in either enriched endpoint, by offense classification "
            f"(classification from {s} `/games`):", "",
            _table(["offense", "covered"], [[k, ratio(*v)] for k, v in cov.items()]), ""]
    n, max_diff = ppa_agreement(plays25, pas25 + rus25)
    out += [f"`ppa` on enriched rows vs `/plays`: {n} matched plays, max |difference| = {max_diff}.", ""]

    # Status distributions
    out += [f"## 5. Status fields ({es} week 1)", "", _table(["field", "values"], [
        ["passing parseStatus", dict(Counter(r["parseStatus"] for r in pas25))],
        ["rushing parseStatus", dict(Counter(r["parseStatus"] for r in rus25))],
        ["rushing attributionStatus", dict(Counter(r["attributionStatus"] for r in rus25))],
        ["passing outcome", dict(Counter(r["outcome"] for r in pas25))],
        ["passing airYards non-null", ratio(sum(r["airYards"] is not None for r in pas25), len(pas25))],
        ["passing passerId non-null", ratio(sum(r["passerId"] is not None for r in pas25), len(pas25))],
        ["passing targetId non-null", ratio(sum(r["targetId"] is not None for r in pas25), len(pas25))],
        ["rushing isSack / isKneel / isTeamRush",
         f"{sum(r['isSack'] for r in rus25)} / {sum(r['isKneel'] for r in rus25)} / {sum(r['isTeamRush'] for r in rus25)}"],
    ]), ""]
    m, w, t = sack_rusher_is_passer(pas25, rus25)
    out += [f"Sacks: {t} sack rows, {w} with a `rusherId`, and {m} of those are a passer for the same game and offense.", ""]

    # Attribution from /plays/stats
    out += ["## 6. Player attribution from `/plays/stats`", ""]
    keys = [(s, f"stats_{s}_power"), (s, f"stats_{s}_g5"), (s, f"stats_{s}_fcs"), (cs, f"stats_{cs}_power"),
            (es, f"stats_{es}_power")]
    covs = {key: attribution_coverage(ctx.get(f"plays_{season}", []), ctx.get(key, [])) for season, key in keys if key in ctx}
    metrics = list(next(iter(covs.values())).keys()) if covs else []
    out += [_table(["role"] + list(covs), [[m] + [ratio(*covs[k][m]) for k in covs] for m in metrics]), ""]
    types = Counter(r["statType"] for _, key in keys for r in ctx.get(key, []))
    out += [f"Stat types seen across these games: {dict(types.most_common())}.",
            f"Defensive types never seen: {sorted(DEFENSIVE_STAT_TYPES - set(types))}.", ""]
    sacks = ctx.get(f"sacks_{s}", [])
    out += [f"Week-level pull `statTypeId=Sack` ({s} week 1): {len(sacks)} rows (cap 2,000); "
            f"`/plays` has {sum(p['playType'] == 'Sack' for p in ctx.get(f'plays_{s}', []))} sack plays that week.", ""]
    sizes = manifest.filter(pl.col("endpoint") == "plays_stats").select("params", "row_count")
    out += [f"Rows per game: {sorted(r for r in sizes['row_count'].to_list() if r is not None)}.", ""]

    # ppa
    out += ["## 7. `ppa` coverage in `/plays`", "", _table(["season", "scrimmage plays with ppa", "other plays with ppa"], [
        [season, ratio(*ppa_nulls(ctx.get(f"plays_{season}", []))["scrimmage plays with ppa"]),
         ratio(*ppa_nulls(ctx.get(f"plays_{season}", []))["other plays with ppa"])] for season in (cs, s, es)]), ""]

    # Lines, WP, Elo (leakage)
    lines, wp = ctx.get(f"lines_{s}", []), ctx.get(f"wp_{s}", [])
    conv = spread_convention(lines)
    out += ["## 8. Baselines and lookahead checks", "",
            f"- Spread sign: {ratio(*conv)} negative spreads name the home team as favorite in `formattedSpread`. "
            "Spreads are home-perspective, negative = home favored.",
            f"- Opening spreads present: {ratio(sum(l['spreadOpen'] is not None for g in lines for l in g['lines']), sum(1 for g in lines for l in g['lines']))}.",
            f"- `/metrics/wp/pregame` spread equals each provider's closing spread: "
            + ", ".join(f"{k} {ratio(*v)}" for k, v in wp_spread_source(wp, lines).items()) + "."]
    for season in (cs, s):
        e = elo_chain(ctx.get(f"games_{season}", []))
        corr = "n/a" if e["gap_margin_corr"] is None else f"{e['gap_margin_corr']:.2f}"
        out.append(f"- {season} pregame Elo equals the team's previous postgame Elo: {e['match']} match, {e['differ']} differ "
                   f"(gap vs this game's margin: corr {corr}), {e['first']} first games, {e['missing']} missing.")
    out.append("")
    return "\n".join(out)


def write_report(settings: Settings) -> Path:
    path = settings.paths.reports / "phase1a_probe.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(settings))
    return path


# ---------- fixtures ----------

FIXTURE_GAMES = {PROBE_SEASON: 401628322, COVERAGE_SEASON: 400547743, ENRICHED_SEASON: 401752665}
ELO_TEAMS = ("Florida", "Miami", "SMU", "Arkansas")


def write_fixtures(settings: Settings, out_dir: Path) -> list[Path]:
    ctx, _ = load_landed(settings)
    s, cs, es = PROBE_SEASON, COVERAGE_SEASON, ENRICHED_SEASON
    files: dict[str, Any] = {}

    def game(rows, gid, key="gameId"):
        return [r for r in rows if r.get(key) == gid]

    for season, key in ((s, f"stats_{s}_power"), (cs, f"stats_{cs}_power"), (es, f"stats_{es}_power")):
        gid = FIXTURE_GAMES[season]
        files[f"plays_{season}_{gid}"] = game(ctx[f"plays_{season}"], gid)
        files[f"plays_stats_{season}_{gid}"] = ctx[key]
    gid = FIXTURE_GAMES[es]
    files[f"passing_plays_{es}_{gid}"] = game(ctx[f"passing_{es}"], gid)
    files[f"rushing_plays_{es}_{gid}"] = game(ctx[f"rushing_{es}"], gid)
    files[f"passing_plays_{s}_empty"] = []

    games = [g for g in ctx[f"games_{s}"] if g["homeTeam"] in ELO_TEAMS or g["awayTeam"] in ELO_TEAMS]
    ids = {g["id"] for g in games}
    files[f"games_{s}_sample"] = games
    files[f"lines_{s}_sample"] = [g for g in ctx[f"lines_{s}"] if g["id"] in ids]
    files[f"metrics_wp_pregame_{s}_sample"] = [w for w in ctx[f"wp_{s}"] if w["gameId"] in ids]
    files[f"games_players_{s}_{FIXTURE_GAMES[s]}"] = game(ctx[f"box_{s}"], FIXTURE_GAMES[s], key="id")
    files[f"ppa_players_games_{s}_sample"] = [r for r in ctx[f"ppa_games_{s}"] if r["team"] in ("Florida", "Miami")]
    files[f"plays_stats_week_sacks_{s}_sample"] = ctx[f"sacks_{s}"][:25]
    files[f"calendar_{s}"] = ctx[f"calendar_{s}"]
    files["plays_types"] = ctx["plays_types"]
    files["plays_stats_types"] = ctx["plays_stats_types"]
    info, _ = load_infos(settings)
    if info:  # same shape, placeholder values
        files["info_sanitized"] = {**info, "patronLevel": 0, "tierName": "Tier X", "monthlyLimit": 1000,
                                   "remainingCalls": 900, "usedCalls": 100, "resetAt": "2026-01-01T00:00:00.000Z"}

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, payload in files.items():
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")))
        written.append(path)
    return written
