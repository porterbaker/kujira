"""Phase 1a probe: a small, fixed set of live calls that checks the spec before bulk ingestion.

Most steps are ordinary planner requests, so they land in the canonical raw tree and bulk
ingestion reuses them. Experiments that bulk never makes (sacks for a whole week, /info)
land under ``data/raw/_probe/``. Steps already on disk are read locally, not re-pulled.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cfb.api.budget import check_budget, describe_info
from cfb.api.client import CFBDClient
from cfb.config import Settings
from cfb.ingest import raw_store, runner
from cfb.ingest.manifest import Manifest
from cfb.ingest.planner import (
    PlannedRequest,
    game_request,
    once_request,
    season_request,
    season_type_request,
    week_request,
)
from cfb.ingest.raw_store import season_part, week_parts
from cfb.ingest.registry import BY_SLUG, EndpointSpec

PROBE_SEASON, PROBE_WEEK = 2024, 1  # largest week, most FBS-vs-FCS games
COVERAGE_SEASON, COVERAGE_WEEK = 2014, 5  # first season of the chosen range
ENRICHED_SEASON, ENRICHED_WEEK = 2025, 1  # /passing/plays and /rushing/plays are empty before 2025
POWER_CONFERENCES = frozenset({"SEC", "Big Ten", "Big 12", "ACC", "Pac-12"})

INFO = EndpointSpec("/info", "once", "lookup", availability="static")
INFO_CALLS = 2  # /info before (budget check) and after (to see what the run cost)


class Unresolved(Exception):
    """A step's inputs come from an earlier step that isn't available yet."""


@dataclass(frozen=True)
class Step:
    label: str
    describe: str
    build: Callable[[dict[str, Any]], PlannedRequest | str]  # str = skip with this reason
    probe_only: bool = False


def _need(ctx: dict[str, Any], label: str) -> Any:
    if label not in ctx:
        raise Unresolved(label)
    return ctx[label]


def pick_game(games: list[dict], week: int, kind: str) -> int | str:
    """Lowest completed game id of a kind: power (P4/P5 vs P4/P5), g5 (FBS non-power), fcs (FBS vs FCS)."""
    ids = []
    for g in games:
        if g.get("week") != week or not g.get("completed"):
            continue
        classes = (g.get("homeClassification"), g.get("awayClassification"))
        power = [c in POWER_CONFERENCES for c in (g.get("homeConference"), g.get("awayConference"))]
        if kind == "power":
            ok = classes == ("fbs", "fbs") and all(power)
        elif kind == "g5":
            ok = classes == ("fbs", "fbs") and not any(power)
        else:
            ok = sorted(c or "" for c in classes) == ["fbs", "fcs"]
        if ok:
            ids.append(int(g["id"]))
    return min(ids) if ids else f"no completed {kind} game in week {week}"


def sack_stat_type_id(stat_types: list[dict]) -> int | str:
    for row in stat_types:
        if str(row.get("name", "")).strip().lower() == "sack":
            return int(row["id"])
    return "no stat type named 'Sack'"


def _power_game_from_plays(plays: list[dict]) -> int | str:
    ids = [p["gameId"] for p in plays
           if p.get("offenseConference") in POWER_CONFERENCES and p.get("defenseConference") in POWER_CONFERENCES]
    return min(ids) if ids else "no power-vs-power game in plays"


def _game_step(label: str, games_label: str, season: int, week: int, kind: str) -> Step:
    def build(ctx):
        picked = pick_game(_need(ctx, games_label), week, kind)
        return picked if isinstance(picked, str) else game_request(BY_SLUG["plays_stats"], season, picked)

    return Step(label, f"GET /plays/stats?gameId=<{kind} game, {season} week {week}, chosen from /games>", build)


def _sacks_build(ctx):
    sack = sack_stat_type_id(_need(ctx, "plays_stats_types"))
    if isinstance(sack, str):
        return sack
    params = {"year": PROBE_SEASON, "week": PROBE_WEEK, "seasonType": "regular", "statTypeId": sack}
    part = (season_part(PROBE_SEASON),) + week_parts("regular", PROBE_WEEK) + (("stat_type", str(sack)),)
    return PlannedRequest(BY_SLUG["plays_stats"], params, part)


def _fixed(label: str, req: PlannedRequest) -> Step:
    query = "&".join(f"{k}={v}" for k, v in req.params.items())
    return Step(label, f"GET {req.spec.path}{'?' + query if query else ''}", lambda ctx: req)


def steps() -> list[Step]:
    s, w, cs, cw, es, ew = PROBE_SEASON, PROBE_WEEK, COVERAGE_SEASON, COVERAGE_WEEK, ENRICHED_SEASON, ENRICHED_WEEK
    ep = BY_SLUG

    def enriched_game(ctx):
        picked = _power_game_from_plays(_need(ctx, f"plays_{es}"))
        return picked if isinstance(picked, str) else game_request(ep["plays_stats"], es, picked)

    return [
        _fixed("plays_types", once_request(ep["plays_types"])),
        _fixed("plays_stats_types", once_request(ep["plays_stats_types"])),
        _fixed(f"calendar_{s}", season_request(ep["calendar"], s)),
        _fixed(f"games_{s}", season_type_request(ep["games"], s, "regular")),
        _fixed(f"plays_{s}", week_request(ep["plays"], s, "regular", w)),
        _fixed(f"box_{s}", week_request(ep["games_players"], s, "regular", w)),
        _fixed(f"ppa_games_{s}", week_request(ep["ppa_players_games"], s, "regular", w)),
        _fixed(f"lines_{s}", season_type_request(ep["lines"], s, "regular")),
        _fixed(f"wp_{s}", season_type_request(ep["metrics_wp_pregame"], s, "regular")),
        _game_step(f"stats_{s}_power", f"games_{s}", s, w, "power"),
        _game_step(f"stats_{s}_g5", f"games_{s}", s, w, "g5"),
        _game_step(f"stats_{s}_fcs", f"games_{s}", s, w, "fcs"),
        Step(f"sacks_{s}", f"GET /plays/stats?year={s}&week={w}&seasonType=regular&statTypeId=<Sack>",
             _sacks_build, probe_only=True),
        _fixed(f"games_{cs}", season_type_request(ep["games"], cs, "regular")),
        _fixed(f"plays_{cs}", week_request(ep["plays"], cs, "regular", cw)),
        _game_step(f"stats_{cs}_power", f"games_{cs}", cs, cw, "power"),
        _fixed(f"plays_{es}", week_request(ep["plays"], es, "regular", ew)),
        _fixed(f"passing_{es}", week_request(ep["passing_plays"], es, "regular", ew)),
        _fixed(f"rushing_{es}", week_request(ep["rushing_plays"], es, "regular", ew)),
        Step(f"stats_{es}_power", f"GET /plays/stats?gameId=<power game, {es} week {ew}, chosen from /plays>",
             enriched_game),
    ]


def probe_raw_root(settings: Settings) -> Path:
    return settings.paths.raw / "_probe"


def _root_for(step: Step, settings: Settings) -> Path:
    return probe_raw_root(settings) if step.probe_only else settings.paths.raw


@dataclass
class StepStatus:
    step: Step
    request: PlannedRequest | None  # None = unresolved (dry run) or skipped
    landed: bool
    skip_reason: str | None = None


def load_landed(settings: Settings) -> tuple[dict[str, Any], list[StepStatus]]:
    """Resolve steps against what is on disk. Pure and offline."""
    ctx: dict[str, Any] = {}
    statuses = []
    for step in steps():
        try:
            built = step.build(ctx)
        except Unresolved:
            statuses.append(StepStatus(step, None, False))
            continue
        if isinstance(built, str):
            statuses.append(StepStatus(step, None, False, built))
            continue
        path = raw_store.response_path(_root_for(step, settings), built.spec.slug, built.partition)
        if path.is_file():
            ctx[step.label] = raw_store.read_json(path)
        statuses.append(StepStatus(step, built, path.is_file()))
    return ctx, statuses


def planned_calls(statuses: list[StepStatus]) -> int:
    return INFO_CALLS + sum(1 for s in statuses if not s.landed and s.skip_reason is None)


def format_probe_plan(settings: Settings) -> str:
    _, statuses = load_landed(settings)
    lines = [f"Phase 1a probe: {PROBE_SEASON} week {PROBE_WEEK}, coverage check {COVERAGE_SEASON} week {COVERAGE_WEEK}, "
             f"enriched endpoints {ENRICHED_SEASON} week {ENRICHED_WEEK}", ""]
    lines.append("  GET /info  (budget check before the run)")
    for st in statuses:
        mark = "landed, read locally" if st.landed else (f"skip: {st.skip_reason}" if st.skip_reason else "")
        where = " [_probe]" if st.step.probe_only else ""
        lines.append(f"  {st.step.describe}{where}{'  (' + mark + ')' if mark else ''}")
    lines.append("  GET /info  (after the run, to measure what it cost)")
    lines += ["", f"Planned calls: {planned_calls(statuses)} (before retries)"]
    return "\n".join(lines)


def run_probe(settings: Settings, client: CFBDClient, manifest: Manifest) -> dict[str, Any]:
    """Execute the probe. Returns the context: payloads by step label, plus /info before/after."""
    run_id = runner.new_run_id()
    _, statuses = load_landed(settings)
    planned = planned_calls(statuses)

    def info_step(when: str) -> dict:
        req = PlannedRequest(INFO, {}, (("run", run_id), ("when", when)))
        out = runner.execute(req, client, settings, manifest, run_id, raw_root=probe_raw_root(settings))
        if out.status != 200 or not out.payload:
            raise SystemExit(f"/info failed (HTTP {out.status}); is CFB_APIKEY valid?")
        return out.payload

    info_before = info_step("before")
    print("account:", describe_info(info_before))
    budget = check_budget(info_before, planned - 1, settings.api.budget_buffer)
    print("budget:", budget.message())
    if not budget.ok:
        manifest.compact()
        raise SystemExit("aborting: planned calls exceed remaining calls minus buffer")

    ctx: dict[str, Any] = {}
    for step in steps():
        try:
            built = step.build(ctx)
        except Unresolved as exc:
            print(f"  skip {step.label}: input {exc} unavailable")
            continue
        if isinstance(built, str):
            print(f"  skip {step.label}: {built}")
            continue
        root = _root_for(step, settings)
        path = raw_store.response_path(root, built.spec.slug, built.partition)
        if path.is_file():
            ctx[step.label] = raw_store.read_json(path)
            print(f"  local {step.label}")
            continue
        out = runner.execute(built, client, settings, manifest, run_id, raw_root=root)
        note = f"{out.row_count} rows" if out.status == 200 else f"HTTP {out.status}"
        print(f"  GET   {step.label}: {note}{' (TRUNCATED?)' if out.truncated else ''}")
        if out.status == 200:
            ctx[step.label] = out.payload

    ctx["_info_before"] = info_before
    ctx["_info_after"] = info_step("after")
    ctx["_calls_made"] = client.calls_made
    ctx["_planned"] = planned
    manifest.compact()
    print(f"calls made: {client.calls_made} (planned {planned})")
    print("account after:", describe_info(ctx["_info_after"]))
    return ctx


def load_infos(settings: Settings) -> tuple[dict | None, dict | None]:
    """Latest /info before/after pair from _probe, for offline report regeneration."""
    runs = sorted((probe_raw_root(settings) / "info").glob("run=*"))
    for run in reversed(runs):
        before, after = run / "when=before" / raw_store.RESPONSE_FILE, run / "when=after" / raw_store.RESPONSE_FILE
        if before.is_file() and after.is_file():
            return raw_store.read_json(before), raw_store.read_json(after)
    return None, None
