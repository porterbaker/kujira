"""Expand the endpoint registry into concrete requests. Pure and offline.

Weekly and per-game requests depend on ``/calendar`` and ``/games``. When those are on
disk, the plan is exact. When they aren't, the planner emits an estimated count instead,
shown with ``~``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from cfb.config import Settings
from cfb.ingest import raw_store
from cfb.ingest.raw_store import Partition, season_part, week_parts
from cfb.ingest.registry import ENDPOINTS, RECRUIT_TYPES, SEASON_TYPES, EndpointSpec


@dataclass(frozen=True)
class PlannedRequest:
    spec: EndpointSpec
    params: dict[str, object]
    partition: Partition


@dataclass(frozen=True)
class EstimatedBlock:
    spec: EndpointSpec
    season: int
    count: int
    reason: str


@dataclass
class Plan:
    seasons: tuple[int, ...]
    include_fcs: bool
    include_weather: bool
    refresh: bool
    requests: list[PlannedRequest] = field(default_factory=list)
    estimates: list[EstimatedBlock] = field(default_factory=list)
    skipped: list[PlannedRequest] = field(default_factory=list)  # already landed

    @property
    def exact_count(self) -> int:
        return len(self.requests)

    @property
    def estimated_count(self) -> int:
        return sum(block.count for block in self.estimates)

    @property
    def total(self) -> int:
        return self.exact_count + self.estimated_count

    def counts_by_endpoint(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = defaultdict(lambda: {"exact": 0, "est": 0, "skipped": 0})
        for req in self.requests:
            out[req.spec.slug]["exact"] += 1
        for block in self.estimates:
            out[block.spec.slug]["est"] += block.count
        for req in self.skipped:
            out[req.spec.slug]["skipped"] += 1
        return out


def once_request(spec: EndpointSpec) -> PlannedRequest:
    return PlannedRequest(spec, dict(spec.params), (season_part("all"),))


def season_request(spec: EndpointSpec, season: int) -> PlannedRequest:
    year = season + spec.year_offset  # partitioned by the year actually requested
    return PlannedRequest(spec, {**spec.params, "year": year}, (season_part(year),))


def recruit_request(spec: EndpointSpec, year: int, recruit_type: str) -> PlannedRequest:
    return PlannedRequest(spec, {**spec.params, "year": year, "classification": recruit_type},
                          (season_part(year), ("recruit_type", recruit_type)))


def season_type_request(spec: EndpointSpec, season: int, season_type: str, division: str = "fbs") -> PlannedRequest:
    return _with_division(spec, {**spec.params, "year": season, "seasonType": season_type},
                          (season_part(season), ("type", season_type)), division)


def week_request(spec: EndpointSpec, season: int, season_type: str, week: int, division: str = "fbs") -> PlannedRequest:
    return _with_division(spec, {**spec.params, "year": season, "week": week, "seasonType": season_type},
                          (season_part(season),) + week_parts(season_type, week), division)


def game_request(spec: EndpointSpec, season: int, game_id: int) -> PlannedRequest:
    return PlannedRequest(spec, {**spec.params, "gameId": game_id}, (season_part(season), ("game", str(game_id))))


def _with_division(spec: EndpointSpec, params: dict[str, object], partition: Partition, division: str) -> PlannedRequest:
    if spec.division_param is None:
        return PlannedRequest(spec, params, partition)
    suffix: Partition = () if division == "fbs" else (("division", division),)
    return PlannedRequest(spec, {**params, spec.division_param: division}, partition + suffix)


def build_plan(
    seasons: tuple[int, ...],
    settings: Settings,
    *,
    endpoints: set[str] | None = None,
    refresh: bool = False,
    now: datetime | None = None,
) -> Plan:
    now = now or datetime.now(timezone.utc)
    ing = settings.ingest
    raw = settings.paths.raw
    plan = Plan(seasons, ing.include_fcs, ing.include_weather, refresh)
    divisions = ("fbs", "fcs") if ing.include_fcs else ("fbs",)

    def add(req: PlannedRequest, season: int | None) -> None:
        refreshable = refresh and season == ing.current_season
        if raw_store.is_landed(raw, req.spec.slug, req.partition) and not refreshable:
            plan.skipped.append(req)
        else:
            plan.requests.append(req)

    for spec in ENDPOINTS:
        if endpoints is not None and spec.slug not in endpoints:
            continue
        if spec.flag and not getattr(ing, spec.flag):
            continue
        spec_divisions = divisions if spec.division_param else ("fbs",)

        if spec.granularity == "once":
            add(once_request(spec), None)

        elif spec.granularity == "recruit_class":
            classes = sorted({y for s in seasons for y in range(s - ing.recruit_lookback, s + 1)})
            for year in (y for y in classes if y <= ing.current_season):
                for rtype in RECRUIT_TYPES:
                    add(recruit_request(spec, year, rtype), year)

        for season in seasons:
            if spec.first_season is not None and season < spec.first_season:
                continue
            if spec.granularity == "season":
                if season + spec.year_offset > ing.current_season:
                    continue  # e.g. the draft for the in-progress season hasn't happened
                add(season_request(spec, season), season)

            elif spec.granularity == "season_type":
                for stype in _season_types(season, ing.current_season, now):
                    for div in spec_divisions:
                        add(season_type_request(spec, season, stype, div), season)

            elif spec.granularity == "week":
                weeks = _calendar_weeks(raw, season, ing.current_season, now)
                if weeks is None:
                    n = _estimated_weeks(season, settings, now) * len(spec_divisions)
                    plan.estimates.append(EstimatedBlock(spec, season, n, "no /calendar on disk"))
                    continue
                for stype, week in weeks:
                    for div in spec_divisions:
                        add(week_request(spec, season, stype, week, div), season)

            elif spec.granularity == "game":
                game_ids = _game_ids(raw, season, divisions, ing.current_season)
                if game_ids is None:
                    n = _estimated_games(season, settings, now)
                    plan.estimates.append(EstimatedBlock(spec, season, n, "no /games on disk"))
                    continue
                for game_id in game_ids:
                    add(game_request(spec, season, game_id), season)

    return plan


def _season_types(season: int, current_season: int, now: datetime) -> tuple[str, ...]:
    if season == current_season and now.date() < _postseason_start(season):
        return ("regular",)
    return SEASON_TYPES


def _postseason_start(season: int) -> date:
    return date(season, 12, 13)


def _calendar_weeks(raw: Path, season: int, current_season: int, now: datetime) -> list[tuple[str, int]] | None:
    path = raw_store.response_path(raw, "calendar", (season_part(season),))
    if not path.is_file():
        return None
    weeks = []
    for row in raw_store.read_json(path):
        stype = row.get("seasonType")
        if stype not in SEASON_TYPES:
            continue
        if season == current_season:
            last_start = row.get("lastGameStart") or row.get("endDate")
            if last_start is None or datetime.fromisoformat(last_start) > now:
                continue  # week not fully kicked off yet
        weeks.append((stype, int(row["week"])))
    return sorted(set(weeks), key=lambda w: (SEASON_TYPES.index(w[0]), w[1]))


def _game_ids(raw: Path, season: int, divisions: tuple[str, ...], current_season: int) -> list[int] | None:
    season_dir = raw_store.partition_dir(raw, "games", (season_part(season),))
    files = sorted(season_dir.glob(f"**/{raw_store.RESPONSE_FILE}"))
    if not files:
        return None
    ids: set[int] = set()
    for path in files:
        for game in raw_store.read_json(path):
            if season == current_season and not game.get("completed"):
                continue
            if game.get("homeClassification") in divisions or game.get("awayClassification") in divisions:
                ids.add(int(game["id"]))
    return sorted(ids)


def _season_progress(season: int, settings: Settings, now: datetime) -> tuple[int, int]:
    """(regular weeks, postseason weeks) expected to have data, for dry-run estimates."""
    ing = settings.ingest
    if season < ing.current_season:
        return ing.est_regular_weeks, ing.est_postseason_weeks
    elapsed = (now.date() - date(season, 8, 24)).days // 7
    regular = max(0, min(ing.est_regular_weeks, elapsed))
    post = ing.est_postseason_weeks if now.date() >= _postseason_start(season) else 0
    return regular, post


def _estimated_weeks(season: int, settings: Settings, now: datetime) -> int:
    regular, post = _season_progress(season, settings, now)
    return regular + post


def _estimated_games(season: int, settings: Settings, now: datetime) -> int:
    ing = settings.ingest
    regular, post = _season_progress(season, settings, now)
    share = regular / ing.est_regular_weeks
    games = round(ing.est_fbs_games_regular * share) + (ing.est_fbs_games_postseason if post else 0)
    if ing.include_fcs:
        games += round(ing.est_fcs_only_games_regular * share) + (ing.est_fcs_only_games_postseason if post else 0)
    return games


def format_plan(plan: Plan) -> str:
    seasons = plan.seasons
    span = f"{seasons[0]}" if len(seasons) == 1 else f"{seasons[0]}-{seasons[-1]} ({len(seasons)} seasons)"
    lines = [
        f"Ingest plan: seasons {span}; fcs={'on' if plan.include_fcs else 'off'}, "
        f"weather={'on' if plan.include_weather else 'off'}, refresh={'on' if plan.refresh else 'off'}",
        "",
        f"{'endpoint':<24}{'role':<11}{'granularity':<15}{'exact':>7}{'est.':>8}{'calls':>8}{'landed':>8}",
    ]
    counts = plan.counts_by_endpoint()
    for spec in ENDPOINTS:
        if spec.slug not in counts:
            continue
        c = counts[spec.slug]
        calls = c["exact"] + c["est"]
        est = f"~{c['est']}" if c["est"] else "0"
        calls_s = f"~{calls}" if c["est"] else str(calls)
        lines.append(f"{spec.slug:<24}{spec.role:<11}{spec.granularity:<15}{c['exact']:>7}{est:>8}{calls_s:>8}{c['skipped']:>8}")
    total = f"~{plan.total}" if plan.estimated_count else str(plan.total)
    lines += [
        "-" * 81,
        f"{'TOTAL':<50}{plan.exact_count:>7}{('~' + str(plan.estimated_count)) if plan.estimated_count else '0':>8}{total:>8}{len(plan.skipped):>8}",
        "",
        "exact = concrete requests; est. = estimated (no /calendar or /games on disk yet; exact after stage A);",
        "landed = partitions already on disk and skipped. /plays/stats re-pulls after 2,000-row pages are not included.",
        "Retries count as calls, so keep the budget buffer in mind.",
    ]
    return "\n".join(lines)
