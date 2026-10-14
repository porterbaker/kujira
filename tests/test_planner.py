from dataclasses import replace
from datetime import datetime, timezone

from cfb.ingest.planner import build_plan
from cfb.ingest.raw_store import season_part, week_parts
from cfb.ingest.registry import ENDPOINTS

from conftest import write_raw

NOW = datetime(2026, 10, 13, 12, tzinfo=timezone.utc)

N_ONCE = sum(s.granularity == "once" for s in ENDPOINTS)
N_SEASON = sum(s.granularity == "season" for s in ENDPOINTS)
N_SEASON_TYPE = sum(s.granularity == "season_type" and s.flag is None for s in ENDPOINTS)


def n_week(season, division_only=False):
    return sum(s.granularity == "week" and (s.first_season or 0) <= season
               and (s.division_param is not None or not division_only) for s in ENDPOINTS)


N_WEEK, N_WEEK_DIVISION = n_week(2024), n_week(2024, division_only=True)


def _with(settings, **ingest):
    return replace(settings, ingest=replace(settings.ingest, **ingest))


def _by_slug(plan, slug):
    return [r for r in plan.requests if r.spec.slug == slug]


def test_completed_season_estimate_matches_formula(settings):
    ing = settings.ingest
    plan = build_plan((2024,), settings, now=NOW)
    weeks = ing.est_regular_weeks + ing.est_postseason_weeks
    games = ing.est_fbs_games_regular + ing.est_fbs_games_postseason
    recruit_classes = ing.recruit_lookback + 1
    expected = N_ONCE + N_SEASON + 2 * N_SEASON_TYPE + N_WEEK * weeks + games + 3 * recruit_classes
    assert plan.total == expected
    assert plan.estimated_count == N_WEEK * weeks + games


def test_fcs_and_weather_flags_add_calls(settings):
    ing = settings.ingest
    base = build_plan((2024,), settings, now=NOW).total
    fcs = build_plan((2024,), _with(settings, include_fcs=True), now=NOW).total
    weeks = ing.est_regular_weeks + ing.est_postseason_weeks
    fcs_games = ing.est_fcs_only_games_regular + ing.est_fcs_only_games_postseason
    games_endpoint_extra = 2  # /games: an fcs request per season type
    assert fcs - base == N_WEEK_DIVISION * weeks + games_endpoint_extra + fcs_games
    weather = build_plan((2024,), _with(settings, include_weather=True), now=NOW).total
    assert weather - base == 2


def test_enriched_play_endpoints_start_in_2025(settings):
    plan = build_plan((2024, 2025), settings, now=NOW)
    seasons = {b.season for b in plan.estimates if b.spec.slug in ("passing_plays", "rushing_plays")}
    assert seasons == {2025}
    assert n_week(2025) == N_WEEK + 2


def test_every_season_request_passes_year_explicitly(settings):
    plan = build_plan((2022, 2023), settings, now=NOW)
    for req in plan.requests:
        if req.spec.granularity != "once":
            assert "year" in req.params, req.spec.path
    roster_years = sorted(r.params["year"] for r in _by_slug(plan, "roster"))
    assert roster_years == [2022, 2023]
    draft_years = sorted(r.params["year"] for r in _by_slug(plan, "draft_picks"))
    assert draft_years == [2023, 2024]  # draft year = season + 1


def test_exact_plan_from_calendar_and_games(settings):
    write_raw(settings, "calendar", (season_part(2024),), [
        {"season": 2024, "week": 1, "seasonType": "regular"},
        {"season": 2024, "week": 2, "seasonType": "regular"},
        {"season": 2024, "week": 1, "seasonType": "postseason"},
        {"season": 2024, "week": 1, "seasonType": "allstar"},
    ])
    write_raw(settings, "games", (season_part(2024), ("type", "regular")), [
        {"id": 1, "homeClassification": "fbs", "awayClassification": "fbs", "completed": True},
        {"id": 2, "homeClassification": "fbs", "awayClassification": "fcs", "completed": True},
        {"id": 3, "homeClassification": "fcs", "awayClassification": "fcs", "completed": True},
    ])
    plan = build_plan((2024,), settings, endpoints={"plays", "plays_stats"}, now=NOW)
    assert plan.estimated_count == 0

    plays = _by_slug(plan, "plays")
    assert [(r.params["seasonType"], r.params["week"]) for r in plays] == [
        ("regular", 1), ("regular", 2), ("postseason", 1)]
    assert all(r.params["classification"] == "fbs" for r in plays)
    partitions = {r.partition for r in plays}
    assert (season_part(2024),) + week_parts("regular", 1) in partitions
    assert (season_part(2024),) + week_parts("postseason", 1) in partitions
    assert len(partitions) == 3

    assert sorted(r.params["gameId"] for r in _by_slug(plan, "plays_stats")) == [1, 2]
    fcs = build_plan((2024,), _with(settings, include_fcs=True), endpoints={"plays_stats"}, now=NOW)
    assert sorted(r.params["gameId"] for r in fcs.requests) == [1, 2, 3]


def test_current_season_only_plans_what_has_happened(settings):
    write_raw(settings, "calendar", (season_part(2026),), [
        {"week": 6, "seasonType": "regular", "lastGameStart": "2026-10-03T23:00:00.000Z"},
        {"week": 7, "seasonType": "regular", "lastGameStart": "2026-10-10T23:00:00.000Z"},
        {"week": 8, "seasonType": "regular", "lastGameStart": "2026-10-17T23:00:00.000Z"},
    ])
    write_raw(settings, "games", (season_part(2026), ("type", "regular")), [
        {"id": 10, "homeClassification": "fbs", "awayClassification": "fbs", "completed": True},
        {"id": 11, "homeClassification": "fbs", "awayClassification": "fbs", "completed": False},
    ])
    plan = build_plan((2026,), settings, now=NOW)
    assert [r.params["week"] for r in _by_slug(plan, "plays")] == [6, 7]
    assert [r.params["gameId"] for r in _by_slug(plan, "plays_stats")] == [10]
    assert _by_slug(plan, "draft_picks") == []  # 2027 draft hasn't happened
    games = [r for r in plan.requests + plan.skipped if r.spec.slug == "games"]
    assert {r.params["seasonType"] for r in games} == {"regular"}  # no postseason before mid-December


def test_landed_partitions_skipped_and_refresh_only_current_season(settings):
    write_raw(settings, "roster", (season_part(2024),), [])
    write_raw(settings, "roster", (season_part(2026),), [])
    plan = build_plan((2024, 2026), settings, endpoints={"roster"}, now=NOW)
    assert plan.requests == [] and len(plan.skipped) == 2

    refreshed = build_plan((2024, 2026), settings, endpoints={"roster"}, refresh=True, now=NOW)
    assert [r.params["year"] for r in refreshed.requests] == [2026]
    assert [r.params["year"] for r in refreshed.skipped] == [2024]
