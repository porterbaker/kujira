"""Probe findings, pinned against trimmed live responses in tests/fixtures/."""

import pytest

from cfb.ingest import probe_report as pr
from cfb.ingest.schema_check import compare

from conftest import FIXTURES, load_fixture

MODEL_BY_FIXTURE_PREFIX = {
    "plays_stats_week": "play_stat",
    "plays_stats_types": "play_stat_type",
    "plays_stats": "play_stat",
    "plays_types": "play_type",
    "plays": "play",
    "passing_plays": "passing_play",
    "rushing_plays": "rushing_play",
    "games_players": "game_player_stats",
    "games": "game",
    "lines": "betting_game",
    "metrics_wp_pregame": "pregame_win_probability",
    "ppa_players_games": "player_game_predicted_points_added",
    "calendar": "calendar_week",
    "info": "user_info",
}

GAMES = {2024: 401628322, 2014: 400547743, 2025: 401752665}


def _model_for(name: str) -> str:
    return next(model for prefix, model in MODEL_BY_FIXTURE_PREFIX.items() if name.startswith(prefix))


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.json")), ids=lambda p: p.stem)
def test_fixtures_match_vendored_models(path):
    payload = load_fixture(path.stem)
    if not payload:
        pytest.skip("empty response")
    diff = compare(payload, _model_for(path.stem))
    assert diff.clean, diff


@pytest.mark.parametrize("season", sorted(GAMES))
def test_plays_stats_join_and_attribution(season):
    gid = GAMES[season]
    plays, stats = load_fixture(f"plays_{season}_{gid}"), load_fixture(f"plays_stats_{season}_{gid}")
    matched, total = pr.join_rate([s["playId"] for s in stats], {p["id"] for p in plays})
    assert matched == total > 0
    cov = pr.attribution_coverage(plays, stats)
    for role in ("passer on pass plays", "receiver on completions", "QB on sacks"):
        n, d = cov[role]
        assert n == d > 0, role
    n, d = cov["rusher on rushes"]
    assert n / d > 0.9


def test_no_tackle_level_defensive_credits_in_plays_stats():
    types = {s["statType"] for season, gid in GAMES.items() for s in load_fixture(f"plays_stats_{season}_{gid}")}
    assert not types & {"Tackle", "Pass Breakup", "QB Hurry", "Fumble Forced"}


def test_enriched_endpoints_join_plays_and_agree_on_ppa():
    gid = GAMES[2025]
    plays = load_fixture(f"plays_2025_{gid}")
    passing, rushing = load_fixture(f"passing_plays_2025_{gid}"), load_fixture(f"rushing_plays_2025_{gid}")
    ids = {p["id"] for p in plays}
    assert pr.join_rate([r["playId"] for r in passing + rushing], ids) == (len(passing) + len(rushing),) * 2
    n, max_diff = pr.ppa_agreement(plays, passing + rushing)
    assert n > 100 and max_diff == 0.0


def test_sack_rows_credit_the_quarterback():
    gid = GAMES[2025]
    matched, with_id, sacks = pr.sack_rusher_is_passer(load_fixture(f"passing_plays_2025_{gid}"),
                                                       load_fixture(f"rushing_plays_2025_{gid}"))
    assert sacks > 0 and matched == with_id


def test_ppa_present_on_scrimmage_plays_back_to_2014():
    for season, gid in GAMES.items():
        n, d = pr.ppa_nulls(load_fixture(f"plays_{season}_{gid}"))["scrimmage plays with ppa"]
        assert n / d > 0.99


def test_spread_is_home_perspective():
    n, d = pr.spread_convention(load_fixture("lines_2024_sample"))
    assert n == d > 0


def test_pregame_wp_spread_is_bovada_closing():
    n, d = pr.wp_spread_source(load_fixture("metrics_wp_pregame_2024_sample"), load_fixture("lines_2024_sample"))["Bovada"]
    assert n == d > 0
