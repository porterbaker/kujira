"""Lookahead guards at the raw-data level. Feature-level as-of tests come in Phase 4."""

from cfb.ingest.probe_report import elo_chain
from cfb.ingest.registry import ENDPOINTS
from cfb.leakage import (
    GAME_POSTGAME_FIELDS,
    GAME_PREGAME_FIELDS,
    never_feature_endpoints,
    same_season_forbidden_endpoints,
)

from conftest import load_fixture


def test_every_games_field_is_classified_exactly_once():
    fields = {key for game in load_fixture("games_2024_sample") for key in game}
    assert not GAME_PREGAME_FIELDS & GAME_POSTGAME_FIELDS
    unclassified = fields - GAME_PREGAME_FIELDS - GAME_POSTGAME_FIELDS
    assert unclassified == set(), "new /games field: decide whether it is known before kickoff"


def test_benchmarks_lines_and_draft_are_never_features():
    assert {"lines", "metrics_wp_pregame", "draft_picks", "ppa_players_games", "ppa_players_season",
            "wepa_players_passing", "wepa_players_rushing", "wepa_players_kicking",
            "ratings_sp", "ratings_srs", "ratings_elo", "ratings_fpi"} == never_feature_endpoints()
    assert all(spec.never_feature for spec in ENDPOINTS if spec.role == "benchmark")


def test_season_level_endpoints_are_flagged():
    # Everything season_end is either never a feature or (coaches) usable only from prior seasons.
    assert same_season_forbidden_endpoints() - never_feature_endpoints() == {"coaches"}


def test_pregame_elo_does_not_contain_the_result():
    games = load_fixture("games_2024_sample")
    chain = elo_chain(games)
    assert chain["match"] > 10
    # Any break in the chain must not track this game's margin (it sits next to FCS games).
    assert chain["gap_margin_corr"] is None or chain["gap_margin_corr"] < 0.5
    fcs_adjacent = {g["id"] for g in games if "fcs" in (g["homeClassification"], g["awayClassification"])}
    assert fcs_adjacent  # the sample includes the FCS-adjacent cases on purpose
