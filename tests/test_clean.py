"""Clean layer: an end-to-end build on trimmed real data (tests/fixtures/clean) plus unit tests."""

from datetime import datetime, timezone

import polars as pl
import pytest

from cfb.clean import context, plays, transfers
from cfb.clean.box import split_stat
from cfb.clean.build import build_all
from cfb.clean.common import normalize_name, position_group, read_table
from cfb.clean.fixtures import install_fixtures
from cfb.config import Paths, Settings
from cfb.leakage import (
    CLEAN_GAME_POSTGAME_COLUMNS,
    CLEAN_GAME_PREGAME_COLUMNS,
    PLAY_OUTCOME_COLUMNS,
    PLAY_STATE_COLUMNS,
)

from conftest import FIXTURES

SEASONS = (2014, 2024, 2025)
GAMES = {2014: 400547743, 2024: 401628322, 2025: 401752665}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    settings = Settings(paths=Paths(root=tmp_path_factory.mktemp("clean")))
    install_fixtures(FIXTURES / "clean", settings)
    build_all(settings, SEASONS, log=lambda m: None)
    return settings


def t(settings, name):
    return read_table(settings, name)


# ---------- end to end ----------

def test_every_table_is_built(built):
    names = {p.stem for p in built.paths.clean.glob("*.parquet")}
    assert {"games", "plays", "pass_plays", "rush_plays", "play_participants", "player_game_box", "players",
            "player_seasons", "recruits", "recruit_links", "transfers", "transfer_matches", "draft", "lines",
            "venues", "coaches", "team_coaching", "benchmarks", "team_context", "teams"} <= names
    assert (built.paths.reports / "transfer_review.csv").is_file()


def test_plays_have_ids_and_presnap_scores(built):
    p = t(built, "plays")
    assert set(p["game_id"].unique().to_list()) == set(GAMES.values())
    assert p["offense_id"].null_count() == 0 and p["play_id"].is_unique().all()
    first = p.sort("play_id").group_by("game_id").first()
    assert (first["offense_score"] == 0).all() and (first["defense_score"] == 0).all()
    tds = p.filter(pl.col("play_type").is_in(["Passing Touchdown", "Rushing Touchdown"]))
    # /plays scores include the play's own touchdown; the clean pre-snap score must not.
    assert (tds["offense_score_after"] > tds["offense_score"]).mean() > 0.8
    assert (p["score_diff"] == p["offense_score"] - p["defense_score"]).all()


def test_garbage_time_uses_presnap_margin(built):
    p = t(built, "plays")
    th = built.thresholds.garbage_time
    expected = pl.lit(False)
    for q, m in th.items():
        expected = pl.when((pl.col("period") == q) & (pl.col("score_diff").abs() > m)).then(True).otherwise(expected)
    assert p.select((pl.col("garbage_time") == expected).all()).item()


def test_model_eligibility_reasons(built):
    p = t(built, "plays")
    assert (p.filter(pl.col("model_eligible"))["category"].is_in(["pass", "sack", "rush"])).all()
    assert p.filter(pl.col("model_eligible"))["ppa"].null_count() == 0
    assert not p.filter(pl.col("model_eligible"))["is_kneel"].any()
    assert set(p["exclude_reason"].drop_nulls().unique()) <= {"non_scrimmage", "kneel", "spike", "no_ppa", "invalid_parse"}


def test_attribution_from_stats_matches_enriched_2025(built):
    pp, rp = t(built, "pass_plays"), t(built, "rush_plays")
    for season in SEASONS:
        s = pp.filter((pl.col("season") == season) & ~pl.col("is_sack"))
        assert (s["passer_attribution"] == "individual").mean() > 0.95
        assert s.filter(pl.col("outcome") == "completion")["target_id"].null_count() == 0
    p25, r25 = pp.filter(pl.col("season") == 2025), rp.filter(pl.col("season") == 2025)
    assert p25["passer_agrees_enriched"].drop_nulls().all() and p25["passer_agrees_enriched"].count() > 40
    assert r25["rusher_agrees_enriched"].drop_nulls().all() and r25["rusher_agrees_enriched"].count() > 40


def test_participants_all_join_plays_with_a_side(built):
    pa = t(built, "play_participants")
    assert pa["in_plays"].all() and pa["side"].null_count() == 0


def test_players_and_links(built):
    players, links, ps = t(built, "players"), t(built, "recruit_links"), t(built, "player_seasons")
    credited = t(built, "play_participants")["athlete_id"].unique()
    assert credited.is_in(players["athlete_id"].implode()).all()
    assert credited.is_in(ps["athlete_id"].implode()).all()  # every credited athlete is on a roster
    assert links.height > 100 and set(links["path"].unique()) <= {"both", "roster", "recruiting"}
    assert ps.filter(pl.col("class_year_suspect"))["class_year"].null_count() == ps["class_year_suspect"].sum()


def test_transfers_accept_exact_and_fuzzy_at_threshold_only(built):
    m = t(built, "transfer_matches")
    threshold = built.thresholds.transfer_match_threshold
    assert m.height > 10
    fuzzy = m.filter(pl.col("method") == "fuzzy")
    assert (fuzzy["accepted"] == (fuzzy["confidence"] >= threshold)).all()
    assert m.filter(pl.col("method") == "exact")["accepted"].all()
    assert not m.filter(pl.col("method") == "none")["accepted"].any()


def test_context_tables(built):
    lines = t(built, "lines")
    neg = lines.filter(pl.col("spread") < 0)
    assert neg.select(pl.col("formatted_spread").str.starts_with(pl.col("home_team")).all()).item()
    bench = t(built, "benchmarks")
    assert "nationalAverages" not in bench["entity_name"].to_list()
    assert set(bench["source"].unique()) >= {"sp"}
    assert t(built, "team_context")["talent"].drop_nulls().len() == 2
    box = t(built, "player_game_box")
    assert {"C", "ATT"} <= set(box.filter(pl.col("category") == "passing")["stat_name"].to_list())


def test_clean_games_columns_are_classified_for_leakage(built):
    cols = set(t(built, "games").columns)
    assert cols == CLEAN_GAME_PREGAME_COLUMNS | CLEAN_GAME_POSTGAME_COLUMNS
    assert not PLAY_STATE_COLUMNS & PLAY_OUTCOME_COLUMNS
    assert PLAY_STATE_COLUMNS | PLAY_OUTCOME_COLUMNS <= set(t(built, "plays").columns)


# ---------- unit ----------

@pytest.mark.parametrize("raw, norm", [
    ("D'Andre Swift Jr.", "dandre swift"), ("José  Núñez III", "jose nunez"), ("Anthony Brown-Stephens", "anthony brown stephens"),
    ("TJ Branch", "tj branch"), (None, ""),
])
def test_normalize_name(raw, norm):
    assert normalize_name(raw) == norm


def test_position_groups():
    assert position_group("EDGE") == "DL" and position_group("IOL") == "OL" and position_group("CB") == "DB"
    assert position_group("XYZ") == "OTHER" and position_group(None) is None


@pytest.mark.parametrize("text, target", [
    ("Davis Webb pass incomplete to Jakeem Grant", "Jakeem Grant"),
    ("Cade Klubnik pass incomplete to Phil Mafah, broken up by X", "Phil Mafah"),
    ("Carson Beck pass complete to Colbie Young for 7 yds for a TD (Kick)", "Colbie Young"),
    ("Kenneth Walker III pass incomplete to D.J. Moore", "D.J. Moore"),
    ("Tyler Riddell pass incomplete", None),
])
def test_target_text_extraction(text, target):
    got = pl.DataFrame({"t": [text]}).select(pl.col("t").str.extract(plays.TARGET_TEXT, 1)).item()
    assert got == target


def test_kneel_and_spike_rules():
    df = pl.DataFrame({"t": ["Kneel down by SMU at SMU2", "C. Bailey takes a knee", "TEAM run for a loss of 1 yard",
                             "Joe Smith run for 3 yds", "Joe Smith pass incomplete (spiked)"]})
    out = df.select(k=pl.col("t").str.contains(plays.KNEEL_TEXT) | pl.col("t").str.contains(plays.TEAM_RUN_FOR_LOSS),
                    s=pl.col("t").str.contains(plays.SPIKE_TEXT))
    assert out["k"].to_list() == [True, True, True, False, False]
    assert out["s"].to_list() == [False, False, False, False, True]


def test_presnap_scores_shift_within_game():
    raw = pl.DataFrame({
        "play_id": [1, 2, 3, 4], "game_id": [9, 9, 9, 9], "offense": ["A", "A", "B", "B"], "defense": ["B", "B", "A", "A"],
        "home": ["A"] * 4,
        "offense_score_after": [0, 7, 0, 3], "defense_score_after": [0, 0, 7, 7], "period": [1, 1, 2, 4],
        "clock_seconds": [900, 800, 700, 10], "play_type": ["Rush"] * 4,
    })
    games = pl.DataFrame({"game_id": [9], "home_id": [1], "away_id": [2], "home_team": ["A"], "away_team": ["B"],
                          "neutral_site": [False]})
    pt = pl.DataFrame({"play_type": ["Rush"], "play_type_id": [5]}, schema={"play_type": pl.String, "play_type_id": pl.Int16})
    out = plays.build_plays_frame(raw, games, pt, {4: 2}).sort("play_id")
    assert out["offense_score"].to_list() == [0, 0, 0, 0]
    assert out["defense_score"].to_list() == [0, 0, 7, 7]  # B's defense faces A's 7 once A has scored
    assert out["garbage_time"].to_list() == [False, False, False, True]  # Q4, |0-7| > 2
    assert out["game_seconds_remaining"].to_list() == [3600, 3500, 2500, 10]


def test_split_stat():
    assert split_stat("C/ATT", "15/22") == [("C", 15.0), ("ATT", 22.0)]
    assert split_stat("FG", "1/2") == [("FG_MADE", 1.0), ("FG_ATT", 2.0)]
    assert split_stat("YDS", "--") == [("YDS", None)]


def test_transfer_origin_season_and_name_score():
    assert transfers.origin_season(2024, datetime(2023, 12, 4, tzinfo=timezone.utc)) == 2023
    assert transfers.origin_season(2024, datetime(2024, 4, 20, tzinfo=timezone.utc)) == 2023
    assert transfers.origin_season(2024, None) == 2023
    nick, last_ok, first_ok = transfers.name_score("cameron", "threatt", "cam", "threatt")
    other, *_ = transfers.name_score("brock", "montgomery", "chuck", "montgomery")
    wrong, last_wrong, _ = transfers.name_score("jacob", "sykes", "jacob", "psyk")
    assert last_ok and first_ok and nick >= 90
    assert 70 <= other < 80
    assert not last_wrong and wrong < 60


def test_fuzzy_transfers_accepted_only_at_threshold():
    tr = pl.DataFrame({
        "transfer_key": ["a", "b"], "season": [2025, 2025], "first_name": ["Cameron", "Zed"], "last_name": ["Threatt", "Q"],
        "name_norm": ["cameron threatt", "zed q"], "position_group": ["DB", "DB"], "origin_id": [1, 1],
        "origin_season": [2024, 2024], "destination_id": [None, None],
    })
    ps = pl.DataFrame({"athlete_id": [10], "season": [2024], "team_id": [1], "name_norm": ["cam threatt"],
                       "first_name": ["Cam"], "last_name": ["Threatt"], "position": ["CB"], "position_group": ["DB"]})
    none = transfers.match_transfers(tr, ps, None)
    assert none.filter(pl.col("transfer_key") == "a").row(0, named=True)["accepted"] is False
    some = transfers.match_transfers(tr, ps, 90.0)
    row = some.filter(pl.col("transfer_key") == "a").row(0, named=True)
    assert row["method"] == "fuzzy" and row["accepted"] and row["athlete_id"] == 10
    assert some.filter(pl.col("transfer_key") == "b")["method"].item() == "none"


def test_team_coaching_change_is_preseason_only():
    def c(coach, season, hire, games):
        return {"coach_id": coach, "season": season, "team_id": 1, "games": games,
                "hire_date": datetime(hire, 12, 1, tzinfo=timezone.utc) if hire else None}
    rows = [c(1, 2022, 2015, 12),
            c(1, 2023, 2015, 8), c(2, 2023, 2023, 4),   # mid-season firing: starter is the continuing coach
            c(3, 2024, 2023, 12)]                       # new head coach for 2024
    df = pl.DataFrame(rows, schema={"coach_id": pl.Int64, "season": pl.Int16, "team_id": pl.Int64, "games": pl.Int16,
                                    "hire_date": pl.Datetime("us", "UTC")})
    out = {r["season"]: r for r in context.build_team_coaching(df).iter_rows(named=True)}
    assert out[2022]["coaching_change"] is None  # no earlier season to compare
    assert out[2023]["preseason_coach_id"] == 1 and out[2023]["coaching_change"] is False
    assert out[2023]["in_season_change"] is True
    assert out[2024]["preseason_coach_id"] == 3 and out[2024]["coaching_change"] is True
