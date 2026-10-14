"""Lookahead rules shared by the clean, feature, and model layers.

Endpoint-level timing lives in the registry (``availability``, ``never_feature``). This module
holds field-level rules for endpoints that mix pregame and postgame information.
"""

from __future__ import annotations

from cfb.ingest.registry import ENDPOINTS

# /games: known before kickoff. Pregame Elo is probe-verified: it equals the previous postgame
# Elo (2014: 1318/1318); the 2024 exceptions sit next to FCS games and don't track the result.
GAME_PREGAME_FIELDS = frozenset({
    "id", "season", "week", "seasonType", "startDate", "startTimeTBD", "neutralSite", "conferenceGame",
    "venueId", "venue", "playoff",
    "homeId", "homeTeam", "homeClassification", "homeConference", "homePregameElo",
    "awayId", "awayTeam", "awayClassification", "awayConference", "awayPregameElo",
})

# /games: only known after the game (or free text that may describe it). Never features for that game.
GAME_POSTGAME_FIELDS = frozenset({
    "completed", "attendance", "excitementIndex", "highlights", "notes",
    "homePoints", "homeLineScores", "homePostgameWinProbability", "homePostgameElo",
    "awayPoints", "awayLineScores", "awayPostgameWinProbability", "awayPostgameElo",
})


# Clean `games` table: the same split in snake_case.
CLEAN_GAME_PREGAME_COLUMNS = frozenset({
    "game_id", "season", "week", "season_type", "start_date", "start_time_tbd", "neutral_site", "conference_game",
    "venue_id", "venue", "home_id", "home_team", "home_conference", "home_classification", "home_pregame_elo",
    "away_id", "away_team", "away_conference", "away_classification", "away_pregame_elo", "is_fbs_game",
})
CLEAN_GAME_POSTGAME_COLUMNS = frozenset({
    "completed", "home_points", "away_points", "home_postgame_elo", "away_postgame_elo", "home_postgame_wp",
    "away_postgame_wp", "attendance", "excitement_index", "notes",
})

# Clean `plays`: the game state before the snap (usable as play context) versus what the play produced.
# /plays scores are post-play; the clean `offense_score`/`defense_score`/`score_diff` are pre-snap.
PLAY_STATE_COLUMNS = frozenset({
    "period", "clock_seconds", "game_seconds_remaining", "down", "distance", "yards_to_goal", "yardline",
    "offense_score", "defense_score", "score_diff", "offense_timeouts", "defense_timeouts", "is_home_offense",
    "neutral_site", "offense_id", "defense_id",
})
PLAY_OUTCOME_COLUMNS = frozenset({
    "offense_score_after", "defense_score_after", "yards_gained", "scoring", "ppa", "play_text", "play_type",
    "play_type_id", "category", "wallclock",
})


def never_feature_endpoints() -> frozenset[str]:
    return frozenset(spec.slug for spec in ENDPOINTS if spec.never_feature)


def same_season_forbidden_endpoints() -> frozenset[str]:
    """Full-season values: usable only as priors from earlier seasons."""
    return frozenset(spec.slug for spec in ENDPOINTS if spec.availability == "season_end")
