"""Endpoint registry: what to pull from CFBD, at what granularity, with which parameters.

Parameter names are the API's wire names (camelCase), taken from the vendored client's
API modules (``vendor/cfbd-python/cfbd/api``). Facts that were read from the spec and are
not yet confirmed live are marked "spec:" and are checked in the Phase 1a probe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Granularity = Literal["once", "season", "season_type", "week", "game", "recruit_class"]
Role = Literal["lookup", "core", "player", "context", "benchmark"]

# When an endpoint's data becomes knowable. Feature builders use this to enforce as-of rules:
#   static     - lookups, no time dimension
#   pregame    - known before the game it describes (per-field caveats handled in clean)
#   postgame   - describes a game; usable only for later games
#   season_end - full-season values; never a same-season feature, prior seasons are fine
#   snapshot   - state pulled at an unknown date (rosters); only static attributes are usable
#   dated      - rows carry their own date (portal transferDate); usable after that date
Availability = Literal["static", "pregame", "postgame", "season_end", "snapshot", "dated"]

SEASON_TYPES = ("regular", "postseason")
RECRUIT_TYPES = ("HighSchool", "JUCO", "PrepSchool")


@dataclass(frozen=True)
class EndpointSpec:
    path: str
    granularity: Granularity
    role: Role
    params: dict[str, str] = field(default_factory=dict)  # fixed query params
    # Query param that selects FBS/FCS scope; None if the endpoint has no such filter.
    division_param: str | None = None
    year_offset: int = 0  # /draft/picks: draft year = season + 1
    first_season: int | None = None  # seasons before this return no data (probe-verified)
    flag: str | None = None  # opt-in setting name, e.g. "include_weather"
    stage: Literal["A", "B"] = "B"  # A = needed to plan B (calendar, games, lookups)
    availability: Availability = "postgame"
    never_feature: bool = False  # benchmarks, betting lines, validation-only data
    notes: str = ""

    @property
    def slug(self) -> str:
        return self.path.strip("/").replace("/", "_")


ENDPOINTS: tuple[EndpointSpec, ...] = (
    # Lookups
    EndpointSpec("/venues", "once", "lookup", stage="A", availability="static"),
    EndpointSpec("/plays/types", "once", "lookup", stage="A", availability="static"),
    EndpointSpec("/plays/stats/types", "once", "lookup", stage="A", availability="static",
                 notes="name<->id map; /plays/stats rows carry only the stat type name"),
    # Core play data
    EndpointSpec("/calendar", "season", "core", stage="A", availability="pregame"),
    EndpointSpec("/games", "season_type", "core", division_param="classification", stage="A",
                 availability="pregame",
                 notes="schedule + pregame Elo; also postgame fields (points, postgame Elo/WP) to split in clean"),
    EndpointSpec("/plays", "week", "core", division_param="classification",
                 notes="spec: year+week required, no gameId filter; postseason needs seasonType"),
    # Probe: empty before 2025 for any filter; classification=fbs returns FBS-offense plays only.
    EndpointSpec("/passing/plays", "week", "core", division_param="classification", first_season=2025,
                 notes="year + (week|team) required; parseStatus only; airYards null in probe week"),
    EndpointSpec("/rushing/plays", "week", "core", division_param="classification", first_season=2025,
                 notes="includes sacks (rusherId = QB), kneels, team rushes; has attributionStatus"),
    # Probe: the only passer/target/rusher attribution source for 2014-2024. No tackles, PBUs,
    # hurries or forced fumbles at play level; ~170-270 rows per game, far under the 2,000 cap.
    EndpointSpec("/plays/stats", "game", "core",
                 notes="capped at 2,000 rows; re-pull split by statTypeId when a page is full"),
    EndpointSpec("/games/players", "week", "core", division_param="classification",
                 notes="year + (week|team|conference) required; defensive box has TOT/SOLO/SACKS/TFL/PD/QB HUR"),
    # Player context
    EndpointSpec("/teams", "season", "player", availability="pregame",
                 notes="team id/conference by season; location = home stadium (cross-check)"),
    EndpointSpec("/roster", "season", "player", availability="snapshot",
                 notes="year defaults to 2025 if omitted; always pass it. No team = all teams"),
    EndpointSpec("/recruiting/players", "recruit_class", "player", availability="pregame",
                 notes="classification defaults to HighSchool; JUCO/PrepSchool pulled separately"),
    EndpointSpec("/player/portal", "season", "player", availability="dated",
                 notes="no athlete id; names only"),
    EndpointSpec("/draft/picks", "season", "player", year_offset=1, availability="season_end",
                 never_feature=True, notes="validation only"),
    EndpointSpec("/coaches", "season", "player", availability="season_end",
                 notes="season records; coaching-change flag must not leak mid-season firings"),
    # Game context
    EndpointSpec("/lines", "season_type", "context", availability="pregame", never_feature=True,
                 notes="home perspective, negative = home favored; spreadOpen partial"),
    EndpointSpec("/metrics/wp/pregame", "season_type", "context", availability="pregame",
                 never_feature=True, notes="spread = Bovada closing spread (probe: 757/757)"),
    EndpointSpec("/talent", "season", "context", availability="pregame"),
    EndpointSpec("/player/returning", "season", "context", availability="pregame"),
    EndpointSpec("/games/weather", "season_type", "context", division_param="classification",
                 flag="include_weather", notes="Patreon feature; /info features.weather; observed, not forecast"),
    # Benchmarks only
    EndpointSpec("/ppa/players/games", "week", "benchmark", never_feature=True),
    EndpointSpec("/ppa/players/season", "season", "benchmark", availability="season_end", never_feature=True),
    EndpointSpec("/wepa/players/passing", "season", "benchmark", availability="season_end", never_feature=True),
    EndpointSpec("/wepa/players/rushing", "season", "benchmark", availability="season_end", never_feature=True),
    EndpointSpec("/wepa/players/kicking", "season", "benchmark", availability="season_end", never_feature=True),
    EndpointSpec("/ratings/sp", "season", "benchmark", availability="season_end", never_feature=True),
    EndpointSpec("/ratings/srs", "season", "benchmark", availability="season_end", never_feature=True),
    EndpointSpec("/ratings/elo", "season", "benchmark", availability="season_end", never_feature=True,
                 notes="defaults to latest week"),
    EndpointSpec("/ratings/fpi", "season", "benchmark", availability="season_end", never_feature=True),
)

BY_SLUG: dict[str, EndpointSpec] = {spec.slug: spec for spec in ENDPOINTS}
