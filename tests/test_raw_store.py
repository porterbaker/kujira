from pathlib import Path

from cfb.ingest import raw_store
from cfb.ingest.registry import BY_SLUG, ENDPOINTS

from conftest import write_raw


def test_response_path_layout():
    part = (raw_store.season_part(2024),) + raw_store.week_parts("postseason", 1)
    path = raw_store.response_path(Path("data/raw"), "plays", part)
    assert path.as_posix() == "data/raw/plays/season=2024/type=postseason/week=01/response.json.gz"


def test_game_partition_layout():
    path = raw_store.response_path(Path("data/raw"), "plays_stats", (("season", "2024"), ("game", "401628319")))
    assert path.as_posix() == "data/raw/plays_stats/season=2024/game=401628319/response.json.gz"


def test_slugs_are_unique_and_flat():
    assert len(BY_SLUG) == len(ENDPOINTS)
    assert BY_SLUG["plays_stats"].path == "/plays/stats"
    assert all("/" not in slug for slug in BY_SLUG)


def test_landed_and_read_roundtrip(settings):
    part = (raw_store.season_part("all"),)
    assert not raw_store.is_landed(settings.paths.raw, "venues", part)
    path = write_raw(settings, "venues", part, [{"id": 1}])
    assert raw_store.is_landed(settings.paths.raw, "venues", part)
    assert raw_store.read_json(path) == [{"id": 1}]
