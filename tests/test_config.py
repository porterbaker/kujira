import pytest

from cfb.config import API_KEY_ENV_VAR, Paths, Settings, load_api_key, parse_seasons


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("2024", (2024,)),
        ("2021-2023", (2021, 2022, 2023)),
        ("2019,2021-2022", (2019, 2021, 2022)),
        ("2022, 2021-2022", (2021, 2022)),
    ],
)
def test_parse_seasons(spec, expected):
    assert parse_seasons(spec) == expected


@pytest.mark.parametrize("spec", ["", "abc", "2025-2021", "2000", "2027", "2024-"])
def test_parse_seasons_rejects(spec):
    with pytest.raises(ValueError):
        parse_seasons(spec)


def test_owner_decisions_recorded():
    s = Settings()
    assert s.ingest.include_fcs is False
    assert s.ingest.include_weather is False
    assert s.ingest.plays_stats_cap == 2000
    assert s.thresholds.transfer_match_threshold == 85.0
    assert s.thresholds.garbage_time == {2: 38, 3: 28, 4: 22}
    assert s.seasons[0] == 2014 and s.model.burn_in_seasons == (2014,) and s.model.eval_first_season == 2015


def _unset_key(monkeypatch):
    # setenv first so monkeypatch restores the original state (and removes keys .env loading adds)
    monkeypatch.setenv(API_KEY_ENV_VAR, "placeholder")
    monkeypatch.delenv(API_KEY_ENV_VAR)


def test_load_api_key_from_env_file(tmp_path, monkeypatch):
    _unset_key(monkeypatch)
    (tmp_path / ".env").write_text(f"{API_KEY_ENV_VAR}=test-key-123\n")
    assert load_api_key(Paths(root=tmp_path)) == "test-key-123"


def test_load_api_key_missing(tmp_path, monkeypatch):
    _unset_key(monkeypatch)
    with pytest.raises(RuntimeError, match=API_KEY_ENV_VAR):
        load_api_key(Paths(root=tmp_path))
