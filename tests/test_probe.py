import httpx
import pytest

from cfb.api.client import CFBDClient
from cfb.config import ApiSettings
from cfb.ingest import probe, raw_store
from cfb.ingest.manifest import Manifest
from cfb.ingest.planner import build_plan

from conftest import write_raw

GAMES_2024 = [
    {"id": 30, "week": 1, "completed": True, "homeClassification": "fbs", "awayClassification": "fcs",
     "homeConference": "SEC", "awayConference": "MVFC"},
    {"id": 20, "week": 1, "completed": True, "homeClassification": "fbs", "awayClassification": "fbs",
     "homeConference": "MAC", "awayConference": "Sun Belt"},
    {"id": 10, "week": 1, "completed": True, "homeClassification": "fbs", "awayClassification": "fbs",
     "homeConference": "SEC", "awayConference": "Big Ten"},
    {"id": 5, "week": 2, "completed": True, "homeClassification": "fbs", "awayClassification": "fbs",
     "homeConference": "SEC", "awayConference": "ACC"},
]
GAMES_2014 = [{"id": 99, "week": 5, "completed": True, "homeClassification": "fbs", "awayClassification": "fbs",
               "homeConference": "Pac-12", "awayConference": "SEC"}]


def fake_api(request: httpx.Request) -> httpx.Response:
    path, params = request.url.path, dict(request.url.params)
    if path == "/info":
        return httpx.Response(200, json={"remainingCalls": 5000, "monthlyLimit": 30000, "features": {"weather": True}})
    if path == "/plays/stats/types":
        return httpx.Response(200, json=[{"id": 18, "name": "Sack"}, {"id": 7, "name": "Tackle"}])
    if path == "/calendar":
        return httpx.Response(200, json=[{"week": w, "seasonType": "regular"} for w in (1, 2)])
    if path == "/plays" and params["year"] == "2025":
        return httpx.Response(200, json=[{"gameId": 7, "offenseConference": "SEC", "defenseConference": "ACC"}])
    if path == "/games":
        return httpx.Response(200, json=GAMES_2014 if params["year"] == "2014" else GAMES_2024)
    return httpx.Response(200, json=[])


def test_pick_game_kinds():
    assert probe.pick_game(GAMES_2024, 1, "power") == 10
    assert probe.pick_game(GAMES_2024, 1, "g5") == 20
    assert probe.pick_game(GAMES_2024, 1, "fcs") == 30
    assert isinstance(probe.pick_game(GAMES_2024, 3, "power"), str)


def test_probe_runs_end_to_end_against_mock(settings, capsys):
    client = CFBDClient("k", ApiSettings(min_interval_s=0), transport=httpx.MockTransport(fake_api), sleep=lambda s: None)
    ctx = probe.run_probe(settings, client, Manifest(settings.paths))
    assert client.calls_made == 22 == ctx["_planned"]

    raw = settings.paths.raw
    assert raw_store.is_landed(raw, "plays_stats", (("season", "2024"), ("game", "10")))
    assert raw_store.is_landed(raw, "plays_stats", (("season", "2014"), ("game", "99")))
    assert raw_store.is_landed(raw, "plays", (("season", "2024"), ("type", "regular"), ("week", "01")))
    sack = (("season", "2024"), ("type", "regular"), ("week", "01"), ("stat_type", "18"))
    assert raw_store.is_landed(probe.probe_raw_root(settings), "plays_stats", sack)
    assert not raw_store.is_landed(raw, "plays_stats", sack)  # experiments stay out of the canonical tree

    manifest = Manifest(settings.paths).read()
    assert manifest.height == 22 and int(manifest["attempts"].sum()) == 22

    # A second run reads everything locally; only the two /info calls go out.
    client2 = CFBDClient("k", ApiSettings(min_interval_s=0), transport=httpx.MockTransport(fake_api), sleep=lambda s: None)
    probe.run_probe(settings, client2, Manifest(settings.paths))
    assert client2.calls_made == 2


def test_probe_canonical_requests_match_bulk_planner(settings, capsys):
    client = CFBDClient("k", ApiSettings(min_interval_s=0), transport=httpx.MockTransport(fake_api), sleep=lambda s: None)
    probe.run_probe(settings, client, Manifest(settings.paths))
    write_raw(settings, "calendar", (("season", "2014"),), [{"week": 5, "seasonType": "regular"}])
    write_raw(settings, "calendar", (("season", "2025"),), [{"week": 1, "seasonType": "regular"}])
    write_raw(settings, "games", (("season", "2025"), ("type", "regular")),
              [{"id": 7, "week": 1, "completed": True, "homeClassification": "fbs", "awayClassification": "fbs"}])
    _, statuses = probe.load_landed(settings)
    canonical = [s.request for s in statuses if s.request and not s.step.probe_only]
    plan = build_plan((2014, 2024, 2025), settings)
    # Everything the probe landed is a partition the bulk planner would request (so it is skipped).
    skipped = {(r.spec.slug, r.partition): r.params for r in plan.skipped}
    for req in canonical:
        assert skipped.get((req.spec.slug, req.partition)) == req.params, req


def test_probe_aborts_when_budget_too_small(settings, capsys):
    def poor_api(request):
        if request.url.path == "/info":
            return httpx.Response(200, json={"remainingCalls": 100})
        return httpx.Response(200, json=[])

    client = CFBDClient("k", ApiSettings(min_interval_s=0, budget_buffer=250),
                        transport=httpx.MockTransport(poor_api), sleep=lambda s: None)
    with pytest.raises(SystemExit, match="aborting"):
        probe.run_probe(settings, client, Manifest(settings.paths))
    assert client.calls_made == 1
