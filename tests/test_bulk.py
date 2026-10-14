from datetime import datetime, timezone

import httpx
import pytest

from cfb.api.client import CFBDClient
from cfb.config import ApiSettings
from cfb.ingest import bulk, raw_store
from cfb.ingest.manifest import Manifest

NOW = datetime(2026, 10, 13, 12, tzinfo=timezone.utc)
STAT_TYPES = [{"id": 1, "name": "Incompletion"}, {"id": 12, "name": "Sack"}, {"id": 7, "name": "Rush"}]


class FakeApi:
    def __init__(self, remaining=30000, full_page_game=None, fail_path=None, fail_status=500):
        self.remaining = remaining
        self.full_page_game = full_page_game
        self.fail_path, self.fail_status = fail_path, fail_status
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path, params = request.url.path, dict(request.url.params)
        if path == "/info":
            return httpx.Response(200, json={"remainingCalls": self.remaining})
        self.calls.append((path, params))
        self.remaining -= 1
        if path == self.fail_path:
            return httpx.Response(self.fail_status, text="nope")
        if path == "/plays/stats/types":
            return httpx.Response(200, json=STAT_TYPES)
        if path == "/calendar":
            return httpx.Response(200, json=[{"week": 1, "seasonType": "regular"}, {"week": 1, "seasonType": "postseason"}])
        if path == "/games":
            games = [{"id": 11, "completed": True, "homeClassification": "fbs", "awayClassification": "fbs"},
                     {"id": 12, "completed": True, "homeClassification": "fbs", "awayClassification": "fcs"}]
            return httpx.Response(200, json=games if params["seasonType"] == "regular" else [])
        if path == "/plays/stats" and params.get("gameId") == str(self.full_page_game) and "statTypeId" not in params:
            return httpx.Response(200, json=[{}] * 2000)
        return httpx.Response(200, json=[])


def _client(api):
    return CFBDClient("k", ApiSettings(min_interval_s=0, max_attempts=2, backoff_min_s=0, backoff_max_s=0),
                      transport=httpx.MockTransport(api), sleep=lambda s: None)


def _run(settings, api, **kw):
    kw.setdefault("now", NOW)
    return bulk.run(settings, _client(api), Manifest(settings.paths), (2024,), log=lambda m: None, **kw)


def test_stage_a_then_exact_stage_b_then_resume(settings):
    api = FakeApi()
    result = _run(settings, api)
    paths = [p for p, _ in api.calls]
    # Stage A first: lookups, calendar, games; then per-game stats for both games, plus weekly pulls.
    assert paths.index("/calendar") < paths.index("/plays") and paths.index("/games") < paths.index("/plays/stats")
    assert sorted(int(p["gameId"]) for path, p in api.calls if path == "/plays/stats") == [11, 12]
    weekly = [(p["seasonType"], p["week"]) for path, p in api.calls if path == "/plays"]
    assert weekly == [("regular", "1"), ("postseason", "1")]
    assert not any(path == "/passing/plays" for path, _ in api.calls)  # empty before 2025
    assert result.data_calls == len(api.calls) == result.requests and not result.failures

    manifest = Manifest(settings.paths).read()
    assert int(manifest.filter(manifest["endpoint"] != "info")["attempts"].sum()) == len(api.calls)

    again = FakeApi()
    assert _run(settings, again).data_calls == 0 and again.calls == []


def test_truncated_page_is_split_by_stat_type(settings):
    api = FakeApi(full_page_game=11)
    result = _run(settings, api)
    assert [r.params["gameId"] for r in result.truncated] == [11]
    splits = sorted(int(p["statTypeId"]) for path, p in api.calls if path == "/plays/stats" and "statTypeId" in p)
    assert splits == [1, 7, 12]
    part = (("season", "2024"), ("game", "11"), ("stat_type", "12"))
    assert raw_store.is_landed(settings.paths.raw, "plays_stats", part)


def test_owed_splits_recovered_after_interrupted_run(settings):
    api = FakeApi(full_page_game=11)
    _run(settings, api)
    # Simulate a crash that lost two split partitions.
    for t in ("1", "7"):
        raw_store.response_path(settings.paths.raw, "plays_stats",
                                (("season", "2024"), ("game", "11"), ("stat_type", t))).unlink()
    again = FakeApi(full_page_game=11)
    _run(settings, again)
    assert sorted(int(p["statTypeId"]) for _, p in again.calls) == [1, 7]


def test_aborts_before_any_data_call_when_budget_short(settings):
    api = FakeApi(remaining=100)
    with pytest.raises(bulk.Abort, match="remaining"):
        _run(settings, api)
    assert api.calls == []


def test_max_calls_cap_checked_up_front(settings):
    api = FakeApi()
    with pytest.raises(bulk.Abort, match="max-calls"):
        _run(settings, api, max_calls=5)
    assert api.calls == []


def test_rejected_key_aborts(settings):
    api = FakeApi(fail_path="/venues", fail_status=401)
    with pytest.raises(bulk.Abort, match="401"):
        _run(settings, api)
    assert len(api.calls) == 1


def test_isolated_failures_are_recorded_and_skipped(settings):
    api = FakeApi(fail_path="/talent", fail_status=500)
    result = _run(settings, api)
    assert [(slug, status) for slug, _, status in result.failures] == [("talent", 500)]
    assert not raw_store.is_landed(settings.paths.raw, "talent", (("season", "2024"),))
    manifest = Manifest(settings.paths).read()
    talent = manifest.filter(manifest["endpoint"] == "talent").row(0, named=True)
    assert talent["http_status"] == 500 and talent["attempts"] == 2 and talent["file_path"] is None


def test_summary_report_from_a_bulk_run(settings):
    from cfb.ingest.summary import build_summary

    _run(settings, FakeApi(fail_path="/talent", fail_status=500))
    report = build_summary(settings, check_schemas=False)
    assert "1 failed requests; 0 later landed on a re-run; 1 still missing." in report
    assert "`talent`" in report
    assert "| plays_stats | 2 | 2 | 0 |" in report  # two games, both empty in the fake API
    assert "| endpoint | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |" in report  # recruit classes, draft year
