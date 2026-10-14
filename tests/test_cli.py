import pytest

from cfb import cli


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch, settings):
    monkeypatch.setattr(cli, "Settings", lambda: settings)


def test_ingest_plan_is_dry_run(capsys):
    assert cli.main(["ingest", "plan", "--seasons", "2024"]) == 0
    out = capsys.readouterr().out
    assert "TOTAL" in out and "Dry run" in out


def test_ingest_plan_lists_requests(capsys):
    assert cli.main(["ingest", "plan", "--seasons", "2024", "--endpoints", "roster", "--list"]) == 0
    out = capsys.readouterr().out
    assert "GET /roster?year=2024" in out
    assert "data/raw/roster/season=2024" in out


def test_ingest_execute_without_flag_is_dry_run(capsys):
    assert cli.main(["ingest", "execute", "--seasons", "2024"]) == 0
    assert "Re-run with --execute" in capsys.readouterr().out


def test_ingest_execute_with_flag_runs_bulk(monkeypatch, capsys):
    import httpx

    from cfb.api import client as client_mod
    from cfb import config

    calls = []

    def fake_api(request):
        if request.url.path == "/info":
            return httpx.Response(200, json={"remainingCalls": 30000})
        calls.append(request.url.path)
        return httpx.Response(200, json=[])

    real = client_mod.CFBDClient
    monkeypatch.setattr(config, "load_api_key", lambda paths=None: "k")
    monkeypatch.setattr(client_mod, "CFBDClient", lambda key, settings: real(
        key, settings, transport=httpx.MockTransport(fake_api), sleep=lambda s: None))
    code = cli.main(["ingest", "execute", "--seasons", "2024", "--endpoints", "roster,talent", "--execute"])
    assert code == 0
    assert sorted(calls) == ["/roster", "/talent"]
    assert "2 counted calls" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv, phase",
    [
        (["game", "fit"], "Phase 4"),
        (["backtest"], "Phase 5"),
        (["report"], "Phase 5"),
    ],
)
def test_later_phase_commands_exist(argv, phase, capsys):
    assert cli.main(argv) == cli.EXIT_NOT_IMPLEMENTED
    assert phase in capsys.readouterr().err


def test_seasons_default_to_config_range(capsys):
    assert cli.main(["ingest", "plan", "--endpoints", "roster", "--list"]) == 0
    out = capsys.readouterr().out
    assert "GET /roster?year=2014" in out and "GET /roster?year=2026" in out
    assert "year=2013" not in out


def test_probe_dry_run_prints_count(capsys):
    assert cli.main(["ingest", "probe"]) == 0
    out = capsys.readouterr().out
    assert "Planned calls: 22" in out and "Dry run" in out


def test_unknown_endpoint_rejected():
    with pytest.raises(SystemExit, match="unknown endpoint"):
        cli.main(["ingest", "plan", "--seasons", "2024", "--endpoints", "nope"])
