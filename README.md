# cfb: college football player value and game prediction

A two-stage model built on the [CollegeFootballData](https://collegefootballdata.com) API:

1. **Player values.** A hierarchical Bayesian model (NumPyro) estimates a value for every player from play-level EPA (`ppa`).
2. **Game model.** Player values are aggregated into team strength, which predicts game margins and win probabilities.

Status: **Phase 2 (clean layer) done**: 23 typed tables in `data/clean/`, see
[reports/phase2_data_quality.md](reports/phase2_data_quality.md). See [PLAN.md](PLAN.md) for the design and findings,
[reports/phase1a_probe.md](reports/phase1a_probe.md) for the probe, and [reports/phase1b_ingest.md](reports/phase1b_ingest.md)
for what landed.

## Setup

Requires Python 3.11 or newer (developed on 3.14).

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # then put your key in .env
```

### Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env   # then put your key in .env
```

If PowerShell blocks the activate script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

### API key

`.env` holds a single line, `CFB_APIKEY=<your key>`. It is gitignored. The key is read only
through `python-dotenv` and is never logged or written to the manifest.

## Usage

Every command that would call the API is a **dry run by default**. It prints the planned
requests and the call count, then exits. Only `--execute` touches the network.

```bash
cfb ingest plan                                           # call estimate, default seasons 2014-2026
cfb ingest plan --seasons 2024 --endpoints plays,plays_stats --list
cfb ingest execute --seasons 2024                         # dry run
cfb ingest execute --execute --stage A                     # lookups, calendars, games (makes stage B exact)
cfb ingest execute --execute --max-calls 12500            # full run, hard cap on counted calls; resumable
cfb ingest summary                                        # offline: reports/phase1b_ingest.md
cfb ingest probe                                          # probe plan (dry run)
cfb ingest probe --execute                                # Phase 1a live probe (skips what is on disk)
cfb ingest probe --report                                 # offline: probe report + test fixtures
cfb clean build                                           # raw -> data/clean/*.parquet (offline, ~5 min)
cfb clean report                                          # reports/phase2_data_quality.md
cfb player fit                                            # Phase 3
cfb game fit                                              # Phase 4
cfb backtest                                              # Phase 5
cfb report                                                # Phase 5
```

`python -m cfb ...` works the same as `cfb ...`.

## Layout

```
cfb/              package: config, api, ingest, clean, features, player_model, game_model, eval
data/             raw landing, clean parquet, manifest (gitignored)
reports/          generated reports
tests/            offline pytest suite; fixtures in tests/fixtures/
vendor/cfbd-python/  cfbd v5.32.1 client source, used as schema reference only (never imported)
```

## Tests

```bash
pytest
```

The suite runs offline. An autouse fixture blocks all socket connections. Fixtures in `tests/fixtures/` are trimmed live
responses from the probe.
