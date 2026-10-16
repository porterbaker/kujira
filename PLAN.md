# Plan

Design notes: where the CFBD spec disagrees with the original brief, what the Phase 1a probe
found against live data, the draft clean-layer schemas, and the API call estimate. The full
probe evidence is in [reports/phase1a_probe.md](reports/phase1a_probe.md), which
`cfb ingest probe --report` regenerates offline. The project owner's original phased brief is in
[docs/BRIEF.md](docs/BRIEF.md); where it and this file differ, the decisions below win.

## Decisions made

| Decision | Outcome | Date |
|---|---|---|
| Season range | **2014–2026** (config default; `--seasons` overrides) | 2026-10-13 |
| matplotlib for report charts | Approved for Phase 5 | 2026-10-13 |
| Legacy `cfbdata.py`, `requirements.txt` | Deleted | 2026-10-13 |
| Phase 1b bulk run, 2014–2026 FBS scope | Approved at ~11,900 calls; run with `--max-calls 12500` | 2026-10-13 |
| Legacy `venv/`, `data/players_*.json`, `data/teams/` | Deleted (raw re-pulls replace them) | 2026-10-13 |
| Garbage time | CFBD thresholds (Q2 > 38, Q3 > 28, Q4 > 22), flag only; Stage 1 excludes flagged plays | 2026-10-13 |
| Transfer match threshold | 85 | 2026-10-13 |
| Start season | 2014 fitted as burn-in; evaluation from 2015 | 2026-10-13 |
| Defensive credit (Phase 3) | Team-season unit D in the play model, plus a separate box-score defender model (2016+) | 2026-10-13 |
| Season-step scales (Phase 3) | Held at rolling two-season NUTS fits (pairs ending S-3..S-1), like the within-season scales; SVI shrinks them 3–10x | 2026-10-15 |
| Burn-in | **2014 and 2015**; player-model evaluation from 2016 (2015 has no earlier season pair for its step scales) | 2026-10-15 |

## Phase 1a probe findings

Phase 1a used 29 counted calls: 20 for the probe, 8 for diagnostics, and 1 canonical 2025 pull. Tier 2 allows
**30,000 calls a month**, resetting on the 1st. **`/info` is free**: it doesn't lower the counter.

1. **`/passing/plays` and `/rushing/plays` are empty before 2025**, for any filter (week,
   team, gameId, with or without `classification`). They're populated for 2025 and 2026.
   The registry sets `first_season=2025`, which saves about 374 calls.
2. **`/plays/stats` is therefore the attribution source for 2014–2024, and we should use it for
   every season so the model sees one consistent source.** It credits the passer on 100% of
   pass plays, the receiver on 100% of completions, the rusher on 82–97% of rushes, and the QB
   on 100% of sacks, for both teams including FCS. It has 170–270 rows per game, far under the
   2,000 cap. The enriched endpoints add little for 2025+: `airYards` is null on every row of
   the probe week and `parseStatus` is `partial` on every pass.
3. **Targets on incompletions are patchy** in `/plays/stats`: 0–93% per game. Phase 2 will
   recover them by parsing `playText` ("X pass incomplete to Y") and matching the name to the
   offense's roster. Without that, receivers would be credited only for catches.
4. **No play-level tackles, pass breakups, hurries or forced fumbles exist.** Defenders are
   credited at play level only for sacks and interceptions. The 2014 game has no defender
   sack credit; Phase 2 will find the first season that does. `/games/players` box scores do
   carry per-game TOT, SOLO, SACKS, TFL, PD and QB HUR for each defender. That reshapes the
   Phase 3 defensive-credit choice; options come with Phase 3.
5. **Joins are clean.** `play_id` from all four sources joins to `/plays` at 100%. `ppa` on
   the enriched rows is identical to `/plays`. On sacks, `rusherId` is the QB (193/193).
   `classification=fbs` on the enriched endpoints returns only plays where the offense is FBS.
6. **`ppa` is on 99.9–100% of scrimmage plays** in 2014, 2024 and 2025. The nulls are
   kickoffs, punts, penalties and administrative rows.
7. **Every response matches the vendored models**: no missing or extra fields, no unexpected
   nulls, no type mismatches (26 partitions checked).
8. **Baselines:** spreads are home-perspective, negative = home favored (100%). Opening
   spreads exist on 74% of lines. The `/metrics/wp/pregame` spread is Bovada's closing spread
   (757/757), so it is a benchmark only.
9. **Lookahead:** `/games` pregame Elo equals the team's previous postgame Elo (2014:
   1318/1318). The 118 exceptions in 2024 sit next to FCS games, and the gap doesn't track
   the game's result (corr −0.13). `/games` also carries postgame fields (points, postgame Elo
   and WP, excitement, attendance). [cfb/leakage.py](cfb/leakage.py) classifies every `/games`
   field, and a test fails if a new field appears unclassified.
10. **Known limitation:** `ppa` comes from CFBD's expected-points model, which was fitted on
    data that includes seasons after 2014. That's a mild lookahead in the *target*, not in
    features. Refitting our own EP model as-of is possible but out of scope unless you want it.

## Phase 1b bulk ingestion results

Full detail is in [reports/phase1b_ingest.md](reports/phase1b_ingest.md), which
`cfb ingest summary` regenerates.

- **The run completed** (2026-10-13, 134 minutes): 11,759 requests in stage B plus 36 in
  stage A, 1 retry, **0 failures, 0 truncated `/plays/stats` pages**. 247 MB of gzipped raw
  data (about 2.7 GB as JSON).
- **Budget:** 11,825 counted calls across all of Phase 1 by our count. CFBD's counter dropped
  by only 11,333 during stage B, so our count is conservative. **18,588 calls remain until
  Nov 1.**
- **Coverage, 2014–2026:** about 158–166k plays per season (2020: 103k; 2026 through week 6:
  79k) and about 165–196k `/plays/stats` rows per season.
- **Empty responses (87), all explained:**
  - 2019 regular week 16 (no FBS games that week).
  - The portal before 2021, team talent in 2014, kicker WEPA in 2014–15.
  - Small JUCO and PrepSchool recruiting classes.
  - Enriched endpoints in the probe seasons.
  - **67 games (0.6%) with no `/plays/stats`.** 50 of them have no play-by-play either.
    **17 have 3,049 plays but no player attribution**; those plays can only inform unit
    effects.
- **Every file was checked against the vendored models.** Two differences:
  - `/metrics/wp/pregame` has null `spread` and `homeWinProbability` in 4 files, which the
    vendored model forbids. The strict client would have crashed on them; the clean layer
    must allow the nulls.
  - `/ratings/sp` has a `nationalAverages` row with no `conference`. The clean layer drops it.
- **For Phase 2:**
  - Roster row counts jump in 2022, 2025 and 2026 (about 30k vs 16–18k), which probably
    means more divisions are included.
  - Lines and SRS also cover FCS from 2022.
  - Team talent covers about 230 teams through 2023 but only about 135 (FBS) from 2024.
  - 2026 season-level files (ratings, WEPA, PPA, roster) are mid-season snapshots from
    2026-10-13 and need `--refresh` after the season.

## Phase 2 clean layer

`cfb clean build` builds 23 typed parquet tables in `data/clean/` (about 4.5 minutes, peak about
1.6 GB). `cfb clean report` writes [reports/phase2_data_quality.md](reports/phase2_data_quality.md).

**Design choices made from the data:**
- **`pass_plays` and `rush_plays` cover every season**, not only 2025+. Passer, target and
  rusher come from `/plays/stats` credits, then the enriched endpoints (2025+), then an exact
  roster-name match from the play text. For 2025–26 the stats-based passer and rusher agree with
  CFBD's enriched ids on every play (80,002 and 75,787 plays, 0 disagreements). The enriched
  columns (air yards, direction, statuses) are attached where they exist.
- **Pre-snap game state.** `/plays` scores are post-play, which affects 115,517 plays (5.9%).
  The clean `offense_score`, `defense_score` and `score_diff` come from the previous play in the
  game. The raw values keep an `_after` suffix. [cfb/leakage.py](cfb/leakage.py) lists the
  pre-snap state columns, the outcome columns, and the postgame `games` columns, and tests
  enforce the split.
- **Play filters** (`plays.exclude_reason`, `model_eligible`):
  - non-scrimmage, kneel, spike, null `ppa`, and an unreliable enriched status;
  - unreliable means parse status `invalid`, plus the attribution statuses unmatched,
    ambiguous, conflict, unlinked and multi_carrier, decided from the probe;
  - **kneels:** text ("kneel", "takes a knee") or "TEAM run" for no gain or a loss. Against
    CFBD's `isKneel` in 2025–26 this catches 1,419/1,419, plus 87 TEAM-run kneels CFBD misses;
  - **spikes:** text or CFBD's `isSpike` (122/122).
- **Garbage time** (provisional, flag only): pre-snap margin > 38 in Q2, > 28 in Q3, > 22 in Q4.
  That's about 9–11% of scrimmage plays per season.
- **Identity:**
  - every credited athlete is on that season's roster;
  - placeholder (negative) ids are aliased only by exact keys (201 aliased);
  - class year is missing for 92% of 2014 roster rows, falling to 3–6% from 2020, so
    experience must come from roster history.
- **Recruits:** `recruiting.athleteId` covers slightly more (43,216 links vs 38,218 from
  rosters), but for players who record plays the two paths agree. Coverage is 49–79% of
  involved FBS players per season.
- **Transfers:**
  - the origin season comes from each entry's date (97% of exact matches land on that season);
  - 16,550 exact, unique full-name matches are accepted, and the destination roster confirms
    91% of them where the destination is known;
  - 924 fuzzy candidates are in `reports/transfer_review.csv`, and 1,422 have no candidate.
- **Extra tables beyond the brief's list:** `teams`, `play_types`, `stat_types`,
  `recruit_links`, `athlete_aliases`, `transfer_matches`, `team_coaching` (preseason
  coaching change, separate from in-season change), and `team_context` (talent, returning
  production).

**Data limits Phase 3 must handle:**
- Incompletion targets are 75% complete for 2014–2020 but 57/50/43/22% for 2021–2024.
- Defender sack credits are under 10% in 2014–15 and 2020, and 50% in 2021.
- Interceptor credits are 34–47% in 2021–2023.
- About 0.6% of games have no player credits at all.

## Phase 3 status (resumed 2026-10-15 on a 14 GB / 24-core machine)

**Done and tested** (`pytest` passes offline):
- Model, as-of driver, defender model, benchmarks, report generator, and the `cfb player` commands.
- **2019 NUTS validation (v2):** 0 divergences. The game-effect scales mix slowly: rush_tau_g R-hat
  1.11, pass_tau_g 1.05.
- **Mean-field SVI biases the hierarchical scales.** Every variant was compared with NUTS on 2019
  (`data/model/validation/effects_2019.parquet`):
  - non-centered: player tau_c 6–15x too small, posterior sds 0.12–0.36 of NUTS;
  - centered: tau_c overshoots and O's team scale collapses;
  - low-rank guide: no better than AutoNormal.
- **With the scales held at NUTS means, SVI matches NUTS:** player-value correlation 0.988–0.993,
  sd ratio 0.87–0.99, units 0.999.
- **Design adopted (empirical Bayes):** the within-season scales for season S's as-of fits are the mean
  of single-season NUTS fits on S-3..S-1 (`cfb player hyper`; `scales_for` refuses a missing or
  same-season fit). The season-step scales (tau_s, tau_team, rho) are left free in SVI. The defender
  model follows the same rule, with 2016 as its burn-in.
- **The scales drift by season:** 2014 pass_tau_c is 0.097 ± 0.011, against about 0.047 in 2019.

**NUTS fixes (2026-10-15)**, all in `run_chains` (fit.py), shared by the offensive and defender models:
- **Memory.** Chains run one at a time and only the summary sites are kept. Each chain's sampler was also
  compiled with the season's data baked in as constants, and JAX's compile cache kept every copy (about
  0.5 GB per chain); the data are now arguments (`jit_model_args`) and the cache is cleared between chains.
  A single-season scale fit now peaks at about 2.9 GB, and three run side by side.
- **Stuck chains.** With numpyro's default init (uniform(-2, 2) unconstrained, so scales start at 0.1–7),
  3 of 6 runs had one chain that sat at max tree depth and never adapted: 2015, 2018 and the 2014–15
  validation (R-hat on `nu` above 30,000). Chains now start at prior medians. 2015 and 2018 at the same
  seed then converged (worst scale R-hat 1.040 and 1.051, 63 leapfrog steps on every chain, 42 min each).
  2014, 2016 and 2017 converged under the old init and are kept.
- **Guards.** `cfb player hyper` rejects a fit whose worst scale R-hat is above 1.1
  (`hyper/rejected_season=YYYY_seed=N.json`), retries with the next seed, and otherwise writes nothing;
  `cfb player validate` stops before SVI if NUTS did not converge. Both take `--seed`, and every NUTS
  diagnostics file records per-chain steps, divergences and scale means.

**Scale fits done (2026-10-15):**
- Within-season, 2014–2025: worst scale R-hat 1.014–1.088, 0 divergences except 1 in 2022. 2022 failed twice
  at R-hat 1.11 on `rush_tau_g` (ESS about 20, slow mixing, not a stuck chain) and passed with 1000 + 1000
  draws (ESS 158). 2020, the short season, has one chain slightly apart on `pass_tau_c` and `recv_tau_c`.
- Defender, 2016–2025: worst R-hat 1.005–1.015, 0 divergences.
- **Defender validation (2016):** SVI with tau_c free is about 6% low (0.75 vs 0.80 for sacks); held at NUTS,
  spread and sd ratios are 0.99–1.04. Value correlation 0.96–0.99 either way. The bias is small but not zero,
  so the fixed defender scales stay.
- **Two-season validation (2014–15, rerun):** NUTS converged (worst R-hat 1.038). With the season-step scales
  free (svi_partial), SVI shrinks them like the within-season ones: O_tau_team 0.004 vs 0.040, player tau_s
  about 0.007 vs 0.023–0.039, D_rho 0.47 vs 0.69. Player values still correlate 0.965–0.991 with NUTS, but
  rush spread is 1.24x and the O unit sd ratio 0.65. With every scale held (svi_fixed): correlation
  0.965–0.999, spread 1.00–1.07. Hence the season-step decision above.

**Step-scale fits (paused 2026-10-15 overnight):** pairs ending 2015–2021 are done (`hyper_step/`); worst
step-scale R-hat 1.021–1.081, 0 divergences. The 2017–18 pair was rejected at first on `rush_tau_g` alone
(1.140, a within-season scale a step file doesn't store) and accepted once the rule was narrowed to the stored
scales. Each pair takes 90–125 min with three side by side (about 3.5–4.3 GB each).

**Next, in order:**
1. Resume: `seq 2022 2025 | xargs -P 3 -I{} cfb player hyper --step --seasons {}` (about 3.5 h; finished
   pairs are skipped). The 2015 pair (`hyper_step/season=2015.json`) comes from the validation run above.
2. `cfb player fit` (202 weekly as-of fits plus the defender fits), then `cfb player report`, then stop for
   approval. A partial run with free step scales (7 fits) is kept in `data/model/asof_free_step_scales/`.

Defender values used in evaluation start in 2017 (2016 is the defender model's burn-in).

## 1. Where the brief and the spec disagree

| # | Brief | Spec | What we do |
|---|---|---|---|
| 1 | `/passing/plays` and `/rushing/plays` both carry `parse_status` and `attribution_status` | `PassingPlay` has only `parseStatus` (complete / partial / invalid). `RushingPlay` has both; attribution is one of individual, team, multi_carrier, unmatched, ambiguous, conflict, unlinked | Confirmed live. Both endpoints exist only from 2025 (probe finding 1) |
| 2 | `/passing/plays`, `/rushing/plays` need `year` | Both also need `week` or `team`. `/rushing/plays` accepts `gameId` instead. `/games/players` and `/ppa/players/games` have the same requirement | Per-week pulls are required. Payloads are fine: a full week of `/plays` is 17k rows, about 10 MB |
| 3 | Pass plays are the passer's dropbacks | Sacks are in `/rushing/plays` (`isSack`), as are kneels (`isKneel`) and team rushes (`isTeamRush`). Spikes, throwaways and grounding are flags on passing plays | Stage 1 counts sack rows as passer dropbacks; otherwise sack-prone QBs look better than they are. Confirmed: `rusherId` is the QB on sacks. For 2014–2024, `Sack Taken` in `/plays/stats` gives the QB |
| 4 | Raw layout `season=YYYY/[week=WW]` | `/plays` has no `gameId` filter, and postseason weeks restart at 1 under `seasonType=postseason` | Week partitions include the season type (see §4) |
| 5 | `/plays/stats` gives `stat_type` | Rows carry the stat type name only; the request filter is `statTypeId`. `gameId` and `season` are typed float-or-int | Name-to-id map comes from `/plays/stats/types`; ids are cast to Int64 |
| 6 | — | Athlete and play ids are strings in play endpoints but ints in `/draft/picks.collegeAthleteId`. `Play` has no season, week or offense id | Ids are normalized to Int64. Season and week come from the game; team name maps to team id |
| 7 | `/roster` defaults `year` to 2025 | Confirmed. One call without `team` returns every team (22–30k rows). In the 2023–25 roster files already in `data/`, `year` is the class year (1–5), but **4–6% of rows hold the season (e.g. 2023)**, plus a few 0s | The planner always passes `year`. Phase 2 cleans `class_year` and flags suspect rows |
| 8 | — | `/recruiting/players` defaults to `HighSchool` | JUCO and PrepSchool are pulled separately (3 calls per class) |
| 9 | Derive home venue from home games | `/teams` returns each team's `location` (stadium coordinates, timezone, elevation, dome, grass) | Derive from home games as specified, and cross-check against `/teams` (1 call/season; it also gives team id and conference by season) |
| 10 | Weather requires a Patreon tier, behind a flag | Tier 2 includes weather, and `/info` reports `features.weather` | Flag stays default-off |
| 11 | — | `/games/{id}/preview/adjusted` is stored only until the game completes | Not ingested; it's no use for backtests |
| 12 | — | `Venue.elevation` is a string. `/ratings/elo` defaults to the latest week | Elevation is parsed to a float. Elo is benchmark-only; pregame Elo comes from `/games` |
| 13 | Validate with NUTS on one season | This machine has 6 GB of RAM and no GPU | NUTS on a full season may be slow; Phase 3 may validate on a subset |

**Environment notes.** Only Python 3.14 is installed. The full stack installs and imports on
3.14, and JAX runs on CPU. Versions resolved: jax 0.11, numpyro 0.22, polars 2.0, and
**arviz 1.3**. ArviZ 1.x is the new split-package API (`arviz_base`, `arviz_stats`), not the
0.x API most examples use.

## 2. Clean-layer tables (Phase 2 draft)

Ids are Int64. `play_id` stays Utf8 until the probe confirms it is numeric. Play-level tables
carry `game_id, season, week, season_type`.

| Table | Key | Main columns |
|---|---|---|
| `games` | game_id | season, week, season_type, start_date (UTC), neutral_site, conference_game, venue_id, home/away id, team, conference, classification, points, pregame_elo, is_fbs_game |
| `teams` | team_id, season | school, conference, classification, home_venue_id, alternate names (for name→id joins) |
| `plays` | play_id | drive, period, clock_seconds, game_seconds_remaining, offense_id, defense_id, is_home_offense, scores, timeouts, yards_to_goal, down, distance, yards_gained, play_type(_id), play_text, ppa, `garbage_time`, `exclude_reason` |
| `pass_plays` | play_id | passer_id, target_id, outcome, air_yards, depth/direction/location, yac, spike/throwaway/grounding flags, parse_status, ppa, success |
| `rush_plays` | play_id | rusher_id, direction, rushing_yards, is_sack, is_kneel, is_team_rush, attribution_status, parse_status, ppa, success |
| `play_participants` | play_id, athlete_id, stat_type | game_id, team_id, opponent_id, stat_type_id, stat, side (off/def/st) |
| `player_game_box` | game_id, athlete_id, category, stat_name | team_id, value_raw, value (`C/ATT` split into two rows) |
| `players` | athlete_id | names, normalized name, primary_position, height, weight, hometown, first/last roster season |
| `player_seasons` | athlete_id, season | team_id, position, jersey, class_year, class_year_raw, class_year_suspect, height, weight, recruit_ids |
| `recruits` + `recruit_links` | recruit_id; (athlete_id, recruit_id) | type, class year, ranking, stars, rating, position, size, school; link path = roster / recruiting / both |
| `transfers` + `transfer_matches` | transfer_key; (transfer_key, athlete_id) | season, names, position, origin, destination, rating; match score, method, accepted |
| `draft` | year, overall | college_athlete_id, round, pick, position, pre-draft ranking and grade |
| `lines` | game_id, provider | spread, spread_open, over_under(_open), moneylines |
| `venues` | venue_id | lat, lon, elevation_m, timezone, dome, grass, capacity |
| `coaches` | coach_id, team, season | record; derived `coaching_change` per team-season |
| `benchmarks` | long format | source, season, week, entity_type, entity_id, metric, value. Sources: pregame WP, Elo, SP+, SRS, FPI, PPA player game/season, WEPA pass/rush/kick |
| `team_context` | team_id, season | talent, returning production |
| lookups | — | play types, play stat types |

## 3. API call estimate

From `cfb ingest plan` for 2014–2026, FBS scope, after the probe. Week and game counts are
exact for seasons whose `/calendar` and `/games` are on disk, and estimated (17 weeks, ~882
games) for the rest.

| Endpoint group | Granularity | Calls, 2014–2026 |
|---|---|---|
| **`/plays/stats`** | **per game** | **~10,890** |
| `/plays`, `/games/players`, `/ppa/players/games` | per week | ~630 |
| `/passing/plays`, `/rushing/plays` | per week, 2025–2026 only | ~48 |
| `/games`, `/lines`, `/metrics/wp/pregame` | per season type | ~71 |
| 16 season-level endpoints (calendar, teams, roster, portal, draft, coaches, talent, returning, PPA season, 3× WEPA, 4× ratings) | per season | ~200 |
| `/recruiting/players` | per class (2009–2026) × 3 types | 54 |
| **Total still to pull** | | **~11,900** |

- **This fits in the current month's remaining 29,957 calls**, with room for retries and
  `/plays/stats` re-pulls (none expected at 170–270 rows per game).
- `--fcs` would add roughly 700 calls per season. `--weather` adds 2 per season.
- In-season 2026 updates cost about 75 calls per week.
- The per-week rare-stat shortcut from Phase 0 no longer helps: `/plays/stats` per game is
  needed for offensive attribution anyway.

## 4. Ingestion design (Phase 1)

- **Raw paths:** `data/raw/<endpoint>/season=YYYY/type=<regular|postseason>/week=WW/response.json.gz`;
  per game, `.../season=YYYY/game=ID/` with `stat_type=N/` sub-partitions for truncation
  re-pulls; `season=all/` for lookups. FCS-scope requests add `division=fcs/`. Season-level
  endpoints are partitioned by the `year` they request, so `/draft/picks` for the 2024 season
  lands under `season=2025`.
- **Atomic writes:** each response goes to a temp file and is then renamed. A partition is
  complete exactly when its `response.json.gz` exists.
- **Refresh (current season only):** `--refresh` renames the existing file to
  `response.<pulled_at>.json.gz` and writes the new pull as `response.json.gz`. Raw files are
  never edited.
- **Manifest:** one row per request, with an `attempts` column, since retries count against
  the budget. Rows are appended to `data/manifest_journal.jsonl` as requests finish and
  compacted into `data/manifest.parquet` at the end of a run.
- **Two stages:** stage A pulls `/calendar`, `/games`, and the lookups. Stage B is planned
  from the raw files stage A wrote. The budget is checked against `/info` (free) before each
  stage and every 1,000 requests, and against `--max-calls`, a hard cap on counted calls.
- **Failures:** a 401 or an exhausted 429 aborts the run. Five consecutive failures abort it.
  Isolated failures are recorded in the manifest and not landed, so a re-run retries them.
  Interrupted runs resume by skipping landed partitions.
- **Truncation:** a `/plays/stats` page at the 2,000-row cap is re-pulled once per
  `statTypeId` under `game=ID/stat_type=N/`. Re-pulls owed by an interrupted run are
  recovered from the manifest.
- **Summary:** `cfb ingest summary` writes `reports/phase1b_ingest.md`. It's safe while an
  ingest is running because manifest reads never modify files.
- **Key hygiene:** the key comes only from the environment or `.env`. It is redacted from the
  client `repr` and never written to the manifest.

## 5. Phase 1a probe (done)

`cfb ingest probe` covers 2024 week 1 (largest week, 121 FBS-vs-FCS games in the season),
2014 week 5 (the first season of the range), and 2025 week 1 (enriched endpoints). Most pulls
land in the canonical raw tree, so bulk ingestion reuses them. Experiments (the per-week sack
pull, `/info`, the diagnostics) live under `data/raw/_probe/`. Trimmed responses are in
`tests/fixtures/`.

## 6. Decisions pending with the project owner

None open; Phase 3 questions will be listed with its results.
