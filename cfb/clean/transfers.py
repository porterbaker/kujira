"""Transfer portal entries matched to athlete ids.

/player/portal has names only. Each entry is scored against the origin team's roster around its
origin season. The score rests on an exact last-name match plus a compatible first name (prefix,
same initial, rapidfuzz similarity: "Cam"/"Cameron", "Mike"/"Michael"), adjusted for position
group, the destination roster, season distance, and teammates sharing the last name. Only
exact, unique normalized full-name matches are accepted without a threshold.
Fuzzy matches are accepted only at or above ``Thresholds.transfer_match_threshold``. Everything
else goes to ``reports/transfer_review.csv``; no fuzzy match is ever accepted silently.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime

import polars as pl
from rapidfuzz import fuzz

from cfb.clean.common import frame, iter_raw, normalize_name, parse_ts, position_group, to_float, to_int
from cfb.clean.reference import team_name_index
from cfb.config import Settings

TRANSFER_SCHEMA = {
    "transfer_key": pl.String, "season": pl.Int16, "first_name": pl.String, "last_name": pl.String,
    "name_norm": pl.String, "position": pl.String, "position_group": pl.String, "origin": pl.String,
    "origin_id": pl.Int64, "destination": pl.String, "destination_id": pl.Int64,
    "transfer_date": pl.Datetime("us", "UTC"), "origin_season": pl.Int16, "rating": pl.Float64,
    "stars": pl.Int8, "eligibility": pl.String,
}

MATCH_SCHEMA = {
    "transfer_key": pl.String, "athlete_id": pl.Int64, "candidate_name": pl.String,
    "candidate_position": pl.String, "candidate_roster_season": pl.Int16, "name_score": pl.Float64,
    "position_match": pl.Boolean, "destination_confirmed": pl.Boolean, "second_best_score": pl.Float64,
    "confidence": pl.Float64, "method": pl.String, "accepted": pl.Boolean, "n_candidates": pl.Int32,
    "last_name_match": pl.Boolean, "first_name_compatible": pl.Boolean, "shared_last_name": pl.Boolean,
}

MIN_NAME_SCORE = 50.0  # below this the best candidate is not a candidate at all


def name_score(first: str, last: str, cand_first: str, cand_last: str) -> tuple[float, bool, bool]:
    """(score 0-100, last names equal, first names compatible). Last names must match to score above 60."""
    if last and last == cand_last:
        prefix = len(min(first, cand_first, key=len)) >= 2 and (first.startswith(cand_first) or cand_first.startswith(first))
        initial = bool(first) and bool(cand_first) and first[0] == cand_first[0]
        score = 70 + (20 if prefix else 10 if initial else 0) + 10 * fuzz.ratio(first, cand_first) / 100
        return score, True, prefix or initial
    return 0.6 * fuzz.token_sort_ratio(f"{first} {last}", f"{cand_first} {cand_last}"), False, False


def origin_season(season: int, transfer_date: datetime | None) -> int:
    """The season played before entering the portal: Aug-Dec entries belong to that year's season."""
    if transfer_date is None:
        return season - 1
    return transfer_date.year if transfer_date.month >= 8 else transfer_date.year - 1


def transfer_key(season: int, name_norm: str, origin: str | None, date: str | None, position: str | None) -> str:
    raw = f"{season}|{name_norm}|{origin}|{date}|{position}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def build_transfers(settings: Settings, teams: pl.DataFrame) -> pl.DataFrame:
    index = team_name_index(teams)
    rows = []
    for part, payload in iter_raw(settings, "player_portal"):
        season = int(part["season"])
        for t in payload:
            first, last = t.get("firstName"), t.get("lastName")
            norm = normalize_name(f"{first or ''} {last or ''}")
            date = parse_ts(t.get("transferDate"))
            o_season = origin_season(season, date)
            rows.append({
                "transfer_key": transfer_key(season, norm, t.get("origin"), t.get("transferDate"), t.get("position")),
                "season": season, "first_name": first, "last_name": last, "name_norm": norm,
                "position": t.get("position"), "position_group": position_group(t.get("position")),
                "origin": t.get("origin"), "origin_id": index.get(o_season, t.get("origin")),
                "destination": t.get("destination"), "destination_id": index.get(season, t.get("destination")),
                "transfer_date": date, "origin_season": o_season, "rating": to_float(t.get("rating")),
                "stars": to_int(t.get("stars")), "eligibility": t.get("eligibility"),
            })
    return frame(rows, TRANSFER_SCHEMA).unique("transfer_key", keep="first").sort("season", "transfer_key")


def _positions_compatible(a: str | None, b: str | None) -> bool:
    return a is None or b is None or a == b or "ATH" in (a, b)


def match_transfers(transfers: pl.DataFrame, player_seasons: pl.DataFrame, threshold: float | None) -> pl.DataFrame:
    roster: dict[tuple[int, int], list[tuple]] = defaultdict(list)
    on_team: dict[tuple[int, int], set[int]] = defaultdict(set)
    for athlete_id, season, team_id, name, first, last, pos, grp in player_seasons.select(
            "athlete_id", "season", "team_id", "name_norm", "first_name", "last_name", "position",
            "position_group").iter_rows():
        if team_id is not None:
            roster[(team_id, season)].append((athlete_id, name, normalize_name(first), normalize_name(last), pos, grp))
            on_team[(team_id, season)].add(athlete_id)

    out = []
    for t in transfers.iter_rows(named=True):
        base = {"transfer_key": t["transfer_key"], "athlete_id": None, "candidate_name": None,
                "candidate_position": None, "candidate_roster_season": None, "name_score": None,
                "position_match": None, "destination_confirmed": None, "second_best_score": None,
                "confidence": None, "method": "none", "accepted": False, "n_candidates": 0,
                "last_name_match": None, "first_name_compatible": None, "shared_last_name": None}
        if t["origin_id"] is None or not t["name_norm"]:
            out.append(base)
            continue
        cands: dict[int, tuple] = {}
        for offset in (0, -1, 1):  # origin season first, then neighbours
            for athlete_id, name, first, last, pos, grp in roster.get((t["origin_id"], t["origin_season"] + offset), []):
                cands.setdefault(athlete_id, (name, first, last, pos, grp, t["origin_season"] + offset))
        first, last = normalize_name(t["first_name"]), normalize_name(t["last_name"])
        # Ties prefer the origin season, then real ids over placeholders (which never have stats).
        scored = sorted(((*name_score(first, last, c[1], c[2]), aid, c) for aid, c in cands.items()),
                        key=lambda x: (-x[0], abs(x[4][5] - t["origin_season"]), x[3] < 0, x[3]))
        base["n_candidates"] = len(scored)
        if not scored or scored[0][0] < MIN_NAME_SCORE:
            out.append(base)
            continue
        score, last_ok, first_ok, athlete_id, (name, _, cand_last, pos, grp, roster_season) = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        shared_last = sum(1 for c in cands.values() if c[2] == cand_last) > 1
        pos_ok = _positions_compatible(t["position_group"], grp)
        dest_ok = t["destination_id"] is not None and athlete_id in on_team.get((t["destination_id"], t["season"]), set())
        exact_unique = name == t["name_norm"] and athlete_id > 0 and \
            sum(1 for aid, c in cands.items() if c[0] == name and aid > 0) == 1
        confidence = (score + (5 if pos_ok else -10) + (5 if dest_ok else 0)
                      - 5 * abs(roster_season - t["origin_season"])
                      - (15 if shared_last or second >= score - 5 else 0))
        confidence = max(0.0, min(100.0, confidence))
        method = "exact" if exact_unique else "fuzzy"
        accepted = exact_unique or (threshold is not None and confidence >= threshold)
        out.append({**base, "athlete_id": athlete_id, "candidate_name": name, "candidate_position": pos,
                    "candidate_roster_season": roster_season, "name_score": round(score, 1), "position_match": pos_ok,
                    "destination_confirmed": dest_ok, "second_best_score": round(second, 1),
                    "confidence": round(confidence, 1), "method": method, "accepted": accepted,
                    "last_name_match": last_ok, "first_name_compatible": first_ok, "shared_last_name": shared_last})
    return frame(out, MATCH_SCHEMA)


def review_rows(transfers: pl.DataFrame, matches: pl.DataFrame) -> pl.DataFrame:
    """Fuzzy candidates that were not accepted, for human review."""
    return (matches.filter((pl.col("method") == "fuzzy") & ~pl.col("accepted"))
            .join(transfers.select("transfer_key", "season", "transfer_date", "first_name", "last_name", "position",
                                   "origin", "destination"), on="transfer_key")
            .select("transfer_key", "season", "transfer_date", "first_name", "last_name", "position", "origin",
                    "destination", "athlete_id", "candidate_name", "candidate_position", "candidate_roster_season",
                    "name_score", "last_name_match", "first_name_compatible", "shared_last_name", "position_match",
                    "destination_confirmed", "second_best_score", "confidence")
            .sort("confidence", descending=True))
