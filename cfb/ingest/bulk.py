"""Phase 1b bulk ingestion.

Stage A pulls what the planner needs (lookups, ``/calendar``, ``/games``). Stage B is then
planned exactly from raw. The budget is checked against ``/info`` (free) and against an
explicit ``max_calls`` cap before each stage and periodically during it. Runs are resumable:
landed partitions are skipped, so a re-run continues where an interrupted one stopped.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from cfb.api.budget import check_budget, describe_info
from cfb.api.client import CFBDClient
from cfb.config import Settings
from cfb.ingest import raw_store, runner
from cfb.ingest.manifest import Manifest
from cfb.ingest.planner import PlannedRequest, build_plan
from cfb.ingest.probe import INFO
from cfb.ingest.registry import BY_SLUG, ENDPOINTS

STAGES = {"A": frozenset(s.slug for s in ENDPOINTS if s.stage == "A"),
          "B": frozenset(s.slug for s in ENDPOINTS if s.stage == "B")}
MAX_CONSECUTIVE_FAILURES = 5
INFO_CHECK_EVERY = 1000  # requests
PROGRESS_EVERY_S = 60.0


class Abort(Exception):
    pass


@dataclass
class BulkResult:
    run_id: str
    data_calls: int = 0
    requests: int = 0
    failures: list[tuple[str, dict, int]] = field(default_factory=list)  # (endpoint, params, status)
    truncated: list[PlannedRequest] = field(default_factory=list)
    empty: int = 0


def stat_type_ids(settings: Settings) -> list[int]:
    path = raw_store.response_path(settings.paths.raw, "plays_stats_types", (raw_store.season_part("all"),))
    return sorted(int(row["id"]) for row in raw_store.read_json(path))


def split_requests(page: PlannedRequest, type_ids: list[int]) -> list[PlannedRequest]:
    """Re-pull a full /plays/stats page one stat type at a time."""
    return [PlannedRequest(page.spec, {**page.params, "statTypeId": t}, page.partition + (("stat_type", str(t)),))
            for t in type_ids]


def pending_splits(settings: Settings, manifest: Manifest) -> list[PlannedRequest]:
    """Split re-pulls owed by truncated pages from earlier (possibly interrupted) runs."""
    df = manifest.read()
    if df.is_empty():
        return []
    rows = df.filter(df["truncated"] & (df["endpoint"] == "plays_stats")).iter_rows(named=True)
    out: list[PlannedRequest] = []
    for row in rows:
        if not row["file_path"]:
            continue
        part_dir = (settings.paths.root / row["file_path"]).parent.relative_to(settings.paths.raw / "plays_stats")
        partition = tuple(tuple(seg.split("=", 1)) for seg in part_dir.parts)
        if any(k == "stat_type" for k, _ in partition):
            continue
        page = PlannedRequest(BY_SLUG["plays_stats"], {"gameId": int(dict(partition)["game"])}, partition)
        out += [r for r in split_requests(page, stat_type_ids(settings))
                if not raw_store.is_landed(settings.paths.raw, "plays_stats", r.partition)]
    return out


def run(
    settings: Settings,
    client: CFBDClient,
    manifest: Manifest,
    seasons: tuple[int, ...],
    *,
    endpoints: set[str] | None = None,
    stages: tuple[str, ...] = ("A", "B"),
    refresh: bool = False,
    max_calls: int | None = None,
    now: datetime | None = None,
    log: Callable[[str], None] = print,
) -> BulkResult:
    result = BulkResult(runner.new_run_id())
    info_calls = 0

    def info() -> dict:
        nonlocal info_calls
        out = runner.execute(PlannedRequest(INFO, {}, ()), client, settings, manifest, result.run_id, land=False)
        info_calls += out.attempts
        if out.status != 200 or not out.payload:
            raise Abort(f"/info failed (HTTP {out.status}); check CFB_APIKEY")
        return out.payload

    def data_calls() -> int:
        return client.calls_made - info_calls

    def ensure_budget(planned: int, label: str) -> None:
        status = info()
        check = check_budget(status, planned, settings.api.budget_buffer)
        log(f"[{label}] {check.message()}")
        if not check.ok:
            raise Abort(f"stage {label}: planned calls exceed remaining calls minus buffer")
        if max_calls is not None and data_calls() + planned > max_calls:
            raise Abort(f"stage {label}: {data_calls()} made + {planned} planned exceeds --max-calls {max_calls}")

    try:
        log("account: " + describe_info(info()))
        for stage in stages:
            wanted = set(STAGES[stage]) if endpoints is None else STAGES[stage] & endpoints
            plan = build_plan(seasons, settings, endpoints=wanted, refresh=refresh, now=now)
            queue = list(plan.requests)
            if stage == "B" and "plays_stats" in wanted:
                queue += pending_splits(settings, manifest)
            planned = len(queue) + plan.estimated_count
            if stage == "A" and "B" in stages:  # check the whole run up front, B still estimated
                b_wanted = set(STAGES["B"]) if endpoints is None else STAGES["B"] & endpoints
                planned += build_plan(seasons, settings, endpoints=b_wanted, refresh=refresh, now=now).total
            log(f"[{stage}] {len(queue)} requests to make ({len(plan.skipped)} already landed); "
                f"budget check covers {planned} calls")
            ensure_budget(planned, stage)
            _execute_queue(stage, queue, settings, client, manifest, result, data_calls, info, max_calls, refresh, log)
        log("account after: " + describe_info(info()))
    finally:
        result.data_calls = data_calls()
        manifest.compact()
    return result


def _execute_queue(stage, queue, settings, client, manifest, result, data_calls, info, max_calls, refresh, log):
    started = last_log = time.monotonic()
    consecutive = 0
    type_ids: list[int] | None = None
    i = 0
    while i < len(queue):
        req = queue[i]
        i += 1
        if raw_store.is_landed(settings.paths.raw, req.spec.slug, req.partition) and not refresh:
            continue
        out = runner.execute(req, client, settings, manifest, result.run_id, refresh=refresh)
        result.requests += 1

        if out.status == 200:
            consecutive = 0
            result.empty += out.row_count == 0
            if out.truncated:
                type_ids = type_ids or stat_type_ids(settings)
                splits = split_requests(req, type_ids)
                queue.extend(splits)
                result.truncated.append(req)
                log(f"[{stage}] truncated page {req.params}; queued {len(splits)} split re-pulls")
        else:
            consecutive += 1
            result.failures.append((req.spec.slug, req.params, out.status))
            log(f"[{stage}] FAILED {req.spec.path} {req.params}: HTTP {out.status or 'transport error'}")
            if out.status == 401:
                raise Abort("HTTP 401: the API key was rejected")
            if out.status == 429:
                raise Abort("HTTP 429 after all retries: rate limited or monthly quota exhausted")
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                raise Abort(f"{consecutive} consecutive failures")

        if max_calls is not None and data_calls() > max_calls:
            raise Abort(f"--max-calls {max_calls} reached")
        if result.requests % INFO_CHECK_EVERY == 0:
            status = info()
            remaining = status.get("remainingCalls")
            if remaining is not None and remaining < settings.api.budget_buffer:
                raise Abort(f"only {remaining} calls remain (buffer {settings.api.budget_buffer})")
            manifest.compact()
        now = time.monotonic()
        if now - last_log >= PROGRESS_EVERY_S or i == len(queue):
            done, total = i, len(queue)
            rate = (now - started) / max(done, 1)
            eta = rate * (total - done)
            log(f"[{stage}] {done}/{total} ({done / max(total, 1):.0%}) data calls={data_calls()} "
                f"failures={len(result.failures)} elapsed={(now - started) / 60:.1f}m eta={eta / 60:.1f}m "
                f"last={req.spec.slug} {req.params}")
            last_log = now
