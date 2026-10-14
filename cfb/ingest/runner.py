"""Execute planned requests: call the API, land raw responses, record the manifest."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cfb.api.client import CFBDClient, CFBDError
from cfb.config import Settings
from cfb.ingest import raw_store
from cfb.ingest.manifest import Manifest, ManifestRow
from cfb.ingest.planner import PlannedRequest


@dataclass(frozen=True)
class Outcome:
    status: int  # 0 = transport failure after all retries
    payload: Any  # parsed JSON on success, else None
    file_path: Path | None
    row_count: int | None
    truncated: bool
    attempts: int


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def row_count(payload: Any) -> int:
    if isinstance(payload, list):
        return len(payload)
    return 0 if payload is None else 1


def execute(
    req: PlannedRequest,
    client: CFBDClient,
    settings: Settings,
    manifest: Manifest,
    run_id: str,
    *,
    raw_root: Path | None = None,
    refresh: bool = False,
    land: bool = True,
) -> Outcome:
    raw_root = raw_root or settings.paths.raw
    requested_at = datetime.now(timezone.utc)
    calls_before = client.calls_made
    try:
        result = client.get(req.spec.path, req.params)
    except CFBDError as exc:
        attempts = client.calls_made - calls_before
        manifest.append(ManifestRow(
            run_id=run_id, endpoint=req.spec.slug, path=req.spec.path,
            params=json.dumps(req.params, sort_keys=True), requested_at=requested_at.isoformat(),
            http_status=None, attempts=attempts, row_count=None, bytes=None, file_path=None,
            truncated=False, elapsed_ms=None, error=str(exc),
        ))
        return Outcome(0, None, None, None, False, attempts)

    payload = file_path = rows = None
    truncated = False
    error = None
    if result.ok and not land:
        payload = result.json()
        rows = row_count(payload)
    elif result.ok:
        payload = result.json()
        rows = row_count(payload)
        truncated = req.spec.slug == "plays_stats" and rows >= settings.ingest.plays_stats_cap
        file_path = raw_store.write_response(
            raw_store.response_path(raw_root, req.spec.slug, req.partition),
            result.content,
            refresh=refresh,
            pulled_at=requested_at,
        )
    else:
        error = result.content[:500].decode("utf-8", errors="replace")

    manifest.append(ManifestRow(
        run_id=run_id,
        endpoint=req.spec.slug,
        path=req.spec.path,
        params=json.dumps(req.params, sort_keys=True),
        requested_at=requested_at.isoformat(),
        http_status=result.status,
        attempts=result.attempts,
        row_count=rows,
        bytes=len(result.content),
        file_path=file_path.relative_to(settings.paths.root).as_posix() if file_path else None,
        truncated=truncated,
        elapsed_ms=result.elapsed_ms,
        error=error,
    ))
    return Outcome(result.status, payload, file_path, rows, truncated, result.attempts)
