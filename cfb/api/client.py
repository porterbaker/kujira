"""CFBD REST client: bearer auth, retries with backoff on 429/5xx, polite throttling.

Every HTTP attempt, retries included, counts against the API budget and is tallied in
``calls_made``. The key lives only in the request header. It is never part of a URL,
log line, exception, or ``repr``.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import RetryCallState, Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from cfb.config import ApiSettings

# httpx logs every request at INFO; keep it quiet.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class CFBDError(RuntimeError):
    pass


class _RetryableStatus(Exception):
    def __init__(self, response: httpx.Response) -> None:
        super().__init__(f"HTTP {response.status_code}")
        self.response = response


@dataclass(frozen=True)
class ApiResult:
    path: str
    params: dict[str, Any]
    status: int
    content: bytes  # response body, verbatim
    attempts: int  # HTTP calls spent on this request
    elapsed_ms: int

    @property
    def ok(self) -> bool:
        return self.status == 200

    def json(self) -> Any:
        return json.loads(self.content)


class CFBDClient:
    def __init__(
        self,
        api_key: str,
        settings: ApiSettings | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is empty")
        self.settings = settings or ApiSettings()
        self._http = httpx.Client(
            base_url=self.settings.base_url,
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            timeout=self.settings.timeout_s,
            transport=transport,
        )
        self._sleep = sleep
        self._clock = clock
        self._last_call: float | None = None
        self.calls_made = 0

    def __repr__(self) -> str:
        return f"CFBDClient(base_url={self.settings.base_url!r}, api_key=<redacted>, calls_made={self.calls_made})"

    def __enter__(self) -> CFBDClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> ApiResult:
        """GET with retries. Returns the final response, including non-retryable errors like 400/401/403.

        Raises CFBDError only when the transport keeps failing (timeouts, connection errors).
        """
        query = {k: v for k, v in (params or {}).items() if v is not None}
        calls_before = self.calls_made
        started = self._clock()
        retryer = Retrying(
            retry=retry_if_exception(lambda e: isinstance(e, (_RetryableStatus, httpx.TransportError))),
            wait=self._wait,
            stop=stop_after_attempt(self.settings.max_attempts),
            sleep=self._sleep,
            reraise=True,
        )
        try:
            response = retryer(self._send, path, query)
        except _RetryableStatus as exc:
            response = exc.response
        except httpx.TransportError as exc:
            raise CFBDError(
                f"GET {path} failed after {self.calls_made - calls_before} attempts: {type(exc).__name__}"
            ) from None
        return ApiResult(
            path=path,
            params=query,
            status=response.status_code,
            content=response.content,
            attempts=self.calls_made - calls_before,
            elapsed_ms=round((self._clock() - started) * 1000),
        )

    def info(self) -> dict[str, Any]:
        """GET /info: Patreon tier, remaining calls, feature access. Costs one call."""
        result = self.get("/info")
        if not result.ok:
            raise CFBDError(f"GET /info returned HTTP {result.status}")
        payload = result.json()
        if not payload:
            raise CFBDError("GET /info returned null: the API key was not accepted")
        return payload

    def _send(self, path: str, query: dict[str, Any]) -> httpx.Response:
        self._throttle()
        self.calls_made += 1
        response = self._http.get(path, params=query)
        if response.status_code in RETRY_STATUSES:
            raise _RetryableStatus(response)
        return response

    def _throttle(self) -> None:
        if self._last_call is not None:
            wait = self.settings.min_interval_s - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)
        self._last_call = self._clock()

    def _wait(self, state: RetryCallState) -> float:
        exc = state.outcome.exception() if state.outcome else None
        if isinstance(exc, _RetryableStatus):
            retry_after = exc.response.headers.get("Retry-After", "")
            if retry_after.replace(".", "", 1).isdigit():
                return min(float(retry_after), self.settings.backoff_max_s)
        backoff = wait_exponential_jitter(multiplier=self.settings.backoff_min_s, max=self.settings.backoff_max_s)
        return backoff(state)
