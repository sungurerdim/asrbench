"""
In-memory per-client rate limiter — ASGI middleware.

Uses a token-bucket algorithm keyed by client IP. No external dependencies;
state lives in process memory and resets on restart (appropriate for a
single-process local tool like asrbench).

Configuration comes from LimitsConfig but has sensible defaults so the
middleware works even without a config file.

Exemptions:
    - WebSocket upgrades are NOT rate-limited (they're long-lived connections)
    - GET /system/health and GET /system/vram are NOT rate-limited
      (monitoring probes)
    - GET /runs/... and GET /optimize/... polling is NOT rate-limited
      (the UI refreshes detail panes every second or two while a job is
      in flight; the default 120 req/min bucket would 429 any moderately
      active dashboard)

Mutating endpoints (POST /runs/start, POST /optimize/start,
POST /datasets/fetch, POST /models/register, DELETE /*) always remain
subject to the limiter, even under the exempt prefixes, so a stolen or
leaked API key cannot be used to spam expensive jobs.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

_EXEMPT_EXACT: frozenset[tuple[str, str]] = frozenset(
    {
        ("GET", "/system/health"),
        ("GET", "/system/vram"),
    }
)

_EXEMPT_PREFIX_RULES: tuple[tuple[str, str], ...] = (
    ("GET", "/runs/"),
    ("GET", "/optimize/"),
)


def _is_exempt(method: str, path: str) -> bool:
    """Return True when (method, path) matches a polling-exempt rule.

    Only GET requests under the /runs/ and /optimize/ prefixes are exempt;
    the POST /runs/start and POST /optimize/start endpoints fall through
    to the limiter so a client cannot spam benchmark starts.
    """
    if (method, path) in _EXEMPT_EXACT:
        return True
    for exempt_method, prefix in _EXEMPT_PREFIX_RULES:
        if method == exempt_method and path.startswith(prefix):
            return True
    return False


@dataclass
class _Bucket:
    """Token bucket for one client."""

    tokens: float
    last_refill: float


class RateLimitMiddleware:
    """
    ASGI middleware that enforces a per-IP request rate limit.

    Args:
        app: the ASGI application to wrap
        requests_per_minute: maximum sustained request rate per client IP
        burst: maximum burst size (bucket capacity). Defaults to 2x the
               per-minute rate, so a client can burst briefly without being
               throttled as long as the average stays below the rate.

    Response on throttle:
        HTTP 429 Too Many Requests with a JSON body:
        {"detail": "Rate limit exceeded. Try again in {n:.1f}s."}
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        requests_per_minute: int = 120,
        burst: int | None = None,
    ) -> None:
        self.app = app
        self.rate = requests_per_minute / 60.0  # tokens per second
        self.burst = float(burst if burst is not None else requests_per_minute * 2)
        self._buckets: dict[str, _Bucket] = defaultdict(
            lambda: _Bucket(tokens=self.burst, last_refill=time.monotonic())
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            # WebSocket or lifespan — pass through without limiting
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method", "GET")).upper()
        path = scope.get("path", "").rstrip("/")
        if _is_exempt(method, path):
            await self.app(scope, receive, send)
            return

        # Identify client by IP (X-Forwarded-For not trusted for a local tool)
        client = scope.get("client")
        client_ip = client[0] if client else "unknown"

        bucket = self._buckets[client_ip]
        now = time.monotonic()

        # Refill tokens since last request
        elapsed = now - bucket.last_refill
        bucket.tokens = min(self.burst, bucket.tokens + elapsed * self.rate)
        bucket.last_refill = now

        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            await self.app(scope, receive, send)
        else:
            # Throttled — compute wait time for the next token
            wait = (1.0 - bucket.tokens) / self.rate
            response = JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded. Try again in {wait:.1f}s."},
            )
            await response(scope, receive, send)
