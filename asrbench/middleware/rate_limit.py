"""
In-memory per-client rate limiter — ASGI middleware.

Uses a token-bucket algorithm keyed by client IP. No external dependencies;
state lives in process memory and resets on restart (appropriate for a
single-process local tool like asrbench).

Configuration comes from LimitsConfig but has sensible defaults so the
middleware works even without a config file.

Exemptions:
    - WebSocket upgrades are NOT rate-limited (they're long-lived connections)
    - GET /system/health is NOT rate-limited (monitoring probes)
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

# Paths exempt from rate limiting (exact match after stripping trailing slash).
_EXEMPT_PATHS = frozenset({"/system/health", "/system/vram"})


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

        path = scope.get("path", "").rstrip("/")
        if path in _EXEMPT_PATHS:
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
