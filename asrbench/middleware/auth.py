"""
API key auth middleware.

ASRbench is primarily a local single-user tool, but PyPI v1.0 shipped
with support for running on an internet-facing host (``asrbench serve
--host 0.0.0.0 --allow-network``). Once the process listens on a public
interface, unauthenticated access would let anyone on the network start
benchmark jobs, spawn optimizer studies, delete datasets, or read
cached transcripts.

Policy (enforced in ``cli/serve.py`` for the CLI surface and here for
every HTTP request):

* Requests from loopback (``127.0.0.1``, ``localhost``, ``::1``) are
  always allowed — single-user local UX must not require a header.
* Requests from non-loopback clients must carry ``X-API-Key`` matching
  the server's ``ASRBENCH_API_KEY`` env var (or ``asrbench.api_key``
  config key).
* If the server is bound to a non-loopback host but no API key is
  configured, every non-loopback request is rejected with 401 — the
  CLI blocks startup in that case, but the middleware defends against
  env-var loss after launch.

WebSocket upgrades reuse the same logic: loopback-only in dev, auth
header required for remote clients. Starlette exposes the subprotocol
header list on the scope so we reject the upgrade before any handler
runs.
"""

from __future__ import annotations

import logging
import os
from typing import Final

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

LOOPBACK_HOSTS: Final[frozenset[str]] = frozenset({"127.0.0.1", "localhost", "::1", "testclient"})
"""Client addresses treated as loopback — auth is skipped for these.

``testclient`` is the sentinel Starlette's :class:`TestClient` uses for its
synchronous in-process transport. It is never a real network client, so
including it here lets integration tests exercise the full routing stack
without every test having to wire an API key.
"""

_API_KEY_ENV: Final[str] = "ASRBENCH_API_KEY"
_API_KEY_HEADER: Final[bytes] = b"x-api-key"


def get_api_key() -> str | None:
    """Return the configured API key (env var only; process-level secret).

    Read at request time rather than at middleware construction so an
    operator can rotate ``ASRBENCH_API_KEY`` via ``supervisorctl restart``
    without patching the codebase. Empty strings are treated as unset.
    """
    value = os.environ.get(_API_KEY_ENV, "").strip()
    return value or None


def _is_loopback_client(scope: Scope) -> bool:
    client = scope.get("client")
    if not client:
        return False
    return str(client[0]) in LOOPBACK_HOSTS


def _extract_api_key(scope: Scope) -> str | None:
    """Read the ``X-API-Key`` header value, returning ``None`` if absent."""
    for name, value in scope.get("headers", []):
        if name.lower() == _API_KEY_HEADER:
            try:
                return value.decode("latin-1").strip()
            except UnicodeDecodeError:
                return None
    return None


class AuthMiddleware:
    """ASGI middleware enforcing ``X-API-Key`` for non-loopback clients.

    Order matters: install this BEFORE the rate limiter so unauthenticated
    requests never get a bucket entry that a casual attacker could use to
    fingerprint valid-IP ranges.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        scope_type = scope["type"]
        if scope_type == "lifespan":
            await self.app(scope, receive, send)
            return

        if _is_loopback_client(scope):
            await self.app(scope, receive, send)
            return

        expected = get_api_key()
        provided = _extract_api_key(scope)

        if expected is None:
            # Server bound to a non-loopback host without an API key —
            # CLI is supposed to have refused to start, but defend in
            # depth: block every remote request.
            await self._deny(scope, receive, send, "server is not configured for network access")
            return

        if provided is None or provided != expected:
            await self._deny(scope, receive, send, "missing or invalid X-API-Key")
            return

        await self.app(scope, receive, send)

    async def _deny(self, scope: Scope, receive: Receive, send: Send, detail: str) -> None:
        if scope["type"] == "websocket":
            await self._deny_websocket(receive, send)
            return
        response = JSONResponse(status_code=401, content={"detail": detail})
        await response(scope, receive, send)

    async def _deny_websocket(self, receive: Receive, send: Send) -> None:
        """Close a WS upgrade with policy-violation code 1008.

        Starlette requires the connect event to be consumed before we can
        respond, otherwise the ASGI server treats the handler as buggy.
        """
        message = await receive()
        if message["type"] == "websocket.connect":
            await send({"type": "websocket.close", "code": 1008})
