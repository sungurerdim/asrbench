"""``asrbench serve`` — start the FastAPI server."""

from __future__ import annotations

import logging
import os
import sys

import typer

from asrbench.middleware.auth import LOOPBACK_HOSTS

logger = logging.getLogger(__name__)


def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind address."),
    port: int = typer.Option(8765, "--port", "-p", help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development."),
    open_browser: bool = typer.Option(
        False,
        "--open/--no-open",
        help="Open the UI in the default browser once the server is up.",
    ),
    allow_network: bool = typer.Option(
        False,
        "--allow-network",
        help=(
            "Required to bind to a non-loopback host. The server will only "
            "start when ASRBENCH_API_KEY is set so remote clients must "
            "present a matching X-API-Key header."
        ),
    ),
) -> None:
    """Start the ASRbench API server."""
    import uvicorn

    _enforce_network_policy(host=host, allow_network=allow_network)

    if open_browser:
        _schedule_browser_open(host, port)

    uvicorn.run(
        "asrbench.main:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


def _enforce_network_policy(*, host: str, allow_network: bool) -> None:
    """Abort startup when the requested bind address leaks auth or scope.

    Binding to anything other than loopback exposes ASRbench to the local
    network. That is only safe when (a) the user explicitly asked for it
    with ``--allow-network`` and (b) an API key is configured so the new
    HTTP/WS surface is not unauthenticated.

    Fails loud with exit code 1 rather than silently binding, because a
    running tool is much harder to unwind than a refusal to start.
    """
    if host in LOOPBACK_HOSTS:
        return

    if not allow_network:
        typer.secho(
            (
                f"Refusing to bind to {host!r}: non-loopback hosts require "
                "--allow-network. Re-run with --allow-network (and set "
                "ASRBENCH_API_KEY) if you really want to expose this instance."
            ),
            err=True,
            fg=typer.colors.RED,
        )
        sys.exit(1)

    if not os.environ.get("ASRBENCH_API_KEY", "").strip():
        typer.secho(
            (
                "--allow-network requires the ASRBENCH_API_KEY environment "
                "variable. Set it to a long random string (at least 32 bytes "
                "of entropy) before starting the server."
            ),
            err=True,
            fg=typer.colors.RED,
        )
        sys.exit(1)


def _schedule_browser_open(host: str, port: int) -> None:
    """Open the UI in a background thread once the port starts accepting.

    Polls 127.0.0.1:<port> instead of a fixed sleep so we don't open the
    browser before FastAPI is actually ready; gives up after 15 s to avoid
    blocking shutdown if the server never comes up.
    """
    import socket
    import threading
    import time
    import webbrowser

    target = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    url = f"http://{target}:{port}/"

    def _wait_then_open() -> None:
        for _ in range(150):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    webbrowser.open(url)
                    return
            except OSError:
                time.sleep(0.1)
        logger.warning("Server did not start within 15 s; not opening the browser.")

    t = threading.Thread(target=_wait_then_open, daemon=True)
    t.start()
