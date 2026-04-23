"""``asrbench serve`` — start the FastAPI server."""

from __future__ import annotations

import logging

import typer

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
) -> None:
    """Start the ASRbench API server."""
    import uvicorn

    if open_browser:
        _schedule_browser_open(host, port)

    uvicorn.run(
        "asrbench.main:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


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
