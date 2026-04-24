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
    dev: bool = typer.Option(
        False,
        "--dev",
        help=(
            "Development mode. Enables --reload, assumes the Vite dev "
            "server is running at http://localhost:5173, and warns when "
            "the static bundle is missing so `npm run build` can be "
            "deferred while iterating on the UI."
        ),
    ),
) -> None:
    """Start the ASRbench API server."""
    import uvicorn

    _enforce_network_policy(host=host, allow_network=allow_network)
    _warn_on_missing_ui_bundle(dev=dev)

    effective_reload = reload or dev

    if open_browser:
        _schedule_browser_open(host, port, dev=dev)

    uvicorn.run(
        "asrbench.main:create_app",
        factory=True,
        host=host,
        port=port,
        reload=effective_reload,
    )


def _warn_on_missing_ui_bundle(*, dev: bool) -> None:
    """Emit a one-line hint when the static UI bundle is missing.

    In production installs the wheel ships ``asrbench/static/index.html``
    — if it is absent the user either edited the source tree without
    rebuilding or the CI pipeline dropped the artifact. In dev mode
    the missing bundle is expected (the Vite dev server serves the UI)
    so we just remind the user which port to visit.
    """
    from asrbench.main import _ui_static_dir

    static_dir = _ui_static_dir()
    has_bundle = (static_dir / "index.html").is_file()

    if not has_bundle and not dev:
        typer.secho(
            (
                f"UI bundle not found at {static_dir} — the server will "
                "only serve the JSON API. Run `cd ui && npm run build` "
                "or start with `--dev` to use the Vite dev server."
            ),
            err=True,
            fg=typer.colors.YELLOW,
        )
    elif dev:
        typer.secho(
            "Dev mode — UI is expected at http://localhost:5173 "
            "(run `cd ui && npm run dev` in another terminal).",
            err=True,
            fg=typer.colors.CYAN,
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


def _schedule_browser_open(host: str, port: int, *, dev: bool = False) -> None:
    """Open the UI in a background thread once the port starts accepting.

    Polls 127.0.0.1:<port> instead of a fixed sleep so we don't open the
    browser before FastAPI is actually ready; gives up after 15 s to avoid
    blocking shutdown if the server never comes up.

    In dev mode the dev UX is reversed — the Vite dev server runs at
    ``:5173`` and hot-reloads; the API is an auxiliary process. Point
    the browser at the dev server so saving a .svelte file refreshes
    instantly.
    """
    import socket
    import threading
    import time
    import webbrowser

    target = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    ui_port = 5173 if dev else port
    url = f"http://{target}:{ui_port}/"

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
