"""``asrbench serve`` — start the FastAPI server."""

from __future__ import annotations

import typer


def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind address."),
    port: int = typer.Option(8765, "--port", "-p", help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development."),
) -> None:
    """Start the ASRbench API server."""
    import uvicorn

    uvicorn.run(
        "asrbench.main:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )
