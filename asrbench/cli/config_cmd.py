"""``asrbench config`` — inspect and edit ``~/.asrbench/config.toml``."""

from __future__ import annotations

import contextlib
from pathlib import Path

import typer

app = typer.Typer(help="Manage the ASRbench configuration file.")


def _config_path() -> Path:
    from asrbench.config import _default_config_path

    return _default_config_path()


@app.command("path")
def path_cmd() -> None:
    """Print the absolute path to the config file."""
    typer.echo(str(_config_path()))


@app.command("init")
def init_cmd(
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite an existing config file with defaults.",
    ),
) -> None:
    """Create the config file at the default location (if missing)."""
    from asrbench.config import _DEFAULTS_TOML

    path = _config_path()
    if path.exists() and not force:
        typer.echo(f"Config already exists at {path} (use --force to overwrite).")
        raise typer.Exit(code=0)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_DEFAULTS_TOML, encoding="utf-8")
    typer.echo(f"Wrote default config to {path}.")


@app.command("show")
def show_cmd() -> None:
    """Print the current config file contents."""
    path = _config_path()
    if not path.exists():
        typer.echo(
            f"No config file at {path}. Run `asrbench config init` to create one.",
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo(path.read_text(encoding="utf-8"))


@app.command("set")
def set_cmd(
    key: str = typer.Argument(
        ...,
        help="Dotted key, e.g. 'server.port' or 'limits.max_concurrent_runs'.",
    ),
    value: str = typer.Argument(..., help="New value (parsed as int/float/bool where possible)."),
) -> None:
    """Rewrite a single key in the on-disk config (preserves comments for other sections)."""
    path = _config_path()
    if not path.exists():
        typer.echo(
            f"No config file at {path}. Run `asrbench config init` first.",
            err=True,
        )
        raise typer.Exit(code=1)

    if "." not in key:
        typer.echo("Key must use dotted form, e.g. 'server.port'.", err=True)
        raise typer.Exit(code=2)

    section, sub_key = key.split(".", 1)
    new_value_literal = _format_toml_scalar(value)

    lines = path.read_text(encoding="utf-8").splitlines()
    if not _rewrite_existing_key(lines, section, sub_key, new_value_literal):
        _append_key_to_section(lines, section, sub_key, new_value_literal)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    from asrbench.config import get_config

    get_config.cache_clear()

    typer.echo(f"Set {key} = {new_value_literal} in {path}.")


def _rewrite_existing_key(
    lines: list[str], section: str, sub_key: str, new_value_literal: str
) -> bool:
    """Replace a live or commented ``sub_key`` line inside ``[section]``.

    Returns True when the key was found and rewritten.
    """
    in_section = False
    live_prefix = f"{sub_key} ="
    comment_prefix = f"# {sub_key} ="
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("["):
            in_section = stripped == f"[{section}]"
            continue
        if not in_section:
            continue
        if stripped.startswith(live_prefix) or stripped.startswith(comment_prefix):
            lines[i] = f"{sub_key} = {new_value_literal}"
            return True
    return False


def _append_key_to_section(
    lines: list[str], section: str, sub_key: str, new_value_literal: str
) -> None:
    """Insert ``sub_key = value`` into ``[section]``, creating the section if absent."""
    section_start = next(
        (i for i, line in enumerate(lines) if line.strip() == f"[{section}]"),
        None,
    )
    if section_start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"[{section}]")
        lines.append(f"{sub_key} = {new_value_literal}")
        return

    insert_at = section_start + 1
    while insert_at < len(lines) and not lines[insert_at].strip().startswith("["):
        insert_at += 1
    lines.insert(insert_at, f"{sub_key} = {new_value_literal}")


def _format_toml_scalar(raw: str) -> str:
    """Best-effort convert a CLI string into a typed TOML scalar."""
    lower = raw.lower()
    if lower in ("true", "false"):
        return lower
    with contextlib.suppress(ValueError):
        return str(int(raw))
    with contextlib.suppress(ValueError):
        return str(float(raw))
    escaped = raw.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
