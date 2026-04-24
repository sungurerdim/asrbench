"""Regenerate docs/API.md from the live FastAPI app.

Reads ``app.openapi()`` and emits a markdown file grouped by path.
The output is deterministic so CI can diff it against the committed
copy and fail when endpoints drift without a docs update.

Usage (from repo root)::

    python scripts/gen_api_docs.py              # writes docs/API.md
    python scripts/gen_api_docs.py --check      # exit 1 if the file is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_TARGET = _REPO / "docs" / "API.md"


def _load_openapi() -> dict:
    """Import the app and grab the OpenAPI dict.

    Creating the app starts a full lifespan cycle, so wrap it in a
    TestClient to keep DB init on the happy path.
    """
    from fastapi.testclient import TestClient

    from asrbench.main import create_app

    app = create_app()
    with TestClient(app):
        return app.openapi()


def _render(schema: dict) -> str:
    title = schema.get("info", {}).get("title", "ASRbench")
    version = schema.get("info", {}).get("version", "")
    lines: list[str] = [
        f"# {title} API reference",
        "",
        f"Auto-generated from the live FastAPI OpenAPI schema (version `{version}`).",
        "Regenerate with `python scripts/gen_api_docs.py`.",
        "",
    ]

    paths = schema.get("paths", {}) or {}
    groups: dict[str, list[str]] = {}
    for route, methods in sorted(paths.items()):
        group = _group_for(route)
        for method, op in sorted(methods.items()):
            if method.startswith("x-"):
                continue
            groups.setdefault(group, []).append(_render_operation(route, method.upper(), op))

    for group in sorted(groups):
        lines.append(f"## {group}")
        lines.append("")
        lines.extend(groups[group])
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _group_for(route: str) -> str:
    parts = [p for p in route.split("/") if p and not p.startswith("{")]
    return parts[0] if parts else "root"


def _render_operation(route: str, method: str, op: dict) -> str:
    summary = op.get("summary") or op.get("operationId", "")
    description = (op.get("description") or "").strip()
    tags = ", ".join(op.get("tags", []))
    status_codes = sorted((op.get("responses") or {}).keys())

    body = [
        f"### `{method} {route}`",
        "",
    ]
    if summary:
        body.append(f"**{summary}**")
        body.append("")
    if tags:
        body.append(f"*Tags:* {tags}")
        body.append("")
    if description:
        body.append(description)
        body.append("")
    if status_codes:
        body.append(f"*Responses:* {', '.join(status_codes)}")
        body.append("")
    return "\n".join(body)


def _write(content: str) -> None:
    _TARGET.parent.mkdir(parents=True, exist_ok=True)
    _TARGET.write_text(content, encoding="utf-8")


def _check(content: str) -> int:
    existing = _TARGET.read_text(encoding="utf-8") if _TARGET.exists() else ""
    if existing.replace("\r\n", "\n") == content.replace("\r\n", "\n"):
        return 0
    print(
        f"::error::{_TARGET.relative_to(_REPO)} is stale. "
        "Run `python scripts/gen_api_docs.py` and commit the diff.",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when docs/API.md does not match the current schema.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Dump the raw OpenAPI schema instead of the markdown rendering.",
    )
    args = parser.parse_args()

    schema = _load_openapi()
    if args.json:
        sys.stdout.write(json.dumps(schema, indent=2))
        return 0

    rendered = _render(schema)
    if args.check:
        return _check(rendered)
    _write(rendered)
    print(f"Wrote {_TARGET.relative_to(_REPO)} ({len(rendered.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
