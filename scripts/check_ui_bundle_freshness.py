"""Warn when the committed UI bundle looks stale relative to ui/ sources.

Lightweight heuristic used by the pre-commit hook — emits a warning but
does not fail the commit. The authoritative check is the ui-build job
in ``.github/workflows/ci.yml`` which runs the full Vite build and
diffs ``asrbench/static``.

Rules of thumb:
* If any ``ui/src/**`` file is newer than ``asrbench/static/index.html``,
  print a reminder to run ``cd ui && npm run build``.
* If ``asrbench/static/index.html`` does not exist at all, remind the
  user the wheel will ship without a UI.

Keep the script dependency-free so pre-commit does not have to install
anything to run it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_BUNDLE = _REPO / "asrbench" / "static" / "index.html"
_UI_SRC = _REPO / "ui" / "src"


def _latest_source_mtime() -> float:
    latest = 0.0
    if not _UI_SRC.is_dir():
        return latest
    for path in _UI_SRC.rglob("*"):
        if path.is_file():
            latest = max(latest, path.stat().st_mtime)
    return latest


def main() -> int:
    if not _BUNDLE.is_file():
        print(
            f"warning: {_BUNDLE.relative_to(_REPO)} is missing. "
            "Run 'cd ui && npm run build' before committing — the wheel "
            "ships without a UI otherwise.",
            file=sys.stderr,
        )
        return 0

    bundle_mtime = _BUNDLE.stat().st_mtime
    src_mtime = _latest_source_mtime()
    if src_mtime > bundle_mtime:
        print(
            "warning: ui/src has been modified since the committed bundle "
            "was produced. Run 'cd ui && npm run build' and stage the "
            "result alongside your source changes to avoid a CI failure.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
