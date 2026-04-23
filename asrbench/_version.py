"""Single source of truth for the ASRbench package version.

Hatch reads the version from this file via ``[tool.hatch.version]`` so that
``pyproject.toml`` and runtime code never drift. Update this string for each
release; everything else (``asrbench.__version__``, the FastAPI app title,
the CLI ``--version`` flag) picks it up automatically.
"""

from __future__ import annotations

__version__ = "0.1.0"
