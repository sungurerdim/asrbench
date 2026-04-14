"""Backend discovery via entry_points."""

from __future__ import annotations

import importlib.metadata
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from asrbench.backends.base import BaseBackend

logger = logging.getLogger(__name__)


def load_backends() -> dict[str, type[BaseBackend]]:
    """Discover installed ASR backends registered under ``asrbench.backends``.

    Returns a mapping of backend name to class. Backends that fail to import
    (missing optional dependencies) are silently skipped.
    """
    eps = importlib.metadata.entry_points(group="asrbench.backends")
    backends: dict[str, type[BaseBackend]] = {}
    for ep in eps:
        try:
            backends[ep.name] = ep.load()
        except Exception as exc:
            logger.debug("Skipping backend %s: %s", ep.name, exc)
    return backends
