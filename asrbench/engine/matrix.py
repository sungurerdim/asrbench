"""MatrixBuilder — cartesian product of parameter values for param_compare runs.

The user specifies which parameters to sweep as ``{name: [value, ...]}``. The
builder emits the cartesian product, merges in ``default_params`` for keys not
present in the matrix, and marks the FIRST combination as the baseline so the
UI can show "delta vs baseline" deltas without the user choosing which row is
the reference.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

__all__ = ["MatrixBuilder", "MatrixRun"]


@dataclass
class MatrixRun:
    """A single cell of the parameter matrix — fully materialized params + baseline flag."""

    params: dict[str, Any]
    is_baseline: bool = False
    mode: str = "param_compare"
    label: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class MatrixBuilder:
    """Build the cartesian product of parameter values with baseline-first ordering."""

    def build_matrix(
        self,
        matrix: dict[str, list[Any]],
        default_params: dict[str, Any],
        mode: str,
    ) -> list[MatrixRun]:
        """Return the cartesian product of ``matrix`` as MatrixRun objects.

        Raises:
            ValueError: if ``matrix`` is empty or any value list is empty.
        """
        if not matrix:
            raise ValueError(
                "Matrix must define at least one parameter. Example: {'beam_size': [1, 5]}."
            )

        names: list[str] = []
        value_lists: list[list[Any]] = []
        for name, values in matrix.items():
            if not values:
                raise ValueError(
                    f"Matrix parameter '{name}' has no values. "
                    "Provide at least one value or remove the key."
                )
            names.append(name)
            value_lists.append(list(values))

        runs: list[MatrixRun] = []
        for combo_index, combo in enumerate(itertools.product(*value_lists)):
            params: dict[str, Any] = {**default_params}
            for name, val in zip(names, combo, strict=False):
                params[name] = val
            runs.append(
                MatrixRun(
                    params=params,
                    is_baseline=(combo_index == 0),
                    mode=mode,
                )
            )

        return runs
