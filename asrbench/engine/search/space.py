"""
Typed parameter space declaration for the IAMS optimizer.

A ParameterSpace is an ordered collection of ParamSpec entries describing
the search domain. It is the single source of truth for what the optimizer
is allowed to vary and within what bounds.

Supported parameter types:
    - float  : continuous bounded [min, max]
    - int    : discrete bounded [min, max] with optional integer step
    - bool   : {False, True}
    - enum   : finite unordered categorical values

Each ParamSpec carries a `default` which acts as the screening baseline in
Layer 1 and the fallback value in ablation refinement (Layer 5).

Space loading:
    space = ParameterSpace.from_yaml(Path("space.yaml"))
    space = ParameterSpace.from_dict({...})

YAML schema:
    parameters:
      beam_size:   { type: int,   min: 1,   default: 5,   max: 20 }
      temperature: { type: float, min: 0.0, default: 0.0, max: 1.0 }
      vad_filter:  { type: bool,  default: true }
      language:    { type: enum,  values: [tr, en, auto], default: auto }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

ParamType = Literal["float", "int", "bool", "enum"]


@dataclass(frozen=True)
class ParamSpec:
    """
    Single parameter declaration.

    Invariants (enforced by validate()):
    - name is a non-empty identifier
    - type ∈ {float, int, bool, enum}
    - float/int: min ≤ default ≤ max; step (if set) > 0 and fits grid
    - enum: values is non-empty; default ∈ values
    - bool: default ∈ {True, False}
    """

    name: str
    type: ParamType
    default: Any
    min: Any = None
    max: Any = None
    step: Any = None
    values: tuple[Any, ...] | None = None  # only for enum

    def validate(self) -> None:
        """Raise ValueError with a precise, actionable message on any invariant violation."""
        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"ParamSpec.name must be a non-empty string, got: {self.name!r}")
        if self.type == "float":
            self._validate_numeric(kind="float")
        elif self.type == "int":
            self._validate_numeric(kind="int")
        elif self.type == "bool":
            if not isinstance(self.default, bool):
                raise ValueError(
                    f"Param '{self.name}' (bool): default must be True or False, "
                    f"got {self.default!r}"
                )
        elif self.type == "enum":
            if not self.values or len(self.values) == 0:
                raise ValueError(
                    f"Param '{self.name}' (enum): must declare non-empty 'values' list"
                )
            if self.default not in self.values:
                raise ValueError(
                    f"Param '{self.name}' (enum): default {self.default!r} "
                    f"not in values {list(self.values)!r}"
                )
        else:
            raise ValueError(
                f"Param '{self.name}': unknown type {self.type!r}. "
                "Expected one of: float, int, bool, enum."
            )

    def _validate_numeric(self, kind: Literal["int", "float"]) -> None:
        if self.min is None or self.max is None:
            raise ValueError(f"Param '{self.name}' ({self.type}): 'min' and 'max' are required")
        if not isinstance(self.min, (int, float)) or not isinstance(self.max, (int, float)):
            raise ValueError(
                f"Param '{self.name}' ({self.type}): min/max must be numeric, "
                f"got min={self.min!r}, max={self.max!r}"
            )
        if kind == "int":
            if not (isinstance(self.min, int) and isinstance(self.max, int)):
                raise ValueError(
                    f"Param '{self.name}' (int): min and max must be integers, "
                    f"got min={self.min!r}, max={self.max!r}"
                )
            if not isinstance(self.default, int) or isinstance(self.default, bool):
                raise ValueError(
                    f"Param '{self.name}' (int): default must be int, got {self.default!r}"
                )
        else:  # float
            if not isinstance(self.default, (int, float)) or isinstance(self.default, bool):
                raise ValueError(
                    f"Param '{self.name}' (float): default must be numeric, got {self.default!r}"
                )
        if self.min > self.max:
            raise ValueError(f"Param '{self.name}': min ({self.min}) must be <= max ({self.max})")
        if not (self.min <= self.default <= self.max):
            raise ValueError(
                f"Param '{self.name}': default ({self.default}) must satisfy "
                f"min ({self.min}) <= default <= max ({self.max})"
            )
        if self.step is not None:
            if not isinstance(self.step, (int, float)) or self.step <= 0:
                raise ValueError(
                    f"Param '{self.name}': step must be a positive number, got {self.step!r}"
                )

    def enumerate_values(self, max_points: int = 64) -> list[Any]:
        """
        Return all discrete sample points for this parameter, capped at max_points.

        For continuous float without step, returns [min, default, max] — further
        refinement is handled by golden section search, not enumeration.

        For enum/bool, returns the full value list.
        For int with step, returns the arithmetic progression.
        For int without step, returns [min..max] if size ≤ max_points else samples.
        """
        if self.type == "bool":
            return [False, True]
        if self.type == "enum":
            assert self.values is not None
            return list(self.values)
        if self.type == "int":
            step = int(self.step) if self.step is not None else 1
            if step < 1:
                step = 1
            int_values: list[int] = list(range(int(self.min), int(self.max) + 1, step))
            if len(int_values) > max_points:
                # Sample evenly (keep endpoints + default)
                stride = max(1, len(int_values) // max_points)
                sampled = int_values[::stride]
                if int_values[-1] not in sampled:
                    sampled.append(int_values[-1])
                return sampled
            return int_values
        # float
        if self.step is not None:
            float_values: list[float] = []
            v = float(self.min)
            while v <= float(self.max) + 1e-12:
                float_values.append(round(v, 10))
                v += float(self.step)
            if len(float_values) > max_points:
                stride = max(1, len(float_values) // max_points)
                return float_values[::stride]
            return float_values
        # Continuous float without step: just return 3-point probe set.
        return [float(self.min), float(self.default), float(self.max)]

    def clamp(self, value: Any) -> Any:
        """
        Project an arbitrary value back into the parameter domain.

        Used by 1D search routines (e.g., golden section) that may propose
        values outside the allowed range due to floating-point drift.
        """
        if self.type == "float":
            return max(float(self.min), min(float(self.max), float(value)))
        if self.type == "int":
            v = int(round(value))
            return max(int(self.min), min(int(self.max), v))
        if self.type == "bool":
            return bool(value)
        if self.type == "enum":
            assert self.values is not None
            return value if value in self.values else self.default
        raise ValueError(f"Unknown type {self.type!r} for clamp")

    def is_continuous(self) -> bool:
        return self.type == "float" and self.step is None

    def is_discrete(self) -> bool:
        return not self.is_continuous()


@dataclass(frozen=True)
class ParameterSpace:
    """
    Ordered collection of parameter specs plus lookup and projection helpers.

    Immutable: any space transformation (e.g. restricting to sensitive params)
    returns a new ParameterSpace instance.
    """

    parameters: tuple[ParamSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.parameters) == 0:
            raise ValueError(
                "ParameterSpace requires at least one parameter. "
                "Declare parameters in the YAML space file."
            )
        names_seen: set[str] = set()
        for p in self.parameters:
            p.validate()
            if p.name in names_seen:
                raise ValueError(f"Duplicate parameter name '{p.name}' in ParameterSpace.")
            names_seen.add(p.name)
        # O(1) lookup cache for get()
        object.__setattr__(self, "_by_name", {p.name: p for p in self.parameters})

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.parameters]

    def get(self, name: str) -> ParamSpec:
        try:
            return self._by_name[name]  # type: ignore[attr-defined]
        except KeyError:
            raise KeyError(f"Parameter '{name}' not found in space. Available: {self.names}")

    def defaults(self) -> dict[str, Any]:
        """Return {name: default} for all parameters — the screening baseline."""
        return {p.name: p.default for p in self.parameters}

    def restrict(self, names: list[str]) -> ParameterSpace:
        """Return a new space containing only the listed parameters, in their original order."""
        name_set = set(names)
        return ParameterSpace(parameters=tuple(p for p in self.parameters if p.name in name_set))

    def with_config(self, base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
        """
        Merge overrides onto base, clamping each override through its ParamSpec.

        Used by search layers to construct new candidate configs while guaranteeing
        type and range safety.
        """
        result = dict(base)
        for key, value in overrides.items():
            spec = self.get(key)  # raises if unknown
            result[key] = spec.clamp(value)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParameterSpace:
        """
        Build from a plain dict, typically the result of yaml.safe_load().

        Expected shape:
            {"parameters": {"name": {"type": ..., "default": ..., ...}, ...}}
        """
        if "parameters" not in data:
            raise ValueError(
                "Space dict must contain a 'parameters' key with per-parameter entries."
            )
        params_dict = data["parameters"]
        if not isinstance(params_dict, dict) or not params_dict:
            raise ValueError("'parameters' must be a non-empty mapping of name → spec.")
        specs: list[ParamSpec] = []
        for name, entry in params_dict.items():
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Parameter '{name}' entry must be a dict, got {type(entry).__name__}."
                )
            if "type" not in entry or "default" not in entry:
                raise ValueError(f"Parameter '{name}': 'type' and 'default' are required fields.")
            values = entry.get("values")
            specs.append(
                ParamSpec(
                    name=name,
                    type=entry["type"],
                    default=entry["default"],
                    min=entry.get("min"),
                    max=entry.get("max"),
                    step=entry.get("step"),
                    values=tuple(values) if values is not None else None,
                )
            )
        return cls(parameters=tuple(specs))

    @classmethod
    def from_yaml(cls, path: Path) -> ParameterSpace:
        """Load a space declaration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Space file not found: {path}. "
                "Create a YAML file with a top-level 'parameters' key."
            )
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"Space file {path} must contain a top-level mapping, got {type(data).__name__}."
            )
        return cls.from_dict(data)
