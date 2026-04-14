"""Unit tests for ParameterSpace — validation, defaults, clamping, YAML loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from asrbench.engine.search.space import ParameterSpace, ParamSpec


class TestParamSpecValidation:
    def test_float_spec_valid(self) -> None:
        p = ParamSpec(name="lr", type="float", min=0.0, default=0.5, max=1.0)
        p.validate()

    def test_int_spec_valid(self) -> None:
        p = ParamSpec(name="beam", type="int", min=1, default=5, max=20)
        p.validate()

    def test_bool_spec_valid(self) -> None:
        p = ParamSpec(name="vad", type="bool", default=True)
        p.validate()

    def test_enum_spec_valid(self) -> None:
        p = ParamSpec(name="lang", type="enum", default="auto", values=("tr", "en", "auto"))
        p.validate()

    def test_float_rejects_default_out_of_range(self) -> None:
        p = ParamSpec(name="x", type="float", min=0.0, default=2.0, max=1.0)
        with pytest.raises(ValueError, match="default .* must satisfy"):
            p.validate()

    def test_float_rejects_min_gt_max(self) -> None:
        p = ParamSpec(name="x", type="float", min=1.0, default=0.5, max=0.0)
        with pytest.raises(ValueError, match="min .* must be <= max"):
            p.validate()

    def test_int_rejects_non_integer_bounds(self) -> None:
        p = ParamSpec(name="beam", type="int", min=1.5, default=5, max=20)
        with pytest.raises(ValueError, match="min and max must be integers"):
            p.validate()

    def test_int_rejects_float_default(self) -> None:
        p = ParamSpec(name="beam", type="int", min=1, default=5.0, max=20)
        with pytest.raises(ValueError, match="default must be int"):
            p.validate()

    def test_bool_rejects_non_bool_default(self) -> None:
        p = ParamSpec(name="vad", type="bool", default="true")
        with pytest.raises(ValueError, match="default must be True or False"):
            p.validate()

    def test_enum_rejects_missing_values(self) -> None:
        p = ParamSpec(name="lang", type="enum", default="tr")
        with pytest.raises(ValueError, match="must declare non-empty 'values'"):
            p.validate()

    def test_enum_rejects_default_not_in_values(self) -> None:
        p = ParamSpec(name="lang", type="enum", default="de", values=("tr", "en", "auto"))
        with pytest.raises(ValueError, match="not in values"):
            p.validate()

    def test_rejects_unknown_type(self) -> None:
        p = ParamSpec(name="x", type="quaternion", default=0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="unknown type"):
            p.validate()

    def test_rejects_empty_name(self) -> None:
        p = ParamSpec(name="", type="float", min=0.0, default=0.5, max=1.0)
        with pytest.raises(ValueError, match="non-empty string"):
            p.validate()

    def test_step_must_be_positive(self) -> None:
        p = ParamSpec(name="x", type="float", min=0.0, default=0.5, max=1.0, step=-0.1)
        with pytest.raises(ValueError, match="step must be a positive number"):
            p.validate()

    def test_int_default_rejects_bool(self) -> None:
        # Python quirk: bool is a subclass of int. Ensure we reject it explicitly.
        p = ParamSpec(name="beam", type="int", min=0, default=True, max=10)
        with pytest.raises(ValueError, match="default must be int"):
            p.validate()


class TestParamSpecEnumeration:
    def test_bool_enumerate(self) -> None:
        p = ParamSpec(name="x", type="bool", default=True)
        assert p.enumerate_values() == [False, True]

    def test_enum_enumerate(self) -> None:
        p = ParamSpec(name="lang", type="enum", default="tr", values=("tr", "en", "auto"))
        assert p.enumerate_values() == ["tr", "en", "auto"]

    def test_int_enumerate_small_range(self) -> None:
        p = ParamSpec(name="beam", type="int", min=1, default=3, max=5)
        assert p.enumerate_values() == [1, 2, 3, 4, 5]

    def test_int_enumerate_with_step(self) -> None:
        p = ParamSpec(name="beam", type="int", min=0, default=10, max=20, step=5)
        assert p.enumerate_values() == [0, 5, 10, 15, 20]

    def test_int_enumerate_large_range_sampled(self) -> None:
        p = ParamSpec(name="beam", type="int", min=1, default=50, max=100)
        values = p.enumerate_values(max_points=10)
        assert len(values) <= 11  # +1 for endpoint insertion
        assert values[0] == 1
        assert values[-1] == 100

    def test_float_continuous_enumerate_3point(self) -> None:
        p = ParamSpec(name="t", type="float", min=0.0, default=0.3, max=1.0)
        assert p.enumerate_values() == [0.0, 0.3, 1.0]

    def test_float_stepped_enumerate(self) -> None:
        p = ParamSpec(name="t", type="float", min=0.0, default=0.0, max=1.0, step=0.25)
        values = p.enumerate_values()
        assert values == [0.0, 0.25, 0.5, 0.75, 1.0]


class TestParamSpecClamp:
    def test_clamp_float_below(self) -> None:
        p = ParamSpec(name="x", type="float", min=0.0, default=0.5, max=1.0)
        assert p.clamp(-1.0) == 0.0

    def test_clamp_float_above(self) -> None:
        p = ParamSpec(name="x", type="float", min=0.0, default=0.5, max=1.0)
        assert p.clamp(2.5) == 1.0

    def test_clamp_float_inside(self) -> None:
        p = ParamSpec(name="x", type="float", min=0.0, default=0.5, max=1.0)
        assert p.clamp(0.7) == 0.7

    def test_clamp_int_rounds_and_clamps(self) -> None:
        p = ParamSpec(name="beam", type="int", min=1, default=5, max=20)
        assert p.clamp(3.7) == 4
        assert p.clamp(25.2) == 20
        assert p.clamp(-5) == 1

    def test_clamp_bool_coerces(self) -> None:
        p = ParamSpec(name="v", type="bool", default=False)
        assert p.clamp(1) is True
        assert p.clamp(0) is False

    def test_clamp_enum_invalid_returns_default(self) -> None:
        p = ParamSpec(name="lang", type="enum", default="auto", values=("tr", "en", "auto"))
        assert p.clamp("de") == "auto"
        assert p.clamp("tr") == "tr"


class TestParameterSpace:
    def test_defaults_returns_all(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="a", type="int", min=1, default=5, max=10),
                ParamSpec(name="b", type="float", min=0.0, default=0.3, max=1.0),
                ParamSpec(name="c", type="bool", default=True),
            )
        )
        assert space.defaults() == {"a": 5, "b": 0.3, "c": True}

    def test_names_preserves_order(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="z", type="int", min=0, default=0, max=1),
                ParamSpec(name="a", type="int", min=0, default=0, max=1),
                ParamSpec(name="m", type="int", min=0, default=0, max=1),
            )
        )
        assert space.names == ["z", "a", "m"]

    def test_get_unknown_raises(self) -> None:
        space = ParameterSpace(
            parameters=(ParamSpec(name="a", type="int", min=1, default=1, max=5),)
        )
        with pytest.raises(KeyError, match="Parameter 'nope' not found"):
            space.get("nope")

    def test_restrict_preserves_order(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="a", type="int", min=0, default=0, max=1),
                ParamSpec(name="b", type="int", min=0, default=0, max=1),
                ParamSpec(name="c", type="int", min=0, default=0, max=1),
            )
        )
        sub = space.restrict(["c", "a"])
        # Restrict preserves the original space order, not the caller's order
        assert sub.names == ["a", "c"]

    def test_with_config_merges_and_clamps(self) -> None:
        space = ParameterSpace(
            parameters=(
                ParamSpec(name="beam", type="int", min=1, default=5, max=20),
                ParamSpec(name="t", type="float", min=0.0, default=0.0, max=1.0),
                ParamSpec(name="vad", type="bool", default=True),
            )
        )
        base = space.defaults()
        new = space.with_config(base, {"beam": 999, "t": -0.5})
        assert new == {"beam": 20, "t": 0.0, "vad": True}

    def test_with_config_unknown_key_raises(self) -> None:
        space = ParameterSpace(
            parameters=(ParamSpec(name="a", type="int", min=0, default=0, max=1),)
        )
        with pytest.raises(KeyError, match="not found"):
            space.with_config({"a": 0}, {"unknown": 1})

    def test_empty_space_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one parameter"):
            ParameterSpace(parameters=())

    def test_duplicate_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="Duplicate parameter name"):
            ParameterSpace(
                parameters=(
                    ParamSpec(name="a", type="int", min=0, default=0, max=1),
                    ParamSpec(name="a", type="int", min=0, default=0, max=1),
                )
            )


class TestYamlLoading:
    def test_from_dict_basic(self) -> None:
        data = {
            "parameters": {
                "beam_size": {"type": "int", "min": 1, "default": 5, "max": 20},
                "temperature": {
                    "type": "float",
                    "min": 0.0,
                    "default": 0.0,
                    "max": 1.0,
                },
                "vad_filter": {"type": "bool", "default": True},
                "language": {
                    "type": "enum",
                    "values": ["tr", "en", "auto"],
                    "default": "auto",
                },
            }
        }
        space = ParameterSpace.from_dict(data)
        assert space.names == ["beam_size", "temperature", "vad_filter", "language"]
        assert space.defaults() == {
            "beam_size": 5,
            "temperature": 0.0,
            "vad_filter": True,
            "language": "auto",
        }

    def test_from_dict_missing_parameters_key(self) -> None:
        with pytest.raises(ValueError, match="must contain a 'parameters' key"):
            ParameterSpace.from_dict({})

    def test_from_dict_empty_parameters(self) -> None:
        with pytest.raises(ValueError, match="non-empty mapping"):
            ParameterSpace.from_dict({"parameters": {}})

    def test_from_dict_missing_type_or_default(self) -> None:
        with pytest.raises(ValueError, match="'type' and 'default' are required"):
            ParameterSpace.from_dict({"parameters": {"x": {"type": "int"}}})

    def test_from_yaml_roundtrip(self, tmp_path: Path) -> None:
        space_file = tmp_path / "space.yaml"
        content = {
            "parameters": {
                "beam": {"type": "int", "min": 1, "default": 5, "max": 10},
                "t": {"type": "float", "min": 0.0, "default": 0.5, "max": 1.0},
            }
        }
        space_file.write_text(yaml.safe_dump(content), encoding="utf-8")
        space = ParameterSpace.from_yaml(space_file)
        assert space.names == ["beam", "t"]
        assert space.defaults() == {"beam": 5, "t": 0.5}

    def test_from_yaml_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Space file not found"):
            ParameterSpace.from_yaml(tmp_path / "nope.yaml")
