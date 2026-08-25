from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from molcascade.config import (
    PipelineConfig,
    StageConfig,
    canonical_json,
    canonical_sha256,
    load_pipeline_config,
    load_yaml,
    parse_pipeline_config,
)
from molcascade.errors import ConfigError, DuplicateKeyError


def _pipeline_yaml(*, path: str = "/runtime/input/../candidates.smi") -> str:
    return f"""
schema_version: 1
name: example
metadata:
  owner: screening
  nested:
    weights: [1.0, 2.0]
stages:
  - id: ingest
    slot: source
    plugin: io.delimited@1.0.0
    config:
      path: {path}
  - id: gate
    slot: hard_gate
    plugin: chemistry.gate@2.1.0
    config:
      enabled_rules: [valence, elements]
"""


def test_parse_pipeline_preserves_stage_order_and_path_spelling() -> None:
    config = parse_pipeline_config(_pipeline_yaml())

    assert isinstance(config.stages, tuple)
    assert [stage.id for stage in config.stages] == ["ingest", "gate"]
    assert config.stages[0].config["path"] == "/runtime/input/../candidates.smi"


def test_pipeline_and_nested_json_are_immutable() -> None:
    config = parse_pipeline_config(_pipeline_yaml())

    with pytest.raises(ValidationError, match="frozen"):
        config.name = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError, match="immutable"):
        config.metadata["owner"] = "changed"
    with pytest.raises(TypeError, match="immutable"):
        config.metadata["nested"]["weights"].append(3.0)  # type: ignore[union-attr]
    with pytest.raises(TypeError, match="immutable"):
        config.stages[1].config["enabled_rules"].pop()  # type: ignore[union-attr]

    # Frozen JSON snapshots can safely be shared by a Pydantic deep copy.
    copied = config.model_copy(deep=True)
    assert copied == config
    assert copied.metadata is config.metadata


@pytest.mark.parametrize(
    "text",
    [
        "name: one\nname: two\nstages: []\n",
        "name: one\nstages:\n  - id: x\n    id: y\n    slot: source\n    plugin: p\n",
        "name: one\nmetadata:\n  nested:\n    value: 1\n    value: 2\nstages: []\n",
    ],
)
def test_yaml_duplicate_keys_are_rejected_at_every_depth(text: str) -> None:
    with pytest.raises(DuplicateKeyError) as caught:
        load_yaml(text)

    assert caught.value.code == "CONFIG_DUPLICATE_KEY"
    assert caught.value.context["line"] >= 2


def test_yaml_12_boolean_words_are_not_silently_changed() -> None:
    parsed = load_yaml("value_on: on\nvalue_yes: yes\nreal_bool: true\n")

    assert parsed == {"value_on": "on", "value_yes": "yes", "real_bool": True}


@pytest.mark.parametrize("token", [".nan", ".NaN", ".inf", "-.Inf"])
def test_non_finite_yaml_is_rejected_recursively(token: str) -> None:
    text = f"""
schema_version: 1
name: non-finite
metadata:
  nested:
    score: {token}
stages:
  - id: source
    slot: source
    plugin: io.source@1
"""

    with pytest.raises(ConfigError) as caught:
        parse_pipeline_config(text)

    assert caught.value.code == "CONFIG_NON_FINITE_NUMBER"


def test_non_finite_metadata_is_rejected_by_direct_model_validation() -> None:
    with pytest.raises(ValidationError, match="finite"):
        PipelineConfig.model_validate(
            {
                "name": "bad",
                "stages": (
                    {"id": "source", "slot": "source", "plugin": "io.source@1"},
                ),
                "metadata": {"nested": {"score": float("inf")}},
            }
        )


def test_strict_types_and_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(
            {
                "schema_version": "1",
                "name": "strict",
                "stages": (
                    {"id": "source", "slot": "source", "plugin": "io.source@1"},
                ),
            }
        )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        StageConfig.model_validate(
            {
                "id": "source",
                "slot": "source",
                "plugin": "io.source@1",
                "typo": True,
            }
        )


def test_only_schema_version_one_is_accepted() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(
            {
                "schema_version": 2,
                "name": "future",
                "stages": (
                    {"id": "source", "slot": "source", "plugin": "io.source@1"},
                ),
            }
        )


def test_stage_ids_are_unique_and_one_stage_must_be_enabled() -> None:
    with pytest.raises(ValidationError, match="stage ids must be unique"):
        PipelineConfig.model_validate(
            {
                "name": "duplicates",
                "stages": (
                    {"id": "same", "slot": "source", "plugin": "a"},
                    {"id": "same", "slot": "export", "plugin": "b"},
                ),
            }
        )

    with pytest.raises(ValidationError, match="at least one enabled stage"):
        PipelineConfig.model_validate(
            {
                "name": "disabled",
                "stages": (
                    {
                        "id": "source",
                        "slot": "source",
                        "plugin": "a",
                        "enabled": False,
                    },
                ),
            }
        )


@pytest.mark.parametrize("plugin", ["", "   ", "plugin name", "plugin\nname"])
def test_plugin_identifier_must_be_nonempty_without_whitespace(plugin: str) -> None:
    with pytest.raises(ValidationError):
        StageConfig(id="source", slot="source", plugin=plugin)


def test_canonical_json_sorts_mappings_but_preserves_arrays() -> None:
    left = {"z": 3, "a": {"right": 2, "left": 1}, "stages": ["first", "second"]}
    right = {"stages": ["first", "second"], "a": {"left": 1, "right": 2}, "z": 3}

    assert canonical_json(left) == canonical_json(right)
    assert canonical_sha256(left) == canonical_sha256(right)
    assert '"stages":["first","second"]' in canonical_json(left)
    assert canonical_sha256({**left, "stages": ["second", "first"]}) != canonical_sha256(left)


def test_canonical_json_rejects_non_finite_and_non_string_keys() -> None:
    with pytest.raises(ConfigError, match="non-finite"):
        canonical_json({"nested": [float("nan")]})
    with pytest.raises(ConfigError, match="must be a string"):
        canonical_json({1: "ambiguous"})


def test_canonical_json_rejects_recursive_values() -> None:
    recursive: list[object] = []
    recursive.append(recursive)

    with pytest.raises(ConfigError) as caught:
        canonical_json({"recursive": recursive})

    assert caught.value.code == "CONFIG_RECURSIVE_VALUE"


def test_canonical_json_does_not_normalize_absolute_paths() -> None:
    lexical_path = Path("/runtime/input/../candidates.smi")

    assert canonical_json({"path": lexical_path}) == (
        '{"path":"/runtime/input/../candidates.smi"}'
    )
    assert canonical_sha256({"path": lexical_path}) != canonical_sha256(
        {"path": Path("/runtime/candidates.smi")}
    )


def test_file_loader_handles_utf8_bom_without_touching_runtime_path(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("\ufeff" + _pipeline_yaml(), encoding="utf-8")

    config = load_pipeline_config(config_path)

    assert config.stages[0].config["path"] == "/runtime/input/../candidates.smi"


def test_yaml_only_values_and_non_string_keys_are_rejected() -> None:
    with pytest.raises(ConfigError) as date_error:
        load_yaml("created: 2026-08-22\n")
    assert date_error.value.code == "CONFIG_NOT_JSON"

    with pytest.raises(ConfigError) as key_error:
        load_yaml("1: value\n")
    assert key_error.value.code == "CONFIG_NON_STRING_KEY"


def test_explicit_input_bindings_only_reference_earlier_enabled_stages() -> None:
    config = PipelineConfig.model_validate(
        {
            "schema_version": 1,
            "name": "retained side tables",
            "stages": [
                {"id": "source", "slot": "source", "plugin": "source.test@1.0.0"},
                {
                    "id": "select",
                    "slot": "select",
                    "plugin": "select.test@1.0.0",
                    "inputs": [
                        {
                            "request_port": "molecules",
                            "stage": "source",
                            "port": "primary",
                        }
                    ],
                },
            ],
        }
    )
    assert config.stages[1].inputs[0].stage == "source"
    with pytest.raises(ValidationError, match="frozen"):
        config.stages[1].inputs[0].port = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "binding",
    [
        {"request_port": "molecules", "stage": "missing", "port": "primary"},
        {"request_port": "molecules", "stage": "select", "port": "primary"},
    ],
)
def test_explicit_input_binding_rejects_unknown_or_future_stage(
    binding: dict[str, str],
) -> None:
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(
            {
                "schema_version": 1,
                "name": "bad retained context",
                "stages": [
                    {"id": "source", "slot": "source", "plugin": "source.test@1.0.0"},
                    {
                        "id": "select",
                        "slot": "select",
                        "plugin": "select.test@1.0.0",
                        "inputs": [binding],
                    },
                ],
            }
        )
