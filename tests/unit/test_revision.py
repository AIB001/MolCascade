from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from molcascade.config import PipelineConfig, canonical_json, parse_pipeline_config
from molcascade.errors import PipelineError
from molcascade.pipeline import (
    PipelineRevision,
    freeze_pipeline,
    revision_id_for,
    verify_revision,
)


def _pipeline(order: tuple[str, ...] = ("source", "gate")) -> PipelineConfig:
    stages = {
        "source": {
            "id": "source",
            "slot": "source",
            "plugin": "io.source@1.0.0",
            "config": {"path": "/data/in/../input.smi", "delimiter": "\\t"},
        },
        "gate": {
            "id": "gate",
            "slot": "hard_gate",
            "plugin": "chemistry.gate@1.0.0",
            "config": {"rules": ["valence", "elements"]},
        },
    }
    return PipelineConfig.model_validate(
        {
            "schema_version": 1,
            "name": "revision-test",
            "stages": tuple(stages[name] for name in order),
            "metadata": {"b": 2, "a": 1},
        }
    )


def test_semantically_equal_mapping_order_has_same_revision_id() -> None:
    yaml_a = """
schema_version: 1
name: stable
metadata: {z: 2, a: 1}
stages:
  - id: source
    slot: source
    plugin: io.source@1
    config: {path: /tmp/input.smi, delimiter: comma}
"""
    yaml_b = """
name: stable
stages:
  - plugin: io.source@1
    config: {delimiter: comma, path: /tmp/input.smi}
    slot: source
    id: source
metadata: {a: 1, z: 2}
schema_version: 1
"""

    revision_a = freeze_pipeline(parse_pipeline_config(yaml_a))
    revision_b = freeze_pipeline(parse_pipeline_config(yaml_b))

    assert revision_a.revision_id == revision_b.revision_id
    assert revision_a.canonical_config == revision_b.canonical_config


def test_stage_order_is_part_of_revision_identity() -> None:
    assert revision_id_for(_pipeline(("source", "gate"))) != revision_id_for(
        _pipeline(("gate", "source"))
    )


def test_runtime_path_spelling_is_part_of_revision_identity() -> None:
    first = _pipeline()
    second = PipelineConfig.model_validate_json(
        canonical_json(first).replace("/data/in/../input.smi", "/data/input.smi")
    )

    assert revision_id_for(first) != revision_id_for(second)
    assert freeze_pipeline(first).config.stages[0].config["path"] == (
        "/data/in/../input.smi"
    )


def test_freeze_detaches_from_mutable_input() -> None:
    raw = {
        "name": "snapshot",
        "stages": [
            {
                "id": "source",
                "slot": "source",
                "plugin": "io.source@1",
                "config": {"columns": ["smiles"]},
            }
        ],
    }
    revision = freeze_pipeline(raw)
    original_id = revision.revision_id

    raw["name"] = "changed"
    raw["stages"][0]["config"]["columns"].append("id")

    assert revision.config.name == "snapshot"
    assert revision.config.stages[0].config["columns"] == ["smiles"]
    assert revision.revision_id == original_id
    assert verify_revision(revision)


def test_revision_and_retained_config_are_frozen() -> None:
    revision = freeze_pipeline(_pipeline())

    with pytest.raises(ValidationError, match="frozen"):
        revision.revision_id = "0" * 64  # type: ignore[misc]
    with pytest.raises(TypeError, match="immutable"):
        revision.config.stages[0].config["path"] = "/other"


def test_revision_rejects_wrong_or_malformed_identifier() -> None:
    config = _pipeline()
    with pytest.raises(ValidationError, match="lowercase SHA-256"):
        PipelineRevision(revision_id="G" * 64, config=config)
    with pytest.raises(ValidationError, match="does not match canonical config"):
        PipelineRevision(revision_id="0" * 64, config=config)


def test_revision_json_round_trip_verifies_identity() -> None:
    revision = freeze_pipeline(_pipeline())
    encoded = revision.model_dump_json()
    restored = PipelineRevision.model_validate_json(encoded)

    assert restored == revision
    assert restored.canonical_json == canonical_json(restored.config)
    assert restored.pipeline is restored.config
    assert json.loads(encoded)["revision_id"] == revision.revision_id


def test_freeze_invalid_mapping_raises_structured_pipeline_error() -> None:
    with pytest.raises(PipelineError) as caught:
        freeze_pipeline(
            {
                "name": "invalid",
                "stages": [
                    {"id": "source", "slot": "source", "plugin": ""},
                ],
            }
        )

    assert caught.value.code == "PIPELINE_CONFIG_INVALID"
    assert caught.value.category.value == "CONFIG"
