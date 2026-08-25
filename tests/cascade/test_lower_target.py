"""Lowering pushes one target into every stage that can read it, and nowhere else.

Three docking engines each carrying their own receptor setting is three chances
to dock into three different preparations of the same protein and then report
the disagreement as a consensus.  So the receptor is named once, on the cascade,
and lowering writes it into the stages -- which means the interesting questions
are all about *reach*: which stages get written to, which of their fields get
written, and what happens to a plugin that has never heard of a protein.

The rule under test is the one ``_finalize_settings`` already established: write
a value only where the plugin's own config model declares the field.  Applied to
targets it means a docking engine gets exactly the subset it asked for, and a
property filter three tiers up is not handed a receptor it would reject.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from molcascade.cascade.lower import lower_cascade
from molcascade.cascade.models import BoxConfig, CascadeConfig, TargetConfig
from molcascade.plugins import (
    Cardinality,
    Determinism,
    PluginDescriptor,
    PluginKind,
    PluginRegistry,
    StageContext,
    StageRequest,
    StageResponse,
)

_RECEPTOR = "/targets/stk17b/receptor.pdb"
_LIGAND = "/targets/stk17b/reference.sdf"
_PREPARED = "/targets/stk17b/receptor.pdbqt"
_DIGEST = "b" * 64

_BOX = BoxConfig(
    center_x=1.0, center_y=2.0, center_z=3.0, size_x=20.0, size_y=21.0, size_z=22.0
)


class _BoxEngineConfig(BaseModel):
    """What Uni-Dock and GNINA declare: a receptor and six numbers."""

    receptor_path: str
    receptor_sha256: str | None = None
    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.0
    size_x: float = 0.0
    size_y: float = 0.0
    size_z: float = 0.0
    receptor_pdbqt_path: str | None = None


class _LigandEngineConfig(BaseModel):
    """What KarmaDock declares: a receptor and the ligand that locates the site.

    Deliberately without the box fields.  A cascade holding both engines must
    still write one target, and this is the half that proves the write is
    field-directed rather than wholesale.
    """

    receptor_path: str
    reference_ligand_path: str
    pocket_pdb_path: str | None = None


class _PropertyConfig(BaseModel):
    """A plugin that has never heard of a protein -- and forbids what it does not know."""

    model_config = {"extra": "forbid"}

    max_weight: float = 500.0


class _JoinConfig(BaseModel):
    """The decision join lowering inserts under a parallel tier."""

    model_config = {"extra": "forbid"}

    mode: str = "all"
    min_pass_count: int = 1


class _Plugin:
    def __init__(self, descriptor: PluginDescriptor, config_model: type[BaseModel]) -> None:
        self.descriptor = descriptor
        self.config_model = config_model

    def execute(self, request: StageRequest, context: StageContext) -> StageResponse:
        raise AssertionError("lowering must not execute a stage")


def _plugin(
    plugin_id: str,
    kind: PluginKind,
    *,
    config_model: type[BaseModel],
    inputs: tuple[str, ...] = ("parent/v1",),
    output_ports: dict[str, str] | None = None,
    cardinality: Cardinality = Cardinality.ONE_TO_ONE,
) -> _Plugin:
    ports = output_ports or {"primary": "parent/v1", "decisions": "decision/v1"}
    return _Plugin(
        PluginDescriptor(
            id=plugin_id,
            version="0.1.0",
            kind=kind,
            inputs=inputs,
            outputs=tuple(dict.fromkeys(ports.values())),
            output_ports=ports,
            cardinality=cardinality,
            determinism=Determinism.DETERMINISTIC,
        ),
        config_model,
    )


def _registry(*extra: _Plugin) -> PluginRegistry:
    source = _plugin(
        "source.delimited_smiles",
        PluginKind.SOURCE,
        config_model=_PropertyConfig,
        inputs=(),
        output_ports={"primary": "parent/v1"},
        cardinality=Cardinality.ONE_TO_MANY,
    )
    # Lowering inserts this itself under a parallel tier, so a registry without
    # it cannot express the two-engine cascade these tests are about.
    join = _plugin(
        "policy.native_decision_join",
        PluginKind.GATE,
        config_model=_JoinConfig,
        inputs=("parent/v1", "decision/v1"),
        cardinality=Cardinality.FILTER,
    )
    return PluginRegistry((source, join, *extra))


def _box_engine(plugin_id: str = "docking.unidock") -> _Plugin:
    return _plugin(
        plugin_id,
        PluginKind.DOCK,
        config_model=_BoxEngineConfig,
        output_ports={
            "primary": "parent/v1",
            "scores": "docking_score/v1",
            "decisions": "decision/v1",
        },
    )


def _ligand_engine(plugin_id: str = "docking.karmadock") -> _Plugin:
    return _plugin(
        plugin_id,
        PluginKind.DOCK,
        config_model=_LigandEngineConfig,
        output_ports={
            "primary": "parent/v1",
            "scores": "docking_score/v1",
            "decisions": "decision/v1",
        },
    )


def _property_filter(plugin_id: str = "rdkit.properties") -> _Plugin:
    return _plugin(
        plugin_id,
        PluginKind.GATE,
        config_model=_PropertyConfig,
        cardinality=Cardinality.FILTER,
    )


def _target(**overrides: Any) -> TargetConfig:
    fields: dict[str, Any] = {
        "name": "stk17b",
        "receptor_path": _RECEPTOR,
        "receptor_sha256": _DIGEST,
        "box": _BOX,
        "reference_ligand_path": _LIGAND,
        "receptor_pdbqt_path": _PREPARED,
    }
    fields.update(overrides)
    return TargetConfig(**fields)


def _cascade(*tiers: dict[str, Any], target: TargetConfig | None = None) -> CascadeConfig:
    payload: dict[str, Any] = {
        "name": "target-push",
        "ingest": {"id": "ingest", "backend": "source.delimited_smiles@0.1.0"},
        "tiers": list(tiers),
    }
    if target is not None:
        payload["target"] = target.model_dump(mode="python")
    return CascadeConfig.model_validate(payload)


def _tier(tier_id: str, *criteria: dict[str, Any], mode: str = "serial") -> dict[str, Any]:
    return {
        "id": tier_id,
        "title": tier_id.replace("_", " ").title(),
        "mode": mode,
        "criteria": list(criteria),
    }


def _config(pipeline: Any, stage_id: str) -> dict[str, Any]:
    stages = [stage for stage in pipeline.stages if stage.id == stage_id]
    assert stages, f"stage {stage_id!r} was not lowered"
    return dict(stages[0].config)


def _docking_cascade(target: TargetConfig | None) -> CascadeConfig:
    return _cascade(
        _tier("properties", {"id": "weight", "backend": "rdkit.properties@0.1.0"}),
        _tier(
            "dock",
            {"id": "unidock", "backend": "docking.unidock@0.1.0"},
            {"id": "karmadock", "backend": "docking.karmadock@0.1.0"},
            mode="all",
        ),
        target=target,
    )


def _lower(target: TargetConfig | None, *, run_target: TargetConfig | None = None) -> Any:
    return lower_cascade(
        _docking_cascade(target),
        registry=_registry(_box_engine(), _ligand_engine(), _property_filter()),
        allow_missing_library=True,
        target=run_target,
    ).pipeline


def test_each_engine_receives_the_fields_its_own_config_declares() -> None:
    """One target, two shapes.  Neither engine sees the other's vocabulary."""

    pipeline = _lower(_target())

    unidock = _config(pipeline, "unidock")
    assert unidock["receptor_path"] == _RECEPTOR
    assert unidock["receptor_sha256"] == _DIGEST
    assert unidock["receptor_pdbqt_path"] == _PREPARED
    assert unidock["center_x"] == 1.0
    assert unidock["size_z"] == 22.0
    assert "reference_ligand_path" not in unidock

    karmadock = _config(pipeline, "karmadock")
    assert karmadock["receptor_path"] == _RECEPTOR
    assert karmadock["reference_ligand_path"] == _LIGAND
    assert "center_x" not in karmadock
    assert "receptor_pdbqt_path" not in karmadock
    # Not declared, so not written -- a target that never had a pocket file
    # must not leave the field behind as a null the engine would have to read.
    assert "pocket_pdb_path" not in karmadock


def test_a_plugin_that_does_not_dock_is_left_alone() -> None:
    """It forbids extra keys, so a wholesale write would fail its own validation.

    Which is the mild version.  The one that matters is a plugin accepting extra
    keys and silently changing its cache key because a receptor was named three
    tiers below it.
    """

    pipeline = _lower(_target())
    assert _config(pipeline, "weight") == {}


def test_a_cascade_with_no_target_leaves_the_engines_as_written() -> None:
    """The stage is incomplete on purpose; the preflight is what says so.

    Filling in a placeholder receptor here would produce a pipeline that
    compiles and then docks into nothing.
    """

    pipeline = _lower(None)
    assert "receptor_path" not in _config(pipeline, "unidock")


def test_the_run_time_target_overrides_the_one_in_the_cascade() -> None:
    """A per-engine receptor is a disagreement, not a customisation.

    The cascade is the screening policy and is meant to be reused; the receptor
    is what changes between campaigns, so the one named at the run wins outright
    rather than being merged field by field.
    """

    stored = _target(
        receptor_path="/targets/old/receptor.pdb",
        receptor_sha256="a" * 64,
        receptor_pdbqt_path="/targets/old/receptor.pdbqt",
        reference_ligand_path="/targets/old/reference.sdf",
    )
    pipeline = _lower(stored, run_target=_target())
    unidock = _config(pipeline, "unidock")
    assert unidock["receptor_path"] == _RECEPTOR
    assert unidock["receptor_sha256"] == _DIGEST
    assert _config(pipeline, "karmadock")["reference_ligand_path"] == _LIGAND


def test_a_target_without_a_digest_writes_no_digest_rather_than_a_null() -> None:
    """Absent means "establish it from the bytes", and null would not say that.

    ``receptor_sha256`` also reaches the stage config, so writing an explicit
    null would change the cache key of every docking stage relative to a run
    that simply did not pin one.
    """

    pipeline = _lower(_target(receptor_sha256=None))
    assert "receptor_sha256" not in _config(pipeline, "unidock")


def test_a_target_reaching_a_finalize_step_follows_the_same_rule() -> None:
    """Finalize steps are lowered by a different function, on the same rule.

    Worth pinning: an exporter that wanted to name the receptor in its provenance
    would be wired here, and the two paths drifting apart is how one of them
    ends up writing a field nobody declared.
    """

    cascade = CascadeConfig.model_validate(
        {
            "name": "target-push",
            "ingest": {"id": "ingest", "backend": "source.delimited_smiles@0.1.0"},
            "tiers": [_tier("dock", {"id": "unidock", "backend": "docking.unidock@0.1.0"})],
            "finalize": {
                "steps": [{"id": "export", "backend": "docking.karmadock@0.1.0"}],
            },
            "target": _target().model_dump(mode="python"),
        }
    )
    pipeline = lower_cascade(
        cascade,
        registry=_registry(_box_engine(), _ligand_engine(), _property_filter()),
        allow_missing_library=True,
    ).pipeline
    export = _config(pipeline, "export")
    assert export["receptor_path"] == _RECEPTOR
    assert "center_x" not in export


@pytest.mark.parametrize("stage_id", ["unidock", "karmadock"])
def test_lowering_is_byte_stable_for_the_same_target(stage_id: str) -> None:
    """The stage config is hashed into the cache key, so its contents must settle.

    A dict whose key order or float formatting varied between two lowerings of
    the same cascade would re-dock the whole library on the second run.
    """

    first = _config(_lower(_target()), stage_id)
    second = _config(_lower(_target()), stage_id)
    assert list(first.items()) == list(second.items())
