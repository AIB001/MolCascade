from __future__ import annotations

from pathlib import Path

import pytest

from molcascade.config import PipelineConfig
from molcascade.config.models import StrictFrozenModel
from molcascade.errors import PipelineError, PluginError
from molcascade.pipeline.compiler import (
    DEFAULT_SLOT_KINDS,
    CompiledPipeline,
    PipelineCompiler,
    compile_pipeline,
)
from molcascade.plugins import (
    Cardinality,
    Determinism,
    PendingOutput,
    PluginDescriptor,
    PluginKind,
    PluginRegistry,
    StageContext,
    StageRequest,
    StageResponse,
)


class _SourceConfig(StrictFrozenModel):
    path: str
    batch_size: int = 1_024


class _Plugin:
    def __init__(
        self,
        descriptor: PluginDescriptor,
        *,
        config_model: type[StrictFrozenModel] | None = None,
    ) -> None:
        self.descriptor = descriptor
        if config_model is not None:
            self.config_model = config_model

    def execute(self, request: StageRequest, context: StageContext) -> StageResponse:
        del request
        path = context.staging_root / "unused.parquet"
        return StageResponse(
            outputs={
                "primary": PendingOutput(
                    contract_id=self.descriptor.outputs[0],
                    file_paths=(path.name,),
                )
            }
        )


def _descriptor(
    plugin_id: str,
    kind: PluginKind,
    *,
    inputs: tuple[str, ...] = (),
    outputs: tuple[str, ...],
    output_ports: dict[str, str] | None = None,
    version: str = "1.0.0",
) -> PluginDescriptor:
    return PluginDescriptor(
        id=plugin_id,
        version=version,
        kind=kind,
        inputs=inputs,
        outputs=outputs,
        output_ports=(
            ({"primary": outputs[0]} if len(outputs) == 1 else {
                "primary": outputs[0],
                **{
                    f"side_{index}": contract_id
                    for index, contract_id in enumerate(outputs[1:], start=1)
                },
            })
            if output_ports is None
            else output_ports
        ),
        cardinality=(
            Cardinality.ONE_TO_MANY
            if kind is PluginKind.SOURCE
            else Cardinality.ONE_TO_ONE
        ),
        determinism=Determinism.DETERMINISTIC,
    )


def _source(*, outputs: tuple[str, ...] = ("raw_molecule/v1",)) -> _Plugin:
    return _Plugin(
        _descriptor("io.source", PluginKind.SOURCE, outputs=outputs),
        config_model=_SourceConfig,
    )


def _standardizer(
    *,
    inputs: tuple[str, ...] = ("raw_molecule/v1",),
    outputs: tuple[str, ...] = (
        "parent/v1",
        "parent_source_map/v1",
        "decision/v1",
    ),
) -> _Plugin:
    return _Plugin(
        _descriptor(
            "chemistry.standardize",
            PluginKind.STANDARDIZER,
            inputs=inputs,
            outputs=outputs,
        )
    )


def _gate(
    *,
    inputs: tuple[str, ...] = ("parent/v1",),
    outputs: tuple[str, ...] = ("parent/v1", "decision/v1"),
) -> _Plugin:
    return _Plugin(
        _descriptor(
            "chemistry.gate",
            PluginKind.GATE,
            inputs=inputs,
            outputs=outputs,
        )
    )


def _predictor(plugin_id: str) -> _Plugin:
    return _Plugin(
        _descriptor(
            plugin_id,
            PluginKind.PREDICTOR,
            inputs=("parent/v1",),
            outputs=("parent/v1", "decision/v1"),
            output_ports={"primary": "parent/v1", "decisions": "decision/v1"},
        )
    )


def _tier_neutral_policy() -> _Plugin:
    return _Plugin(
        PluginDescriptor(
            id="policy.join",
            version="1.0.0",
            kind=PluginKind.GATE,
            inputs=("parent/v1", "decision/v1"),
            outputs=("parent/v1", "decision/v1"),
            output_ports={"primary": "parent/v1", "decisions": "decision/v1"},
            cardinality=Cardinality.FILTER,
            determinism=Determinism.DETERMINISTIC,
            tier_neutral_policy=True,
        )
    )


def _registry(*plugins: _Plugin) -> PluginRegistry:
    return PluginRegistry(plugins)


def _pipeline(*stages: dict[str, object]) -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "name": "compiler-test",
            "stages": stages,
        }
    )


def test_compiler_keeps_enabled_order_and_emits_primary_linear_contracts() -> None:
    registry = _registry(_source(), _standardizer(), _gate())
    pipeline = _pipeline(
        {
            "id": "ingest",
            "slot": "source",
            "plugin": "io.source@1.0.0",
            "config": {"path": "/data/input.smi"},
        },
        {
            "id": "disabled",
            "slot": "not-a-real-slot",
            "plugin": "not.registered",
            "enabled": False,
        },
        {
            "id": "normalize",
            "slot": "standardize",
            "plugin": "chemistry.standardize@1.0.0",
        },
        {
            "id": "filter",
            "slot": "hard_gate",
            "plugin": "chemistry.gate@1.0.0",
        },
    )

    compiled = compile_pipeline(pipeline, registry)

    assert isinstance(compiled, CompiledPipeline)
    assert compiled.revision.config == pipeline
    assert [stage.stage_id for stage in compiled.stages] == [
        "ingest",
        "normalize",
        "filter",
    ]
    source, standardizer, gate = compiled.stages
    assert source.plugin_key == "io.source@1.0.0"
    assert source.descriptor is registry.get(source.plugin_key).descriptor
    assert source.input_contract is None
    assert source.input_port is None
    assert source.output_contract == "raw_molecule/v1"
    assert source.output_port == "primary"
    assert source.config == {"path": "/data/input.smi", "batch_size": 1_024}
    assert standardizer.input_contract == "raw_molecule/v1"
    assert standardizer.output_contract == "parent/v1"
    assert standardizer.input_port == "primary"
    assert gate.input_contract == "parent/v1"
    assert gate.output_contract == "parent/v1"
    assert compiled.revision_id == compiled.revision.revision_id


def test_compiler_preserves_an_existing_revision() -> None:
    from molcascade.pipeline import freeze_pipeline

    revision = freeze_pipeline(
        _pipeline(
            {
                "id": "ingest",
                "slot": "source",
                "plugin": "io.source@1.0.0",
                "config": {"path": "/data/input.smi"},
            }
        )
    )

    compiled = PipelineCompiler(_registry(_source())).compile(revision)

    assert compiled.revision is revision


def test_default_slots_are_explicit_plugin_kind_mappings() -> None:
    assert DEFAULT_SLOT_KINDS["source"] is PluginKind.SOURCE
    assert DEFAULT_SLOT_KINDS["standardize"] is PluginKind.STANDARDIZER
    assert DEFAULT_SLOT_KINDS["hard_gate"] is PluginKind.GATE
    assert DEFAULT_SLOT_KINDS["synthesis"] is PluginKind.SYNTHESIS
    assert DEFAULT_SLOT_KINDS["predict"] is PluginKind.PREDICTOR
    assert DEFAULT_SLOT_KINDS["scaffold"] is PluginKind.SCAFFOLDER
    with pytest.raises(TypeError):
        DEFAULT_SLOT_KINDS["arbitrary"] = PluginKind.GATE  # type: ignore[index]


def test_explicit_tier_neutral_policy_compiles_inside_prediction_tier() -> None:
    registry = _registry(
        _source(),
        _standardizer(),
        _predictor("prediction.first"),
        _predictor("prediction.second"),
        _tier_neutral_policy(),
    )
    pipeline = _pipeline(
        {
            "id": "ingest",
            "slot": "source",
            "plugin": "io.source@1.0.0",
            "config": {"path": "/data/input.smi"},
        },
        {
            "id": "normalize",
            "slot": "standardize",
            "plugin": "chemistry.standardize@1.0.0",
            "inputs": [
                {"request_port": "primary", "stage": "ingest", "port": "primary"}
            ],
        },
        {
            "id": "first",
            "slot": "predict",
            "plugin": "prediction.first@1.0.0",
            "inputs": [
                {"request_port": "primary", "stage": "normalize", "port": "primary"}
            ],
        },
        {
            "id": "second",
            "slot": "predict",
            "plugin": "prediction.second@1.0.0",
            "inputs": [
                {"request_port": "primary", "stage": "normalize", "port": "primary"}
            ],
        },
        {
            "id": "all_predictions",
            "slot": "predict",
            "plugin": "policy.join@1.0.0",
            "inputs": [
                {"request_port": "parents", "stage": "normalize", "port": "primary"},
                {"request_port": "decision_first", "stage": "first", "port": "decisions"},
                {
                    "request_port": "decision_second",
                    "stage": "second",
                    "port": "decisions",
                },
            ],
        },
    )

    compiled = compile_pipeline(pipeline, registry)

    policy = compiled.stages[-1]
    assert policy.stage_id == "all_predictions"
    assert policy.descriptor.tier_neutral_policy is True
    assert len(policy.input_bindings) == 3


def test_plugin_reference_must_pin_an_exact_version_even_when_registry_is_unambiguous() -> None:
    pipeline = _pipeline(
        {
            "id": "ingest",
            "slot": "source",
            "plugin": "io.source",
            "config": {"path": "/data/input.smi"},
        }
    )

    with pytest.raises(PipelineError) as caught:
        compile_pipeline(pipeline, _registry(_source()))

    assert caught.value.code == "PIPELINE_PLUGIN_VERSION_REQUIRED"
    assert caught.value.context == {"stage_id": "ingest", "plugin": "io.source"}


@pytest.mark.parametrize(
    ("slot", "expected_code"),
    [
        ("unknown_slot", "PIPELINE_SLOT_UNKNOWN"),
        ("hard_gate", "PIPELINE_SLOT_KIND_MISMATCH"),
    ],
)
def test_slot_must_be_known_and_match_plugin_kind(slot: str, expected_code: str) -> None:
    pipeline = _pipeline(
        {
            "id": "ingest",
            "slot": slot,
            "plugin": "io.source@1.0.0",
            "config": {"path": "/data/input.smi"},
        }
    )

    with pytest.raises(PipelineError) as caught:
        compile_pipeline(pipeline, _registry(_source()))

    assert caught.value.code == expected_code
    assert caught.value.context["stage_id"] == "ingest"


def test_first_enabled_stage_must_be_a_source() -> None:
    pipeline = _pipeline(
        {
            "id": "filter",
            "slot": "hard_gate",
            "plugin": "chemistry.gate@1.0.0",
        }
    )

    with pytest.raises(PipelineError) as caught:
        compile_pipeline(pipeline, _registry(_gate()))

    assert caught.value.code == "PIPELINE_SOURCE_REQUIRED"


def test_incompatible_adjacent_contracts_are_rejected() -> None:
    pipeline = _pipeline(
        {
            "id": "ingest",
            "slot": "source",
            "plugin": "io.source@1.0.0",
            "config": {"path": "/data/input.smi"},
        },
        {
            "id": "filter",
            "slot": "hard_gate",
            "plugin": "chemistry.gate@1.0.0",
        },
    )

    with pytest.raises(PluginError) as caught:
        compile_pipeline(pipeline, _registry(_source(), _gate()))

    assert caught.value.code == "PLUGIN_CONTRACT_INCOMPATIBLE"


def test_named_primary_port_disambiguates_multiple_output_contracts() -> None:
    source = _source(outputs=("raw_molecule/v1", "parent/v1"))
    consumer = _gate(inputs=("raw_molecule/v1", "parent/v1"))
    pipeline = _pipeline(
        {
            "id": "ingest",
            "slot": "source",
            "plugin": "io.source@1.0.0",
            "config": {"path": "/data/input.smi"},
        },
        {
            "id": "filter",
            "slot": "hard_gate",
            "plugin": "chemistry.gate@1.0.0",
        },
    )

    compiled = compile_pipeline(pipeline, _registry(source, consumer))

    binding = compiled.stages[-1].input_bindings[0]
    assert binding.source_key == ("ingest", "primary")
    assert binding.contract_id == "raw_molecule/v1"


def test_explicit_binding_retains_a_named_side_output() -> None:
    source = _source(outputs=("raw_molecule/v1", "parent/v1"))
    consumer = _gate(inputs=("parent/v1",))
    pipeline = _pipeline(
        {
            "id": "ingest",
            "slot": "source",
            "plugin": "io.source@1.0.0",
            "config": {"path": "/data/input.smi"},
        },
        {
            "id": "filter",
            "slot": "hard_gate",
            "plugin": "chemistry.gate@1.0.0",
            "inputs": [
                {
                    "request_port": "parents",
                    "stage": "ingest",
                    "port": "side_1",
                }
            ],
        },
    )

    compiled = compile_pipeline(pipeline, _registry(source, consumer))

    binding = compiled.stages[-1].input_bindings[0]
    assert binding.request_port == "parents"
    assert binding.source_key == ("ingest", "side_1")
    assert binding.contract_id == "parent/v1"


def test_retained_context_keys_cannot_collide_on_dotted_names() -> None:
    source = _Plugin(
        _descriptor(
            "io.source",
            PluginKind.SOURCE,
            outputs=("raw_molecule/v1", "decision/v1"),
            output_ports={"primary": "raw_molecule/v1", "b.c": "decision/v1"},
        ),
        config_model=_SourceConfig,
    )
    gate = _Plugin(
        _descriptor(
            "chemistry.gate",
            PluginKind.GATE,
            inputs=("raw_molecule/v1",),
            outputs=("raw_molecule/v1", "decision/v1"),
            output_ports={"primary": "raw_molecule/v1", "c": "decision/v1"},
        )
    )
    selector = _Plugin(
        _descriptor(
            "chemistry.selector",
            PluginKind.SELECTOR,
            inputs=("decision/v1",),
            outputs=("raw_molecule/v1",),
        )
    )
    pipeline = _pipeline(
        {
            "id": "a",
            "slot": "source",
            "plugin": "io.source@1.0.0",
            "config": {"path": "/data/input.smi"},
        },
        {
            "id": "a.b",
            "slot": "hard_gate",
            "plugin": "chemistry.gate@1.0.0",
        },
        {
            "id": "select",
            "slot": "select",
            "plugin": "chemistry.selector@1.0.0",
            "inputs": [
                {"request_port": "left", "stage": "a", "port": "b.c"},
                {"request_port": "right", "stage": "a.b", "port": "c"},
            ],
        },
    )

    compiled = compile_pipeline(pipeline, _registry(source, gate, selector))

    assert {binding.source_key for binding in compiled.stages[-1].input_bindings} == {
        ("a", "b.c"),
        ("a.b", "c"),
    }


def test_explicit_binding_rejects_unknown_source_port() -> None:
    pipeline = _pipeline(
        {
            "id": "ingest",
            "slot": "source",
            "plugin": "io.source@1.0.0",
            "config": {"path": "/data/input.smi"},
        },
        {
            "id": "normalize",
            "slot": "standardize",
            "plugin": "chemistry.standardize@1.0.0",
            "inputs": [
                {
                    "request_port": "molecules",
                    "stage": "ingest",
                    "port": "missing",
                }
            ],
        },
    )

    with pytest.raises(PipelineError) as caught:
        compile_pipeline(pipeline, _registry(_source(), _standardizer()))

    assert caught.value.code == "PIPELINE_BINDING_PORT_UNKNOWN"


def test_automatic_binding_selects_most_recent_compatible_primary() -> None:
    consumer = _gate(inputs=("raw_molecule/v1",))
    pipeline = _pipeline(
        {
            "id": "ingest",
            "slot": "source",
            "plugin": "io.source@1.0.0",
            "config": {"path": "/data/input.smi"},
        },
        {
            "id": "normalize",
            "slot": "standardize",
            "plugin": "chemistry.standardize@1.0.0",
        },
        {
            "id": "filter_raw",
            "slot": "hard_gate",
            "plugin": "chemistry.gate@1.0.0",
        },
    )

    compiled = compile_pipeline(
        pipeline,
        _registry(_source(), _standardizer(), consumer),
    )

    binding = compiled.stages[-1].input_bindings[0]
    assert binding.source_stage_id == "ingest"
    assert binding.contract_id == "raw_molecule/v1"


def test_plugin_specific_config_is_validated_during_compilation() -> None:
    pipeline = _pipeline(
        {
            "id": "ingest",
            "slot": "source",
            "plugin": "io.source@1.0.0",
            "config": {"path": 123},
        }
    )

    with pytest.raises(PipelineError) as caught:
        compile_pipeline(pipeline, _registry(_source()))

    assert caught.value.code == "PIPELINE_STAGE_CONFIG_INVALID"
    assert caught.value.context["stage_id"] == "ingest"
    assert caught.value.context["error_count"] == 1


def test_custom_slot_map_is_snapshotted_and_used_explicitly() -> None:
    slots = {"input_library": PluginKind.SOURCE}
    compiler = PipelineCompiler(_registry(_source()), slot_kinds=slots)
    slots["input_library"] = PluginKind.GATE
    pipeline = _pipeline(
        {
            "id": "ingest",
            "slot": "input_library",
            "plugin": "io.source@1.0.0",
            "config": {"path": "/data/input.smi"},
        }
    )

    compiled = compiler.compile(pipeline)

    assert compiled.stages[0].slot == "input_library"
    assert compiler.slot_kinds["input_library"] is PluginKind.SOURCE


def test_compiled_stage_config_is_immutable() -> None:
    compiled = compile_pipeline(
        _pipeline(
            {
                "id": "ingest",
                "slot": "source",
                "plugin": "io.source@1.0.0",
                "config": {"path": "/data/input.smi"},
            }
        ),
        _registry(_source()),
    )

    with pytest.raises(TypeError, match="immutable"):
        compiled.stages[0].config["path"] = str(Path("/other"))  # type: ignore[index]
