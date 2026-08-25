from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from molcascade.artifacts.models import ArtifactDatasetRef, ArtifactOutput, ArtifactRef
from molcascade.errors import PluginError
from molcascade.plugins import (
    Cardinality,
    Determinism,
    PendingOutput,
    PluginDescriptor,
    PluginKind,
    PluginRegistry,
    StageContext,
    StageInput,
    StageRequest,
    StageResponse,
)


class DummyPlugin:
    def __init__(self, descriptor: PluginDescriptor) -> None:
        self.descriptor = descriptor

    def execute(self, request: StageRequest, context: StageContext) -> StageResponse:
        del request, context
        return StageResponse(
            outputs={
                "data": PendingOutput(
                    contract_id=self.descriptor.outputs[0],
                    file_paths=("datasets/data.parquet",),
                )
            }
        )


def _source(*, version: str = "1.0.0") -> DummyPlugin:
    return DummyPlugin(
        PluginDescriptor(
            id="io.delimited",
            version=version,
            kind=PluginKind.SOURCE,
            outputs=("raw_molecule/v1",),
            cardinality=Cardinality.ONE_TO_MANY,
            determinism=Determinism.DETERMINISTIC,
        )
    )


def _standardizer() -> DummyPlugin:
    return DummyPlugin(
        PluginDescriptor(
            id="chemistry.standardize",
            version="1.0.0",
            kind=PluginKind.STANDARDIZER,
            inputs=("raw_molecule/v1",),
            outputs=("parent/v1", "parent_source_map/v1"),
            output_ports={
                "primary": "parent/v1",
                "mapping": "parent_source_map/v1",
            },
            cardinality=Cardinality.MANY_TO_ONE,
            determinism=Determinism.DETERMINISTIC,
        )
    )


def _gate() -> DummyPlugin:
    return DummyPlugin(
        PluginDescriptor(
            id="chemistry.gate",
            version="1.0.0",
            kind=PluginKind.GATE,
            inputs=("parent/v1",),
            outputs=("decision/v1",),
            cardinality=Cardinality.ONE_TO_ONE,
            determinism=Determinism.DETERMINISTIC,
        )
    )


def test_descriptor_accepts_json_arrays_but_is_strict_and_versioned() -> None:
    descriptor = PluginDescriptor.model_validate(
        {
            "id": "io.delimited",
            "version": "1.2.3",
            "kind": "source",
            "inputs": [],
            "outputs": ["raw_molecule/v1"],
            "cardinality": "one_to_many",
            "determinism": "deterministic",
        }
    )

    assert descriptor.key == "io.delimited@1.2.3"
    assert descriptor.outputs == ("raw_molecule/v1",)
    with pytest.raises(ValidationError):
        PluginDescriptor.model_validate({**descriptor.model_dump(), "version": "1"})
    with pytest.raises(ValidationError, match="Extra inputs"):
        PluginDescriptor.model_validate({**descriptor.model_dump(), "typo": True})


def test_filter_cardinality_is_a_strict_supported_value() -> None:
    descriptor = _gate().descriptor.model_copy(update={"cardinality": Cardinality.FILTER})

    reparsed = PluginDescriptor.model_validate(descriptor.model_dump())

    assert reparsed.cardinality is Cardinality.FILTER
    with pytest.raises(ValidationError):
        PluginDescriptor.model_validate(
            {**descriptor.model_dump(), "cardinality": "one_to_zero_or_many"}
        )


def test_descriptor_rejects_invalid_ports() -> None:
    with pytest.raises(ValidationError, match="source plugins cannot"):
        PluginDescriptor(
            id="bad.source",
            version="1.0.0",
            kind=PluginKind.SOURCE,
            inputs=("raw_molecule/v1",),
            outputs=("parent/v1",),
            cardinality=Cardinality.ONE_TO_ONE,
            determinism=Determinism.DETERMINISTIC,
        )
    with pytest.raises(ValidationError, match="non-source plugins"):
        PluginDescriptor(
            id="bad.gate",
            version="1.0.0",
            kind=PluginKind.GATE,
            outputs=("decision/v1",),
            cardinality=Cardinality.ONE_TO_ONE,
            determinism=Determinism.DETERMINISTIC,
        )


def test_tier_neutral_policy_descriptor_has_strict_contract_shape() -> None:
    valid = PluginDescriptor(
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
    assert valid.tier_neutral_policy is True

    with pytest.raises(ValidationError, match="must have gate kind"):
        PluginDescriptor(
            id="policy.bad",
            version="1.0.0",
            kind=PluginKind.PREDICTOR,
            inputs=("parent/v1", "decision/v1"),
            outputs=("parent/v1", "decision/v1"),
            output_ports={"primary": "parent/v1", "decisions": "decision/v1"},
            cardinality=Cardinality.FILTER,
            determinism=Determinism.DETERMINISTIC,
            tier_neutral_policy=True,
        )

    with pytest.raises(ValidationError, match="must consume"):
        PluginDescriptor(
            id="policy.bad",
            version="1.0.0",
            kind=PluginKind.GATE,
            inputs=("parent/v1",),
            outputs=("parent/v1", "decision/v1"),
            output_ports={"primary": "parent/v1", "decisions": "decision/v1"},
            cardinality=Cardinality.FILTER,
            determinism=Determinism.DETERMINISTIC,
            tier_neutral_policy=True,
        )
def test_builtin_registration_is_trusted_and_duplicate_never_replaces() -> None:
    registry = PluginRegistry()
    original = _source()
    registered = registry.register_builtin(original)

    assert registered.trusted
    assert registry.get("io.delimited@1.0.0") is original
    with pytest.raises(PluginError) as caught:
        registry.register_builtin(_source())
    assert caught.value.code == "PLUGIN_DUPLICATE"
    assert registry.get("io.delimited@1.0.0") is original


def test_manual_registration_is_untrusted_by_default() -> None:
    registry = PluginRegistry()
    entry = registry.register(_source())

    assert entry.origin == "manual"
    assert not entry.trusted
    with pytest.raises(PluginError) as caught:
        registry.get(entry.key)
    assert caught.value.code == "PLUGIN_NOT_TRUSTED"


def test_multiple_versions_require_exact_resolution() -> None:
    registry = PluginRegistry([_source(version="1.0.0"), _source(version="2.0.0")])

    with pytest.raises(PluginError) as caught:
        registry.get("io.delimited")

    assert caught.value.code == "PLUGIN_VERSION_REQUIRED"
    assert caught.value.context["versions"] == ["1.0.0", "2.0.0"]


def test_unknown_contract_is_rejected_at_registration() -> None:
    plugin = DummyPlugin(
        PluginDescriptor(
            id="external.unknown",
            version="1.0.0",
            kind=PluginKind.GATE,
            inputs=("unknown/v1",),
            outputs=("decision/v1",),
            cardinality=Cardinality.ONE_TO_ONE,
            determinism=Determinism.DETERMINISTIC,
        )
    )

    with pytest.raises(PluginError) as caught:
        PluginRegistry().register(plugin, origin="test")

    assert caught.value.code == "PLUGIN_CONTRACT_UNKNOWN"
    assert caught.value.context["unknown_contracts"] == ["unknown/v1"]


def test_contract_compatibility_is_exact_and_structured() -> None:
    registry = PluginRegistry([_source(), _standardizer(), _gate()])

    assert registry.ensure_compatible("io.delimited@1.0.0", "chemistry.standardize@1.0.0") == (
        "raw_molecule/v1",
    )
    with pytest.raises(PluginError) as caught:
        registry.ensure_compatible("io.delimited@1.0.0", "chemistry.gate@1.0.0")
    assert caught.value.code == "PLUGIN_CONTRACT_INCOMPATIBLE"
    assert caught.value.context["upstream_outputs"] == ["raw_molecule/v1"]


class _FakeDistribution:
    name = "example-plugin-package"


class _FakeEntryPoint:
    group = "molcascade.plugins"
    value = "example:plugin"
    dist = _FakeDistribution()

    def __init__(self, name: str, plugin: DummyPlugin) -> None:
        self.name = name
        self._plugin = plugin
        self.loads = 0

    def load(self) -> DummyPlugin:
        self.loads += 1
        return self._plugin


class _FakeEntryPoints(list[_FakeEntryPoint]):
    def select(self, *, group: str) -> _FakeEntryPoints:
        return _FakeEntryPoints(item for item in self if item.group == group)


def test_registry_construction_does_not_scan_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_scan() -> None:
        raise AssertionError("entry points must not be scanned implicitly")

    monkeypatch.setattr("molcascade.plugins.registry.importlib_metadata.entry_points", fail_scan)

    assert len(PluginRegistry()) == 0


def test_entry_point_discovery_is_allowlisted_and_untrusted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = _FakeEntryPoint("allowed", _source())
    skipped = _FakeEntryPoint("not-allowed", _standardizer())
    monkeypatch.setattr(
        "molcascade.plugins.registry.importlib_metadata.entry_points",
        lambda: _FakeEntryPoints([skipped, allowed]),
    )
    registry = PluginRegistry()

    discovered = registry.discover_entry_points(
        allowlist={"allowed": "example_plugin_package"}
    )

    assert len(discovered) == 1
    assert not discovered[0].trusted
    assert allowed.loads == 1
    assert skipped.loads == 0
    with pytest.raises(PluginError) as blocked:
        registry.get("io.delimited@1.0.0")
    assert blocked.value.code == "PLUGIN_NOT_TRUSTED"

    registry.trust("io.delimited@1.0.0")
    assert registry.get("io.delimited@1.0.0") is allowed._plugin


def test_entry_point_distribution_pin_is_checked_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_point = _FakeEntryPoint("allowed", _source())
    monkeypatch.setattr(
        "molcascade.plugins.registry.importlib_metadata.entry_points",
        lambda: _FakeEntryPoints([entry_point]),
    )

    with pytest.raises(PluginError) as caught:
        PluginRegistry().discover_entry_points(allowlist={"allowed": "different-package"})

    assert caught.value.code == "PLUGIN_DISTRIBUTION_MISMATCH"
    assert entry_point.loads == 0


def test_pending_output_rejects_unsafe_or_duplicate_paths() -> None:
    with pytest.raises(ValueError, match="relative"):
        PendingOutput(contract_id="raw_molecule/v1", file_paths=("/escape.parquet",))
    with pytest.raises(ValueError, match="unique"):
        PendingOutput(
            contract_id="raw_molecule/v1",
            file_paths=("data.parquet", "data.parquet"),
        )


def test_stage_input_contract_must_match_artifact_kind() -> None:
    ref = ArtifactRef(
        artifact_id="artifact:sha256:" + "0" * 64,
        kind="raw_molecule/v1",
    )

    with pytest.raises(ValueError, match="artifact reference kind"):
        StageInput(ref=ref, root=Path("/artifacts/input"), contract_id="parent/v1")


def test_stage_input_derives_port_contract_and_files_from_dataset_ref() -> None:
    ref = ArtifactDatasetRef(
        artifact_id="artifact:sha256:" + "1" * 64,
        artifact_kind="stage-output",
        port="mapping",
        contract_id="parent_source_map/v1",
        file_paths=("datasets/mapping/part-00000.parquet",),
    )

    stage_input = StageInput(ref=ref, root=Path("/artifacts/input"))

    assert stage_input.port == "mapping"
    assert stage_input.contract_id == "parent_source_map/v1"
    assert stage_input.file_paths == ("datasets/mapping/part-00000.parquet",)
    with pytest.raises(ValueError, match="dataset reference"):
        StageInput(
            ref=ref,
            root=Path("/artifacts/input"),
            contract_id="parent/v1",
        )


def test_stage_response_declares_one_atomic_multi_dataset_inventory() -> None:
    response = StageResponse(
        outputs={
            "parents": PendingOutput(
                contract_id="parent/v1",
                file_paths=("parents/data.parquet",),
            ),
            "mapping": PendingOutput(
                contract_id="parent_source_map/v1",
                file_paths=("mapping/data.parquet",),
            ),
        },
        metadata={"identity_policy_id": "rdkit-registration-hash/v1"},
    )

    assert response.file_paths == (
        "mapping/data.parquet",
        "parents/data.parquet",
    )
    assert response.artifact_outputs == (
        ArtifactOutput(
            port="mapping",
            contract_id="parent_source_map/v1",
            file_paths=("mapping/data.parquet",),
        ),
        ArtifactOutput(
            port="parents",
            contract_id="parent/v1",
            file_paths=("parents/data.parquet",),
        ),
    )


def test_multi_output_descriptor_requires_typed_complete_port_map() -> None:
    with pytest.raises(ValidationError, match="output_ports"):
        PluginDescriptor(
            id="test.multi",
            version="1.0.0",
            kind=PluginKind.STANDARDIZER,
            inputs=("raw_molecule/v1",),
            outputs=("parent/v1", "decision/v1"),
            cardinality=Cardinality.MANY_TO_ONE,
            determinism=Determinism.DETERMINISTIC,
        )

    descriptor = PluginDescriptor(
        id="test.multi",
        version="1.0.0",
        kind=PluginKind.STANDARDIZER,
        inputs=("raw_molecule/v1",),
        outputs=("parent/v1", "decision/v1"),
        output_ports={"primary": "parent/v1", "decisions": "decision/v1"},
        cardinality=Cardinality.MANY_TO_ONE,
        determinism=Determinism.DETERMINISTIC,
    )
    assert descriptor.primary_contract == "parent/v1"
    with pytest.raises(TypeError, match="immutable"):
        descriptor.output_ports["other"] = "decision/v1"


def test_builtin_registry_contains_reviewed_local_vertical_slice() -> None:
    from molcascade.plugins.registry import create_builtin_registry

    registry = create_builtin_registry()
    expected = {
        "source.delimited_smiles@0.1.0",
        "source.sdf@0.1.0",
        "source.raw_molecule_parquet@0.1.0",
        "chemistry.rdkit_standardize@0.1.0",
        "chemistry.rdkit_hard_gate@0.1.0",
    }
    assert expected <= {entry.key for entry in registry.entries()}
    assert all(entry.trusted and entry.origin == "builtin" for entry in registry.entries())
    with pytest.raises(ValueError, match="declared by both"):
        StageResponse(
            outputs={
                "parents": PendingOutput(
                    contract_id="parent/v1",
                    file_paths=("shared.parquet",),
                ),
                "mapping": PendingOutput(
                    contract_id="parent_source_map/v1",
                    file_paths=("shared.parquet",),
                ),
            }
        )
