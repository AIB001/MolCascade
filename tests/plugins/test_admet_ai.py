from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import PARENT_V1, PREDICTION_V1
from molcascade.errors import PluginError
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin import admet as admet_module
from molcascade.plugins.builtin.admet import (
    DEFAULT_MODELS_REFERENCE,
    ADMETAIConfig,
    ADMETAIV2PredictorPlugin,
    inspect_admet_ai_installation,
    inspect_admet_ai_model_assets,
)
from molcascade.plugins.manifest import Determinism
from molcascade.plugins.registry import create_builtin_registry


def _parent(parent_id: str, smiles: str) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "identity_policy_id": "identity-policy:test",
        "parent_smiles": smiles,
        "registration_key": f"registration:{parent_id}",
        "stereo_key": None,
        "formula": None,
        "duplicate_count": 1,
    }


def _input(tmp_path: Path) -> StageInput:
    root = tmp_path / "input"
    root.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                _parent("ethanol", "CCO"),
                _parent("propanol", "CCCO"),
                _parent("butanol", "CCCCO"),
            ],
            schema=PARENT_V1.schema,
        ),
        root / "parents.parquet",
    )
    return StageInput(
        ref=ArtifactRef(
            artifact_id="artifact:sha256:" + "a" * 64,
            kind=PARENT_V1.id,
        ),
        root=root,
        contract_id=PARENT_V1.id,
    )


def _assets(tmp_path: Path) -> tuple[Path, dict[str, object], Path, str]:
    models = tmp_path / "models"
    checkpoint = models / "ensemble-a" / "model-0.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"trusted fake checkpoint")
    (checkpoint.parent / "output_columns.txt").write_text(
        "Caco2_Wang\nHIA_Hou\n",
        encoding="utf-8",
    )
    inspected = inspect_admet_ai_model_assets(models)

    package_root = tmp_path / "fake-admet-ai-package"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("__version__ = '2.0.1'\n", encoding="utf-8")
    package_hash, _ = admet_module._hash_existing_tree(
        package_root,
        maximum_files=100,
        maximum_bytes=1024 * 1024,
        suffixes=admet_module._PACKAGE_CODE_SUFFIXES,
    )
    return models, inspected, package_root, package_hash


def _config(
    models: Path,
    inspected: dict[str, object],
    package_hash: str,
    **updates: object,
) -> dict[str, object]:
    checkpoint = inspected["checkpoints"][0]  # type: ignore[index]
    config: dict[str, object] = {
        "models_dir": str(models),
        "expected_model_manifest_sha256": inspected["model_manifest_sha256"],
        "expected_package_code_sha256": package_hash,
        "expected_checkpoints": [
            {
                "relative_path": checkpoint["relative_path"],
                "sha256": checkpoint["sha256"],
            }
        ],
        "endpoints": [
            {
                "output_column": "Caco2_Wang",
                "endpoint_id": "admet-ai-v2/Caco2_Wang",
            },
            {
                "output_column": "HIA_Hou",
                "endpoint_id": "admet-ai-v2/HIA_Hou",
            },
        ],
        "batch_size": 2,
        "num_workers": 0,
        "allow_unsafe_model_deserialization": True,
    }
    config.update(updates)
    return config


class _FakeFrame:
    columns = ("Caco2_Wang", "HIA_Hou")

    def __init__(
        self,
        rows: list[dict[str, float]],
        *,
        index: list[str] | None = None,
    ) -> None:
        self._rows = rows
        if index is not None:
            self.index = index

    def to_dict(self, *, orient: str) -> list[dict[str, float]]:
        assert orient == "records"
        return list(self._rows)


def test_optional_adapter_is_lazy_registered_and_best_effort() -> None:
    source = inspect.getsource(admet_module)
    tree = ast.parse(source)
    eager_external_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "admet_ai" or alias.name.startswith("admet_ai.")
    }
    eager_external_imports.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and (node.module == "admet_ai" or (node.module or "").startswith("admet_ai."))
    )
    assert eager_external_imports == set()

    plugin = create_builtin_registry().get("prediction.admet_ai_v2@0.1.0")
    assert isinstance(plugin, ADMETAIV2PredictorPlugin)
    assert plugin.descriptor.determinism is Determinism.BEST_EFFORT
    assert plugin.descriptor.inputs == (PARENT_V1.id,)
    assert plugin.descriptor.output_ports == {
        "primary": PARENT_V1.id,
        "predictions": PREDICTION_V1.id,
    }


def test_model_inspection_hashes_recursive_assets_without_loading_backend(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    (models / "z").mkdir(parents=True)
    (models / "z" / "weights.pt").write_bytes(b"weights")
    (models / "z" / "a.json").write_text("{}\n", encoding="utf-8")

    first = inspect_admet_ai_model_assets(models)
    second = inspect_admet_ai_model_assets(models)
    assert first == second
    assert first["model_file_count"] == 2
    assert first["model_size_bytes"] == 10
    assert first["checkpoint_count"] == 1
    assert first["model_manifest_sha256"] != first["checkpoint_manifest_sha256"]
    assert first["checkpoints"] == [
        {
            "relative_path": "z/weights.pt",
            "sha256": hashlib.sha256(b"weights").hexdigest(),
            "size_bytes": 7,
        }
    ]


def test_installation_inspection_hashes_code_without_importing_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "admet_ai"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("VERSION = 'fake'\n", encoding="utf-8")
    (package_root / "ignored.csv").write_text("not,code\n", encoding="utf-8")

    class FakeDistribution:
        version = "2.0.1"

        @staticmethod
        def read_text(name: str) -> str | None:
            assert name == "RECORD"
            return "admet_ai/__init__.py,,\n"

    monkeypatch.setattr(
        admet_module.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(submodule_search_locations=[str(package_root)])
        if name == "admet_ai"
        else None,
    )
    monkeypatch.setattr(
        admet_module.importlib.metadata,
        "distribution",
        lambda name: FakeDistribution() if name == "admet-ai" else None,
    )
    monkeypatch.setattr(
        admet_module.importlib.metadata,
        "version",
        lambda name: f"fake-{name}-version",
    )

    inspected = inspect_admet_ai_installation()
    package_hash, _ = admet_module._hash_existing_tree(
        package_root,
        maximum_files=10_000,
        maximum_bytes=2 * 1024 * 1024 * 1024,
        suffixes=admet_module._PACKAGE_CODE_SUFFIXES,
    )
    assert inspected == {
        "package_root": str(package_root.resolve()),
        "version": "2.0.1",
        "package_code_sha256": package_hash,
        "distribution_record_sha256": hashlib.sha256(
            b"admet_ai/__init__.py,,\n"
        ).hexdigest(),
        "dependency_versions": {
            name: f"fake-{name}-version"
            for name in admet_module._SCIENTIFIC_DISTRIBUTIONS
        },
    }


def test_model_inspection_rejects_layout_the_v2_loader_cannot_use(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "root-level.pt").write_bytes(b"not inside an ensemble")
    with pytest.raises(PluginError) as captured:
        inspect_admet_ai_model_assets(models)
    assert captured.value.code == "ADMET_AI_MODEL_LAYOUT_INVALID"


def test_model_and_package_pins_are_both_required_and_strict() -> None:
    with pytest.raises(ValidationError):
        ADMETAIConfig.model_validate(
            {
                "models_dir": "models",
                "expected_model_manifest_sha256": "0" * 64,
                "endpoints": [
                    {"output_column": "Caco2_Wang", "endpoint_id": "caco2"}
                ],
            }
        )
    with pytest.raises(ValidationError):
        ADMETAIConfig.model_validate(
            {
                "models_dir": "models",
                "expected_package_code_sha256": "0" * 64,
                "endpoints": [
                    {"output_column": "Caco2_Wang", "endpoint_id": "caco2"}
                ],
            }
        )
    with pytest.raises(ValidationError):
        ADMETAIConfig.model_validate(
            {
                "models_dir": "models",
                "expected_model_manifest_sha256": "A" * 64,
                "expected_package_code_sha256": "0" * 64,
                "endpoints": [
                    {"output_column": "Caco2_Wang", "endpoint_id": "caco2"}
                ],
            }
        )
    with pytest.raises(ValidationError):
        ADMETAIConfig.model_validate(
            {
                "models_dir": "models",
                "expected_model_manifest_sha256": "0" * 64,
                "expected_package_code_sha256": "A" * 64,
                "endpoints": [
                    {"output_column": "Caco2_Wang", "endpoint_id": "caco2"}
                ],
            }
        )


def test_prediction_requires_explicit_checkpoint_deserialization_trust(
    tmp_path: Path,
) -> None:
    request = StageRequest(
        stage_id="admet",
        inputs={"parents": _input(tmp_path)},
        config={
            "models_dir": str(tmp_path / "not-even-inspected"),
            "expected_model_manifest_sha256": "0" * 64,
            "expected_package_code_sha256": "0" * 64,
            "endpoints": [
                {"output_column": "Caco2_Wang", "endpoint_id": "caco2"}
            ],
        },
    )
    with pytest.raises(PluginError) as captured:
        ADMETAIV2PredictorPlugin().execute(request, StageContext(tmp_path / "out"))
    assert captured.value.code == "ADMET_AI_MODEL_TRUST_REQUIRED"
    assert captured.value.context["risk"] == "executable-model-artifacts"


def test_missing_package_pin_fails_before_backend_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_import(_name: str) -> object:
        raise AssertionError("invalid pin configuration must stop before import")

    monkeypatch.setattr(admet_module.importlib, "import_module", forbidden_import)
    request = StageRequest(
        stage_id="admet",
        inputs={"parents": _input(tmp_path)},
        config={
            "models_dir": str(tmp_path / "not-inspected"),
            "expected_model_manifest_sha256": "0" * 64,
            "endpoints": [
                {"output_column": "Caco2_Wang", "endpoint_id": "caco2"}
            ],
            "allow_unsafe_model_deserialization": True,
        },
    )
    with pytest.raises(PluginError) as captured:
        ADMETAIV2PredictorPlugin().execute(request, StageContext(tmp_path / "out"))
    assert captured.value.code == "PLUGIN_CONFIG_INVALID"


def test_admet_ai_v2_runs_in_bounded_batches_and_emits_typed_long_form(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models, inspected, package_root, package_hash = _assets(tmp_path)
    package = {
        "package_root": package_root,
        "version": "2.0.1",
        "package_code_sha256": package_hash,
        "distribution_record_sha256": "b" * 64,
    }
    monkeypatch.setattr(admet_module, "_package_provenance", lambda _config: package)

    constructor_calls: list[dict[str, Any]] = []
    prediction_batches: list[list[str]] = []

    class FakeADMETModel:
        def __init__(self, **kwargs: Any) -> None:
            constructor_calls.append(dict(kwargs))
            snapshot = Path(kwargs["models_dir"])
            assert snapshot.name == ".admet-ai-model-snapshot"
            assert (snapshot / "ensemble-a" / "model-0.pt").read_bytes() == (
                b"trusted fake checkpoint"
            )

        def predict(self, smiles: list[str]) -> _FakeFrame:
            prediction_batches.append(list(smiles))
            return _FakeFrame(
                [
                    {
                        "Caco2_Wang": 0.1 + index,
                        "HIA_Hou": 0.8 - index * 0.1,
                    }
                    for index, _ in enumerate(smiles)
                ],
                index=list(smiles),
            )

    fake_module = SimpleNamespace(ADMETModel=FakeADMETModel)
    monkeypatch.setattr(
        admet_module.importlib,
        "import_module",
        lambda name: fake_module if name == "admet_ai" else None,
    )

    output = tmp_path / "output"
    response = ADMETAIV2PredictorPlugin().execute(
        StageRequest(
            stage_id="admet",
            inputs={"parents": _input(tmp_path)},
            config=_config(models, inspected, package_hash),
        ),
        StageContext(output),
    )

    assert prediction_batches == [["CCO", "CCCO"], ["CCCCO"]]
    assert constructor_calls == [
        {
            "models_dir": output / ".admet-ai-model-snapshot",
            "include_physchem": False,
            "drugbank_path": None,
            "num_workers": 0,
        }
    ]
    assert not (output / ".admet-ai-model-snapshot").exists()

    parents = pq.read_table(output / response.outputs["primary"].file_paths[0])
    predictions = pq.read_table(
        output / response.outputs["predictions"].file_paths[0]
    )
    PARENT_V1.validate(parents)
    PREDICTION_V1.validate(predictions)
    assert parents.num_rows == 3
    assert predictions.num_rows == 6
    assert {row["endpoint_id"] for row in predictions.to_pylist()} == {
        "admet-ai-v2/Caco2_Wang",
        "admet-ai-v2/HIA_Hou",
    }
    assert len({row["model_id"] for row in predictions.to_pylist()}) == 1
    assert all(row["prediction_std"] is None for row in predictions.to_pylist())
    assert all(row["calibration_id"] is None for row in predictions.to_pylist())
    assert response.metadata["backend_version"] == "2.0.1"
    assert response.metadata["package_code_sha256"] == package_hash
    assert response.metadata["model_manifest_sha256"] == inspected[
        "model_manifest_sha256"
    ]
    assert response.metadata["checkpoint_manifest_sha256"] == inspected[
        "checkpoint_manifest_sha256"
    ]
    assert response.metadata["network_or_download_invoked_by_adapter"] is False
    assert response.metadata["uncertainty_available"] is False
    assert response.metadata["calibration_available"] is False


def test_same_cardinality_dataframe_reorder_is_rejected_before_row_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models, inspected, package_root, package_hash = _assets(tmp_path)
    package = {
        "package_root": package_root,
        "version": "2.0.1",
        "package_code_sha256": package_hash,
        "distribution_record_sha256": None,
    }
    monkeypatch.setattr(admet_module, "_package_provenance", lambda _config: package)

    class ReorderedModel:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def predict(self, smiles: list[str]) -> _FakeFrame:
            reversed_smiles = list(reversed(smiles))
            return _FakeFrame(
                [
                    {"Caco2_Wang": 0.2, "HIA_Hou": 0.7}
                    for _ in reversed_smiles
                ],
                index=reversed_smiles,
            )

    monkeypatch.setattr(
        admet_module.importlib,
        "import_module",
        lambda _name: SimpleNamespace(ADMETModel=ReorderedModel),
    )
    output = tmp_path / "output"
    with pytest.raises(PluginError) as captured:
        ADMETAIV2PredictorPlugin().execute(
            StageRequest(
                stage_id="admet",
                inputs={"parents": _input(tmp_path)},
                config=_config(models, inspected, package_hash),
            ),
            StageContext(output),
        )
    assert captured.value.code == "ADMET_AI_OUTPUT_INDEX_MISMATCH"
    assert captured.value.context == {
        "position": 0,
        "expected_smiles": "CCO",
        "actual_index": "CCCO",
    }
    assert not (output / "datasets/passed_parents/part-00000.parquet").exists()
    assert not (output / "datasets/admet_predictions/part-00000.parquet").exists()
    assert not (output / ".admet-ai-model-snapshot").exists()


def test_wrong_model_digest_stops_before_import_or_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models, inspected, package_root, package_hash = _assets(tmp_path)
    package = {
        "package_root": package_root,
        "version": "2.0.1",
        "package_code_sha256": package_hash,
        "distribution_record_sha256": None,
    }
    monkeypatch.setattr(admet_module, "_package_provenance", lambda _config: package)

    def forbidden_import(_name: str) -> object:
        raise AssertionError("digest mismatch must stop before importing ADMET-AI")

    monkeypatch.setattr(admet_module.importlib, "import_module", forbidden_import)
    config = _config(
        models,
        inspected,
        package_hash,
        expected_model_manifest_sha256="0" * 64,
    )
    output = tmp_path / "output"
    with pytest.raises(PluginError) as captured:
        ADMETAIV2PredictorPlugin().execute(
            StageRequest(
                stage_id="admet",
                inputs={"parents": _input(tmp_path)},
                config=config,
            ),
            StageContext(output),
        )
    assert captured.value.code == "ADMET_AI_MODEL_HASH_MISMATCH"
    assert captured.value.context["actual_sha256"] == inspected[
        "model_manifest_sha256"
    ]
    assert not (output / ".admet-ai-model-snapshot").exists()


@pytest.mark.parametrize(
    ("overrides", "expected_code"),
    [
        (
            {"expected_package_code_sha256": "0" * 64},
            "ADMET_AI_PACKAGE_HASH_MISMATCH",
        ),
        (
            {
                "expected_checkpoints": [
                    {
                        "relative_path": "ensemble-a/model-0.pt",
                        "sha256": "0" * 64,
                    }
                ]
            },
            "ADMET_AI_CHECKPOINT_HASH_MISMATCH",
        ),
    ],
)
def test_package_and_checkpoint_pins_stop_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    expected_code: str,
) -> None:
    models, inspected, package_root, package_hash = _assets(tmp_path)
    package = {
        "package_root": package_root,
        "version": "2.0.1",
        "package_code_sha256": package_hash,
        "distribution_record_sha256": None,
    }
    monkeypatch.setattr(admet_module, "_package_provenance", lambda _config: package)

    def forbidden_import(_name: str) -> object:
        raise AssertionError("pin mismatch must stop before importing ADMET-AI")

    monkeypatch.setattr(admet_module.importlib, "import_module", forbidden_import)
    output = tmp_path / "output"
    with pytest.raises(PluginError) as captured:
        ADMETAIV2PredictorPlugin().execute(
            StageRequest(
                stage_id="admet",
                inputs={"parents": _input(tmp_path)},
                config=_config(models, inspected, package_hash, **overrides),
            ),
            StageContext(output),
        )
    assert captured.value.code == expected_code
    assert not (output / ".admet-ai-model-snapshot").exists()


def test_missing_endpoint_is_a_structured_failure_and_removes_partial_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models, inspected, package_root, package_hash = _assets(tmp_path)
    package = {
        "package_root": package_root,
        "version": "2.0.1",
        "package_code_sha256": package_hash,
        "distribution_record_sha256": None,
    }
    monkeypatch.setattr(admet_module, "_package_provenance", lambda _config: package)

    class MissingEndpointModel:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def predict(self, smiles: list[str]) -> _FakeFrame:
            result = _FakeFrame([{"Caco2_Wang": 0.5} for _ in smiles])
            result.columns = ("Caco2_Wang",)  # type: ignore[misc]
            return result

    monkeypatch.setattr(
        admet_module.importlib,
        "import_module",
        lambda _name: SimpleNamespace(ADMETModel=MissingEndpointModel),
    )
    output = tmp_path / "output"
    with pytest.raises(PluginError) as captured:
        ADMETAIV2PredictorPlugin().execute(
            StageRequest(
                stage_id="admet",
                inputs={"parents": _input(tmp_path)},
                config=_config(models, inspected, package_hash),
            ),
            StageContext(output),
        )
    assert captured.value.code == "ADMET_AI_ENDPOINT_MISSING"
    assert captured.value.context["missing_columns"] == ["HIA_Hou"]
    assert not (output / "datasets/passed_parents/part-00000.parquet").exists()
    assert not (output / "datasets/admet_predictions/part-00000.parquet").exists()
    assert not (output / ".admet-ai-model-snapshot").exists()


def test_a_package_reference_resolves_to_the_wheel_and_refuses_to_leave_it(
    tmp_path: Path,
) -> None:
    """``models_dir`` must mean the same thing on the laptop and on the GPU node.

    An absolute ``site-packages`` path is correct exactly once -- on the machine
    that wrote it -- so the shipped default is a reference instead.  What has to
    hold is that the reference lands on the same bytes an absolute path would,
    and that nothing dressed up as a reference can point somewhere else.
    """

    pytest.importorskip("admet_ai")
    import admet_ai

    absolute = Path(admet_ai.__file__).parent / "resources" / "models"
    assert admet_module._local_model_root(DEFAULT_MODELS_REFERENCE) == absolute.resolve()
    # The digest is what the pin is compared against, so it is the thing that
    # actually has to agree -- not just the path string.
    assert (
        inspect_admet_ai_model_assets(DEFAULT_MODELS_REFERENCE)["model_manifest_sha256"]
        == inspect_admet_ai_model_assets(absolute)["model_manifest_sha256"]
    )

    escapes = [
        "package:admet_ai/../../etc",  # traversal
        "package:os/path",  # a different package
        "package:admet_ai",  # no member at all
        "package:admet_ai/",  # empty member
        "package:admet_ai//resources",  # empty path part
        "package:admet_ai/./resources",  # non-canonical
    ]
    for reference in escapes:
        with pytest.raises(PluginError) as captured:
            admet_module._local_model_root(reference)
        assert captured.value.code == "ADMET_AI_MODEL_PATH_INVALID", reference

    # A plain path still works; the scheme is an addition, not a replacement.
    plain, _, _, _ = _assets(tmp_path)
    assert admet_module._local_model_root(str(plain)) == plain.resolve()


def test_the_catalogue_ships_pins_that_match_the_installed_release() -> None:
    """A default that cannot run is worse than no default.

    The builder offers this option pre-configured, and the adapter refuses to
    start unless the digests match the bytes on disk.  So the shipped digests
    are a claim about the installed wheel, and this is where that claim is
    checked -- otherwise the first person to find out is someone forty minutes
    into a screen.
    """

    pytest.importorskip("admet_ai")
    from molcascade.cascade.catalog import CRITERIA_BY_ID

    option = next(
        item
        for item in CRITERIA_BY_ID["admet_endpoint"].options
        if item.id == "admet_ai_v2"
    )
    config = ADMETAIConfig.model_validate(dict(option.defaults))
    installed = inspect_admet_ai_installation()
    assets = inspect_admet_ai_model_assets(config.models_dir)
    remedy = "run 'molcascade pins admet-ai' and update cascade/catalog.py"
    assert config.expected_package_code_sha256 == installed["package_code_sha256"], remedy
    assert config.expected_model_manifest_sha256 == assets["model_manifest_sha256"], remedy

    # Every column the panel names has to be one ADMET-AI really emits, or the
    # stage dies on ADMET_AI_ENDPOINT_MISSING after paying for the forward pass.
    import csv

    import admet_ai

    table = Path(admet_ai.__file__).parent / "resources" / "data" / "admet.csv"
    with table.open(encoding="utf-8") as handle:
        known = {row["id"] for row in csv.DictReader(handle)}
    assert {endpoint.output_column for endpoint in config.endpoints} <= known


def test_the_shipped_default_neither_grants_trust_nor_offers_a_useless_head() -> None:
    """Two things the builder must not decide on the user's behalf."""

    from molcascade.cascade.catalog import CRITERIA_BY_ID

    option = next(
        item
        for item in CRITERIA_BY_ID["admet_endpoint"].options
        if item.id == "admet_ai_v2"
    )
    # Loading a Torch checkpoint runs code. Shipping this pre-ticked would move
    # that decision from the user to the catalogue.
    assert option.defaults["allow_unsafe_model_deserialization"] is False
    trust = next(
        field for field in option.thresholds
        if field.name == "allow_unsafe_model_deserialization"
    )
    assert trust.kind == "boolean" and trust.target == "criterion"

    recorded = {entry["endpoint_id"] for entry in option.defaults["endpoints"]}
    gateable = {
        value
        for value, _ in next(
            field for field in option.thresholds if field.name == "endpoint_id"
        ).choices
    }
    # A gate can only reference evidence that was written.
    assert gateable <= recorded
    # ADMET-AI's own held-out R^2 for these two is negative: predicting the
    # training mean would beat them. They are recorded, never gated on.
    assert recorded - gateable == {"half_life", "volume_of_distribution"}
    assert option.gate_defaults["endpoint_id"] in gateable
