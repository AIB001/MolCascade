"""The bring-your-own-model interface.

ONNX Runtime is an optional dependency and is not installed here, so the
graph-execution seam (`_load_onnx_session`) is replaced by a fake session in
the tests that exercise inference.  Everything on either side of that seam --
bundle hashing, manifest validation, featurisation, output interpretation, row
alignment and the emitted ``prediction/v1`` -- is the real code.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from molcascade.artifacts import ArtifactRef
from molcascade.chemistry.featurizers import Featurizer, RepresentationSpec
from molcascade.contracts import PARENT_V1, PREDICTION_V1
from molcascade.errors import PluginError
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin import custom_model as module
from molcascade.plugins.builtin.custom_model import (
    MANIFEST_FILENAME,
    CustomModelPredictorPlugin,
    ModelBundleManifest,
    inspect_model_bundle,
)
from molcascade.plugins.manifest import Determinism
from molcascade.plugins.registry import create_builtin_registry

_SMILES = ["CCO", "c1ccccc1O", "not a molecule", "CC(=O)Oc1ccccc1C(=O)O"]


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


def _input(tmp_path: Path, smiles: list[str] | None = None) -> StageInput:
    root = tmp_path / "input"
    root.mkdir(parents=True, exist_ok=True)
    rows = [_parent(f"m{index}", entry) for index, entry in enumerate(smiles or _SMILES)]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=PARENT_V1.schema),
        root / "parents.parquet",
    )
    return StageInput(
        ref=ArtifactRef(artifact_id="artifact:sha256:" + "a" * 64, kind=PARENT_V1.id),
        root=root,
        contract_id=PARENT_V1.id,
    )


def _bundle(
    tmp_path: Path,
    *,
    model_bytes: bytes = b"fake onnx graph",
    filename: str = "model.onnx",
    task: str = "regression",
    extra: str = "",
    n_features: int | None = None,
) -> Path:
    root = tmp_path / "bundle"
    root.mkdir(parents=True, exist_ok=True)
    (root / filename).write_bytes(model_bytes)
    width = 64 + 3 if n_features is None else n_features
    (root / MANIFEST_FILENAME).write_text(
        textwrap.dedent(
            f"""\
            schema_version: 1
            model_name: solubility-demo
            endpoint_id: custom/logS
            task: {task}
            representation:
              blocks:
                - kind: morgan
                  radius: 2
                  n_bits: 64
                - kind: descriptors
                  names: [MolWt, MolLogP, TPSA]
            model:
              file: {filename}
              sha256: {hashlib.sha256(model_bytes).hexdigest()}
              n_features: {width}
              dtype: float32
            """
        )
        + extra,
        encoding="utf-8",
    )
    return root


class _FakeSession:
    """Stands in for onnxruntime.InferenceSession.

    ``predict`` receives the real feature matrix MolCascade computed, so a
    featurisation change that silently altered widths or ordering still shows
    up here as a shape error.
    """

    def __init__(
        self,
        width: int,
        predict: Any,
        *,
        outputs: tuple[str, ...] = ("variable",),
        declared_width: int | None = None,
    ) -> None:
        self._width = width
        self._predict = predict
        self._outputs = outputs
        self._declared = width if declared_width is None else declared_width
        self.seen: list[np.ndarray] = []

    def get_inputs(self) -> list[Any]:
        return [type("I", (), {"name": "X", "shape": [None, self._declared]})()]

    def get_outputs(self) -> list[Any]:
        return [type("O", (), {"name": name, "shape": [None, 1]})() for name in self._outputs]

    def run(self, names: list[str], feed: dict[str, np.ndarray]) -> list[Any]:
        matrix = feed["X"]
        assert matrix.shape[1] == self._width, "featuriser produced the wrong width"
        self.seen.append(matrix)
        return self._predict(matrix, names)


def _install(monkeypatch: pytest.MonkeyPatch, session: Any) -> None:
    monkeypatch.setattr(module, "_load_onnx_session", lambda _path: session)


def _run(
    tmp_path: Path,
    bundle: Path,
    session: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    smiles: list[str] | None = None,
    **overrides: object,
) -> tuple[Any, list[dict[str, Any]]]:
    _install(monkeypatch, session)
    config: dict[str, object] = {
        "bundle_dir": str(bundle),
        "expected_bundle_sha256": inspect_model_bundle(bundle)["bundle_sha256"],
    }
    config.update(overrides)
    staging = tmp_path / "out"
    response = CustomModelPredictorPlugin().execute(
        StageRequest(
            stage_id="custom", inputs={"parents": _input(tmp_path, smiles)}, config=config
        ),
        StageContext(staging),
    )
    table = pq.read_table(staging / "datasets/custom_predictions/part-00000.parquet")
    return response, table.to_pylist()


# ------------------------------------------------------------------ contract


def test_adapter_is_registered_and_never_imports_the_runtime_eagerly() -> None:
    tree = ast.parse(inspect.getsource(module))
    eager = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if isinstance(node.col_offset, int) and node.col_offset == 0
    }
    eager |= {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.col_offset == 0
    }
    assert "onnxruntime" not in eager

    plugin = create_builtin_registry().get("prediction.custom_model@0.1.0")
    assert isinstance(plugin, CustomModelPredictorPlugin)
    assert plugin.descriptor.determinism is Determinism.BEST_EFFORT
    assert plugin.descriptor.output_ports == {
        "primary": PARENT_V1.id,
        "predictions": PREDICTION_V1.id,
    }


def test_missing_runtime_fails_closed_with_an_actionable_message(tmp_path: Path) -> None:
    # Nothing is monkeypatched here, so where onnxruntime is genuinely absent
    # this is the exact message a user will see.
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("onnxruntime is installed, so the unavailable path cannot run")
    with pytest.raises(PluginError) as captured:
        module._load_onnx_session(tmp_path / "model.onnx")
    assert captured.value.code == "CUSTOM_MODEL_RUNTIME_UNAVAILABLE"
    assert "never installs" in (captured.value.hint or "")


# -------------------------------------------------------------- bundle rules


def test_inspection_reports_pastable_digests_without_executing_anything(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    first = inspect_model_bundle(bundle)
    second = inspect_model_bundle(bundle)
    assert first == second
    assert first["file_count"] == 2
    assert first["representation_width"] == first["n_features"] == 67
    assert first["model_id"].startswith("solubility-demo:sha256:")
    assert len(first["bundle_sha256"]) == 64

    # Touching any byte of the bundle changes the digest the config pins.
    (bundle / "model.onnx").write_bytes(b"fake onnx graph!")
    with pytest.raises(PluginError) as captured:
        inspect_model_bundle(bundle)
    assert captured.value.code == "CUSTOM_MODEL_FILE_HASH_MISMATCH"


def test_a_bundle_digest_covers_every_file_not_just_the_model(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    before = inspect_model_bundle(bundle)["bundle_sha256"]
    (bundle / "training_notes.txt").write_text("added later\n", encoding="utf-8")
    assert inspect_model_bundle(bundle)["bundle_sha256"] != before


def test_executable_model_formats_are_refused_by_name(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, filename="model.pkl")
    with pytest.raises(PluginError) as captured:
        inspect_model_bundle(bundle)
    assert captured.value.code == "CUSTOM_MODEL_FORMAT_EXECUTABLE"
    # The error has to be useful, not just correct.
    assert "skl2onnx" in (captured.value.hint or "")
    assert "zipmap" in (captured.value.hint or "")


def test_symlinks_and_subdirectories_are_refused(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (tmp_path / "elsewhere.bin").write_bytes(b"x")
    try:
        (bundle / "link.onnx").symlink_to(tmp_path / "elsewhere.bin")
    except (OSError, NotImplementedError):
        pytest.skip("this platform does not allow creating symlinks")
    with pytest.raises(PluginError) as captured:
        inspect_model_bundle(bundle)
    assert captured.value.code == "CUSTOM_MODEL_BUNDLE_UNREADABLE"

    (bundle / "link.onnx").unlink()
    (bundle / "nested").mkdir()
    with pytest.raises(PluginError) as captured:
        inspect_model_bundle(bundle)
    assert captured.value.code == "CUSTOM_MODEL_BUNDLE_UNREADABLE"


def test_representation_must_match_the_exported_estimator(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, n_features=999)
    with pytest.raises(PluginError) as captured:
        inspect_model_bundle(bundle)
    assert captured.value.code == "CUSTOM_MODEL_MANIFEST_INVALID"
    assert "does not match the exported estimator" in str(captured.value)


def test_duplicate_manifest_keys_are_rejected(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    text = (bundle / MANIFEST_FILENAME).read_text(encoding="utf-8")
    (bundle / MANIFEST_FILENAME).write_text(text + "task: regression\n", encoding="utf-8")
    with pytest.raises(Exception) as captured:
        inspect_model_bundle(bundle)
    assert "duplicate" in str(captured.value).casefold()


def test_model_identity_binds_the_representation_not_only_the_bytes(
    tmp_path: Path,
) -> None:
    """Two bundles with identical model bytes but different features must differ.

    Otherwise a run could attribute predictions computed from one feature
    vector to a model identity established with another.
    """

    first = inspect_model_bundle(_bundle(tmp_path / "a"))["model_id"]
    other = _bundle(tmp_path / "b")
    manifest = (other / MANIFEST_FILENAME).read_text(encoding="utf-8")
    (other / MANIFEST_FILENAME).write_text(
        manifest.replace("radius: 2", "radius: 3"), encoding="utf-8"
    )
    assert inspect_model_bundle(other)["model_id"] != first


# ------------------------------------------------------------------ scoring


def test_predictions_are_emitted_for_every_featurisable_molecule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    session = _FakeSession(67, lambda matrix, _names: [matrix[:, 64:65] / 100.0])
    response, rows = _run(tmp_path, bundle, session, monkeypatch)

    # Four molecules in, one unparseable, so three predictions and four parents.
    assert response.metadata["input_count"] == 4
    assert response.metadata["output_count"] == 4
    assert response.metadata["prediction_count"] == 3
    assert response.metadata["unfeaturizable_count"] == 1
    assert response.metadata["non_finite_prediction_count"] == 0
    assert response.metadata["network_or_download_invoked_by_adapter"] is False
    assert response.metadata["model_deserialization_trust_acknowledged"] is False

    # The rows that survived are the parseable ones, in order, and each score
    # corresponds to its own molecule rather than to a shifted neighbour.
    assert [row["parent_id"] for row in rows] == ["m0", "m1", "m3"]
    assert rows[0]["prediction_mean"] == pytest.approx(46.069 / 100.0, abs=1e-4)
    assert rows[2]["prediction_mean"] == pytest.approx(180.159 / 100.0, abs=1e-4)
    assert {row["endpoint_id"] for row in rows} == {"custom/logS"}
    assert all(row["model_id"].startswith("solubility-demo:sha256:") for row in rows)
    assert all(row["prediction_std"] is None for row in rows)


def test_a_wrong_bundle_digest_stops_the_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    session = _FakeSession(67, lambda matrix, _names: [matrix[:, :1]])
    with pytest.raises(PluginError) as captured:
        _run(
            tmp_path,
            bundle,
            session,
            monkeypatch,
            expected_bundle_sha256="0" * 64,
        )
    assert captured.value.code == "CUSTOM_MODEL_BUNDLE_HASH_MISMATCH"
    assert "model-bundle" in (captured.value.hint or "")


def test_non_finite_scores_produce_no_evidence_rather_than_a_number(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)

    def predict(matrix: np.ndarray, _names: list[str]) -> list[np.ndarray]:
        out = matrix[:, 64:65].copy()
        out[0, 0] = np.nan
        return [out]

    response, rows = _run(tmp_path, bundle, _FakeSession(67, predict), monkeypatch)
    assert response.metadata["non_finite_prediction_count"] == 1
    assert [row["parent_id"] for row in rows] == ["m1", "m3"]


def test_zipmap_classifier_output_is_diagnosed_precisely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path, task="binary_probability")
    session = _FakeSession(67, lambda matrix, _n: [[{0: 0.4, 1: 0.6}] * matrix.shape[0]])
    with pytest.raises(PluginError) as captured:
        _run(tmp_path, bundle, session, monkeypatch)
    assert captured.value.code == "CUSTOM_MODEL_OUTPUT_NOT_TENSOR"
    assert "zipmap" in (captured.value.hint or "").casefold()


def test_a_multi_column_output_must_say_which_column_it_means(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path, task="binary_probability")
    probabilities = lambda matrix, _n: [  # noqa: E731
        np.column_stack([np.full(matrix.shape[0], 0.3), np.full(matrix.shape[0], 0.7)])
    ]
    with pytest.raises(PluginError) as captured:
        _run(tmp_path, bundle, _FakeSession(67, probabilities), monkeypatch)
    assert captured.value.code == "CUSTOM_MODEL_OUTPUT_COLUMN_INVALID"
    assert "column 1" in (captured.value.hint or "")

    # Declaring the column makes it work, and the positive class is used.
    chosen = _bundle(tmp_path / "b", task="binary_probability", extra="  output_column: 1\n")
    _, rows = _run(tmp_path / "b", chosen, _FakeSession(67, probabilities), monkeypatch)
    assert [row["prediction_mean"] for row in rows] == [pytest.approx(0.7)] * len(rows)
    assert len(rows) == 3


def test_a_probability_model_returning_labels_is_caught(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path, task="binary_probability")
    labels = lambda matrix, _n: [np.full((matrix.shape[0], 1), 7.0)]  # noqa: E731
    with pytest.raises(PluginError) as captured:
        _run(tmp_path, bundle, _FakeSession(67, labels), monkeypatch)
    assert captured.value.code == "CUSTOM_MODEL_PROBABILITY_OUT_OF_RANGE"
    assert "label" in (captured.value.hint or "")


def test_graph_width_disagreement_is_caught_before_any_molecule_is_scored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    session = _FakeSession(
        67, lambda *_: pytest.fail("must not reach inference"), declared_width=99
    )
    with pytest.raises(PluginError) as captured:
        _run(tmp_path, bundle, session, monkeypatch)
    assert captured.value.code == "CUSTOM_MODEL_FEATURE_WIDTH_MISMATCH"
    assert session.seen == []


def test_optional_uncertainty_output_flows_into_prediction_std(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(
        tmp_path,
        extra="uncertainty:\n  output_name: spread\n  scale: variance\n",
    )
    session = _FakeSession(
        67,
        lambda matrix, _n: [matrix[:, 64:65] / 100.0, np.full((matrix.shape[0], 1), 0.25)],
        outputs=("variable", "spread"),
    )
    response, rows = _run(tmp_path, bundle, session, monkeypatch)
    assert response.metadata["uncertainty_available"] is True
    # A declared variance is reported as the standard deviation the contract
    # asks for, so 0.25 becomes 0.5 rather than being passed through.
    assert all(row["prediction_std"] == pytest.approx(0.5) for row in rows)


def test_a_row_count_change_from_the_model_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    session = _FakeSession(67, lambda matrix, _n: [matrix[:1, 64:65]])
    with pytest.raises(PluginError) as captured:
        _run(tmp_path, bundle, session, monkeypatch)
    assert captured.value.code == "CUSTOM_MODEL_OUTPUT_COUNT_MISMATCH"


def test_features_reaching_the_model_are_the_declared_representation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The matrix the graph sees must equal what the featuriser alone produces."""

    bundle = _bundle(tmp_path)
    session = _FakeSession(67, lambda matrix, _n: [matrix[:, :1]])
    _run(tmp_path, bundle, session, monkeypatch)

    manifest = ModelBundleManifest.model_validate(
        __import__("molcascade.config.load", fromlist=["load_yaml"]).load_yaml(
            (bundle / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
    )
    expected, failed = Featurizer(manifest.representation).transform(
        _SMILES, dtype=manifest.model.dtype
    )
    assert failed == [2]
    assert np.array_equal(session.seen[0], expected)
    assert session.seen[0].dtype == np.float32


def test_a_float64_bundle_feeds_the_graph_doubles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    manifest = (bundle / MANIFEST_FILENAME).read_text(encoding="utf-8")
    (bundle / MANIFEST_FILENAME).write_text(
        manifest.replace("dtype: float32", "dtype: float64"), encoding="utf-8"
    )
    session = _FakeSession(67, lambda matrix, _n: [matrix[:, :1]])
    _run(tmp_path, bundle, session, monkeypatch)
    assert session.seen[0].dtype == np.float64


def test_batching_does_not_change_the_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    predict = lambda matrix, _n: [matrix[:, 64:65] / 100.0]  # noqa: E731
    whole = _FakeSession(67, predict)
    _, single = _run(tmp_path / "one", bundle, whole, monkeypatch)
    split = _FakeSession(67, predict)
    _, chunked = _run(tmp_path / "two", bundle, split, monkeypatch, batch_size=2)
    assert single == chunked
    assert len(whole.seen) == 1 and len(split.seen) == 2


@pytest.mark.parametrize(
    ("label", "blocks"),
    [
        ("unknown descriptor", [{"kind": "descriptors", "names": ["NoSuchDescriptor"]}]),
        ("repeated descriptor", [{"kind": "descriptors", "names": ["MolWt", "MolWt"]}]),
        ("unknown block kind", [{"kind": "nonexistent"}]),
        ("no blocks at all", []),
    ],
)
def test_representation_spec_rejects_unknown_and_repeated_descriptors(
    label: str, blocks: list[dict[str, object]]
) -> None:
    """``ValidationError``, not ``Exception``.

    Catching the base class would let this pass on a typo in the spec being
    built, on an import that stopped resolving, or on any unrelated failure
    inside pydantic -- which is the opposite of what a rejection test is for.
    """

    with pytest.raises(ValidationError):
        RepresentationSpec.model_validate({"blocks": blocks})
