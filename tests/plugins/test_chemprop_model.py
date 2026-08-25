"""The bring-your-own-Chemprop-checkpoint interface.

Unlike the ONNX adapter next door, the runtime here *is* installed, so nothing
is faked: these tests train nothing but they build real Chemprop models, save
real checkpoints, and run real inference through the plugin.  Building an
untrained network costs about a tenth of a second where training even a toy one
costs half a minute, and an untrained network exercises every code path the
adapter has -- it just predicts noise, which no assertion here depends on.

Importing Chemprop is the expensive part (roughly half a minute), so the
checkpoints are built once per module.
"""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import PARENT_V1, PREDICTION_V1
from molcascade.errors import PluginError
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.chemprop_model import (
    MANIFEST_FILENAME,
    ChempropCheckpointPlugin,
    inspect_chemprop_bundle,
)
from molcascade.plugins.manifest import Determinism
from molcascade.plugins.registry import create_builtin_registry

pytest.importorskip("chemprop", reason="chemprop is not installed in this environment")

_SMILES = ["CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "CCN", "CCCC"]


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


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory: pytest.TempPathFactory) -> dict[str, bytes]:
    """One saved checkpoint per head shape, built once and reused as bytes."""

    from chemprop import models, nn

    root = tmp_path_factory.mktemp("checkpoints")
    saved: dict[str, bytes] = {}
    for label, predictor in (
        ("regression", nn.RegressionFFN(input_dim=32, n_tasks=1)),
        ("regression_3", nn.RegressionFFN(input_dim=32, n_tasks=3)),
        ("binary", nn.BinaryClassificationFFN(input_dim=32, n_tasks=1)),
    ):
        model = models.MPNN(
            nn.BondMessagePassing(d_h=32, depth=2),
            nn.MeanAggregation(),
            predictor,
            batch_norm=False,
        )
        path = root / f"{label}.pt"
        models.save_model(path, model)
        saved[label] = path.read_bytes()
    return saved


def _bundle(
    tmp_path: Path,
    checkpoints: dict[str, bytes],
    *,
    kind: str = "regression",
    task: str = "regression",
    filename: str = "model.pt",
    n_tasks: int = 1,
    task_index: int = 0,
    payload: bytes | None = None,
    sha256: str | None = None,
) -> Path:
    root = tmp_path / "bundle"
    root.mkdir(parents=True, exist_ok=True)
    body = checkpoints[kind] if payload is None else payload
    (root / filename).write_bytes(body)
    (root / MANIFEST_FILENAME).write_text(
        textwrap.dedent(
            f"""\
            schema_version: 1
            model_name: solubility-demo
            endpoint_id: custom/logS
            task: {task}
            units: log10 mol/L
            model:
              file: {filename}
              sha256: {sha256 or hashlib.sha256(body).hexdigest()}
              format: chemprop_pt
              n_tasks: {n_tasks}
              task_index: {task_index}
            """
        ),
        encoding="utf-8",
    )
    return root


def _run(
    bundle: Path,
    tmp_path: Path,
    *,
    smiles: list[str] | None = None,
    **overrides: object,
) -> tuple[object, Path]:
    config: dict[str, object] = {
        "bundle_dir": str(bundle),
        "allow_unsafe_model_deserialization": True,
        "batch_size": 4,
    }
    config.update(overrides)
    if "expected_bundle_sha256" not in config:
        # Inspected lazily: a test that deliberately breaks the bundle must
        # reach the plugin, not fail inside this helper.
        config["expected_bundle_sha256"] = inspect_chemprop_bundle(bundle)["bundle_sha256"]
    staging = tmp_path / "staging"
    response = ChempropCheckpointPlugin().execute(
        StageRequest(
            stage_id="predict",
            inputs={"parents": _input(tmp_path, smiles)},
            config=config,
        ),
        StageContext(staging),
    )
    return response, staging


def _predictions(staging: Path) -> pa.Table:
    return pq.read_table(staging / "datasets/chemprop_predictions/part-00000.parquet")


# --------------------------------------------------------------------------
# The bundle
# --------------------------------------------------------------------------


def test_the_bundle_digest_covers_every_file(tmp_path: Path, checkpoints: dict) -> None:
    bundle = _bundle(tmp_path, checkpoints)
    first = inspect_chemprop_bundle(bundle)

    assert first["backend"] == "chemprop"
    assert first["model_name"] == "solubility-demo"
    assert first["task"] == "regression"
    assert first["units"] == "log10 mol/L"
    assert len(first["bundle_sha256"]) == 64
    assert first["model_id"].startswith("solubility-demo:sha256:")

    # A file nobody declared still changes the digest: the pin is over the
    # directory, not over the checkpoint the manifest happens to name.
    (bundle / "notes.txt").write_text("added later", encoding="utf-8")
    assert inspect_chemprop_bundle(bundle)["bundle_sha256"] != first["bundle_sha256"]


def test_inspection_does_not_import_the_runtime(tmp_path: Path, checkpoints: dict) -> None:
    """The digest has to be computable before deciding to run the file it pins.

    Loading a Chemprop checkpoint executes code, so an inspection that imported
    Chemprop to describe a bundle would defeat the point of inspecting one.
    """

    import sys

    bundle = _bundle(tmp_path, checkpoints)
    torch_modules = {name for name in sys.modules if name.startswith("torch")}
    inspect_chemprop_bundle(bundle)
    assert {name for name in sys.modules if name.startswith("torch")} == torch_modules


def test_a_checkpoint_that_does_not_match_its_digest_is_refused(
    tmp_path: Path, checkpoints: dict
) -> None:
    bundle = _bundle(tmp_path, checkpoints, sha256="a" * 64)
    with pytest.raises(PluginError) as caught:
        inspect_chemprop_bundle(bundle)
    assert caught.value.code == "CHEMPROP_FILE_HASH_MISMATCH"


def test_a_bundle_without_a_manifest_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "model.pt").write_bytes(b"bytes")
    with pytest.raises(PluginError) as caught:
        inspect_chemprop_bundle(root)
    assert caught.value.code == "CHEMPROP_MANIFEST_MISSING"


def test_a_task_index_outside_the_declared_head_is_refused(
    tmp_path: Path, checkpoints: dict
) -> None:
    """Catching this in the manifest keeps the error about the endpoint.

    Left to inference it would surface as an index error deep in a tensor, long
    after the run started.
    """

    bundle = _bundle(tmp_path, checkpoints, n_tasks=2, task_index=5)
    with pytest.raises(PluginError) as caught:
        inspect_chemprop_bundle(bundle)
    assert caught.value.code == "CHEMPROP_MANIFEST_INVALID"


def test_a_bundle_directory_may_not_be_nested(tmp_path: Path, checkpoints: dict) -> None:
    bundle = _bundle(tmp_path, checkpoints)
    (bundle / "extra").mkdir()
    with pytest.raises(PluginError) as caught:
        inspect_chemprop_bundle(bundle)
    assert caught.value.code == "CHEMPROP_BUNDLE_UNREADABLE"


# --------------------------------------------------------------------------
# Trust
# --------------------------------------------------------------------------


def test_the_stage_refuses_to_start_without_an_explicit_trust_acknowledgement(
    tmp_path: Path, checkpoints: dict
) -> None:
    """A digest proves which bytes will run, not that running them is safe.

    ``chemprop.models.load_model`` calls ``torch.load(weights_only=False)``, so
    deserializing a checkpoint executes code from it.  The ONNX adapter refuses
    ``.pt`` outright for this reason; this one cannot, so it asks instead.
    """

    bundle = _bundle(tmp_path, checkpoints)
    with pytest.raises(PluginError) as caught:
        _run(bundle, tmp_path, allow_unsafe_model_deserialization=False)
    assert caught.value.code == "CHEMPROP_MODEL_TRUST_REQUIRED"


def test_trust_is_checked_before_the_bundle_is_read(
    tmp_path: Path, checkpoints: dict
) -> None:
    """An untrusted stage must not touch the checkpoint at all.

    If the digest were verified first, a bundle that failed to hash would report
    a hashing problem and hide the fact that it was never going to be allowed to
    run in the first place.
    """

    bundle = _bundle(tmp_path, checkpoints)
    (bundle / "model.pt").unlink()
    with pytest.raises(PluginError) as caught:
        _run(
            bundle,
            tmp_path,
            allow_unsafe_model_deserialization=False,
            expected_bundle_sha256="0" * 64,
        )
    assert caught.value.code == "CHEMPROP_MODEL_TRUST_REQUIRED"


def test_a_bundle_that_does_not_match_the_pinned_digest_is_refused(
    tmp_path: Path, checkpoints: dict
) -> None:
    bundle = _bundle(tmp_path, checkpoints)
    with pytest.raises(PluginError) as caught:
        _run(bundle, tmp_path, expected_bundle_sha256="0" * 64)
    assert caught.value.code == "CHEMPROP_BUNDLE_HASH_MISMATCH"


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------


def test_every_parent_is_scored_and_passed_through(
    tmp_path: Path, checkpoints: dict
) -> None:
    response, staging = _run(tmp_path=tmp_path, bundle=_bundle(tmp_path, checkpoints))

    assert response.metadata["input_count"] == len(_SMILES)
    assert response.metadata["prediction_count"] == len(_SMILES)
    assert response.metadata["unfeaturizable_count"] == 0
    assert response.metadata["backend"] == "chemprop-checkpoint"
    assert response.metadata["units"] == "log10 mol/L"
    assert response.metadata["model_deserialization_trust_acknowledged"] is True
    assert response.metadata["network_or_download_invoked_by_adapter"] is False

    table = _predictions(staging)
    assert table.schema == PREDICTION_V1.schema
    assert table.num_rows == len(_SMILES)
    assert set(table.column("endpoint_id").to_pylist()) == {"custom/logS"}
    assert set(table.column("parent_id").to_pylist()) == {
        f"m{index}" for index in range(len(_SMILES))
    }
    # No uncertainty head, so the spread columns are honestly empty rather than
    # filled with a number that would read as a measured one.
    assert set(table.column("prediction_std").to_pylist()) == {None}


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 8])
def test_no_molecule_is_dropped_at_any_batch_boundary(
    tmp_path: Path, checkpoints: dict, batch_size: int
) -> None:
    """Chemprop's dataloader drops a trailing batch of one by default.

    That protects batch-norm statistics while training and silently loses a
    molecule while predicting -- and which molecule depends only on how the
    count divides, so it would pass every test that used a convenient number.
    Five molecules over these batch sizes hits the remainder-of-one case twice.
    """

    _, staging = _run(
        tmp_path=tmp_path, bundle=_bundle(tmp_path, checkpoints), batch_size=batch_size
    )
    assert _predictions(staging).num_rows == len(_SMILES)


def test_an_unparseable_smiles_is_missing_evidence_not_a_failed_run(
    tmp_path: Path, checkpoints: dict
) -> None:
    """Chemprop raises on a SMILES RDKit rejects; a screen must not die on one.

    The molecule is passed through as a parent and simply has no prediction,
    which the downstream gate turns into an explicit reject.
    """

    response, staging = _run(
        _bundle(tmp_path, checkpoints),
        tmp_path,
        smiles=["CCO", "not a molecule", "CCN"],
    )

    assert response.metadata["input_count"] == 3
    assert response.metadata["prediction_count"] == 2
    assert response.metadata["unfeaturizable_count"] == 1
    assert _predictions(staging).column("parent_id").to_pylist() == ["m0", "m2"]


def test_a_multitask_checkpoint_scores_the_declared_column(
    tmp_path: Path, checkpoints: dict
) -> None:
    """One tier gates on one endpoint, chosen deliberately.

    A three-task model must not have its columns averaged, and must not
    default to column zero just because that is what indexing does.
    """

    bundle = _bundle(tmp_path, checkpoints, kind="regression_3", n_tasks=3, task_index=2)
    _, staging = _run(bundle=bundle, tmp_path=tmp_path)
    first = _predictions(staging).column("prediction_mean").to_pylist()

    other = _bundle(
        tmp_path / "other", checkpoints, kind="regression_3", n_tasks=3, task_index=0
    )
    _, other_staging = _run(bundle=other, tmp_path=tmp_path / "other")
    assert first != _predictions(other_staging).column("prediction_mean").to_pylist()


def test_a_checkpoint_shaped_differently_than_the_manifest_says_is_refused(
    tmp_path: Path, checkpoints: dict
) -> None:
    """The manifest's n_tasks is a claim about the file, and it is checked.

    A checkpoint swapped for a differently-shaped one would otherwise keep
    scoring, with task_index quietly selecting a different endpoint.
    """

    bundle = _bundle(tmp_path, checkpoints, kind="regression_3", n_tasks=1, task_index=0)
    with pytest.raises(PluginError) as caught:
        _run(bundle=bundle, tmp_path=tmp_path)
    assert caught.value.code == "CHEMPROP_TASK_COUNT_MISMATCH"


def test_declaring_the_wrong_head_kind_is_refused(
    tmp_path: Path, checkpoints: dict
) -> None:
    """A regression head's raw output is not a probability.

    Declaring ``binary_probability`` over a regression checkpoint would
    reinterpret every number the stage emits, and a threshold meant for 0-1
    would then cut on a scale that has no such bounds.
    """

    bundle = _bundle(tmp_path, checkpoints, task="binary_probability")
    with pytest.raises(PluginError) as caught:
        _run(bundle=bundle, tmp_path=tmp_path)
    assert caught.value.code == "CHEMPROP_TASK_KIND_MISMATCH"


def test_a_classification_head_emits_probabilities(
    tmp_path: Path, checkpoints: dict
) -> None:
    bundle = _bundle(tmp_path, checkpoints, kind="binary", task="binary_probability")
    _, staging = _run(bundle=bundle, tmp_path=tmp_path)
    values = _predictions(staging).column("prediction_mean").to_pylist()
    assert values and all(0.0 <= value <= 1.0 for value in values)


def test_an_empty_input_is_an_error_not_an_empty_shortlist(
    tmp_path: Path, checkpoints: dict
) -> None:
    root = tmp_path / "input"
    root.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([], schema=PARENT_V1.schema), root / "parents.parquet"
    )
    bundle = _bundle(tmp_path, checkpoints)
    request = StageRequest(
        stage_id="predict",
        config={
            "bundle_dir": str(bundle),
            "expected_bundle_sha256": inspect_chemprop_bundle(bundle)["bundle_sha256"],
            "allow_unsafe_model_deserialization": True,
        },
        inputs={
            "parents": StageInput(
                ref=ArtifactRef(
                    artifact_id="artifact:sha256:" + "a" * 64, kind=PARENT_V1.id
                ),
                root=root,
                contract_id=PARENT_V1.id,
            )
        },
    )
    with pytest.raises(PluginError) as caught:
        ChempropCheckpointPlugin().execute(request, StageContext(tmp_path / "s"))
    assert caught.value.code == "CHEMPROP_EMPTY_INPUT"


# --------------------------------------------------------------------------
# Identity and registration
# --------------------------------------------------------------------------


def test_the_model_id_changes_with_the_bytes_that_produced_the_numbers(
    tmp_path: Path, checkpoints: dict
) -> None:
    """``prediction/v1`` promises the identifier names how a number was made.

    The featurizer here belongs to Chemprop rather than to MolCascade, so the
    release is part of the identity in a way it is not for an ONNX bundle.
    """

    bundle = _bundle(tmp_path, checkpoints)
    baseline = inspect_chemprop_bundle(bundle)["model_id"]

    other = _bundle(tmp_path / "other", checkpoints, kind="binary", task="binary_probability")
    assert inspect_chemprop_bundle(other)["model_id"] != baseline

    reindexed = _bundle(tmp_path / "third", checkpoints, kind="regression_3", n_tasks=3, task_index=1)
    assert inspect_chemprop_bundle(reindexed)["model_id"] != baseline


def test_the_plugin_is_registered_and_declares_best_effort_determinism() -> None:
    plugin = create_builtin_registry().get("prediction.chemprop_checkpoint@0.1.0")
    assert plugin.descriptor.outputs == (PARENT_V1.id, PREDICTION_V1.id)
    # Torch float kernels differ between builds, CPU instruction sets and
    # devices, so the same checkpoint is not bit-identical everywhere.
    assert plugin.descriptor.determinism is Determinism.BEST_EFFORT
