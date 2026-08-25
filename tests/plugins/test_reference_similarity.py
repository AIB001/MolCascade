from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import APPLICABILITY_V1, DECISION_V1, PARENT_V1
from molcascade.errors import PluginError
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.similarity import RDKitReferenceSimilarityPlugin


def _parent(parent_id: str, smiles: str) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "identity_policy_id": "identity:test",
        "parent_smiles": smiles,
        "registration_key": f"key:{parent_id}",
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
                _parent("benzene", "c1ccccc1"),
            ],
            schema=PARENT_V1.schema,
        ),
        root / "parents.parquet",
    )
    return StageInput(
        ref=ArtifactRef(artifact_id="artifact:sha256:" + "d" * 64, kind=PARENT_V1.id),
        root=root,
        contract_id=PARENT_V1.id,
    )


def _run(tmp_path: Path, config: dict[str, object]):
    return RDKitReferenceSimilarityPlugin().execute(
        StageRequest(
            stage_id="lead_similarity",
            inputs={"parents": _input(tmp_path)},
            config={"reference_path": None, **config},
        ),
        StageContext(tmp_path / "out"),
    )


def test_similarity_emits_exact_nearest_reference_and_domain(tmp_path: Path) -> None:
    response = _run(
        tmp_path,
        {
            "references": [
                {"reference_id": "lead-ethanol", "smiles": "CCO"},
                {"reference_id": "lead-benzene", "smiles": "c1ccccc1"},
            ],
            "domain_similarity_threshold": 0.5,
        },
    )
    evidence = pq.read_table(
        tmp_path / "out" / response.outputs["applicability"].file_paths[0]
    )
    decisions = pq.read_table(
        tmp_path / "out" / response.outputs["decisions"].file_paths[0]
    )
    APPLICABILITY_V1.validate(evidence)
    DECISION_V1.validate(decisions)
    rows = {row["parent_id"]: row for row in evidence.to_pylist()}
    assert rows["ethanol"]["nearest_reference_id"] == "lead-ethanol"
    assert rows["ethanol"]["nearest_similarity"] == pytest.approx(1.0)
    assert rows["benzene"]["nearest_reference_id"] == "lead-benzene"
    assert response.metadata["comparison_count"] == 6


def test_novelty_objective_can_reject_close_analogues(tmp_path: Path) -> None:
    response = _run(
        tmp_path,
        {
            "references": [{"reference_id": "lead", "smiles": "CCO"}],
            "objective": "novel",
            "novelty_max_similarity": 0.8,
            "failure_action": "reject",
        },
    )
    parents = pq.read_table(tmp_path / "out" / response.outputs["primary"].file_paths[0])
    assert "ethanol" not in {row["parent_id"] for row in parents.to_pylist()}
    assert response.metadata["reject_count"] == 1


def test_file_reference_hash_is_verified(tmp_path: Path) -> None:
    reference = tmp_path / "leads.csv"
    reference.write_text("id,smiles\nlead,CCO\n", encoding="utf-8")
    with pytest.raises(PluginError) as caught:
        RDKitReferenceSimilarityPlugin().execute(
            StageRequest(
                stage_id="similarity",
                inputs={"parents": _input(tmp_path)},
                config={
                    "reference_path": str(reference),
                    "reference_sha256": "0" * 64,
                },
            ),
            StageContext(tmp_path / "out"),
        )
    assert caught.value.code == "REFERENCE_SET_HASH_MISMATCH"


def test_exact_backend_refuses_excessive_comparisons(tmp_path: Path) -> None:
    with pytest.raises(PluginError) as caught:
        _run(
            tmp_path,
            {
                "references": [
                    {"reference_id": "one", "smiles": "CCO"},
                    {"reference_id": "two", "smiles": "CCN"},
                ],
                "max_total_comparisons": 3,
            },
        )
    assert caught.value.code == "REFERENCE_SIMILARITY_BUDGET_EXCEEDED"


def test_reference_set_identity_is_independent_of_file_location(tmp_path: Path) -> None:
    contents = b"id,smiles\nlead,CCO\n"
    digest = hashlib.sha256(contents).hexdigest()
    identities: list[str] = []
    for index, directory_name in enumerate(("first", "relocated"), start=1):
        directory = tmp_path / directory_name
        directory.mkdir()
        reference = directory / "leads.csv"
        reference.write_bytes(contents)
        response = RDKitReferenceSimilarityPlugin().execute(
            StageRequest(
                stage_id=f"similarity_{index}",
                inputs={"parents": _input(directory)},
                config={
                    "reference_path": str(reference),
                    "reference_sha256": digest,
                },
            ),
            StageContext(directory / "out"),
        )
        identities.append(str(response.metadata["reference_set_id"]))

    assert identities[0] == identities[1]
