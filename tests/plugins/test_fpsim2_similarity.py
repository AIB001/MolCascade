"""The indexed backend has to answer the exact backend's question the same way.

That is the whole burden of proof for offering two engines for one criterion.
If FPSim2 and the exact scan disagree about which reference is nearest, or about
whether a molecule clears a novelty threshold, then choosing an engine for speed
has quietly become choosing a filter, and the cascade someone designed in the
builder is not the cascade that ran.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import APPLICABILITY_V1, DECISION_V1, PARENT_V1
from molcascade.errors import PluginError
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.similarity import RDKitReferenceSimilarityPlugin
from molcascade.plugins.builtin.similarity_fpsim2 import (
    _REFERENCE_INDEX,
    _REFERENCE_SMI,
    Fpsim2ReferenceSimilarityPlugin,
)

pytest.importorskip("FPSim2")

CANDIDATES = [
    ("ethanol", "CCO"),
    ("propanol", "CCCO"),
    ("benzene", "c1ccccc1"),
    ("phenol", "Oc1ccccc1"),
    ("toluene", "Cc1ccccc1"),
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
]

LEADS = [
    {"reference_id": "lead-a-ethanol", "smiles": "CCO"},
    {"reference_id": "lead-b-benzene", "smiles": "c1ccccc1"},
    {"reference_id": "lead-c-phenol", "smiles": "Oc1ccccc1"},
]


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


def _input(root: Path, molecules=CANDIDATES) -> StageInput:
    root.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [_parent(name, smiles) for name, smiles in molecules],
            schema=PARENT_V1.schema,
        ),
        root / "parents.parquet",
    )
    return StageInput(
        ref=ArtifactRef(artifact_id="artifact:sha256:" + "d" * 64, kind=PARENT_V1.id),
        root=root,
        contract_id=PARENT_V1.id,
    )


def _run(plugin, tmp_path: Path, config: dict[str, object], *, name: str, molecules=CANDIDATES):
    staging = tmp_path / name
    return plugin.execute(
        StageRequest(
            stage_id="lead_similarity",
            inputs={"parents": _input(tmp_path / f"{name}-input", molecules)},
            config={"reference_path": None, **config},
        ),
        StageContext(staging),
    ), staging


def _evidence(response, staging: Path) -> dict[str, dict]:
    table = pq.read_table(staging / response.outputs["applicability"].file_paths[0])
    APPLICABILITY_V1.validate(table)
    return {row["parent_id"]: row for row in table.to_pylist()}


def test_indexed_backend_finds_the_same_nearest_reference_as_the_exact_one(
    tmp_path: Path,
) -> None:
    config = {"references": LEADS, "domain_similarity_threshold": 0.5}
    exact, exact_staging = _run(
        RDKitReferenceSimilarityPlugin(), tmp_path, config, name="exact"
    )
    indexed, indexed_staging = _run(
        Fpsim2ReferenceSimilarityPlugin(), tmp_path, config, name="indexed"
    )

    exact_rows = _evidence(exact, exact_staging)
    indexed_rows = _evidence(indexed, indexed_staging)
    assert set(exact_rows) == set(indexed_rows)
    for parent_id, exact_row in exact_rows.items():
        indexed_row = indexed_rows[parent_id]
        assert indexed_row["nearest_reference_id"] == exact_row["nearest_reference_id"], (
            f"{parent_id}: engines disagree about the nearest reference"
        )
        # Single versus double precision, and nothing else, separates these.
        assert indexed_row["nearest_similarity"] == pytest.approx(
            exact_row["nearest_similarity"], abs=1e-6
        )
        assert indexed_row["in_domain"] == exact_row["in_domain"]


def test_the_reference_set_has_one_identity_but_each_engine_has_its_own_method(
    tmp_path: Path,
) -> None:
    config = {"references": LEADS}
    exact, _ = _run(RDKitReferenceSimilarityPlugin(), tmp_path, config, name="exact")
    indexed, _ = _run(Fpsim2ReferenceSimilarityPlugin(), tmp_path, config, name="indexed")

    # Same molecules read by two engines are the same reference set...
    assert exact.metadata["reference_set_id"] == indexed.metadata["reference_set_id"]
    # ...but a float32 coefficient from a bounded tie window is not the same
    # measurement as a float64 one from an exhaustive scan, and the identity
    # has to say so or a cached result from one would be reused for the other.
    assert exact.metadata["method_id"] != indexed.metadata["method_id"]
    assert indexed.metadata["backend"] == "fpsim2"


def test_a_tie_resolves_to_the_same_reference_every_run(tmp_path: Path) -> None:
    # Three references that are all exactly as similar to the candidate, given
    # to the engine in an order that is not their sorted order.
    ties = [
        {"reference_id": "lead-z", "smiles": "CCO"},
        {"reference_id": "lead-a", "smiles": "CCO"},
        {"reference_id": "lead-m", "smiles": "CCO"},
    ]
    seen = set()
    for attempt in range(3):
        response, staging = _run(
            Fpsim2ReferenceSimilarityPlugin(),
            tmp_path,
            {"references": ties},
            name=f"tie-{attempt}",
            molecules=[("ethanol", "CCO")],
        )
        rows = _evidence(response, staging)
        assert rows["ethanol"]["nearest_similarity"] == pytest.approx(1.0)
        seen.add(rows["ethanol"]["nearest_reference_id"])
    # Deterministic, and the same rule the exact backend follows: first by id.
    assert seen == {"lead-a"}


def test_novelty_objective_rejects_the_same_molecules_as_the_exact_backend(
    tmp_path: Path,
) -> None:
    config = {
        "references": LEADS,
        "objective": "novel",
        "novelty_max_similarity": 0.8,
        "failure_action": "reject",
    }
    exact, exact_staging = _run(
        RDKitReferenceSimilarityPlugin(), tmp_path, config, name="exact"
    )
    indexed, indexed_staging = _run(
        Fpsim2ReferenceSimilarityPlugin(), tmp_path, config, name="indexed"
    )

    def survivors(response, staging):
        table = pq.read_table(staging / response.outputs["primary"].file_paths[0])
        return {row["parent_id"] for row in table.to_pylist()}

    kept = survivors(exact, exact_staging)
    assert survivors(indexed, indexed_staging) == kept
    # The test is worthless if the threshold happened to remove nothing.
    assert 0 < len(kept) < len(CANDIDATES)
    assert indexed.metadata["reject_count"] == exact.metadata["reject_count"]


def test_a_warned_molecule_still_emits_a_terminal_pass(tmp_path: Path) -> None:
    response, staging = _run(
        Fpsim2ReferenceSimilarityPlugin(),
        tmp_path,
        {
            "references": LEADS,
            "objective": "analogue",
            "analogue_min_similarity": 0.99,
            "failure_action": "warn",
        },
        name="warn",
        molecules=[("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C")],
    )
    table = pq.read_table(staging / response.outputs["decisions"].file_paths[0])
    DECISION_V1.validate(table)
    outcomes = [row["outcome"] for row in table.to_pylist()]
    assert outcomes == ["WARN", "PASS"]
    assert response.metadata["warning_count"] == 1
    assert response.metadata["reject_count"] == 0


def test_the_index_does_not_survive_the_stage(tmp_path: Path) -> None:
    _, staging = _run(
        Fpsim2ReferenceSimilarityPlugin(), tmp_path, {"references": LEADS}, name="scratch"
    )
    # A leftover .h5 is reference-set-sized and would read as a stage output.
    assert not (staging / _REFERENCE_INDEX).exists()
    assert not (staging / _REFERENCE_SMI).exists()


def test_the_indexed_backend_accepts_reference_sets_the_exact_one_refuses() -> None:
    from molcascade.plugins.builtin.similarity import ReferenceSimilarityConfig
    from molcascade.plugins.builtin.similarity_fpsim2 import Fpsim2SimilarityConfig

    exact = ReferenceSimilarityConfig(reference_path="leads.csv")
    indexed = Fpsim2SimilarityConfig(reference_path="leads.csv")
    assert indexed.max_reference_count > exact.max_reference_count
    # And it stops advising itself as the escape hatch from its own limit.
    assert "FPSim2" in exact.over_limit_hint
    assert "FPSim2" not in indexed.over_limit_hint


def test_empty_input_is_an_error_not_an_empty_shortlist(tmp_path: Path) -> None:
    with pytest.raises(PluginError) as caught:
        _run(
            Fpsim2ReferenceSimilarityPlugin(),
            tmp_path,
            {"references": LEADS},
            name="empty",
            molecules=[],
        )
    assert caught.value.code == "REFERENCE_SIMILARITY_EMPTY_INPUT"
