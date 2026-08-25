"""The ChEMBL structure checker as a gate.

The tests that matter here are the two claims the adapter's docstring makes,
because both are the kind of thing that is easy to assert in prose and wrong in
practice: that a parent registered from SMILES does not collect a geometry
penalty for having no coordinates, and that several advisory findings never add
up into a rejection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import DECISION_V1, PARENT_V1
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.chembl_check import ChemblStructureCheckPlugin

pytest.importorskip("chembl_structure_pipeline")


#: A carbon radical.  Upstream scores "radical not in the known list" at 6,
#: the band it reserves for structures nobody meant to draw.
CARBON_RADICAL = "[CH2]C"

#: Undefined double-bond stereo.  InChI reports it, upstream scores it 2, and
#: it describes a large fraction of any generated library.
UNDEFINED_STEREO = "CC=CC"


def _row(parent_id: str, smiles: str) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "identity_policy_id": "identity:test",
        "parent_smiles": smiles,
        "registration_key": f"key:{parent_id}",
        "stereo_key": None,
        "formula": None,
        "duplicate_count": 1,
    }


def _input(tmp_path: Path, rows: list[dict[str, object]]) -> StageInput:
    root = tmp_path / "input"
    root.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=PARENT_V1.schema),
        root / "parents.parquet",
    )
    return StageInput(
        ref=ArtifactRef(artifact_id="artifact:sha256:" + "4" * 64, kind=PARENT_V1.id),
        root=root,
        contract_id=PARENT_V1.id,
    )


def _run(tmp_path: Path, rows: list[dict[str, object]], **config: object):
    plugin = ChemblStructureCheckPlugin()
    response = plugin.execute(
        StageRequest(
            stage_id="chembl_check",
            inputs={"parents": _input(tmp_path, rows)},
            config=config,
        ),
        StageContext(tmp_path / "out"),
    )
    parents = pq.read_table(tmp_path / "out" / response.outputs["primary"].file_paths[0])
    decisions = pq.read_table(
        tmp_path / "out" / response.outputs["decisions"].file_paths[0]
    )
    PARENT_V1.validate(parents)
    DECISION_V1.validate(decisions)
    return response, parents.to_pylist(), decisions.to_pylist()


def test_ordinary_molecules_registered_from_smiles_are_not_penalised(
    tmp_path: Path,
) -> None:
    """The claim the adapter is built on.

    ``check_molblock`` scores 2D geometry, and a parent registered from SMILES
    carries no conformer.  If ``MolToMolBlock`` did not lay out coordinates on
    the way past, every one of these would collect the zero-coordinate penalty
    and the whole backend would be a molecule shredder.
    """

    rows = [
        _row("ethanol", "CCO"),
        _row("caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
        _row("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        # Fused, stereo-bearing, and the sort of thing a generator emits.
        _row("risperidone", "Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCCC2"),
    ]

    _, parents, decisions = _run(tmp_path, rows)

    assert {row["parent_id"] for row in parents} == {
        "ethanol",
        "caffeine",
        "aspirin",
        "risperidone",
    }
    assert {row["outcome"] for row in decisions} == {"PASS"}
    for row in decisions:
        assert json.loads(row["detail"])["max_penalty"] == 0


def test_a_radical_is_rejected_and_the_reason_names_the_penalty(
    tmp_path: Path,
) -> None:
    _, parents, decisions = _run(
        tmp_path, [_row("clean", "CCO"), _row("radical", CARBON_RADICAL)]
    )

    assert {row["parent_id"] for row in parents} == {"clean"}
    rejected = next(row for row in decisions if row["outcome"] == "REJECT")
    assert rejected["entity_id"] == "radical"
    # The penalty is upstream's published integer, so it belongs in the reason
    # code where a report can group on it.
    assert rejected["reason_code"] == "CHEMBL_STRUCTURE_PENALTY_6"
    assert json.loads(rejected["detail"])["max_penalty"] == 6


def test_undefined_stereo_is_recorded_but_survives_the_default_threshold(
    tmp_path: Path,
) -> None:
    """Advisory findings are chemistry to read, not grounds to reject.

    A generated library is mostly molecules whose stereochemistry nobody has
    committed to yet.  Rejecting at penalty 2 would empty the funnel at tier
    one, so the finding is carried on the PASS row instead.
    """

    response, parents, decisions = _run(tmp_path, [_row("alkene", UNDEFINED_STEREO)])

    assert [row["parent_id"] for row in parents] == ["alkene"]
    decision = decisions[0]
    assert decision["outcome"] == "PASS"
    detail = json.loads(decision["detail"])
    assert detail["max_penalty"] == 2
    assert detail["findings"], "the finding must survive into the record"
    # A molecule kept despite a finding is the number to look at if the
    # threshold is ever argued about.
    assert response.metadata["advisory_count"] == 1


def test_lowering_the_threshold_turns_an_advisory_into_a_rejection(
    tmp_path: Path,
) -> None:
    """The knob is real, and it is the user's to set.

    Some campaigns do want defined stereochemistry throughout.  Pinning both
    directions here stops the default from silently becoming the only
    behaviour the backend has.
    """

    _, parents, decisions = _run(
        tmp_path, [_row("alkene", UNDEFINED_STEREO)], reject_at_penalty=2
    )

    assert parents == []
    assert decisions[0]["reason_code"] == "CHEMBL_STRUCTURE_PENALTY_2"


def test_several_advisories_never_add_up_to_a_rejection(tmp_path: Path) -> None:
    """Severity is the worst finding, never the sum.

    Upstream documents the penalty as the seriousness of one issue on a 0-9
    scale.  Cisplatin collects two separate InChI advisories worth 2 each; a
    backend that summed them would produce a 4, and on a stricter setting a
    6 -- a hard reject manufactured out of arithmetic rather than chemistry.
    """

    _, parents, decisions = _run(tmp_path, [_row("cisplatin", "N.N.Cl[Pt]Cl")])

    detail = json.loads(decisions[0]["detail"])
    assert len(detail["findings"]) >= 2
    assert detail["max_penalty"] == max(f["penalty"] for f in detail["findings"])
    assert decisions[0]["outcome"] == "PASS"
    assert [row["parent_id"] for row in parents] == ["cisplatin"]


def test_the_policy_id_binds_the_threshold_and_both_toolkit_versions(
    tmp_path: Path,
) -> None:
    """Two runs that filtered differently must not claim the same policy.

    The InChI verdicts come out of RDKit's bundled InChI build, so the RDKit
    version is part of what produced the answer -- not background detail.
    """

    rows = [_row("ethanol", "CCO")]
    default, _, _ = _run(tmp_path / "a", rows)
    strict, _, _ = _run(tmp_path / "b", rows, reject_at_penalty=2)

    assert default.metadata["policy_id"] != strict.metadata["policy_id"]
    assert default.metadata["backend"] == "chembl_structure_pipeline"
    assert default.metadata["rdkit_version"]
    assert default.metadata["backend_version"] != "unknown"


def test_an_empty_input_is_a_failure_rather_than_an_empty_pass(
    tmp_path: Path,
) -> None:
    with pytest.raises(Exception) as caught:
        _run(tmp_path, [])
    assert "CHEMBL_CHECK_EMPTY_INPUT" in str(getattr(caught.value, "code", caught.value))
