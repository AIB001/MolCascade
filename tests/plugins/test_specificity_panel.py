"""How much of a docking score was about this pocket rather than about docking.

A score is a number a program produced against one receptor, and nothing in it
says whether the same molecule would have scored just as well against a pocket
it has no reason to bind.  These tests pin down that comparison: the arithmetic
of the gap, the count of decoys that beat the target, and -- more importantly --
the several ways the comparison can be meaningless while still producing a
plausible number.  Subtracting a Vinardo score from a Vina one, subtracting a
score whose direction runs the other way, averaging over a panel that only half
answered, or measuring the wrong engine's score because two of them were bound:
each of those yields a float, and each of those floats is wrong.

The stand-in binary is a fake of the *protocol*, as in ``test_unidock.py``: it
parses the real flags, reads the real ``--ligand_index``, and writes the real
``REMARK VINA RESULT`` output.  Its score for a ligand is that ligand's own atom
count plus a per-receptor offset it reads out of the receptor file, which is the
trick that makes both joins checkable at once -- the test can recompute which
number belongs to which molecule *and* which receptor produced it, so a delta
built from a neighbour's score or from the wrong pocket is visible as a number
rather than hidden behind a fixture of invented values.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import DERIVED_METRIC_V1, DOCKING_SCORE_V1, PARENT_V1
from molcascade.errors import PluginError
from molcascade.parallel import StageResources
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.specificity_panel import (
    SpecificityPanelConfig,
    SpecificityPanelPlugin,
)

pytest.importorskip("rdkit")

#: meeko converts the stored pose into the PDBQT the engine re-docks.  It is an
#: optional extra, so every test that reaches the engine says so; the ones about
#: refusing a config never get that far and stay armed either way.
_needs_meeko = pytest.mark.skipif(
    importlib.util.find_spec("meeko") is None,
    reason="meeko is an optional docking extra and is not installed",
)

#: Distinct atom counts on purpose -- the fake engine's score is a function of
#: atom count, so a score landing on the wrong molecule shows up as a number.
_SMILES = {
    "p1": "c1ccccc1",
    "p2": "CC(=O)Oc1ccccc1C(=O)O",
    "p3": "Cc1ccccc1",
}
#: Deliberately not round, and not equal to any atom count: a target score that
#: happened to coincide with a panel score would make ``delta == 0`` for the
#: wrong reason.
_TARGET_SCORES = {"p1": -7.5, "p2": -9.25, "p3": -5.75}
_RECEPTOR = "receptor:sha256:" + "c" * 64


# --------------------------------------------------------------- the fake engine

_FAKE_ENGINE = """\
import json
import sys
from pathlib import Path

arguments = {}
key = None
for token in sys.argv[1:]:
    if token.startswith("--"):
        key = token[2:]
        arguments[key] = []
    else:
        arguments[key].append(token)

receptor = Path(arguments["receptor"][0])
behaviour = {"offset": 0.0}
for line in receptor.read_text().splitlines():
    if line.startswith("REMARK PANEL "):
        behaviour.update(json.loads(line[len("REMARK PANEL "):]))

record_path = Path(_RECORD)
record = json.loads(record_path.read_text()) if record_path.exists() else {"calls": []}
record["calls"].append(arguments)
record_path.write_text(json.dumps(record))

if behaviour.get("exit_code"):
    sys.stdout.write(behaviour.get("message", "panel receptor parse error near residue 88"))
    sys.exit(behaviour["exit_code"])

paths = [line for line in Path(arguments["ligand_index"][0]).read_text().split("\\n") if line]
output_dir = Path(arguments["dir"][0])
offset = float(behaviour["offset"])
skip = set(behaviour.get("skip", []))

for ordinal, text_path in enumerate(paths):
    if ordinal in skip:
        continue
    path = Path(text_path)
    text = path.read_text()
    atoms = [line for line in text.splitlines() if line.startswith(("ATOM", "HETATM"))]
    score = -float(len(atoms)) + offset
    (output_dir / (path.stem + "_out.pdbqt")).write_text(
        "MODEL 1\\nREMARK VINA RESULT:  %.3f      0.000      0.000\\n%sENDMDL\\n" % (score, text)
    )
"""


class _Engine(NamedTuple):
    """The fake binary and the file it writes its argv into."""

    path: Path
    record_path: Path


@pytest.fixture
def engine(tmp_path: Path) -> _Engine:
    """Write the fake ``unidock`` and return it beside its call record."""

    executable = tmp_path / "unidock"
    record_path = tmp_path / "record.json"
    executable.write_text(
        f"#!{sys.executable}\n_RECORD = {json.dumps(str(record_path))}\n{_FAKE_ENGINE}",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return _Engine(executable, record_path)


def _calls(engine: _Engine) -> list[dict[str, Any]]:
    if not engine.record_path.exists():
        return []
    calls = json.loads(engine.record_path.read_text(encoding="utf-8"))["calls"]
    return [dict(call) for call in calls]


# ------------------------------------------------------------------- the panel


def _receptor(
    tmp_path: Path,
    name: str,
    *,
    offset: float = 0.0,
    skip: tuple[int, ...] = (),
    exit_code: int = 0,
    size: float = 20.0,
) -> dict[str, object]:
    """One panel entry, handed over already prepared.

    ``prepared_pdbqt_path`` rather than ``structure_path`` on purpose: what is
    under test here is the comparison, not meeko's opinion of a stub protein,
    and this is the one path where the bytes the engine opens are exactly the
    bytes the operator supplied -- which is also where the fake reads its
    behaviour from.
    """

    behaviour: dict[str, object] = {"offset": offset}
    if skip:
        behaviour["skip"] = list(skip)
    if exit_code:
        behaviour["exit_code"] = exit_code
    path = tmp_path / f"panel-{name}.pdbqt"
    path.write_text(
        f"REMARK PANEL {json.dumps(behaviour, sort_keys=True)}\nATOM      1  N   ALA A   1\n",
        encoding="utf-8",
    )
    return {
        "name": name,
        "prepared_pdbqt_path": str(path),
        "center_x": 1.0,
        "center_y": -2.5,
        "center_z": 3.0,
        "size_x": size,
        "size_y": size,
        "size_z": size,
    }


def _atom_count(molblock: str) -> int:
    """What the fake engine will count, computed the same way the adapter feeds it."""

    from molcascade.plugins.builtin.docking.common import ligand_pdbqt

    pdbqt = ligand_pdbqt(molblock)
    assert pdbqt is not None
    return sum(1 for line in pdbqt.splitlines() if line.startswith(("ATOM", "HETATM")))


def _panel_score(molblock: str, offset: float) -> float:
    return -float(_atom_count(molblock)) + offset


# ------------------------------------------------------------------- the inputs


def _molblock(smiles: str, *, seed: int = 7) -> str:
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    rdDistGeom.EmbedMolecule(molecule, params)
    rdForceFieldHelpers.MMFFOptimizeMolecule(molecule, mmffVariant="MMFF94s", maxIters=800)
    return str(Chem.MolToMolBlock(molecule))


def _parent(parent_id: str) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "identity_policy_id": "identity:test",
        "parent_smiles": _SMILES[parent_id],
        "registration_key": f"registration:{parent_id}",
        "stereo_key": None,
        "formula": None,
        "duplicate_count": 1,
    }


def _pose(
    parent_id: str,
    *,
    molblock: str | None,
    score: float | None = None,
    pose_rank: int = 0,
    engine_id: str = "unidock",
    receptor_id: str = _RECEPTOR,
    score_kind: str = "VINA_KCAL_MOL",
    direction: str = "LOWER_STRONGER",
) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "engine_id": engine_id,
        "receptor_id": receptor_id,
        "pose_rank": pose_rank,
        "score": _TARGET_SCORES[parent_id] if score is None else score,
        "score_kind": score_kind,
        "direction": direction,
        "secondary_score": None,
        "pose_molblock": molblock,
        "method_id": f"{engine_id}:test",
    }


def _dataset(
    tmp_path: Path,
    name: str,
    contract: Any,
    rows: list[dict[str, object]],
    *,
    artifact_hex: str,
) -> StageInput:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=contract.schema), root / "part.parquet")
    return StageInput(
        ref=ArtifactRef(artifact_id="artifact:sha256:" + artifact_hex * 64, kind=contract.id),
        root=root,
        contract_id=contract.id,
    )


def _run(
    tmp_path: Path,
    engine: _Engine,
    panel: list[dict[str, object]],
    poses: list[dict[str, object]],
    *,
    parents: list[str] = ["p1", "p2", "p3"],  # noqa: B006 -- read-only, and named in the id
    settings: dict[str, object] | None = None,
    suffix: str = "",
) -> tuple[Path, Any]:
    config: dict[str, object] = {
        "executable": str(engine.path),
        "panel": panel,
        "timeout_per_molecule_seconds": 60.0,
    }
    config.update(settings or {})
    output = tmp_path / f"out{suffix}"
    response = SpecificityPanelPlugin().execute(
        StageRequest(
            stage_id=f"specificity_panel{suffix}",
            inputs={
                "parents": _dataset(
                    tmp_path,
                    f"parents{suffix}",
                    PARENT_V1,
                    [_parent(key) for key in parents],
                    artifact_hex="a",
                ),
                "evidence": _dataset(
                    tmp_path, f"poses{suffix}", DOCKING_SCORE_V1, poses, artifact_hex="b"
                ),
            },
            config=config,
        ),
        StageContext(output, StageResources(devices=("cuda:0",), shard_rows=1_000)),
    )
    return output, response


def _metrics(output: Path, response: Any) -> dict[tuple[str, str], dict[str, object]]:
    table = pa.concat_tables(
        [pq.read_table(output / path) for path in response.outputs["derived_metrics"].file_paths]
    )
    DERIVED_METRIC_V1.validate(table)
    return {(str(row["parent_id"]), str(row["metric_id"])): row for row in table.to_pylist()}


# --------------------------------------------------------------- the comparison


@_needs_meeko
def test_the_gap_is_the_target_score_minus_its_background(tmp_path: Path, engine: _Engine) -> None:
    """Two decoys, three molecules, and every number recomputed from the fake.

    The offsets differ, so a delta built from one receptor's score twice, or
    from the mean of the wrong pair, does not survive; the atom counts differ,
    so a delta built from a neighbour's molecule does not either.
    """

    blocks = {key: _molblock(value) for key, value in _SMILES.items()}
    output, response = _run(
        tmp_path,
        engine,
        [_receptor(tmp_path, "alpha", offset=0.0), _receptor(tmp_path, "beta", offset=3.25)],
        [_pose(key, molblock=blocks[key]) for key in _SMILES],
    )
    rows = _metrics(output, response)

    for key, block in blocks.items():
        alpha = _panel_score(block, 0.0)
        beta = _panel_score(block, 3.25)
        target = _TARGET_SCORES[key]

        delta = rows[(key, "delta_score")]
        assert delta["units"] == "KCAL_PER_MOL"
        assert delta["direction"] == "LOWER_BETTER"
        assert delta["status"] == "OK"
        assert delta["status_detail"] is None
        assert float(delta["value"]) == pytest.approx(target - (alpha + beta) / 2, abs=1e-6)

        wins = rows[(key, "panel_win_count")]
        assert wins["units"] == "COUNT"
        assert wins["direction"] == "LOWER_BETTER"
        assert wins["status"] == "OK"
        assert float(wins["value"]) == float(sum(1 for score in (alpha, beta) if score <= target))

        # ...and the delta is traceable back to which pocket produced what.
        source = json.loads(str(delta["source_json"]))
        assert source["panel_scores"] == {
            "alpha": pytest.approx(alpha, abs=1e-6),
            "beta": pytest.approx(beta, abs=1e-6),
        }
        assert source["target_score"] == pytest.approx(target, abs=1e-6)
        assert source["score_kind"] == "VINA_KCAL_MOL"
        assert set(source["panel_receptor_ids"]) == {"alpha", "beta"}

    # The two atom counts the arithmetic above rests on have to differ, or none
    # of it proves anything about which molecule got which score.
    assert len({_atom_count(block) for block in blocks.values()}) == len(blocks)
    assert response.metadata["panel_size"] == 2
    assert response.metadata["measured_count"] == 3
    assert response.metadata["metric_ids"] == ["delta_score", "panel_win_count"]
    assert response.metadata["target_engine_id"] == "unidock"
    assert response.metadata["panel_engine_id"] == "unidock"
    assert response.metadata["delta_method_id"] != response.metadata["win_count_method_id"]


@_needs_meeko
def test_each_decoy_is_searched_in_its_own_box(tmp_path: Path, engine: _Engine) -> None:
    """A panel receptor docked in the target's box would be a different experiment."""

    blocks = {key: _molblock(value) for key, value in _SMILES.items()}
    alpha = _receptor(tmp_path, "alpha", offset=0.0, size=18.0)
    beta = _receptor(tmp_path, "beta", offset=1.0, size=22.0)
    _output, _response = _run(
        tmp_path,
        engine,
        [alpha, beta],
        [_pose(key, molblock=blocks[key]) for key in _SMILES],
        settings={"scoring": "vina", "search_mode": "detail", "num_modes": 2, "seed": 99},
    )

    calls = _calls(engine)
    assert len(calls) == 2  # one per receptor per batch, not one per molecule
    by_receptor = {call["receptor"][0]: call for call in calls}
    assert set(by_receptor) == {alpha["prepared_pdbqt_path"], beta["prepared_pdbqt_path"]}
    assert by_receptor[str(alpha["prepared_pdbqt_path"])]["size_x"] == ["18.0"]
    assert by_receptor[str(beta["prepared_pdbqt_path"])]["size_x"] == ["22.0"]
    for call in calls:
        assert call["center_x"] == ["1.0"]
        assert call["center_y"] == ["-2.5"]
        assert call["scoring"] == ["vina"]
        assert call["search_mode"] == ["detail"]
        assert call["num_modes"] == ["2"]
        assert call["seed"] == ["99"]
        assert Path(call["ligand_index"][0]).name == "ligand_index.txt"


# ------------------------------------------------- the four ways to be unmeasured


@_needs_meeko
def test_a_molecule_the_bound_engine_never_scored_is_not_applicable(
    tmp_path: Path, engine: _Engine
) -> None:
    blocks = {key: _molblock(value) for key, value in _SMILES.items()}
    output, response = _run(
        tmp_path,
        engine,
        [_receptor(tmp_path, "alpha")],
        [_pose(key, molblock=blocks[key]) for key in ("p1", "p2")],
    )
    rows = _metrics(output, response)

    for metric in ("delta_score", "panel_win_count"):
        row = rows[("p3", metric)]
        assert row["status"] == "NOT_APPLICABLE"
        assert row["value"] is None
        assert row["source_json"] is None
        assert "no unidock score" in str(row["status_detail"])
        # ...and the units are still declared, because a row with no value is
        # still a row about a quantity.
        assert row["units"] == ("KCAL_PER_MOL" if metric == "delta_score" else "COUNT")
    assert rows[("p1", "delta_score")]["status"] == "OK"
    assert response.metadata["unscored_count"] == 1
    assert response.metadata["measured_count"] == 2


@_needs_meeko
def test_a_pose_stored_without_geometry_cannot_be_re_docked(
    tmp_path: Path, engine: _Engine
) -> None:
    """Scored, accepted, and unmeasurable here -- which is not the same as bad."""

    blocks = {key: _molblock(value) for key, value in _SMILES.items()}
    output, response = _run(
        tmp_path,
        engine,
        [_receptor(tmp_path, "alpha")],
        [
            _pose("p1", molblock=blocks["p1"]),
            _pose("p2", molblock=None),
            _pose("p3", molblock=blocks["p3"]),
        ],
    )
    rows = _metrics(output, response)

    row = rows[("p2", "delta_score")]
    assert row["status"] == "NOT_APPLICABLE"
    assert row["value"] is None
    assert "without geometry" in str(row["status_detail"])
    assert rows[("p1", "delta_score")]["status"] == "OK"
    assert rows[("p3", "delta_score")]["status"] == "OK"
    assert response.metadata["missing_geometry_count"] == 1
    assert response.metadata["unscored_count"] == 0


@_needs_meeko
def test_an_incomplete_panel_keeps_its_value_and_says_which_panel_it_is(
    tmp_path: Path, engine: _Engine
) -> None:
    """A delta over one of two decoys is real, and it is not what ``method_id`` names.

    So the number is kept -- discarding a measurement because it is narrower
    than intended loses information -- and the row is flagged out of domain so a
    threshold does not read it as the panel it claims to be.
    """

    blocks = {key: _molblock(value) for key, value in _SMILES.items()}
    output, response = _run(
        tmp_path,
        engine,
        [
            _receptor(tmp_path, "alpha", offset=0.0),
            # The second decoy declines to score the middle ligand only.
            _receptor(tmp_path, "beta", offset=3.25, skip=(1,)),
        ],
        [_pose(key, molblock=blocks[key]) for key in _SMILES],
    )
    rows = _metrics(output, response)

    row = rows[("p2", "delta_score")]
    assert row["status"] == "OUT_OF_DOMAIN"
    assert float(row["value"]) == pytest.approx(
        _TARGET_SCORES["p2"] - _panel_score(blocks["p2"], 0.0), abs=1e-6
    )
    assert "only 1 of 2 panel receptors" in str(row["status_detail"])
    assert json.loads(str(row["source_json"]))["panel_scores"].keys() == {"alpha"}
    assert rows[("p2", "panel_win_count")]["status"] == "OUT_OF_DOMAIN"
    assert rows[("p1", "delta_score")]["status"] == "OK"
    assert rows[("p3", "delta_score")]["status"] == "OK"
    assert response.metadata["out_of_domain_count"] == 1
    assert response.metadata["measured_count"] == 2


@_needs_meeko
def test_a_molecule_no_decoy_scored_is_a_backend_failure_not_a_zero(
    tmp_path: Path, engine: _Engine
) -> None:
    blocks = {key: _molblock(value) for key, value in _SMILES.items()}
    output, response = _run(
        tmp_path,
        engine,
        [_receptor(tmp_path, "alpha", skip=(0,))],
        [_pose(key, molblock=blocks[key]) for key in _SMILES],
    )
    rows = _metrics(output, response)

    row = rows[("p1", "delta_score")]
    assert row["status"] == "BACKEND_FAILED"
    assert row["value"] is None
    assert "no panel receptor returned a score" in str(row["status_detail"])
    assert rows[("p2", "delta_score")]["status"] == "OK"
    assert response.metadata["backend_failed_count"] == 1


@_needs_meeko
def test_the_engine_failing_arrives_in_the_engines_own_words(
    tmp_path: Path, engine: _Engine
) -> None:
    blocks = {key: _molblock(value) for key, value in _SMILES.items()}
    with pytest.raises(PluginError) as caught:
        _run(
            tmp_path,
            engine,
            [_receptor(tmp_path, "alpha", exit_code=3)],
            [_pose(key, molblock=blocks[key]) for key in _SMILES],
        )

    assert caught.value.code == "SPECIFICITY_PANEL_RUN_FAILED"
    assert "residue 88" in str(caught.value.context["output_tail"])
    assert caught.value.context["receptor_name"] == "alpha"


# ------------------------------------------------ what cannot be subtracted at all


@_needs_meeko
def test_two_scales_are_refused_before_a_single_dock(tmp_path: Path, engine: _Engine) -> None:
    """Vina minus Vinardo is a number, and it is not a binding energy."""

    blocks = {key: _molblock(value) for key, value in _SMILES.items()}
    with pytest.raises(PluginError) as caught:
        _run(
            tmp_path,
            engine,
            [_receptor(tmp_path, "alpha")],
            [_pose(key, molblock=blocks[key], score_kind="VINARDO_KCAL_MOL") for key in _SMILES],
        )

    assert caught.value.code == "SPECIFICITY_PANEL_SCALE_MISMATCH"
    assert caught.value.context["panel_score_kind"] == "VINA_KCAL_MOL"
    assert caught.value.context["observed_score_kinds"] == ["VINARDO_KCAL_MOL"]
    assert _calls(engine) == []  # nothing was docked to find this out

    # Naming the scoring function the bound stage actually used is the fix, and
    # it is one setting rather than a second field that has to agree with this one.
    _output, response = _run(
        tmp_path,
        engine,
        [_receptor(tmp_path, "alpha")],
        [_pose(key, molblock=blocks[key], score_kind="VINARDO_KCAL_MOL") for key in _SMILES],
        settings={"scoring": "vinardo"},
        suffix="-fixed",
    )
    assert response.metadata["score_kind"] == "VINARDO_KCAL_MOL"
    assert response.metadata["measured_count"] == 3


@_needs_meeko
def test_evidence_on_the_opposite_convention_is_refused(tmp_path: Path, engine: _Engine) -> None:
    blocks = {key: _molblock(value) for key, value in _SMILES.items()}
    with pytest.raises(PluginError) as caught:
        _run(
            tmp_path,
            engine,
            [_receptor(tmp_path, "alpha")],
            [_pose(key, molblock=blocks[key], direction="HIGHER_STRONGER") for key in _SMILES],
        )

    assert caught.value.code == "SPECIFICITY_PANEL_DIRECTION_UNSUPPORTED"
    assert caught.value.context["observed_directions"] == ["HIGHER_STRONGER"]
    assert _calls(engine) == []


@_needs_meeko
def test_a_population_whose_poses_were_never_stored_is_refused_up_front(
    tmp_path: Path, engine: _Engine
) -> None:
    """The remedy is on the *other* stage, so it has to be named there."""

    with pytest.raises(PluginError) as caught:
        _run(
            tmp_path,
            engine,
            [_receptor(tmp_path, "alpha")],
            [_pose(key, molblock=None) for key in _SMILES],
        )

    assert caught.value.code == "SPECIFICITY_PANEL_POSES_UNAVAILABLE"
    assert "keep_poses" in str(caught.value.hint)
    assert _calls(engine) == []


@_needs_meeko
def test_two_engines_and_no_pin_is_ambiguous_rather_than_arbitrary(
    tmp_path: Path, engine: _Engine
) -> None:
    """The failure mode this whole tier exists below: which score is the target's.

    Two docking engines in the tier above both write ``docking_score/v1``.
    Silently taking one of them would produce a complete table of deltas against
    a score the operator never chose -- and, since the two engines are on
    different scales, deltas that are not energies at all.
    """

    blocks = {key: _molblock(value) for key, value in _SMILES.items()}
    poses = [_pose(key, molblock=blocks[key]) for key in _SMILES] + [
        _pose(
            key,
            molblock=blocks[key],
            score=4.0,
            engine_id="karmadock",
            score_kind="KARMADOCK_MDN",
            direction="HIGHER_STRONGER",
        )
        for key in _SMILES
    ]
    with pytest.raises(PluginError) as caught:
        _run(tmp_path, engine, [_receptor(tmp_path, "alpha")], poses)

    assert caught.value.code == "SPECIFICITY_PANEL_IDENTITY_AMBIGUOUS"
    assert caught.value.context["identity"] == "engine_id"
    assert caught.value.context["observed_examples"] == ["karmadock", "unidock"]

    # Pinned, the other engine's rows are not merely deprioritised -- they are
    # not in the arithmetic, which is why the scale check passes despite an
    # unsubtractable score sitting in the same table.
    output, response = _run(
        tmp_path,
        engine,
        [_receptor(tmp_path, "alpha", offset=0.0)],
        poses,
        settings={"target_engine_id": "unidock"},
        suffix="-pinned",
    )
    rows = _metrics(output, response)
    assert response.metadata["target_engine_id"] == "unidock"
    assert float(rows[("p1", "delta_score")]["value"]) == pytest.approx(
        _TARGET_SCORES["p1"] - _panel_score(blocks["p1"], 0.0), abs=1e-6
    )


@_needs_meeko
def test_the_method_id_names_the_panel_the_delta_was_measured_against(
    tmp_path: Path, engine: _Engine
) -> None:
    """Two panels are two quantities, and they share a primary key.

    ``(parent_id, metric_id, method_id)`` is what keeps a delta against three
    kinases from overwriting a delta against three GPCRs, so the panel's
    composition -- and the volume each pocket was searched in -- has to be
    inside ``method_id`` rather than merely recorded beside it.
    """

    blocks = {key: _molblock(value) for key, value in _SMILES.items()}
    poses = [_pose(key, molblock=blocks[key]) for key in _SMILES]

    def method(panel: list[dict[str, object]], suffix: str) -> str:
        _output, response = _run(tmp_path, engine, panel, poses, suffix=suffix)
        return str(response.metadata["delta_method_id"])

    one = method([_receptor(tmp_path, "alpha")], "-one")
    again = method([_receptor(tmp_path, "alpha")], "-again")
    two = method([_receptor(tmp_path, "alpha"), _receptor(tmp_path, "beta", offset=3.25)], "-two")
    wider = method([_receptor(tmp_path, "alpha", size=24.0)], "-wider")

    assert one == again  # the same measurement twice, not two measurements
    assert one != two
    assert one != wider
    assert one.startswith("specificity-panel:sha256:")


@_needs_meeko
def test_it_needs_exactly_one_population_and_one_body_of_evidence(
    tmp_path: Path, engine: _Engine
) -> None:
    panel = [_receptor(tmp_path, "alpha")]
    poses = [_pose("p1", molblock=_molblock(_SMILES["p1"]))]
    with pytest.raises(PluginError) as caught:
        SpecificityPanelPlugin().execute(
            StageRequest(
                stage_id="specificity_panel",
                inputs={
                    "parents": _dataset(
                        tmp_path, "solo-parents", PARENT_V1, [_parent("p1")], artifact_hex="a"
                    ),
                    "evidence": _dataset(
                        tmp_path, "solo-poses", DOCKING_SCORE_V1, poses, artifact_hex="b"
                    ),
                    "extra": _dataset(
                        tmp_path, "extra-poses", DOCKING_SCORE_V1, poses, artifact_hex="d"
                    ),
                },
                config={"executable": str(engine.path), "panel": panel},
            ),
            StageContext(tmp_path / "solo", StageResources(devices=("cuda:0",))),
        )

    assert caught.value.code == "SPECIFICITY_PANEL_INPUT_CARDINALITY_INVALID"
    assert caught.value.context["evidence_input_count"] == 2


# ------------------------------------------------------- a decoy that cannot be read


@_needs_meeko
def test_an_unreadable_decoy_stops_the_run_before_anything_is_docked(
    tmp_path: Path, engine: _Engine
) -> None:
    """Prepared once in the parent, so a bad pocket costs nothing to discover.

    A panel is prepared before the first molecule is written, which is the whole
    reason preparation does not live in the workers: a receptor meeko refuses
    would otherwise surface as N simultaneous shard failures after an hour of
    docking had already been paid for.
    """

    blocks = {key: _molblock(value) for key, value in _SMILES.items()}
    receptor = _receptor(tmp_path, "alpha")
    receptor["prepared_pdbqt_path"] = str(tmp_path / "not-written.pdbqt")
    with pytest.raises(PluginError) as caught:
        _run(
            tmp_path,
            engine,
            [receptor],
            [_pose(key, molblock=blocks[key]) for key in _SMILES],
        )

    assert caught.value.code == "SPECIFICITY_PANEL_RECEPTOR_UNREADABLE"
    assert _calls(engine) == []


@_needs_meeko
def test_a_site_that_cannot_be_measured_names_the_decoy_that_owns_it(
    tmp_path: Path, engine: _Engine
) -> None:
    """The cascade layer calls this a configuration error; here it has a receptor.

    ``box_from_structure`` is shared with the campaign's target, where "the site
    is unreadable" is unambiguous.  In a panel of sixteen it is not, so the
    diagnosis is re-raised with the pocket's own name attached and the original
    code kept as the cause.
    """

    blocks = {key: _molblock(value) for key, value in _SMILES.items()}
    receptor = _receptor(tmp_path, "beta")
    for key in ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z"):
        receptor.pop(key)
    receptor["reference_ligand_path"] = str(tmp_path / "missing-ligand.sdf")
    with pytest.raises(PluginError) as caught:
        _run(
            tmp_path,
            engine,
            [receptor],
            [_pose(key, molblock=blocks[key]) for key in _SMILES],
        )

    assert caught.value.code == "SPECIFICITY_PANEL_SITE_UNREADABLE"
    assert caught.value.context["receptor_name"] == "beta"
    assert caught.value.context["cause"] == "DOCKING_SITE_UNREADABLE"
    assert _calls(engine) == []


# ---------------------------------------------------------- the builder shorthand


def _config(**overrides: object) -> SpecificityPanelConfig:
    return SpecificityPanelConfig.model_validate(dict(overrides))


def _site() -> dict[str, object]:
    return {
        "center_x": 1.0,
        "center_y": 2.0,
        "center_z": 3.0,
        "size_x": 20.0,
        "size_y": 20.0,
        "size_z": 20.0,
    }


def test_the_flat_keys_the_builder_writes_become_a_one_receptor_panel() -> None:
    """The HTML builder's threshold fields are flat, and a panel entry is not."""

    config = _config(structure_path="/data/decoy.pdb", **_site())
    assert len(config.panel) == 1
    assert config.panel[0].name == "decoy"
    assert config.panel[0].structure_path == "/data/decoy.pdb"
    assert config.panel[0].box_settings()["size_x"] == 20.0

    named = _config(receptor_name="paralog", structure_path="/data/decoy.pdb", **_site())
    assert named.panel[0].name == "paralog"

    # And the promotion is a rewrite of the input, not a second set of fields:
    # a top-level name a target injector would recognise must not exist here.
    assert "structure_path" not in SpecificityPanelConfig.model_fields
    assert "reference_ligand_path" not in SpecificityPanelConfig.model_fields


def test_the_two_spellings_of_a_panel_are_refused_rather_than_merged() -> None:
    with pytest.raises(ValidationError) as caught:
        _config(
            panel=[{"name": "alpha", "structure_path": "/data/a.pdb", **_site()}],
            structure_path="/data/b.pdb",
            **_site(),
        )

    assert "not both" in str(caught.value)


def test_a_form_nobody_filled_in_reports_a_missing_panel() -> None:
    """A rendered form left alone writes every key as an empty string.

    Promoting that would report a receptor with no structure, when the truth is
    that no panel was configured -- which is a different instruction to the
    operator, pointing at a different field.
    """

    with pytest.raises(ValidationError) as caught:
        _config(receptor_name="", structure_path="", reference_ligand_path="")

    assert [(error["loc"], error["type"]) for error in caught.value.errors()] == [
        (("panel",), "missing")
    ]


def test_a_decoy_needs_a_structure_and_exactly_one_site() -> None:
    with pytest.raises(ValidationError, match="has no structure"):
        _config(panel=[{"name": "alpha", **_site()}])
    with pytest.raises(ValidationError, match="exactly one site definition"):
        _config(panel=[{"name": "alpha", "structure_path": "/data/a.pdb"}])
    with pytest.raises(ValidationError, match="exactly one site definition"):
        _config(
            panel=[
                {
                    "name": "alpha",
                    "structure_path": "/data/a.pdb",
                    "reference_ligand_path": "/data/ref.sdf",
                    **_site(),
                }
            ]
        )
    with pytest.raises(ValidationError, match="partial box"):
        _config(panel=[{"name": "alpha", "structure_path": "/data/a.pdb", "center_x": 1.0}])
    with pytest.raises(ValidationError, match="must be distinct"):
        _config(
            panel=[
                {"name": "alpha", "structure_path": "/data/a.pdb", **_site()},
                {"name": "alpha", "structure_path": "/data/b.pdb", **_site()},
            ]
        )
