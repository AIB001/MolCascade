"""A number for the pair, from a model that never saw our pose.

Every tier above this one measures the geometry some docking program produced.
This one throws that away: it hands Boltz a sequence and a SMILES, gets a
complex folded from scratch, and reads an affinity off the head.  Which means
the failure modes are not the ones the docking tests cover.  The interesting
ones here are about *what the number is attached to* -- the target it was
folded against, the alignment that fold leaned on, the molecule the result
belongs to -- and about the several ways this stage can quietly cost an entire
night instead of failing.

The stand-in binary is a fake of the CLI's protocol, as in
``test_specificity_panel.py``: it parses the real flags, walks the real input
directory, reads the real YAML, and writes affinity JSON where Boltz writes it.
Its predicted affinity for a ligand is a function of that ligand's own SMILES
length, which is the trick that makes the join checkable -- a number landing on
the wrong molecule is visible as a number rather than hidden behind a fixture
of invented values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from pydantic import ValidationError

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import PARENT_V1, PREDICTION_V1
from molcascade.errors import PluginError
from molcascade.parallel import StageResources
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.boltz2 import (
    Boltz2AffinityPlugin,
    Boltz2Config,
    _ensemble_interval,
    _read_affinity,
)

pytest.importorskip("rdkit")

#: Distinct lengths on purpose -- the fake's affinity is a function of SMILES
#: length, so a prediction landing on the wrong parent shows up as a number.
_SMILES = {
    "p1": "c1ccccc1",
    "p2": "CC(=O)Oc1ccccc1C(=O)O",
    "p3": "Cc1ccc(cc1)S(=O)(=O)N",
}
#: Under 128 atoms with hydrogens, and far enough under to stay under a lowered
#: ceiling in the one test that lowers it.
_SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"


# --------------------------------------------------------------- the fake engine

_FAKE_ENGINE = """
import json
import sys
from pathlib import Path

argv = sys.argv[1:]
flags = {}
key = None
positional = []
for token in argv:
    if token.startswith("--"):
        key = token[2:]
        flags[key] = True
        continue
    if key is not None and flags.get(key) is True:
        flags[key] = token
        key = None
    else:
        positional.append(token)

inputs = Path(positional[1])
out_dir = Path(flags["out_dir"])
documents = {}
for path in sorted(inputs.glob("*.yaml")):
    documents[path.stem] = path.read_text(encoding="utf-8")

Path(_RECORD).write_text(
    json.dumps({"argv": argv, "flags": flags, "documents": documents, "cwd": str(Path.cwd())}),
    encoding="utf-8",
)

if _MODE == "silent":
    sys.exit(0)
if _MODE == "crash":
    sys.stderr.write("CUDA out of memory while sampling the affinity head\\n")
    sys.exit(2)

for record, text in documents.items():
    if record in _SKIP:
        continue
    smiles = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("smiles:"):
            smiles = stripped.split(":", 1)[1].strip().strip("'\\"")
    # Deterministic, and a function of the ligand rather than of position.
    value = -0.1 * len(smiles)
    folder = out_dir / "boltz_results_inputs" / "predictions" / record
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ("affinity_" + record + ".json")).write_text(
        json.dumps(
            {
                "affinity_pred_value": value,
                "affinity_probability_binary": 0.5,
                "affinity_pred_value1": value - 0.25,
                "affinity_pred_value2": value + 0.25,
                "affinity_probability_binary1": 0.4,
                "affinity_probability_binary2": 0.6,
            }
        ),
        encoding="utf-8",
    )
    # Boltz writes structures beside the affinity file; the collector must not
    # trip over them.
    (folder / (record + "_model_0.cif")).write_text("data_x\\n", encoding="utf-8")
sys.exit(0)
"""


class _Engine(NamedTuple):
    """The fake binary and the file it writes its call record into."""

    path: Path
    record_path: Path


def _write_engine(
    tmp_path: Path,
    *,
    mode: str = "normal",
    skip: tuple[str, ...] = (),
    name: str = "boltz",
) -> _Engine:
    executable = tmp_path / name
    record_path = tmp_path / f"{name}-record.json"
    executable.write_text(
        f"#!{sys.executable}\n"
        f"_RECORD = {json.dumps(str(record_path))}\n"
        f"_MODE = {json.dumps(mode)}\n"
        f"_SKIP = {json.dumps(list(skip))}\n"
        f"{_FAKE_ENGINE}",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return _Engine(executable, record_path)


@pytest.fixture
def engine(tmp_path: Path) -> _Engine:
    return _write_engine(tmp_path)


def _call(engine: _Engine) -> dict[str, Any]:
    return dict(json.loads(engine.record_path.read_text(encoding="utf-8")))


def _documents(engine: _Engine) -> dict[str, Any]:
    """Every input YAML the engine was handed, parsed back."""

    return {record: yaml.safe_load(text) for record, text in _call(engine)["documents"].items()}


# ------------------------------------------------------------------- the fixtures


@pytest.fixture
def target(tmp_path: Path) -> Path:
    path = tmp_path / "target.fasta"
    path.write_text(f">STK17B kinase domain\n{_SEQUENCE}\n", encoding="utf-8")
    return path


@pytest.fixture
def alignment(tmp_path: Path) -> Path:
    path = tmp_path / "target.a3m"
    path.write_text(f">query\n{_SEQUENCE}\n>hit\n{_SEQUENCE}\n", encoding="utf-8")
    return path


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    path = tmp_path / "boltz-cache"
    path.mkdir()
    (path / "boltz2_conf.ckpt").write_bytes(b"not really three gigabytes")
    return path


def _parent_row(parent_id: str, smiles: str) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "identity_policy_id": "identity:test",
        "parent_smiles": smiles,
        "registration_key": f"registration:{parent_id}",
        "stereo_key": None,
        "formula": None,
        "duplicate_count": 1,
    }


def _parents(tmp_path: Path, keys: list[str], *, suffix: str = "") -> StageInput:
    root = tmp_path / f"parents{suffix}"
    root.mkdir(parents=True, exist_ok=True)
    rows = [_parent_row(key, _SMILES[key]) for key in keys]
    pq.write_table(pa.Table.from_pylist(rows, schema=PARENT_V1.schema), root / "part.parquet")
    return StageInput(
        ref=ArtifactRef(artifact_id="artifact:sha256:" + "a" * 64, kind=PARENT_V1.id),
        root=root,
        contract_id=PARENT_V1.id,
    )


def _run(
    tmp_path: Path,
    engine: _Engine,
    target: Path,
    cache: Path,
    *,
    settings: dict[str, object],
    keys: list[str] | None = None,
    suffix: str = "",
) -> tuple[Path, Any]:
    config: dict[str, object] = {
        "executable": str(engine.path),
        "cache_dir": str(cache),
        "target_fasta_path": str(target),
        "timeout_per_molecule_seconds": 120.0,
        "scratch_dir": str(tmp_path),
    }
    config.update(settings)
    output = tmp_path / f"out{suffix}"
    response = Boltz2AffinityPlugin().execute(
        StageRequest(
            stage_id=f"boltz2{suffix}",
            inputs={
                "parents": _parents(tmp_path, keys or ["p1", "p2", "p3"], suffix=suffix),
            },
            config=config,
        ),
        StageContext(output, StageResources(devices=("cuda:0",))),
    )
    return output, response


def _predictions(output: Path, response: Any) -> dict[tuple[str, str], dict[str, Any]]:
    table = pa.concat_tables(
        [pq.read_table(output / path) for path in response.outputs["predictions"].file_paths]
    )
    PREDICTION_V1.validate(table)
    return {(str(row["parent_id"]), str(row["endpoint_id"])): row for row in table.to_pylist()}


# ------------------------------------------------------------- what gets folded


def test_the_input_yaml_is_boltz_own_schema_with_the_affinity_property(
    tmp_path: Path, engine: _Engine, target: Path, alignment: Path, cache: Path
) -> None:
    """One complex per file, and the affinity head asked for by name.

    The head only runs when ``properties`` names a binder chain, so a YAML that
    folds correctly and predicts nothing is a real and silent failure.  This
    pins the whole document, not just that key, because the chain ids in it are
    what the binder reference points at.
    """

    _run(tmp_path, engine, target, cache, settings={"msa_a3m_path": str(alignment)})

    documents = _documents(engine)
    assert len(documents) == 3
    document = documents["mol-000000"]
    assert document["version"] == 1
    protein = document["sequences"][0]["protein"]
    assert protein["id"] == "A"
    assert protein["sequence"] == _SEQUENCE
    assert protein["msa"] == str(alignment)
    ligand = document["sequences"][1]["ligand"]
    assert ligand["id"] == "B"
    assert ligand["smiles"] in set(_SMILES.values())
    assert document["properties"] == [{"affinity": {"binder": "B"}}]


def test_a_smiles_full_of_yaml_metacharacters_survives_the_round_trip(
    tmp_path: Path, engine: _Engine, target: Path, alignment: Path, cache: Path
) -> None:
    """Serialized by a YAML dumper, not by a format string.

    ``#`` opens a comment, a leading ``[`` opens a flow sequence, and both are
    ordinary characters in a SMILES.  A hand-quoted writer works until one of
    these arrives and then folds a truncated molecule without complaining.
    """

    root = tmp_path / "awkward"
    root.mkdir()
    awkward = "[nH]1cccc1C#N"
    pq.write_table(
        pa.Table.from_pylist([_parent_row("odd", awkward)], schema=PARENT_V1.schema),
        root / "part.parquet",
    )
    Boltz2AffinityPlugin().execute(
        StageRequest(
            stage_id="boltz2",
            inputs={
                "parents": StageInput(
                    ref=ArtifactRef(artifact_id="artifact:sha256:" + "b" * 64, kind=PARENT_V1.id),
                    root=root,
                    contract_id=PARENT_V1.id,
                )
            },
            config={
                "executable": str(engine.path),
                "cache_dir": str(cache),
                "target_fasta_path": str(target),
                "msa_a3m_path": str(alignment),
                "scratch_dir": str(tmp_path),
            },
        ),
        StageContext(tmp_path / "out-awkward", StageResources(devices=("cpu",))),
    )

    document = _documents(engine)["mol-000000"]
    assert document["sequences"][1]["ligand"]["smiles"] == awkward


def test_a_ligand_over_the_atom_ceiling_never_reaches_a_yaml(
    tmp_path: Path, engine: _Engine, target: Path, alignment: Path, cache: Path
) -> None:
    """Filtered before the fold, because the fold is the expensive part.

    Boltz would refuse it anyway.  Refusing it here is the difference between
    losing nothing and losing a minute of GPU time per oversized molecule --
    and the count has to survive into the metadata, because a molecule with no
    prediction row is one the gate above rejects for missing evidence, and an
    operator needs to be able to tell that apart from a weak prediction.
    """

    _, response = _run(
        tmp_path,
        engine,
        target,
        cache,
        settings={"msa_a3m_path": str(alignment), "max_ligand_atoms": 16},
    )

    documents = _documents(engine)
    assert set(documents) == {"mol-000000"}  # benzene: 12 atoms with hydrogens
    assert documents["mol-000000"]["sequences"][1]["ligand"]["smiles"] == _SMILES["p1"]
    assert response.metadata["skipped_too_large_count"] == 2
    assert sorted(response.metadata["skipped_too_large_examples"]) == ["p2", "p3"]
    assert response.metadata["input_count"] == 3


def test_an_unreadable_smiles_is_counted_apart_from_an_oversized_one(
    tmp_path: Path, engine: _Engine, target: Path, cache: Path, alignment: Path
) -> None:
    """Same consequence, different problem, and the metadata says which.

    Both end up without a prediction row.  One means "this library has
    molecules the affinity head was not built for"; the other means "something
    upstream wrote a string that is not a molecule".  Collapsing them would
    hide the second behind the first.
    """

    root = tmp_path / "broken"
    root.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [_parent_row("good", _SMILES["p1"]), _parent_row("bad", "c1ccccc")],
            schema=PARENT_V1.schema,
        ),
        root / "part.parquet",
    )
    response = Boltz2AffinityPlugin().execute(
        StageRequest(
            stage_id="boltz2",
            inputs={
                "parents": StageInput(
                    ref=ArtifactRef(artifact_id="artifact:sha256:" + "c" * 64, kind=PARENT_V1.id),
                    root=root,
                    contract_id=PARENT_V1.id,
                )
            },
            config={
                "executable": str(engine.path),
                "cache_dir": str(cache),
                "target_fasta_path": str(target),
                "msa_a3m_path": str(alignment),
                "scratch_dir": str(tmp_path),
            },
        ),
        StageContext(tmp_path / "out-broken", StageResources(devices=("cpu",))),
    )

    assert response.metadata["skipped_unparseable_count"] == 1
    assert response.metadata["skipped_unparseable_examples"] == ["bad"]
    assert response.metadata["skipped_too_large_count"] == 0


def test_the_budget_stops_the_run_before_the_first_fold(
    tmp_path: Path, engine: _Engine, target: Path, alignment: Path, cache: Path
) -> None:
    """The ceiling is a refusal, not a truncation.

    Nothing in the compiler forces this tier last -- it consumes no evidence at
    all, so it could legally be first -- and only its cost says otherwise.  A
    budget that silently folded the first N would turn a misplaced block into a
    partial answer, which is worse than an error: the rows that came back would
    look like a completed stage.
    """

    with pytest.raises(PluginError) as caught:
        _run(
            tmp_path,
            engine,
            target,
            cache,
            settings={"msa_a3m_path": str(alignment), "max_molecules": 2},
        )

    assert caught.value.code == "BOLTZ2_INPUT_TOO_LARGE"
    assert caught.value.context["max_molecules"] == 2
    assert not engine.record_path.exists()


# ------------------------------------------------------------------- the answer


def test_each_parent_gets_its_own_two_endpoints(
    tmp_path: Path, engine: _Engine, target: Path, alignment: Path, cache: Path
) -> None:
    """Affinity and binder probability, joined by record name rather than order.

    The fake's affinity is a function of the SMILES it was given, so this
    recomputes which number belongs to which molecule.  A join on position
    would pass a test with equal-length inputs and fail on a real run where the
    engine reordered or deduplicated its output.
    """

    output, response = _run(
        tmp_path, engine, target, cache, settings={"msa_a3m_path": str(alignment)}
    )

    rows = _predictions(output, response)
    assert len(rows) == 6
    for key, smiles in _SMILES.items():
        affinity = rows[(key, "boltz2_affinity_log10_ic50_um")]
        assert affinity["prediction_mean"] == pytest.approx(-0.1 * len(smiles))
        assert affinity["model_id"] == response.metadata["model_id"]
        assert rows[(key, "boltz2_binder_probability")]["prediction_mean"] == pytest.approx(0.5)
    assert response.metadata["affinity_units"] == "log10(IC50 / uM)"
    assert response.metadata["affinity_direction"] == "LOWER_STRONGER"
    assert response.metadata["calibrated"] is False


def test_the_ensemble_members_become_an_interval_and_never_a_deviation(
    tmp_path: Path, engine: _Engine, target: Path, alignment: Path, cache: Path
) -> None:
    """Two numbers are a range; they are not a sample of a distribution.

    ``affinity_pred_value1`` and ``affinity_pred_value2`` are the ensemble's
    two members.  Reporting their half-difference as a standard deviation would
    put a shape on two points that two points do not have, and the contract has
    a column that means exactly what they are.
    """

    output, response = _run(
        tmp_path, engine, target, cache, settings={"msa_a3m_path": str(alignment)}
    )

    row = _predictions(output, response)[("p1", "boltz2_affinity_log10_ic50_um")]
    expected = -0.1 * len(_SMILES["p1"])
    assert row["interval_lower"] == pytest.approx(expected - 0.25)
    assert row["interval_upper"] == pytest.approx(expected + 0.25)
    assert row["prediction_std"] is None
    assert row["calibration_id"] is None


def test_a_lone_ensemble_member_leaves_the_interval_empty() -> None:
    """An interval needs two ends, and one number is not an interval."""

    assert _ensemble_interval({"affinity_pred_value1": -1.0}, "affinity_pred_value") == (
        None,
        None,
    )
    assert _ensemble_interval(
        {"affinity_pred_value1": -1.0, "affinity_pred_value2": -2.0},
        "affinity_pred_value",
    ) == (-2.0, -1.0)


def test_a_molecule_the_engine_skipped_gets_no_row_rather_than_a_number(
    tmp_path: Path, target: Path, alignment: Path, cache: Path
) -> None:
    """No prediction is not a weak prediction.

    ``prediction/v1`` cannot say "unknown" -- ``prediction_mean`` is
    non-nullable -- so the only honest response to a molecule the head failed
    on is silence, and the fail-closed gate above turns silence into a
    rejection.  Writing a placeholder here would turn it into a pass.
    """

    engine = _write_engine(tmp_path, skip=("mol-000001",))
    output, response = _run(
        tmp_path, engine, target, cache, settings={"msa_a3m_path": str(alignment)}
    )

    rows = _predictions(output, response)
    assert {key for key, _ in rows} == {"p1", "p3"}
    assert response.metadata["predicted_count"] == 2
    assert response.metadata["prediction_row_count"] == 4
    # The parent passes through regardless: this stage predicts, it does not filter.
    parents = pq.read_table(output / response.outputs["primary"].file_paths[0])
    assert set(parents.column("parent_id").to_pylist()) == {"p1", "p2", "p3"}


def test_a_run_that_writes_nothing_at_all_is_an_error_not_an_empty_answer(
    tmp_path: Path, target: Path, alignment: Path, cache: Path
) -> None:
    """A clean exit with no results is a layout problem, and it has to say so.

    Boltz has moved its output tree between releases.  The collector globs for
    that reason, and the case it cannot glob its way out of -- nothing there at
    all -- must not read as "no molecule bound".
    """

    engine = _write_engine(tmp_path, mode="silent")
    with pytest.raises(PluginError) as caught:
        _run(tmp_path, engine, target, cache, settings={"msa_a3m_path": str(alignment)})

    assert caught.value.code == "BOLTZ2_OUTPUT_MISSING"
    assert caught.value.context["expected"] == 3


def test_a_crashing_engine_reports_its_own_console_tail(
    tmp_path: Path, target: Path, alignment: Path, cache: Path
) -> None:
    """The message an operator needs is Boltz's, not ours.

    Out of memory at the sampling step is the usual way this fails, and it is
    fixed by lowering ``diffusion_samples_affinity`` -- which nobody can work
    out from an exit status alone.
    """

    engine = _write_engine(tmp_path, mode="crash")
    with pytest.raises(PluginError) as caught:
        _run(tmp_path, engine, target, cache, settings={"msa_a3m_path": str(alignment)})

    assert caught.value.code == "BOLTZ2_RUN_FAILED"
    assert "CUDA out of memory" in str(caught.value.context["output_tail"])


def test_a_failed_run_leaves_no_half_written_outputs(
    tmp_path: Path, target: Path, alignment: Path, cache: Path
) -> None:
    """The parents are written before the fold, so a crash must undo them.

    A parent file with no prediction file beside it is a stage that looks like
    it ran and passed everything through, which is the one outcome a failure
    must never resemble.
    """

    engine = _write_engine(tmp_path, mode="crash")
    output = tmp_path / "out-crash"
    with pytest.raises(PluginError):
        Boltz2AffinityPlugin().execute(
            StageRequest(
                stage_id="boltz2",
                inputs={"parents": _parents(tmp_path, ["p1"])},
                config={
                    "executable": str(engine.path),
                    "cache_dir": str(cache),
                    "target_fasta_path": str(target),
                    "msa_a3m_path": str(alignment),
                    "scratch_dir": str(tmp_path),
                },
            ),
            StageContext(output, StageResources(devices=("cpu",))),
        )

    assert not (output / "datasets/parents/part-00000.parquet").exists()
    assert not (output / "datasets/predictions/part-00000.parquet").exists()


# ------------------------------------------------------ what the number is about


def test_the_alignment_is_required_and_the_error_gives_both_ways_out(
    tmp_path: Path, engine: _Engine, target: Path, cache: Path
) -> None:
    """Accuracy leans on the MSA, so its absence stops the run rather than costing.

    Boltz will happily predict from a single sequence, and its own
    documentation discourages it.  A quiet fallback would spend the same GPU
    hours on a worse answer with nothing in the output saying so, which is why
    this is a refusal -- and why the refusal names both of the two ways to
    proceed, including the one that uploads the sequence to a third party.
    """

    with pytest.raises(PluginError) as caught:
        _run(tmp_path, engine, target, cache, settings={})

    assert caught.value.code == "PLUGIN_CONFIG_INVALID"
    message = str(caught.value)
    assert "msa_a3m_path" in message
    assert "empty" in message
    assert "colabfold" in message


def test_uploading_the_target_is_something_you_ask_for(
    tmp_path: Path, engine: _Engine, target: Path, alignment: Path, cache: Path
) -> None:
    """``--use_msa_server`` publishes the sequence, so nothing may imply it.

    It is reachable, deliberately -- an operator with a public target and no
    local MMseqs2 has a real use for it -- but only by naming the mode.  The
    default run must not contain the flag, and the server mode must not also
    carry a local path, because then which one was used is a coin toss.
    """

    _run(tmp_path, engine, target, cache, settings={"msa_a3m_path": str(alignment)})
    assert "use_msa_server" not in _call(engine)["flags"]

    _, _ = _run(
        tmp_path,
        engine,
        target,
        cache,
        settings={"msa_mode": "colabfold_server"},
        suffix="-server",
    )
    assert _call(engine)["flags"]["use_msa_server"] is True
    assert "msa" not in _documents(engine)["mol-000000"]["sequences"][0]["protein"]

    with pytest.raises(ValidationError, match="would not be read"):
        Boltz2Config(
            executable=str(engine.path),
            cache_dir=str(cache),
            target_fasta_path=str(target),
            msa_mode="colabfold_server",
            msa_a3m_path=str(alignment),
        )


def test_single_sequence_mode_says_so_in_boltz_own_spelling(
    tmp_path: Path, engine: _Engine, target: Path, cache: Path
) -> None:
    """``msa: empty`` is how Boltz is told to skip the alignment.

    Omitting the key entirely means something else -- it is what the server
    mode does -- so the explicit downgrade has to be written out.
    """

    _run(tmp_path, engine, target, cache, settings={"msa_mode": "empty"})

    assert _documents(engine)["mol-000000"]["sequences"][0]["protein"]["msa"] == "empty"
    assert "use_msa_server" not in _call(engine)["flags"]


def test_the_identity_of_the_prediction_covers_everything_that_changed_it(
    tmp_path: Path, engine: _Engine, target: Path, alignment: Path, cache: Path
) -> None:
    """A different target, a different alignment or a different seed is a different number.

    All three are hashed into ``model_id``, which is part of the prediction's
    primary key.  Two runs that disagree because the alignment was rebuilt
    would otherwise collide on that key and look like a contradiction in one
    measurement rather than two measurements.
    """

    _, first = _run(tmp_path, engine, target, cache, settings={"msa_a3m_path": str(alignment)})
    _, again = _run(
        tmp_path,
        engine,
        target,
        cache,
        settings={"msa_a3m_path": str(alignment)},
        suffix="-again",
    )
    assert first.metadata["model_id"] == again.metadata["model_id"]

    other = tmp_path / "other.a3m"
    other.write_text(f">query\n{_SEQUENCE}\n", encoding="utf-8")
    _, rebuilt = _run(
        tmp_path, engine, target, cache, settings={"msa_a3m_path": str(other)}, suffix="-msa"
    )
    assert rebuilt.metadata["model_id"] != first.metadata["model_id"]

    _, reseeded = _run(
        tmp_path,
        engine,
        target,
        cache,
        settings={"msa_a3m_path": str(alignment), "seed": 7},
        suffix="-seed",
    )
    assert reseeded.metadata["model_id"] != first.metadata["model_id"]


def test_an_unpinned_backend_version_is_reported_rather_than_implied(
    tmp_path: Path, engine: _Engine, target: Path, alignment: Path, cache: Path
) -> None:
    """The release is part of the measurement, and we cannot read it honestly.

    ``boltz --version`` would describe the machine at the moment of asking, not
    the model that produced an archived number.  So it is the operator's to
    state, and the absence is a warning rather than a silence.
    """

    _, unpinned = _run(tmp_path, engine, target, cache, settings={"msa_a3m_path": str(alignment)})
    assert unpinned.metadata["warnings"] == ["BOLTZ2_BACKEND_VERSION_UNPINNED"]

    _, pinned = _run(
        tmp_path,
        engine,
        target,
        cache,
        settings={"msa_a3m_path": str(alignment), "backend_version": "2.2.1"},
        suffix="-pinned",
    )
    assert pinned.metadata["warnings"] == []
    assert pinned.metadata["model_id"] != unpinned.metadata["model_id"]


def test_the_target_is_a_sequence_and_never_a_campaign_receptor() -> None:
    """No field named ``receptor_path``, and that name is the whole mechanism.

    ``lower.py`` injects the campaign's receptor into any plugin whose config
    declares that exact name.  Taking it would give this stage a PDB, and a
    sequence read off ATOM records has every unresolved loop silently deleted
    -- a fold against a protein missing a loop is wrong in a way no error
    mentions.  The FASTA is a separate, deliberate input.
    """

    fields = set(Boltz2Config.model_fields)
    assert "receptor_path" not in fields
    assert "target_fasta_path" in fields


def test_a_multi_chain_fasta_is_refused_with_both_headers_named(
    tmp_path: Path, engine: _Engine, alignment: Path, cache: Path
) -> None:
    """One binder, one receptor chain: two records leaves the adapter guessing.

    Picking the first would be a guess that produces a plausible number against
    the wrong protein, which is the entire class of error this input exists to
    avoid.
    """

    fasta = tmp_path / "two.fasta"
    fasta.write_text(f">chain A\n{_SEQUENCE}\n>chain B\n{_SEQUENCE}\n", encoding="utf-8")

    with pytest.raises(PluginError) as caught:
        _run(tmp_path, engine, fasta, cache, settings={"msa_a3m_path": str(alignment)})

    assert caught.value.code == "BOLTZ2_TARGET_SEQUENCE_AMBIGUOUS"
    assert caught.value.context["records"] == ["chain A", "chain B"]


def test_a_nucleotide_fasta_lands_here_and_is_named_as_such(
    tmp_path: Path, engine: _Engine, alignment: Path, cache: Path
) -> None:
    """The wrong file with the right extension is the common mistake.

    And the alphabet cannot catch it: A, C, G and T are all amino-acid letters
    too, so a gene is a perfectly valid protein sequence as far as the
    characters go.  Boltz would fold it into a meaningless chain and return a
    number.  Composition is the only tell -- twenty residues drawn from nothing
    but that set is not a protein -- so this is the one input check here that is
    a judgement rather than a rule, and it says so in the message.
    """

    fasta = tmp_path / "dna.fasta"
    fasta.write_text(">gene\nATGCGTACGTTAGCATGCGTACGTTAGC\n", encoding="utf-8")

    with pytest.raises(PluginError) as caught:
        _run(tmp_path, engine, fasta, cache, settings={"msa_a3m_path": str(alignment)})

    assert caught.value.code == "BOLTZ2_TARGET_SEQUENCE_UNREADABLE"
    assert "nucleotide" in str(caught.value)
    assert "composition" in str(caught.value.hint)


# -------------------------------------------------------------- the environment


def test_an_empty_weight_cache_stops_the_run_instead_of_downloading(
    tmp_path: Path, engine: _Engine, target: Path, alignment: Path
) -> None:
    """Three gigabytes fetched mid-screen is a network call nobody asked for.

    It would happen on whichever worker got there first, on a machine that may
    have no route out, at a point where the failure looks like a prediction
    failure.  The warmup is a separate, explicit act, and the remedy says so.
    """

    empty = tmp_path / "empty-cache"
    empty.mkdir()

    with pytest.raises(PluginError) as caught:
        _run(tmp_path, engine, target, empty, settings={"msa_a3m_path": str(alignment)})

    assert caught.value.code == "BOLTZ2_WEIGHTS_MISSING"
    assert "bootstrap.sh boltz2" in str(caught.value.hint)


def test_a_missing_executable_names_the_command_that_creates_it(
    tmp_path: Path, target: Path, alignment: Path, cache: Path
) -> None:
    """A path into an environment that was never built is the first-run failure."""

    missing = _Engine(tmp_path / "nowhere" / "boltz", tmp_path / "unused.json")
    with pytest.raises(PluginError) as caught:
        _run(tmp_path, missing, target, cache, settings={"msa_a3m_path": str(alignment)})

    assert caught.value.code == "BOLTZ2_EXECUTABLE_MISSING"


def test_the_child_runs_without_this_process_python_pointing_into_it(
    tmp_path: Path,
    engine: _Engine,
    target: Path,
    alignment: Path,
    cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The isolation is about Torch, not Python.

    Both environments are 3.12.  What differs is the CUDA build underneath, and
    a ``PYTHONPATH`` leaking from this process would put our Torch on the
    child's import path -- which does not fail, it silently loads the wrong
    one.  ``BOLTZ_CACHE`` is set for the same reason the flag is passed: the
    library reads the variable in places the flag does not reach.
    """

    monkeypatch.setenv("PYTHONPATH", "/somewhere/that/should/not/leak")
    _run(tmp_path, engine, target, cache, settings={"msa_a3m_path": str(alignment)})

    call = _call(engine)
    assert call["flags"]["cache"] == str(cache)
    assert call["flags"]["accelerator"] == "gpu"
    assert call["flags"]["seed"] == "42"
    assert call["flags"]["diffusion_samples_affinity"] == "5"
    assert "affinity_mw_correction" not in call["flags"]


def test_a_cpu_only_worker_asks_for_the_cpu_accelerator(
    tmp_path: Path, engine: _Engine, target: Path, alignment: Path, cache: Path
) -> None:
    """``--accelerator gpu`` on a machine with no GPU is a crash, not a slow run."""

    Boltz2AffinityPlugin().execute(
        StageRequest(
            stage_id="boltz2",
            inputs={"parents": _parents(tmp_path, ["p1"])},
            config={
                "executable": str(engine.path),
                "cache_dir": str(cache),
                "target_fasta_path": str(target),
                "msa_a3m_path": str(alignment),
                "scratch_dir": str(tmp_path),
            },
        ),
        StageContext(tmp_path / "out-cpu", StageResources(devices=("cpu",))),
    )

    assert _call(engine)["flags"]["accelerator"] == "cpu"


def test_the_remedy_for_every_machine_path_is_a_command_you_can_paste(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both paths point at an environment the repository knows how to build.

    Discovery finds them when the environment exists.  When it does not, what
    the operator gets is the line that creates it, verbatim -- so the failure
    is one copy-paste from being fixed rather than a research task.
    """

    monkeypatch.setenv("BOLTZ_CACHE", str(tmp_path / "chosen"))
    paths = Boltz2Config.installed_paths()

    assert [path.field for path in paths] == ["executable", "cache_dir"]
    for path in paths:
        assert path.remedy == "bash envs/bootstrap.sh boltz2"
        assert path.candidates
    cache_path = paths[1]
    assert cache_path.kind == "directory"
    assert cache_path.contents == ("boltz2_conf.ckpt",)
    assert cache_path.candidates == (tmp_path / "chosen",)
    assert cache_path.variable("boltz2") == "MOLCASCADE_BOLTZ2_CACHE_DIR"


def test_an_environment_override_supplies_a_path_that_was_left_blank(
    tmp_path: Path,
    engine: _Engine,
    target: Path,
    alignment: Path,
    cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cascade exported on one machine runs on another without being edited.

    The JSON carries an empty string; the machine supplies the path.  That is
    what makes a shared cascade file portable, and it is the same mechanism
    every other isolated engine uses.
    """

    monkeypatch.setenv("MOLCASCADE_BOLTZ2_EXECUTABLE", str(engine.path))
    monkeypatch.setenv("MOLCASCADE_BOLTZ2_CACHE_DIR", str(cache))
    config = Boltz2Config(target_fasta_path=str(target), msa_a3m_path=str(alignment))

    assert config.executable == str(engine.path)
    assert config.cache_dir == str(cache)


# ---------------------------------------------------------------- reading a result


def test_an_affinity_json_without_the_head_is_named_as_such(tmp_path: Path) -> None:
    """A fold that ran and never scored writes a file with structure keys only.

    That happens when the input's ``properties`` block did not ask for the
    affinity head, and it is worth telling apart from a corrupt file, because
    the fix is in the input rather than in the run.
    """

    path = tmp_path / "affinity_mol-000000.json"
    path.write_text(json.dumps({"confidence_score": 0.8, "ptm": 0.7}), encoding="utf-8")

    with pytest.raises(PluginError) as caught:
        _read_affinity(path)

    assert caught.value.code == "BOLTZ2_OUTPUT_UNREADABLE"
    assert "properties" in str(caught.value.hint)
    assert caught.value.context["keys"] == ["confidence_score", "ptm"]


def test_a_null_affinity_is_as_absent_as_a_missing_one(tmp_path: Path) -> None:
    """``NaN`` is JSON that parses and a number that is not one.

    Python's own decoder accepts ``NaN`` without complaint.  Letting it through
    would write a mean no gate can compare against, and comparison is the only
    thing this column is for.
    """

    path = tmp_path / "affinity_mol-000000.json"
    path.write_text(
        '{"affinity_pred_value": NaN, "affinity_probability_binary": 0.5}', encoding="utf-8"
    )

    with pytest.raises(PluginError) as caught:
        _read_affinity(path)

    assert caught.value.code == "BOLTZ2_OUTPUT_UNREADABLE"


# ------------------------------------------- the fork hazard, and the two lines that close it


def test_the_subprocess_gets_a_one_thread_openmp_team(tmp_path: Path) -> None:
    """Not a performance setting: this is what stops the pool deadlocking.

    Boltz preprocesses through a fork-based ``multiprocessing.Pool`` started
    after its parent has imported Torch, built a CUDA context and run
    OpenMP-parallel code.  ``fork()`` copies only the calling thread, so a child
    inherits a libgomp that believes it still owns a team of workers -- and a
    child that enters a parallel region then spins in libgomp's busy-wait
    barrier for ever.  Nothing recovers from it: ``pool.imap`` yields in order,
    so the parent blocks on the first stuck index, the manifest that gates
    inference is never written, and the GPU is never asked for a batch.  A
    one-thread team has no barrier to spin on.
    """

    from molcascade.plugins.builtin.boltz2 import _isolated_environment

    environment = _isolated_environment("cuda:0", tmp_path)
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["MKL_NUM_THREADS"] == "1"


def test_the_preprocessing_pool_does_not_scale_with_the_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boltz's own default is ``cpu_count()``; that is the number to not inherit.

    Every fork is another chance at the hazard above, and preprocessing runs at
    roughly 0.4 s/molecule, so the cap costs nothing a screen can measure.  A
    host with fewer cores than the cap keeps its own count -- Boltz clamps to
    the number of inputs after that, so there is nothing else to decide here.
    """

    from molcascade.plugins.builtin import boltz2 as module

    monkeypatch.setattr(module.os, "cpu_count", lambda: 32)
    assert module._preprocessing_threads() == module.MAX_PREPROCESSING_THREADS

    monkeypatch.setattr(module.os, "cpu_count", lambda: 2)
    assert module._preprocessing_threads() == 2

    monkeypatch.setattr(module.os, "cpu_count", lambda: None)
    assert module._preprocessing_threads() == 1
