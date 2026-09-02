"""A second opinion, from a model in somebody else's environment.

The tier this feeds joins two models with ``any``, which means a molecule is
deleted only when both object -- and that only works if each number is really
attached to the molecule it was computed for.  So the interesting failures here
are about *identity*: a prediction landing on the wrong parent, a released
model swapped underneath a run, an uncertainty column that is empty because one
model cannot have a spread and a mean invented because a row came back blank.

The stand-in binary is a fake of the real CLI's protocol, as in
``test_boltz2.py``: it parses the real flags, reads the real input CSV, and
writes a real prediction CSV.  Its predicted pIC50 is a function of the
molecule's own SMILES length, which is the trick that makes the join checkable
-- a number on the wrong molecule shows up as a number rather than hiding
behind a fixture of invented values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import PARENT_V1, PREDICTION_V1
from molcascade.errors import PluginError
from molcascade.parallel import StageResources
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.openadmet import (
    OpenADMETConfig,
    OpenADMETPredictorPlugin,
    _finite,
)

pytest.importorskip("rdkit")

#: Distinct lengths on purpose -- the fake's prediction is a function of SMILES
#: length, so a value landing on the wrong parent shows up as a number.
_SMILES = {
    "p1": "c1ccccc1",
    "p2": "CC(=O)Oc1ccccc1C(=O)O",
    "p3": "Cc1ccc(cc1)S(=O)(=O)N",
}

_PRED_COLUMN = "OADMET_PRED_chemprop-chembl_pchembl_value_mean"
_STD_COLUMN = "OADMET_STD_chemprop-chembl_pchembl_value_mean"


def _expected(smiles: str) -> float:
    return round(0.1 * len(smiles), 4)


# --------------------------------------------------------------- the fake engine

_FAKE_ENGINE = """
import csv
import json
import os
import sys
from pathlib import Path

argv = sys.argv[1:]
flags = {}
models = []
key = None
positional = []
for token in argv:
    if token.startswith("--"):
        key = token[2:]
        flags[key] = True
        continue
    if key is not None and flags.get(key) is True:
        if key == "model-dir":
            models.append(token)
        flags[key] = token
        key = None
    else:
        positional.append(token)

input_path = Path(flags["input-path"])
with input_path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

Path(_RECORD).write_text(
    json.dumps(
        {
            "argv": argv,
            "flags": flags,
            "models": models,
            "rows": rows,
            "cwd": str(Path.cwd()),
            "env": {
                name: os.environ.get(name)
                for name in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "OMP_NUM_THREADS")
            },
        }
    ),
    encoding="utf-8",
)

if _MODE == "silent":
    sys.exit(0)
if _MODE == "crash":
    sys.stdout.write("RuntimeError: CUDA out of memory loading the checkpoint\\n")
    sys.exit(2)
if _MODE == "empty":
    Path(flags["output-csv"]).write_text("", encoding="utf-8")
    sys.exit(0)

column = flags["input-col"]
out = []
for index, row in enumerate(rows):
    smiles = row[column]
    value = "" if _MODE == "blank_mean" else repr(round(0.1 * len(smiles), 4))
    if _MODE == "nan_mean":
        value = "NaN"
    record = {}
    if _MODE not in ("drop_id", "short"):
        record["molcascade_parent_id"] = row.get("molcascade_parent_id", "")
    record[_PRED] = value
    record[_STD] = "" if _MODE == "single_model" else repr(round(0.01 * (index + 1), 4))
    if _MODE == "wrong_columns":
        record = {"molcascade_parent_id": record.get("molcascade_parent_id", ""), "score": value}
    out.append(record)

if _MODE == "shuffled":
    out.reverse()
if _MODE == "short":
    out = out[:-1]

target = Path(flags["output-csv"])
with target.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(out[0].keys()))
    writer.writeheader()
    writer.writerows(out)
sys.exit(0)
"""


class _Engine(NamedTuple):
    """The fake binary and the file it writes its call record into."""

    path: Path
    record_path: Path


def _write_engine(tmp_path: Path, *, mode: str = "normal", name: str = "openadmet") -> _Engine:
    executable = tmp_path / name
    record_path = tmp_path / f"{name}-record.json"
    executable.write_text(
        f"#!{sys.executable}\n"
        f"_RECORD = {json.dumps(str(record_path))}\n"
        f"_MODE = {json.dumps(mode)}\n"
        f"_PRED = {json.dumps(_PRED_COLUMN)}\n"
        f"_STD = {json.dumps(_STD_COLUMN)}\n"
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


# ------------------------------------------------------------------- the fixtures


def _model(tmp_path: Path, name: str = "herg-chemeleon-baseline", *, body: str = "weights") -> Path:
    """A released clone, and the directory inside it the CLI is actually given.

    ``--model-dir`` wants ``anvil_training``, which is what the model card's own
    inference script passes, so that is what these tests hand over -- one level
    below the clone the bootstrap step creates.
    """

    root = tmp_path / name / "anvil_training"
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.pth").write_text(body, encoding="utf-8")
    (root / "model.json").write_text('{"model": "chemprop"}', encoding="utf-8")
    return root


@pytest.fixture
def model(tmp_path: Path) -> Path:
    return _model(tmp_path)


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


def _parents(
    tmp_path: Path,
    keys: list[str],
    *,
    suffix: str = "",
    extra: dict[str, str] | None = None,
) -> StageInput:
    root = tmp_path / f"parents{suffix}"
    root.mkdir(parents=True, exist_ok=True)
    rows = [_parent_row(key, _SMILES[key]) for key in keys]
    rows += [_parent_row(key, smiles) for key, smiles in (extra or {}).items()]
    pq.write_table(pa.Table.from_pylist(rows, schema=PARENT_V1.schema), root / "part.parquet")
    return StageInput(
        ref=ArtifactRef(artifact_id="artifact:sha256:" + "a" * 64, kind=PARENT_V1.id),
        root=root,
        contract_id=PARENT_V1.id,
    )


_ENDPOINTS: list[dict[str, object]] = [
    {
        "output_column": _PRED_COLUMN,
        "std_column": _STD_COLUMN,
        "endpoint_id": "herg_pic50",
    }
]


def _run(
    tmp_path: Path,
    engine: _Engine,
    model: Path,
    *,
    settings: dict[str, object] | None = None,
    keys: list[str] | None = None,
    extra: dict[str, str] | None = None,
    devices: tuple[str, ...] = ("cuda:0",),
    suffix: str = "",
) -> tuple[Path, Any]:
    config: dict[str, object] = {
        "executable": str(engine.path),
        "model_dir": str(model),
        "endpoints": _ENDPOINTS,
        "scratch_dir": str(tmp_path),
    }
    config.update(settings or {})
    output = tmp_path / f"out{suffix}"
    parents = _parents(tmp_path, keys or ["p1", "p2", "p3"], suffix=suffix, extra=extra)
    response = OpenADMETPredictorPlugin().execute(
        StageRequest(
            stage_id=f"openadmet{suffix}",
            inputs={"parents": parents},
            config=config,
        ),
        StageContext(output, StageResources(devices=devices)),
    )
    return output, response


def _predictions(output: Path, response: Any) -> dict[tuple[str, str], dict[str, Any]]:
    table = pa.concat_tables(
        [pq.read_table(output / path) for path in response.outputs["predictions"].file_paths]
    )
    PREDICTION_V1.validate(table)
    return {(str(row["parent_id"]), str(row["endpoint_id"])): row for row in table.to_pylist()}


# ------------------------------------------------------- what comes back, and whose


def test_the_predicted_columns_become_one_prediction_row_each(
    tmp_path: Path, engine: _Engine, model: Path
) -> None:
    """The mean and its spread land in the two contract columns, per molecule."""

    output, response = _run(tmp_path, engine, model)

    rows = _predictions(output, response)
    assert set(rows) == {(key, "herg_pic50") for key in _SMILES}
    for key, smiles in _SMILES.items():
        row = rows[(key, "herg_pic50")]
        assert row["prediction_mean"] == pytest.approx(_expected(smiles))
        assert row["prediction_std"] is not None
        # Nothing here is calibrated, and the contract says so rather than
        # implying otherwise with an empty string.
        assert row["calibration_id"] is None
        assert row["interval_lower"] is None
    assert response.metadata["calibrated"] is False
    assert response.metadata["endpoints"] == ["herg_pic50"]
    assert response.metadata["prediction_row_count"] == 3


def test_a_reordered_output_still_lands_on_the_right_molecule(tmp_path: Path, model: Path) -> None:
    """The join is on the id column this adapter wrote, not on row order.

    A future release that sorted, deduplicated or dropped rows would break a
    positional match silently, so the fake reverses its output and the values
    still have to follow their own molecules.
    """

    engine = _write_engine(tmp_path, mode="shuffled")
    output, response = _run(tmp_path, engine, model)

    rows = _predictions(output, response)
    for key, smiles in _SMILES.items():
        assert rows[(key, "herg_pic50")]["prediction_mean"] == pytest.approx(_expected(smiles))
    assert "OPENADMET_JOINED_ON_POSITION" not in response.metadata["warnings"]


def test_row_order_is_a_fallback_and_says_so(tmp_path: Path, model: Path) -> None:
    """Without the id column, position is accepted -- and recorded as weaker.

    The id column travels through somebody else's dataframe, so it is not
    guaranteed to come back.  Position is a real answer when the counts agree,
    but it is a smaller claim than a join, and a reader of the artifact is
    entitled to see which one was made.
    """

    engine = _write_engine(tmp_path, mode="drop_id")
    output, response = _run(tmp_path, engine, model)

    rows = _predictions(output, response)
    for key, smiles in _SMILES.items():
        assert rows[(key, "herg_pic50")]["prediction_mean"] == pytest.approx(_expected(smiles))
    assert "OPENADMET_JOINED_ON_POSITION" in response.metadata["warnings"]


def test_a_short_output_with_nothing_to_join_on_is_refused(tmp_path: Path, model: Path) -> None:
    """Position is only a key while the counts agree; otherwise it is a guess.

    Note what is *not* an error: a short output that still carries the id column
    is a molecule the model declined to score, and that is recorded as a missing
    prediction.  It is only when there is nothing to join on that a shorter file
    stops being readable at all.
    """

    engine = _write_engine(tmp_path, mode="short")
    with pytest.raises(PluginError) as error:
        _run(tmp_path, engine, model)
    assert error.value.code == "OPENADMET_OUTPUT_COUNT_MISMATCH"
    assert error.value.context["input_count"] == 3
    assert error.value.context["output_count"] == 2


# ------------------------------------------------- an uncertainty only an ensemble has


def test_one_model_writes_a_mean_and_a_null_spread(tmp_path: Path, model: Path) -> None:
    """An empty standard deviation is null, not zero.

    Zero would claim the models agreed perfectly. There is only one model, so
    there is nothing to agree; ``prediction_std`` is nullable precisely so this
    can be said.
    """

    engine = _write_engine(tmp_path, mode="single_model")
    output, response = _run(tmp_path, engine, model)

    rows = _predictions(output, response)
    for key in _SMILES:
        row = rows[(key, "herg_pic50")]
        assert row["prediction_mean"] is not None
        assert row["prediction_std"] is None
    assert response.metadata["ensemble_size"] == 1


def test_every_model_directory_is_passed_and_counted(
    tmp_path: Path, engine: _Engine, model: Path
) -> None:
    """Repeating ``--model-dir`` is the only way upstream produces a spread."""

    second = _model(tmp_path, "herg-chemeleon-fold2", body="other weights")
    output, response = _run(
        tmp_path,
        engine,
        model,
        settings={"model_dir": [str(model), str(second)]},
    )

    call = _call(engine)
    assert call["models"] == [str(model), str(second)]
    assert call["argv"].count("--model-dir") == 2
    assert response.metadata["ensemble_size"] == 2
    assert _predictions(output, response)


def test_the_same_directory_twice_is_not_an_ensemble(tmp_path: Path, model: Path) -> None:
    """One model repeated is one model with a standard deviation of zero."""

    with pytest.raises(ValidationError, match="same directory twice"):
        OpenADMETConfig(
            executable="/usr/bin/env",
            model_dir=[str(model), str(model)],
            endpoints=_ENDPOINTS,
        )


# --------------------------------------------------------- what the number is attached to


def test_swapping_the_checkpoint_changes_the_model_id(
    tmp_path: Path, engine: _Engine, model: Path
) -> None:
    """No trust flag, so provenance has to come from the bytes themselves.

    OpenADMET deserializes its own checkpoint in its own interpreter, which is
    why this adapter asks for no consent -- but that means nothing else pins
    *which* model produced a number. Digesting the directory is what makes a
    swap a different measurement rather than a silent one.
    """

    _, first = _run(tmp_path, engine, model, suffix="-a")
    (model / "model.pth").write_text("retrained", encoding="utf-8")
    _, second = _run(tmp_path, engine, model, suffix="-b")

    assert first.metadata["model_id"] != second.metadata["model_id"]
    assert first.metadata["models_sha256"] != second.metadata["models_sha256"]
    assert first.metadata["model_id"].startswith("openadmet:sha256:")


def test_an_unpinned_backend_version_is_reported_rather_than_implied(
    tmp_path: Path, engine: _Engine, model: Path
) -> None:
    """A missing release number is a gap in the record, so the record says so."""

    _, unpinned = _run(tmp_path, engine, model, suffix="-u")
    assert "OPENADMET_BACKEND_VERSION_UNPINNED" in unpinned.metadata["warnings"]

    _, pinned = _run(tmp_path, engine, model, settings={"backend_version": "0.1.0"}, suffix="-p")
    assert pinned.metadata["warnings"] == []
    assert pinned.metadata["model_id"] != unpinned.metadata["model_id"]


# ------------------------------------------------------------- absent evidence stays absent


def test_a_molecule_with_no_finite_prediction_gets_no_row(tmp_path: Path, model: Path) -> None:
    """``prediction_mean`` is not nullable, so the only honest record is none.

    A sentinel would rank rather than reject, and a fail-closed gate turns a
    missing row into a rejection, which is the correct handling of "we do not
    know".
    """

    engine = _write_engine(tmp_path, mode="nan_mean")
    output, response = _run(tmp_path, engine, model)

    assert _predictions(output, response) == {}
    assert response.metadata["missing_prediction_count"] == 3
    assert response.metadata["prediction_row_count"] == 0
    # The parents still pass through untouched: this stage predicts, the gate
    # after it decides.
    parents = pq.read_table(output / response.outputs["primary"].file_paths[0])
    assert parents.num_rows == 3


def test_an_unreadable_smiles_is_skipped_before_the_child_sees_it(
    tmp_path: Path, engine: _Engine, model: Path
) -> None:
    """One bad row costs one row, not the whole invocation.

    The CLI is handed a single CSV for the entire population, so a molecule its
    featurizer cannot build is the kind of thing that ends a batch of thousands
    rather than skipping a line.
    """

    output, response = _run(tmp_path, engine, model, extra={"bad": "not a molecule at all"})

    rows = _predictions(output, response)
    assert set(rows) == {(key, "herg_pic50") for key in _SMILES}
    assert response.metadata["skipped_unparseable_count"] == 1
    assert response.metadata["skipped_unparseable_examples"] == ["bad"]
    assert response.metadata["input_count"] == 4
    # It was never offered to the engine at all.
    assert {row["molcascade_parent_id"] for row in _call(engine)["rows"]} == set(_SMILES)


def test_a_missing_configured_column_names_what_came_back(tmp_path: Path, model: Path) -> None:
    """Column names belong to the released model, so a mismatch is the operator's."""

    engine = _write_engine(tmp_path, mode="wrong_columns")
    with pytest.raises(PluginError) as error:
        _run(tmp_path, engine, model)
    assert error.value.code == "OPENADMET_OUTPUT_COLUMN_MISSING"
    assert error.value.context["missing"] == sorted([_PRED_COLUMN, _STD_COLUMN])
    assert "score" in error.value.context["columns"]


@pytest.mark.parametrize(
    ("mode", "code"),
    [
        ("crash", "OPENADMET_RUN_FAILED"),
        ("silent", "OPENADMET_OUTPUT_MISSING"),
        ("empty", "OPENADMET_OUTPUT_MISSING"),
    ],
)
def test_a_failed_run_carries_the_engine_own_words(
    tmp_path: Path, model: Path, mode: str, code: str
) -> None:
    """The diagnosis is upstream's, so the message has to travel."""

    engine = _write_engine(tmp_path, mode=mode)
    with pytest.raises(PluginError) as error:
        _run(tmp_path, engine, model)
    assert error.value.code == code
    if mode == "crash":
        assert "CUDA out of memory" in error.value.context["output_tail"]


def test_a_failed_run_leaves_no_half_written_artifact(tmp_path: Path, model: Path) -> None:
    """Either both datasets are there or neither is."""

    engine = _write_engine(tmp_path, mode="crash")
    with pytest.raises(PluginError):
        _run(tmp_path, engine, model)
    output = tmp_path / "out"
    assert not list(output.rglob("*.parquet"))


# ------------------------------------------------------------------ which card, whose torch


@pytest.mark.parametrize(
    ("devices", "expected", "visible"),
    [
        (("cuda:1",), "gpu", "1"),
        (("cpu",), "cpu", None),
    ],
)
def test_the_accelerator_follows_the_device_this_stage_was_granted(
    tmp_path: Path,
    engine: _Engine,
    model: Path,
    devices: tuple[str, ...],
    expected: str,
    visible: str | None,
) -> None:
    """``auto`` means the runner's lane, never Lightning's own guess.

    Upstream defaults to ``gpu`` and also accepts ``auto``, both of which pick a
    card without reference to what the scheduler assigned. The CLI has no device
    flag, so the ordinal is fenced with ``CUDA_VISIBLE_DEVICES``.
    """

    _run(tmp_path, engine, model, devices=devices)

    call = _call(engine)
    assert call["flags"]["accelerator"] == expected
    assert call["env"]["CUDA_VISIBLE_DEVICES"] == visible
    # This interpreter's import path must not reach theirs: keeping their torch
    # away from the one ADMET-AI predicts with is the whole reason for the
    # separate environment.
    assert call["env"]["PYTHONPATH"] is None


def test_an_explicit_accelerator_is_honoured_as_written(
    tmp_path: Path, engine: _Engine, model: Path
) -> None:
    """An operator who names one is stating something, not asking for a guess."""

    _run(tmp_path, engine, model, settings={"accelerator": "cpu"}, devices=("cuda:0",))
    assert _call(engine)["flags"]["accelerator"] == "cpu"


def test_the_smiles_column_is_the_one_the_command_declares(
    tmp_path: Path, engine: _Engine, model: Path
) -> None:
    """``--input-col`` is passed rather than left to a default that could move."""

    _run(tmp_path, engine, model, settings={"input_column": "SMILES"})

    call = _call(engine)
    assert call["flags"]["input-col"] == "SMILES"
    assert set(call["rows"][0]) == {"SMILES", "molcascade_parent_id"}


def test_a_smiles_full_of_csv_metacharacters_survives_the_round_trip(
    tmp_path: Path, engine: _Engine, model: Path
) -> None:
    """Written by the csv module, not by a format string.

    A comma inside a bracket expression is an ordinary character in a SMILES and
    a field separator in a CSV. Hand-quoting works until one arrives and then
    predicts on a truncated molecule without complaining.
    """

    awkward = "CC(=O)N[C@@H](Cc1ccccc1)C(=O)O"
    output, response = _run(tmp_path, engine, model, keys=["p1"], extra={"quote": awkward})

    delivered = {row["molcascade_parent_id"]: row for row in _call(engine)["rows"]}
    column = OpenADMETConfig(endpoints=_ENDPOINTS).input_column
    assert delivered["quote"][column] == awkward
    assert ("quote", "herg_pic50") in _predictions(output, response)


# ------------------------------------------------------------------ before the first molecule


def test_a_cascade_authored_without_an_engine_still_validates() -> None:
    """The builder runs on a laptop, so a blank path has to pass validation.

    Nothing is weakened by allowing it: the preflight refuses the run on the
    host that would do the work, naming the field, its variable, where it looked
    and the command that installs it.
    """

    config = OpenADMETConfig(endpoints=_ENDPOINTS)
    assert config.executable == ""
    assert config.model_dir == ()
    assert config.accelerator == "auto"


def test_a_command_name_where_a_path_belongs_is_refused() -> None:
    """That environment is deliberately not on this one's PATH."""

    with pytest.raises(ValidationError, match="MOLCASCADE_OPENADMET_EXECUTABLE"):
        OpenADMETConfig(executable="openadmet", endpoints=_ENDPOINTS)


def test_the_declared_paths_are_the_ones_a_host_can_answer(monkeypatch: Any) -> None:
    """Two paths, both nameable by an environment variable, both discoverable."""

    fields = {spec.field: spec for spec in OpenADMETConfig.installed_paths()}
    assert set(fields) == {"executable", "model_dir"}
    assert fields["model_dir"].kind == "directory"
    # A clone whose git-lfs objects were never pulled is a directory of pointer
    # files that looks complete and loads nothing, so the layout is checked --
    # and checked at the level the CLI is given, not at the clone's root.
    assert fields["model_dir"].contents == ("model.pth", "model.json")
    assert fields["model_dir"].candidates[0].name == "anvil_training"
    for spec in fields.values():
        assert spec.variable("openadmet").startswith("MOLCASCADE_OPENADMET_")
        assert spec.candidates
        assert spec.remedy == "bash envs/bootstrap.sh openadmet"


def test_a_machine_variable_answers_both_paths(
    tmp_path: Path, engine: _Engine, model: Path, monkeypatch: Any
) -> None:
    """Where a tool is installed is a fact about a machine, not about a campaign."""

    monkeypatch.setenv("MOLCASCADE_OPENADMET_EXECUTABLE", str(engine.path))
    monkeypatch.setenv("MOLCASCADE_OPENADMET_MODEL_DIR", str(model))
    config = OpenADMETConfig(endpoints=_ENDPOINTS)
    assert config.executable == str(engine.path)
    assert config.model_dir == (str(model),)


def test_an_absent_model_is_refused_before_a_molecule_is_read(
    tmp_path: Path, engine: _Engine
) -> None:
    """Naming the fix rather than letting the child fail on a pointer file."""

    with pytest.raises(PluginError) as error:
        _run(tmp_path, engine, tmp_path / "never-cloned")
    assert error.value.code == "OPENADMET_MODEL_MISSING"
    assert "bootstrap.sh openadmet" in (error.value.hint or "")
    assert not engine.record_path.exists()


def test_a_clone_whose_large_files_were_never_pulled_is_named_as_that(
    tmp_path: Path, engine: _Engine, model: Path
) -> None:
    """The checkpoint is 49 MB and LFS-tracked, so this is the likely install fault.

    A plain ``git clone`` without the filter leaves a file of exactly the right
    name in exactly the right place holding 130 bytes of pointer text. It is a
    file, so existence passes; the error it eventually produces comes from a
    deserializer complaining about a corrupt archive, which sends the reader
    after the wrong problem entirely.
    """

    (model / "model.pth").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0f7c1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7\n"
        "size 51657071\n",
        encoding="utf-8",
    )

    with pytest.raises(PluginError) as error:
        _run(tmp_path, engine, model)

    assert error.value.code == "OPENADMET_MODEL_NOT_PULLED"
    assert error.value.context["pointers"] == ["model.pth"]
    assert "git lfs pull" in (error.value.hint or "")
    assert not engine.record_path.exists(), "the engine is never asked to load a pointer"


def test_a_missing_executable_names_the_command_that_provisions_it(
    tmp_path: Path, model: Path
) -> None:
    with pytest.raises(PluginError) as error:
        _run(tmp_path, _Engine(tmp_path / "absent", tmp_path / "unused.json"), model)
    assert error.value.code == "OPENADMET_EXECUTABLE_MISSING"
    assert "bootstrap.sh openadmet" in (error.value.hint or "")


def test_a_non_executable_file_is_a_different_mistake(tmp_path: Path, model: Path) -> None:
    """Present but not chmod'ed sends an operator looking somewhere else."""

    path = tmp_path / "openadmet-noexec"
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o644)
    with pytest.raises(PluginError) as error:
        _run(tmp_path, _Engine(path, tmp_path / "unused.json"), model)
    assert error.value.code == "OPENADMET_EXECUTABLE_MISSING"
    assert "executable by this user" in str(error.value)


def test_an_empty_population_is_refused_rather_than_charged_for(
    tmp_path: Path, engine: _Engine, model: Path
) -> None:
    root = tmp_path / "empty-parents"
    root.mkdir()
    pq.write_table(pa.Table.from_pylist([], schema=PARENT_V1.schema), root / "part.parquet")
    with pytest.raises(PluginError) as error:
        OpenADMETPredictorPlugin().execute(
            StageRequest(
                stage_id="openadmet-empty",
                inputs={
                    "parents": StageInput(
                        ref=ArtifactRef(
                            artifact_id="artifact:sha256:" + "b" * 64, kind=PARENT_V1.id
                        ),
                        root=root,
                        contract_id=PARENT_V1.id,
                    )
                },
                config={
                    "executable": str(engine.path),
                    "model_dir": str(model),
                    "endpoints": _ENDPOINTS,
                    "scratch_dir": str(tmp_path),
                },
            ),
            StageContext(tmp_path / "out-empty", StageResources(devices=("cpu",))),
        )
    assert error.value.code == "OPENADMET_EMPTY_INPUT"


# --------------------------------------------------------------------- reading one cell


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("5.4", 5.4),
        (" 5.4 ", 5.4),
        ("", None),
        ("   ", None),
        (None, None),
        ("NaN", None),
        ("nan", None),
        ("inf", None),
        ("n/a", None),
    ],
)
def test_a_blank_or_non_finite_cell_is_not_a_number(cell: str | None, expected: Any) -> None:
    """Empty, NA and NaN all mean the same thing and all become null.

    OpenADMET writes an empty standard deviation for a single model, and
    chemprop writes a NaN for a molecule its featurizer could not build. Neither
    is an error in a batch of thousands, and neither is a number.
    """

    assert _finite(cell) == expected


def test_the_catalogue_defaults_describe_this_adapter() -> None:
    """The builder's boxes and the plugin's fields are one statement, not two."""

    from molcascade.cascade.catalog import CRITERIA_BY_ID

    option = next(
        candidate
        for candidate in CRITERIA_BY_ID["admet_endpoint"].options
        if candidate.id == "openadmet"
    )
    config = OpenADMETConfig.model_validate(dict(option.defaults))
    assert [endpoint.endpoint_id for endpoint in config.endpoints] == ["herg_pic50"]
    assert config.endpoints[0].output_column == _PRED_COLUMN
    assert config.endpoints[0].std_column == _STD_COLUMN
    # A different endpoint id from the ADMET-AI arm, because one is a
    # probability and the other a potency; prediction/v1 exists to keep those
    # from collapsing into one column.
    assert option.gate_defaults["endpoint_id"] == "herg_pic50"
    descriptor = OpenADMETPredictorPlugin.descriptor
    assert option.plugin_ref == f"{descriptor.id}@{descriptor.version}"
