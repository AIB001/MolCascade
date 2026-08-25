"""The AiZynthFinder route-search adapter.

AiZynthFinder itself is not installed here and must not be -- it pins RDKit and
NumPy backwards, which is the whole reason this adapter exists.  So the thing
under test is the process boundary, not the tree search: what the adapter sends
across it, what it refuses to start without, and what it does with what comes
back.

``aizynthcli`` is therefore replaced by a small executable script that reads the
real argument list, writes the real output format (a pandas ``orient="table"``
document), and can be told to lie in each of the ways a future release might.
That is a fake of the *protocol*, which is what this code owns; a fake of the
chemistry would only be a fixture of numbers this project made up.

The alignment tests deserve a note.  Results are joined back to parents by
position because every released ``aizynthcli`` returns one record per input line
in order.  A join that is right by convention and unchecked is how step counts
end up attributed to the wrong molecules, so the adapter verifies each row's
target and stops if it disagrees -- and these tests are the reason to believe it
actually does.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import PARENT_V1, SYNTHESIS_SCORE_V1
from molcascade.errors import PluginError
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.aizynthfinder import (
    AiZynthFinderConfig,
    AiZynthFinderRoutePlugin,
)
from molcascade.plugins.manifest import Cardinality, Determinism, PluginKind
from molcascade.plugins.registry import create_builtin_registry

#: Ethanol, aspirin, caffeine.  Written non-canonically on purpose: the adapter
#: canonicalizes before it writes the SMILES file, and the fake asserts on the
#: canonical forms, so a regression that passed the raw strings through would
#: show up here rather than in a molecule that happens to round-trip.
_SMILES = ["OCC", "CC(=O)Oc1ccccc1C(=O)O", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"]


def _canonical(smiles: list[str]) -> list[str]:
    from rdkit import Chem

    return [Chem.MolToSmiles(Chem.MolFromSmiles(entry)) for entry in smiles]


# --------------------------------------------------------------- the fake CLI

#: A stand-in for ``aizynthcli``.  It parses the flags the adapter passes,
#: echoes the targets it was given back into the results, and reads its
#: behaviour from a literal spliced in above it, so each test can bend one
#: thing without the script having to find a side-file of its own.
_FAKE_CLI = '''\
import json
import sys
from pathlib import Path

behaviour = json.loads(_BEHAVIOUR)

arguments = {}
key = None
for token in sys.argv[1:]:
    if token.startswith("--"):
        key = token[2:]
        arguments[key] = []
    else:
        arguments[key].append(token)

Path(behaviour["argv_path"]).write_text(json.dumps(arguments))

if behaviour.get("exit_code"):
    sys.stderr.write(behaviour.get("message", "policy file not found"))
    sys.exit(behaviour["exit_code"])

targets = Path(arguments["smiles"][0]).read_text().split()
# Recorded here rather than left on disk: the adapter deletes its scratch
# directory, so by the time a test looks, the input file is gone.
arguments["_targets"] = targets
Path(behaviour["argv_path"]).write_text(json.dumps(arguments))

records = []
for index, target in enumerate(targets):
    solved = index not in behaviour.get("unsolved", [])
    records.append(
        {
            "index": index,
            "target": behaviour.get("targets", {}).get(str(index), target),
            "is_solved": solved,
            "number_of_steps": behaviour.get("steps", {}).get(str(index), index + 1)
            if solved
            else 0,
            "top_score": 0.99 if solved else 0.5,
        }
    )
if behaviour.get("drop_last"):
    records.pop()

document = {"schema": {"primaryKey": ["index"]}, "data": records}
Path(arguments["output"][0]).write_text(json.dumps(document))
'''


@pytest.fixture
def cli(tmp_path: Path) -> Any:
    """Return a factory that writes the fake ``aizynthcli`` and its behaviour."""

    executable = tmp_path / "aizynthcli"
    argv_path = tmp_path / "argv.json"

    def configure(**behaviour: Any) -> Path:
        behaviour.setdefault("argv_path", str(argv_path))
        # A shebang rather than a copy of the interpreter: the adapter checks
        # that the configured path is a regular executable file, which is what
        # a console script is.
        executable.write_text(
            f"#!{sys.executable}\n_BEHAVIOUR = {json.dumps(json.dumps(behaviour))}\n{_FAKE_CLI}",
            encoding="utf-8",
        )
        executable.chmod(0o755)
        return executable

    configure.argv_path = argv_path  # type: ignore[attr-defined]
    return configure


@pytest.fixture
def az_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yml"
    path.write_text("expansion:\n  uspto:\n    - policy.onnx\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------- harness


def _parent(parent_id: str, smiles: str) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "identity_policy_id": "identity-policy:test",
        "parent_smiles": smiles,
        "registration_key": f"registration-{parent_id}",
        "stereo_key": None,
        "formula": None,
        "duplicate_count": 1,
    }


def _input(root: Path, smiles: list[str]) -> StageInput:
    input_root = root / "input"
    input_root.mkdir(parents=True, exist_ok=True)
    rows = [_parent(f"m{index}", entry) for index, entry in enumerate(smiles)]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=PARENT_V1.schema),
        input_root / "parents.parquet",
    )
    return StageInput(
        ref=ArtifactRef(artifact_id="artifact:sha256:" + "d" * 64, kind=PARENT_V1.id),
        root=input_root,
        contract_id=PARENT_V1.id,
    )


def _run(
    root: Path,
    config: dict[str, object],
    *,
    smiles: list[str] | None = None,
    staging_name: str = "out",
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    staging = root / staging_name
    response = AiZynthFinderRoutePlugin().execute(
        StageRequest(
            stage_id="retrosynthesis",
            inputs={"parents": _input(root, _SMILES if smiles is None else smiles)},
            config=config,
        ),
        StageContext(staging),
    )
    parents = pq.read_table(staging / response.outputs["primary"].file_paths[0])
    scores = pq.read_table(staging / response.outputs["synthesis_scores"].file_paths[0])
    PARENT_V1.validate(parents)
    SYNTHESIS_SCORE_V1.validate(scores)
    return response, parents.to_pylist(), scores.to_pylist()


def _config(executable: Path, az_config: Path, **overrides: object) -> dict[str, object]:
    settings: dict[str, object] = {
        "schema_version": 1,
        "executable": str(executable),
        "config_path": str(az_config),
    }
    settings.update(overrides)
    return settings


# ------------------------------------------------------------------ contract


def test_the_plugin_registers_as_a_one_to_one_synthesis_stage() -> None:
    registry = create_builtin_registry()
    descriptor = registry.get("synthesis.aizynthfinder@0.1.0").descriptor
    assert descriptor.kind is PluginKind.SYNTHESIS
    # Two runs of the same molecule under a wall-clock budget can find
    # different routes, and the step count moves with them.  Claiming anything
    # stronger would be claiming a reproducibility this backend does not offer.
    assert descriptor.determinism is Determinism.NON_DETERMINISTIC
    # Scoring is evidence, not a verdict: every molecule comes out again and a
    # separate gate decides what the number means.
    assert descriptor.cardinality is Cardinality.ONE_TO_ONE
    assert descriptor.output_ports == {
        "primary": PARENT_V1.id,
        "synthesis_scores": SYNTHESIS_SCORE_V1.id,
    }


def test_the_catalogue_option_points_at_the_registered_adapter() -> None:
    """The builder offers route search only because this adapter exists."""

    from molcascade.cascade.availability import option_availability
    from molcascade.cascade.catalog import CRITERIA

    criterion = next(entry for entry in CRITERIA if entry.id == "retrosynthesis_routes")
    option = next(entry for entry in criterion.options if entry.id == "aizynthfinder")
    assert option.plugin_ref == "synthesis.aizynthfinder@0.1.0"
    # Emphatically not ("aizynthfinder",): that package must be absent from this
    # environment, so probing for it would report the healthy case as broken.
    assert option.requires == ()
    assert option_availability(option, create_builtin_registry()).runnable


# --------------------------------------------------------------- the happy path


def test_a_solved_target_scores_its_step_count(
    tmp_path: Path, cli: Any, az_config: Path
) -> None:
    executable = cli(steps={"0": 1, "1": 2, "2": 5})
    response, parents, scores = _run(tmp_path, _config(executable, az_config))

    assert [row["parent_id"] for row in parents] == ["m0", "m1", "m2"]
    assert [row["score"] for row in scores] == [1.0, 2.0, 5.0]
    assert {row["direction"] for row in scores} == {"HIGHER_HARDER"}
    assert response.metadata["solved_count"] == 3
    assert response.metadata["unsolved_count"] == 0
    # A route search is not a proxy, and the metadata says so where every other
    # synthesis adapter says the opposite.
    assert response.metadata["proxy_not_route"] is False


def test_the_targets_reach_the_command_line_canonicalized_and_in_order(
    tmp_path: Path, cli: Any, az_config: Path
) -> None:
    executable = cli()
    _run(tmp_path, _config(executable, az_config, nproc=2, policy=["uspto"], stocks=["zinc"]))

    arguments = json.loads(cli.argv_path.read_text())
    assert arguments["config"] == [str(az_config)]
    assert arguments["nproc"] == ["2"]
    assert arguments["policy"] == ["uspto"]
    assert arguments["stocks"] == ["zinc"]
    assert arguments["_targets"] == _canonical(_SMILES)


def test_the_method_id_follows_the_configuration_that_defines_the_search(
    tmp_path: Path, cli: Any, az_config: Path
) -> None:
    """Two policies are two methods, whatever else the stage looks like.

    ``synthesis_score/v1`` requires distinct methods to keep distinct
    identities, and for route search the configuration file *is* the method:
    it names the expansion policy and the stock, which together decide both
    what counts as a step and what counts as purchasable.
    """

    executable = cli()
    first, _, _ = _run(tmp_path, _config(executable, az_config), staging_name="a")
    az_config.write_text("expansion:\n  other:\n    - different.onnx\n", encoding="utf-8")
    second, _, _ = _run(tmp_path, _config(executable, az_config), staging_name="b")

    assert first.metadata["method_id"].startswith("synthesis-route:sha256:")
    assert first.metadata["method_id"] != second.metadata["method_id"]
    # And it is distinct from the proxies by construction, not by luck.
    assert not first.metadata["method_id"].startswith("synthesis-sa:")


def test_pinning_the_backend_version_changes_the_method_and_clears_the_warning(
    tmp_path: Path, cli: Any, az_config: Path
) -> None:
    executable = cli()
    unpinned, _, loose = _run(tmp_path, _config(executable, az_config), staging_name="a")
    pinned, _, tight = _run(
        tmp_path,
        _config(executable, az_config, backend_version="4.4.1"),
        staging_name="b",
    )

    assert json.loads(loose[0]["warning_codes_json"]) == ["ROUTE_BACKEND_VERSION_UNPINNED"]
    assert json.loads(tight[0]["warning_codes_json"]) == []
    assert unpinned.metadata["method_id"] != pinned.metadata["method_id"]
    assert pinned.metadata["backend_version"] == "4.4.1"


# ------------------------------------------------------------------- unsolved


def test_an_unsolved_target_is_flagged_rather_than_scored_zero(
    tmp_path: Path, cli: Any, az_config: Path
) -> None:
    """The column is not nullable, so "no route" has to be a number.

    Zero is the wrong number: on a HIGHER_HARDER axis it reads as trivially
    easy, which is the opposite of what happened.  The sentinel sits above any
    real step count and the row carries the reason.
    """

    executable = cli(unsolved=[1])
    response, _, scores = _run(tmp_path, _config(executable, az_config, unsolved_score=42.0))

    assert scores[1]["score"] == 42.0
    assert json.loads(scores[1]["warning_codes_json"]) == [
        "ROUTE_NOT_FOUND",
        "ROUTE_BACKEND_VERSION_UNPINNED",
    ]
    assert "no route" in (scores[1]["domain"] or "")
    assert scores[0]["domain"] != scores[1]["domain"]
    assert response.metadata["unsolved_count"] == 1


def test_the_default_sentinel_is_rejected_by_the_gate_it_ships_with(
    tmp_path: Path, cli: Any, az_config: Path
) -> None:
    """The pairing in the catalogue has to actually drop unsolved molecules."""

    from molcascade.cascade.catalog import CRITERIA

    criterion = next(entry for entry in CRITERIA if entry.id == "retrosynthesis_routes")
    option = next(entry for entry in criterion.options if entry.id == "aizynthfinder")
    assert option.gate_defaults["expected_direction"] == "HIGHER_HARDER"
    assert option.defaults["unsolved_score"] > option.gate_defaults["maximum"]


# ------------------------------------------------------------- fail-closed


def test_a_missing_executable_stops_before_a_molecule_is_read(
    tmp_path: Path, az_config: Path
) -> None:
    with pytest.raises(PluginError) as raised:
        _run(tmp_path, _config(tmp_path / "nowhere" / "aizynthcli", az_config))
    assert raised.value.code == "AIZYNTHFINDER_EXECUTABLE_MISSING"
    assert not (tmp_path / "out").exists()


def test_a_non_executable_file_is_not_an_installation(
    tmp_path: Path, az_config: Path
) -> None:
    impostor = tmp_path / "aizynthcli"
    impostor.write_text("#!/bin/sh\n", encoding="utf-8")
    impostor.chmod(0o644)
    with pytest.raises(PluginError) as raised:
        _run(tmp_path, _config(impostor, az_config))
    assert raised.value.code == "AIZYNTHFINDER_EXECUTABLE_MISSING"


def test_a_bare_command_name_is_refused_rather_than_resolved_on_path(
    tmp_path: Path, az_config: Path
) -> None:
    """PATH is the wrong place to look, so it is never looked at.

    The isolated environment must not be on this process's PATH -- that is what
    isolation means here -- so a name that resolved would be finding the very
    installation that must not be used.
    """

    with pytest.raises(PluginError) as raised:
        _run(tmp_path, _config(Path("aizynthcli"), az_config))
    assert raised.value.code == "PLUGIN_CONFIG_INVALID"


def test_a_missing_configuration_file_stops_the_stage(
    tmp_path: Path, cli: Any
) -> None:
    executable = cli()
    with pytest.raises(PluginError) as raised:
        _run(tmp_path, _config(executable, tmp_path / "absent.yml"))
    assert raised.value.code == "AIZYNTHFINDER_CONFIG_UNREADABLE"


def test_a_failing_search_reports_what_aizynthfinder_said(
    tmp_path: Path, cli: Any, az_config: Path
) -> None:
    executable = cli(exit_code=3, message="FileNotFoundError: uspto_model.onnx")
    with pytest.raises(PluginError) as raised:
        _run(tmp_path, _config(executable, az_config))
    assert raised.value.code == "AIZYNTHFINDER_RUN_FAILED"
    assert "uspto_model.onnx" in str(raised.value.context["output_tail"])
    assert not (tmp_path / "out" / "datasets" / "parents" / "part-00000.parquet").exists()


def test_more_molecules_than_the_cap_stops_before_the_search(
    tmp_path: Path, cli: Any, az_config: Path
) -> None:
    """Tree search is seconds per molecule; the cap is what keeps that visible."""

    executable = cli()
    with pytest.raises(PluginError) as raised:
        _run(tmp_path, _config(executable, az_config, max_molecules=2))
    assert raised.value.code == "AIZYNTHFINDER_INPUT_TOO_LARGE"
    # Not "it ran and then complained": the fake records every invocation, and
    # there was none.
    assert not cli.argv_path.exists()


def test_a_short_result_set_is_a_misalignment_not_a_partial_answer(
    tmp_path: Path, cli: Any, az_config: Path
) -> None:
    executable = cli(drop_last=True)
    with pytest.raises(PluginError) as raised:
        _run(tmp_path, _config(executable, az_config))
    assert raised.value.code == "AIZYNTHFINDER_OUTPUT_MISALIGNED"


def test_a_reordered_result_set_is_caught_by_the_target_check(
    tmp_path: Path, cli: Any, az_config: Path
) -> None:
    """The join is positional, so the position has to be verified.

    A future release that deduplicated or reordered targets would otherwise
    attribute every step count to the wrong molecule with no error at all.
    """

    canonical = _canonical(_SMILES)
    executable = cli(targets={"0": canonical[1], "1": canonical[0]})
    with pytest.raises(PluginError) as raised:
        _run(tmp_path, _config(executable, az_config))
    assert raised.value.code == "AIZYNTHFINDER_OUTPUT_MISALIGNED"
    assert raised.value.context["row"] == 0


def test_an_empty_input_is_refused(tmp_path: Path, cli: Any, az_config: Path) -> None:
    executable = cli()
    with pytest.raises(PluginError) as raised:
        _run(tmp_path, _config(executable, az_config), smiles=[])
    assert raised.value.code == "SYNTHESIS_EMPTY_INPUT"


def test_an_unparseable_parent_stops_the_stage(
    tmp_path: Path, cli: Any, az_config: Path
) -> None:
    executable = cli()
    with pytest.raises(PluginError) as raised:
        _run(tmp_path, _config(executable, az_config), smiles=["CCO", "not-a-molecule"])
    assert raised.value.code == "SYNTHESIS_PARENT_INVALID"


# ------------------------------------------------------------------ settings


def test_the_search_environment_does_not_inherit_this_one_s_import_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PYTHONPATH is exactly how one environment's RDKit reaches another's.

    Since the reason AiZynthFinder runs out of process is that its RDKit and
    NumPy must not be this process's, letting the variable through would
    reintroduce the conflict after the run had already started.
    """

    from molcascade.plugins.builtin.aizynthfinder import _isolated_environment

    monkeypatch.setenv("PYTHONPATH", "/somewhere/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/somewhere")
    monkeypatch.setenv("MOLCASCADE_KEEP_ME", "yes")
    environment = _isolated_environment()
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert environment["MOLCASCADE_KEEP_ME"] == "yes"


@pytest.mark.parametrize("flag", ["--nproc", "-x", "with space", ""])
def test_pass_through_names_cannot_smuggle_in_an_argument(flag: str) -> None:
    """The names go onto an argv list, so they must be names and not flags."""

    with pytest.raises(ValueError):
        AiZynthFinderConfig(
            executable="/opt/aizynth/bin/aizynthcli",
            config_path="/opt/aizynth/config.yml",
            policy=(flag,),
        )


def test_the_whole_call_is_bounded_by_the_per_molecule_budget(
    tmp_path: Path, cli: Any, az_config: Path
) -> None:
    """A hung search must not become a run that never ends.

    The fake sleeps past a one-second budget, which is the only way to prove
    the timeout is wired to the subprocess rather than merely configured.
    """

    script = tmp_path / "sleeper"
    script.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(30)\n", encoding="utf-8")
    script.chmod(0o755)
    with pytest.raises(PluginError) as raised:
        _run(
            tmp_path,
            _config(script, az_config, timeout_per_molecule_seconds=1.0),
        )
    assert raised.value.code == "AIZYNTHFINDER_TIMEOUT"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics")
def test_the_output_is_read_from_the_scratch_directory_that_was_configured(
    tmp_path: Path, cli: Any, az_config: Path
) -> None:
    """A small /tmp is a real deployment, so the location has to be settable."""

    scratch = tmp_path / "big-disk"
    scratch.mkdir()
    executable = cli()
    _run(tmp_path, _config(executable, az_config, scratch_dir=str(scratch)))

    arguments = json.loads(cli.argv_path.read_text())
    assert Path(arguments["output"][0]).is_relative_to(scratch)
    # And it is cleaned up: the trees file is the largest thing this stage
    # touches and leaving it behind would fill the disk it was moved to.
    assert list(scratch.iterdir()) == []
