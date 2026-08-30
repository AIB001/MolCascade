"""What MolCascade changes about a receptor, and what it refuses to change.

The module under test edits the operator's protein, which the target module
spent its whole docstring arguing against.  The argument it was actually making
was against *silent* edits: meeko's ``--allow_bad_res``, the only previous route
past an incomplete residue, deletes that residue backbone and all, and nothing
downstream can tell.  So the tests here are less about "did pdbfixer run" than
about the two rules that separate a repair from a truncation -- nothing absent
from the model is invented, and nothing removed goes unnamed.

The fixture is a real fragment of STK17B, trimmed by hand: LYS43 stops at CB in
the deposition exactly as written here, which is the case this module exists
for.  Around it sit the other four things a crystal structure brings along --
waters, a cryoprotectant, an ion, and a selenomethionine -- plus a one-residue
hole, so that one pass over one file exercises every branch of the report.

Half the tests never import pdbfixer.  Repair is on by default, so its refusal
when the package is absent, its cache keying and its digest bookkeeping are all
reachable on a machine that cannot run it, and those are the paths an operator
meets first.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from molcascade.cascade.models import BoxConfig, ReceptorPreparation, TargetConfig
from molcascade.cascade.receptor import (
    PreparationReport,
    prepare_receptor_pdb,
    recipe_digest,
)
from molcascade.cascade.target import resolve_target
from molcascade.errors import ConfigError

#: Six residues of STK17B's glycine-rich loop, one selenomethionine, and the
#: solvent.  Written out rather than fetched because every assertion below
#: names something in it: LYS43 has no CG onwards, residue 45 is not here at
#: all, and the zinc is a lone atom while the ethylene glycol is not.
_STRUCTURE = """\
ATOM      1  N   GLY A  40      22.961  42.231  -2.089  1.00 20.00           N
ATOM      2  CA  GLY A  40      22.919  41.065  -2.996  1.00 20.00           C
ATOM      3  C   GLY A  40      22.244  41.338  -4.324  1.00 20.00           C
ATOM      4  O   GLY A  40      21.420  42.227  -4.421  1.00 20.00           O
ATOM      5  N   ALA A  41      22.573  40.552  -5.344  1.00 20.00           N
ATOM      6  CA  ALA A  41      21.935  40.673  -6.668  1.00 20.00           C
ATOM      7  C   ALA A  41      21.731  39.290  -7.283  1.00 20.00           C
ATOM      8  O   ALA A  41      22.652  38.528  -7.350  1.00 20.00           O
ATOM      9  CB  ALA A  41      22.811  41.516  -7.613  1.00 20.00           C
ATOM     10  N   GLY A  42      20.527  38.976  -7.723  1.00 20.00           N
ATOM     11  CA  GLY A  42      20.268  37.751  -8.484  1.00 20.00           C
ATOM     12  C   GLY A  42      19.561  38.165  -9.758  1.00 20.00           C
ATOM     13  O   GLY A  42      19.501  39.346 -10.039  1.00 20.00           O
ATOM     14  N   LYS A  43      18.979  37.225 -10.499  1.00 20.00           N
ATOM     15  CA  LYS A  43      18.435  37.554 -11.828  1.00 20.00           C
ATOM     16  C   LYS A  43      17.232  38.503 -11.752  1.00 20.00           C
ATOM     17  O   LYS A  43      17.109  39.405 -12.612  1.00 20.00           O
ATOM     18  CB  LYS A  43      18.059  36.292 -12.629  1.00 20.00           C
ATOM     19  N   ALA A  44      16.373  38.295 -10.744  1.00 20.00           N
ATOM     20  CA  ALA A  44      15.107  39.035 -10.590  1.00 20.00           C
ATOM     21  C   ALA A  44      14.938  39.710  -9.236  1.00 20.00           C
ATOM     22  O   ALA A  44      13.825  40.048  -8.849  1.00 20.00           O
ATOM     23  CB  ALA A  44      13.925  38.101 -10.866  1.00 20.00           C
ATOM     24  N   ALA A  46      17.319  42.375  -6.242  1.00 20.00           N
ATOM     25  CA  ALA A  46      18.513  43.032  -5.730  1.00 20.00           C
ATOM     26  C   ALA A  46      18.202  43.481  -4.311  1.00 20.00           C
ATOM     27  O   ALA A  46      17.103  43.949  -4.017  1.00 20.00           O
ATOM     28  CB  ALA A  46      18.895  44.205  -6.617  1.00 20.00           C
HETATM   29  N   MSE B 182      11.525  29.905  -2.369  1.00 20.00           N
HETATM   30  CA  MSE B 182      11.286  28.480  -2.687  1.00 20.00           C
HETATM   31  C   MSE B 182      10.331  27.759  -1.722  1.00 20.00           C
HETATM   32  O   MSE B 182       9.902  26.662  -2.040  1.00 20.00           O
HETATM   33  CB  MSE B 182      12.600  27.686  -2.685  1.00 20.00           C
HETATM   34  CG  MSE B 182      13.676  28.161  -3.674  1.00 20.00           C
HETATM   35  SE  MSE B 182      13.139  28.338  -5.383  1.00 20.00          SE
HETATM   36  CE  MSE B 182      12.313  26.765  -5.701  1.00 20.00           C
HETATM   37 ZN    ZN A 500      19.000  39.000  -9.000  1.00 20.00          ZN
HETATM   38  O   HOH A 601      25.000  45.000  -1.000  1.00 20.00           O
HETATM   39  O   HOH A 602      26.000  46.000  -2.000  1.00 20.00           O
HETATM   40  C1  EDO A 701      27.000  47.000  -3.000  1.00 20.00           C
HETATM   41  O1  EDO A 701      28.000  47.500  -3.500  1.00 20.00           O
HETATM   42  C2  EDO A 701      26.500  48.200  -2.400  1.00 20.00           C
HETATM   43  O2  EDO A 701      25.600  48.900  -3.200  1.00 20.00           O
END
"""

#: The site, stated outright, so that nothing here depends on box derivation.
_BOX = BoxConfig(center_x=19.0, center_y=39.0, center_z=-9.0, size_x=20.0, size_y=20.0, size_z=20.0)

#: Docking needs the PDB and the site; nothing here asks for a PDBQT, which
#: would need meeko as well.
_NEEDS = frozenset({"receptor_path", "center_x", "size_x"})


@pytest.fixture
def receptor(tmp_path: Path) -> Path:
    path = tmp_path / "protein.pdb"
    path.write_text(_STRUCTURE, encoding="utf-8")
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _residues(path: Path) -> list[str]:
    """``A:LYS43`` for every residue in a PDB, in file order, once each."""

    seen: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        label = f"{line[21]}:{line[17:20].strip()}{line[22:26].strip()}"
        if label not in seen:
            seen.append(label)
    return seen


def _atoms(path: Path, label: str) -> set[str]:
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if f"{line[21]}:{line[17:20].strip()}{line[22:26].strip()}" == label:
            names.add(line[12:16].strip())
    return names


# --------------------------------------------------------------------------
# Reachable without pdbfixer: the decisions, not the chemistry.
# --------------------------------------------------------------------------


def test_the_skip_switch_hands_the_engines_the_operator_s_own_file(
    receptor: Path, tmp_path: Path
) -> None:
    """``--no-receptor-prepare`` has to mean the bytes, not a gentler recipe.

    It is the documented way out for a receptor this module makes worse -- a
    hand-built model, a structure already prepared elsewhere -- so it must leave
    nothing behind at all, not even a cached copy that a later flag could pick
    up.
    """

    resolved = resolve_target(
        TargetConfig(
            name="site",
            receptor_path=str(receptor),
            preparation=ReceptorPreparation(enabled=False),
            box=_BOX,
        ),
        requirements=_NEEDS,
        workspace=tmp_path / "workspace",
    )

    assert resolved is not None
    assert resolved.target.receptor_path == str(receptor)
    assert resolved.notes == ()
    assert not (tmp_path / "workspace").exists()


def test_a_missing_pdbfixer_names_both_ways_to_install_it(
    receptor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail closed, and fail with the command that fixes it.

    Repair is on by default, so on a machine without pdbfixer this refusal is
    the first thing a docking run does.  It has to carry the standalone install
    as well as the extra: someone who did not install MolCascade's optional
    stack still wants the two packages, not the other four.
    """

    monkeypatch.setitem(sys.modules, "pdbfixer", None)

    with pytest.raises(ConfigError) as raised:
        prepare_receptor_pdb(
            receptor, options=ReceptorPreparation(), workspace=tmp_path / "workspace"
        )

    error = raised.value
    assert error.code == "DOCKING_RECEPTOR_REPAIR_UNAVAILABLE"
    hint = error.hint or ""
    assert "pip install pdbfixer" in hint
    assert 'pip install "molcascade[docking]"' in hint
    # And the way to proceed without it, since the two installs are not the
    # only answer to "I cannot install anything right now".
    assert "--no-receptor-prepare" in hint


def test_an_unreadable_structure_is_refused_by_name_not_by_traceback(
    tmp_path: Path,
) -> None:
    """An mmCIF wearing a .pdb suffix is the common way to arrive here."""

    pytest.importorskip("pdbfixer")
    receptor = tmp_path / "protein.pdb"
    receptor.write_text("data_1ABC\n_entry.id 1ABC\n", encoding="utf-8")

    with pytest.raises(ConfigError) as raised:
        prepare_receptor_pdb(
            receptor, options=ReceptorPreparation(), workspace=tmp_path / "workspace"
        )

    assert raised.value.code == "DOCKING_RECEPTOR_REPAIR_FAILED"
    assert raised.value.context["receptor_path"] == str(receptor)


def test_a_lone_ion_is_a_metal_and_the_same_element_in_a_cofactor_is_not() -> None:
    """The whitelist is about what deleting the thing would mean.

    A zinc sitting in the site coordinates the ligand, and a run against a
    pocket without it is a different experiment.  The iron in a haem is not a
    free ion; it belongs to a cofactor that is either kept or removed whole,
    which is what counting the atoms decides.
    """

    from molcascade.cascade.receptor import _is_metal

    def residue(*symbols: str) -> SimpleNamespace:
        atoms = [SimpleNamespace(element=SimpleNamespace(symbol=symbol)) for symbol in symbols]
        return SimpleNamespace(atoms=lambda: iter(atoms))

    assert _is_metal(residue("Zn"))
    assert _is_metal(residue("Mg"))
    assert not _is_metal(residue("S"))
    assert not _is_metal(residue("Fe", "N", "N", "N", "N"))
    # A residue whose element the file never declared is not promoted to a metal
    # on the strength of its name.
    assert not _is_metal(SimpleNamespace(atoms=lambda: iter([SimpleNamespace(element=None)])))


def test_each_recipe_gets_its_own_cache_entry(receptor: Path, tmp_path: Path) -> None:
    """Otherwise a run that kept its waters is served the file that dropped them.

    The prepared structure is a function of the input bytes *and* the options,
    and the input digest alone names only half of that.
    """

    recipes = [
        ReceptorPreparation(),
        ReceptorPreparation(keep_waters=True),
        ReceptorPreparation(keep_heterogens=True),
        ReceptorPreparation(keep_waters=True, keep_heterogens=True),
    ]
    digests = {recipe_digest(recipe) for recipe in recipes}

    assert len(digests) == len(recipes)
    assert all(len(digest) == 8 for digest in digests)
    # Stable across calls, or every run would prepare afresh.
    assert recipe_digest(recipes[0]) == recipe_digest(ReceptorPreparation())


def test_a_report_names_every_change_and_stays_silent_about_the_rest() -> None:
    """These lines are the whole difference from ``--allow_bad_res``.

    They are printed at the top of a run and stored with it, so a score can be
    traced back to the structure that produced it -- including the parts of that
    structure which were not there.
    """

    report = PreparationReport(
        residues_in=11,
        residues_out=8,
        waters_removed=2,
        heterogens_removed=(("EDO", 1),),
        metals_kept=("A:ZN500",),
        nonstandard_replaced=(("B:MSE182", "MET"),),
        atoms_rebuilt=(("A:LYS43", ("CG", "CD", "CE", "NZ")),),
        gaps_not_built=("A:45 unresolved between ALA44 and ALA46",),
    )

    rendered = "\n".join(report.notes())

    assert "11 residues in, 8 out" in rendered
    assert "waters removed: 2" in rendered
    assert "EDO x1" in rendered
    assert "metals kept: A:ZN500" in rendered
    assert "B:MSE182->MET" in rendered
    assert "A:LYS43 CG,CD,CE,NZ" in rendered
    assert "not modelled, not invented" in rendered
    # Nothing was capped, so nothing claims to have been.
    assert "terminals" not in rendered


def test_a_report_with_nothing_to_say_says_that_rather_than_nothing() -> None:
    """Silence reads as "the step did not run", which is a different claim."""

    lines = PreparationReport(residues_in=300, residues_out=300).notes()

    assert lines[-1] == "  no residue was incomplete; nothing was rebuilt"
    assert PreparationReport(residues_in=0, residues_out=0, skipped=True).notes() == (
        "receptor preparation skipped: the structure is docked exactly as supplied",
    )


def test_a_report_survives_the_round_trip_through_its_sidecar() -> None:
    """A cache hit re-reads the report instead of recomputing it.

    If the round trip lost a field, the second run of a cascade would print a
    different account of the same file than the first did.
    """

    report = PreparationReport(
        residues_in=11,
        residues_out=8,
        waters_removed=2,
        heterogens_removed=(("EDO", 1),),
        metals_kept=("A:ZN500",),
        nonstandard_replaced=(("B:MSE182", "MET"),),
        atoms_rebuilt=(("A:LYS43", ("CG", "CD", "CE", "NZ")),),
        terminals_capped=("A:ALA46",),
        gaps_not_built=("A:45",),
    )

    assert PreparationReport.from_json(json.loads(json.dumps(report.to_json()))) == report


def test_the_pin_is_checked_against_the_operator_s_file_and_restated_for_the_workers(
    receptor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``receptor_sha256`` means two files, and both have to be right.

    The operator pinned the structure they looked at, so that is what the pin is
    checked against -- verifying it against the repaired file would break every
    pin the day repair was switched on, for the one reason that is not an error.
    The workers verify the bytes they dock, so the field they are handed is the
    repaired file's digest.  Both appear in the notes; that is the audit chain.
    """

    prepared = tmp_path / "prepared.pdb"
    prepared.write_text("ATOM      1  N   MET A   1       0.000   0.000   0.000\nEND\n")
    report = PreparationReport(residues_in=278, residues_out=278)
    monkeypatch.setattr(
        "molcascade.cascade.target.prepare_receptor_pdb",
        lambda source, *, options, workspace: (prepared, report),
    )

    resolved = resolve_target(
        TargetConfig(
            name="site",
            receptor_path=str(receptor),
            receptor_sha256=_digest(receptor),
            box=_BOX,
        ),
        requirements=_NEEDS,
        workspace=tmp_path / "workspace",
    )

    assert resolved is not None
    assert resolved.target.receptor_path == str(prepared)
    assert resolved.target.receptor_sha256 == _digest(prepared)
    rendered = "\n".join(resolved.notes)
    assert f"receptor read from protein.pdb (sha256 {_digest(receptor)[:12]})" in rendered
    assert f"docking uses prepared.pdb (sha256 {_digest(prepared)[:12]})" in rendered
    assert "278 residues in, 278 out" in rendered


def test_meeko_is_handed_the_repaired_structure_and_not_the_operator_s_file(
    receptor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The receptor has four consumers and the PDBQT branch is the fourth.

    Repair rewrites ``fields["receptor_path"]``; it cannot rewrite ``target``,
    which is frozen.  The PDBQT derivation used to read ``target``, so meeko was
    handed the original while GNINA, KarmaDock and PoseBusters got the repaired
    one -- the "one pocket, two descriptions" split this module exists to avoid.
    On a structure with an incomplete residue it was not even silent: meeko
    refuses those, so the repair ran, succeeded, and the run died anyway on the
    file the repair had already fixed.

    ``receptor_pdbqt_path`` is in the requirements here for that reason.  Every
    other test in this file leaves it out, which is how the bug survived them.
    """

    prepared = tmp_path / "prepared.pdb"
    prepared.write_text("ATOM      1  N   MET A   1       0.000   0.000   0.000\nEND\n")
    monkeypatch.setattr(
        "molcascade.cascade.target.prepare_receptor_pdb",
        lambda source, *, options, workspace: (
            prepared,
            PreparationReport(residues_in=278, residues_out=278),
        ),
    )

    seen: list[Path] = []

    def fake_pdbqt(source: Path, *, workspace: Path) -> tuple[Path, str]:
        seen.append(source)
        return workspace / "receptor.pdbqt", "receptor PDBQT derived with meeko"

    monkeypatch.setattr("molcascade.cascade.target.prepare_receptor_pdbqt", fake_pdbqt)

    resolved = resolve_target(
        TargetConfig(name="site", receptor_path=str(receptor), box=_BOX),
        requirements=_NEEDS | {"receptor_pdbqt_path"},
        workspace=tmp_path / "workspace",
    )

    assert resolved is not None
    assert seen == [prepared], "meeko read the unrepaired structure"


def test_the_skip_switch_leaves_meeko_reading_the_operator_s_file(
    receptor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the rewrite: with repair off there is nothing to swap.

    ``fields["receptor_path"]`` is then the operator's own path, so reading it
    rather than ``target`` has to leave the no-repair case exactly as it was.
    """

    seen: list[Path] = []

    def fake_pdbqt(source: Path, *, workspace: Path) -> tuple[Path, str]:
        seen.append(source)
        return workspace / "receptor.pdbqt", "receptor PDBQT derived with meeko"

    monkeypatch.setattr("molcascade.cascade.target.prepare_receptor_pdbqt", fake_pdbqt)

    resolve_target(
        TargetConfig(
            name="site",
            receptor_path=str(receptor),
            box=_BOX,
            preparation=ReceptorPreparation(enabled=False),
        ),
        requirements=_NEEDS | {"receptor_pdbqt_path"},
        workspace=tmp_path / "workspace",
    )

    assert seen == [receptor]


def test_a_pin_that_does_not_match_stops_the_run_before_anything_is_repaired(
    receptor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pin exists to catch a receptor that changed under a stored cascade.

    Repairing first and then comparing would report the mismatch against a file
    the operator never had, and would have done the work either way.
    """

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the structure was repaired before the pin was checked")

    monkeypatch.setattr("molcascade.cascade.target.prepare_receptor_pdb", refuse)

    with pytest.raises(ConfigError) as raised:
        resolve_target(
            TargetConfig(
                name="site",
                receptor_path=str(receptor),
                receptor_sha256="0" * 64,
                box=_BOX,
            ),
            requirements=_NEEDS,
            workspace=tmp_path / "workspace",
        )

    error = raised.value
    assert error.code == "DOCKING_RECEPTOR_DIGEST_MISMATCH"
    assert error.context["actual_sha256"] == _digest(receptor)


# --------------------------------------------------------------------------
# What pdbfixer actually does to the fixture above.
# --------------------------------------------------------------------------


@pytest.fixture
def prepared(receptor: Path, tmp_path: Path) -> tuple[Path, PreparationReport]:
    pytest.importorskip("pdbfixer")
    return prepare_receptor_pdb(
        receptor, options=ReceptorPreparation(), workspace=tmp_path / "workspace"
    )


def test_a_side_chain_that_stops_at_cb_gets_the_rest_of_itself_back(
    prepared: tuple[Path, PreparationReport],
) -> None:
    """The repair this module exists for.

    LYS43 is truncated at CB in the deposition, which is what made meeko refuse
    STK17B and what ``--allow_bad_res`` answered by deleting the residue.  Here
    the four missing atoms are built and the residue count does not move.
    """

    path, report = prepared

    assert _atoms(path, "A:LYS43") == {"N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"}
    assert ("A:LYS43", ("CG", "CD", "CE", "NZ")) in report.atoms_rebuilt
    # A residue that was already complete is not listed as repaired.
    assert not any(label == "A:ALA44" for label, _ in report.atoms_rebuilt)


def test_an_unresolved_residue_is_reported_and_not_invented(
    prepared: tuple[Path, PreparationReport],
) -> None:
    """The other half of the contract, and the harder half to honour.

    Residue 45 is absent from the model, and a rotamer template would happily
    produce coordinates for it that look exactly as measured as the rest.  The
    fixture carries no SEQRES, so the gap is found the only way it can be: the
    peptide bond between 44 and 46 is 4.7 A long.
    """

    path, report = prepared

    assert "A:ALA45" not in _residues(path)
    assert len(report.gaps_not_built) == 1
    gap = report.gaps_not_built[0]
    assert gap.startswith("A:45 unresolved between ALA44 and ALA46")
    assert "left as a break" in gap


def test_solvent_and_cryoprotectant_go_and_the_ion_stays(
    prepared: tuple[Path, PreparationReport],
) -> None:
    """Waters in the pocket change the docking; a deleted zinc changes the target."""

    path, report = prepared
    remaining = _residues(path)

    assert not [label for label in remaining if "HOH" in label]
    assert not [label for label in remaining if "EDO" in label]
    assert "A:ZN500" in remaining
    assert report.waters_removed == 2
    assert report.heterogens_removed == (("EDO", 1),)
    assert report.metals_kept == ("A:ZN500",)


def test_keeping_the_waters_keeps_them_and_still_drops_the_glycol(
    receptor: Path, tmp_path: Path
) -> None:
    """Structural waters bridge some ligands to their pocket; glycol does not."""

    pytest.importorskip("pdbfixer")

    path, report = prepare_receptor_pdb(
        receptor,
        options=ReceptorPreparation(keep_waters=True),
        workspace=tmp_path / "workspace",
    )
    remaining = _residues(path)

    assert [label for label in remaining if "HOH" in label] == ["A:HOH601", "A:HOH602"]
    assert not [label for label in remaining if "EDO" in label]
    assert report.waters_removed == 0


def test_selenomethionine_becomes_the_methionine_it_stands_for(
    prepared: tuple[Path, PreparationReport],
) -> None:
    """And is converted before heterogens are removed, or it would be deleted.

    MSE is a HETATM and not a standard residue, so the deletion pass would take
    it if the two steps ran the other way round -- losing a real residue out of
    the middle of a chain, which is the failure this whole module is against.
    """

    path, report = prepared

    assert report.nonstandard_replaced == (("B:MSE182", "MET"),)
    assert "B:MET182" in _residues(path)
    assert "SD" in _atoms(path, "B:MET182")
    assert "SE" not in _atoms(path, "B:MET182")


def test_the_author_s_numbering_survives(prepared: tuple[Path, PreparationReport]) -> None:
    """Without ``keepIds`` every chain and residue is renumbered from one.

    Then "LYS43" in the report, in the literature and in every contact analysis
    afterwards would each name a different residue.
    """

    path, _ = prepared

    # Sorted, because the order residues come back in is pdbfixer's business --
    # it moves the ion to the end when it rebuilds the topology -- while their
    # names and numbers are the thing being promised here.
    assert sorted(_residues(path)) == [
        "A:ALA41",
        "A:ALA44",
        "A:ALA46",
        "A:GLY40",
        "A:GLY42",
        "A:LYS43",
        "A:ZN500",
        "B:MET182",
    ]


def test_the_second_run_reads_the_cache_and_still_says_what_changed(
    receptor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache hit that degraded to "reused from cache" would lose the account.

    The report is stored beside the structure for exactly this: the second run
    of a cascade prints the same lines as the first without opening pdbfixer
    again.
    """

    pytest.importorskip("pdbfixer")
    workspace = tmp_path / "workspace"
    first_path, first_report = prepare_receptor_pdb(
        receptor, options=ReceptorPreparation(), workspace=workspace
    )

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the cached structure was prepared a second time")

    monkeypatch.setattr("molcascade.cascade.receptor._open", refuse)
    second_path, second_report = prepare_receptor_pdb(
        receptor, options=ReceptorPreparation(), workspace=workspace
    )

    assert second_path == first_path
    assert second_report == first_report
    assert second_report.notes() == first_report.notes()


def test_a_torn_sidecar_is_a_reason_to_prepare_again_not_to_stop(
    receptor: Path, tmp_path: Path
) -> None:
    """An interrupted run leaves half a file, and the repair is deterministic."""

    pytest.importorskip("pdbfixer")
    workspace = tmp_path / "workspace"
    path, report = prepare_receptor_pdb(
        receptor, options=ReceptorPreparation(), workspace=workspace
    )
    sidecar = path.with_suffix(".json")
    sidecar.write_text('{"residues_in": ', encoding="utf-8")

    again, recovered = prepare_receptor_pdb(
        receptor, options=ReceptorPreparation(), workspace=workspace
    )

    assert again == path
    assert recovered == report
