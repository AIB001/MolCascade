"""What a docking run is told about its protein, and when it is told it is missing.

A cascade is a screening policy and gets reused; the receptor is the one thing
about a docking campaign that changes every time.  So the receptor arrives at
the run rather than in the config, and the whole risk of that arrangement is a
run that starts without one and finds out three tiers deep.

These tests are about the two halves of preventing that.  :mod:`cascade.target`
turns one receptor plus one site definition into the several things three
engines each want; ``preflight_docking_target`` checks the result *before* the
compiler does, because the compiler's own refusal is a Pydantic report that
never mentions the flag that would fix it.

Nothing here docks.  The engines are not installed and could not run if they
were, which is the point: every assertion below is about a decision this
project makes on its own, not about anybody's scoring function.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from molcascade.backends.preflight import preflight_docking_target
from molcascade.cascade.models import BoxConfig, TargetConfig
from molcascade.cascade.target import (
    BOX_PADDING_ANGSTROM,
    box_from_structure,
    merge_target,
    parse_box,
    resolve_target,
)
from molcascade.config import StageConfig
from molcascade.errors import ConfigError, PluginError
from molcascade.plugins.registry import create_builtin_registry

# A protein small enough to read by eye and real enough to hash.  Two of the
# coordinates need four digits before the decimal point on purpose -- that is
# the case where PDB's fixed columns run together and a whitespace split reads
# two numbers as one.
_RECEPTOR_PDB = """\
HEADER    TEST RECEPTOR
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 10.00           C
ATOM      3  C   ALA A   1    1000.000-999.000 999.000  1.00 10.00           C
END
"""

#: Extent 4 A on x, 2 A on y, 6 A on z, centred at (10, 20, 30). Written out
#: rather than computed so the expected box below is an independent statement
#: instead of the implementation restated.
_LIGAND_SDF = """\
reference
     RDKit          3D

  4  0  0  0  0  0  0  0  0  0999 V2000
    8.0000   19.0000   27.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   12.0000   19.0000   27.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    8.0000   21.0000   33.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   12.0000   21.0000   33.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
"""

_LIGAND_MOL2 = """\
@<TRIPOS>MOLECULE
reference
    4     0     1
SMALL
NO_CHARGES

@<TRIPOS>ATOM
      1 C1          8.0000   19.0000   27.0000 C.3     1  LIG        0.0000
      2 C2         12.0000   19.0000   27.0000 C.3     1  LIG        0.0000
      3 C3          8.0000   21.0000   33.0000 C.3     1  LIG        0.0000
      4 O1         12.0000   21.0000   33.0000 O.3     1  LIG        0.0000
@<TRIPOS>BOND
"""

_LIGAND_V3000 = """\
reference
     RDKit          3D

  0  0  0  0  0  0  0  0  0  0999 V3000
M  V30 BEGIN CTAB
M  V30 COUNTS 1 0 0 0 0
M  V30 BEGIN ATOM
M  V30 1 C 8.0000 19.0000 27.0000 0
M  V30 END ATOM
M  V30 END CTAB
M  END
$$$$
"""

#: The box every fixture above describes: the atoms' extent plus padding on
#: each side. Stated once, checked against each format.
_EXPECTED_CENTRE = (10.0, 20.0, 30.0)
_EXPECTED_SIZE = (
    4.0 + 2.0 * BOX_PADDING_ANGSTROM,
    2.0 + 2.0 * BOX_PADDING_ANGSTROM,
    6.0 + 2.0 * BOX_PADDING_ANGSTROM,
)


@pytest.fixture
def receptor(tmp_path: Path) -> Path:
    path = tmp_path / "receptor.pdb"
    path.write_text(_RECEPTOR_PDB, encoding="utf-8")
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage(**config: object) -> StageConfig:
    """A Uni-Dock stage carrying exactly the settings a test wants to talk about.

    Deliberately not built by lowering: the preflight's contract is with the
    *stage*, so a flat pipeline that spells its receptor by hand has to pass the
    same check, and a test that could only build stages through a cascade would
    not notice if that stopped being true.
    """

    return StageConfig(
        id="dock",
        slot="dock",
        plugin="docking.unidock@0.1.0",
        config=dict(config),  # type: ignore[arg-type]
    )


def _box_settings() -> dict[str, float]:
    return BoxConfig(
        center_x=10.0,
        center_y=20.0,
        center_z=30.0,
        size_x=12.0,
        size_y=10.0,
        size_z=14.0,
    ).settings


def test_a_docking_stage_without_a_receptor_names_the_flag() -> None:
    """The failure the whole preflight exists to replace.

    Without it this stage still fails -- in the compiler, as ``receptor_path
    Field required``, which tells an operator what a Pydantic model wanted
    rather than what they should type.
    """

    registry = create_builtin_registry()
    with pytest.raises(ConfigError) as caught:
        preflight_docking_target([_stage(executable="/opt/unidock")], registry=registry)
    error = caught.value
    assert error.code == "DOCKING_TARGET_RECEPTOR_REQUIRED"
    assert "--receptor" in (error.hint or "")
    assert error.context["stage"] == "dock"


def test_a_receptor_without_a_site_names_all_three_accepted_forms(receptor: Path) -> None:
    """Because the operator has a choice here, and only one of them is cheap.

    A reference ligand and a pocket PDB both have to exist as files; the six
    numbers do not.  Naming only one form would send someone looking for a file
    they do not need.
    """

    registry = create_builtin_registry()
    with pytest.raises(ConfigError) as caught:
        preflight_docking_target(
            [_stage(executable="/opt/unidock", receptor_path=str(receptor))],
            registry=registry,
        )
    error = caught.value
    assert error.code == "DOCKING_TARGET_SITE_REQUIRED"
    hint = error.hint or ""
    assert "--reference-ligand" in hint
    assert "--pocket" in hint
    assert "--box" in hint
    # The engine's own missing settings, not a generic list: Uni-Dock needs a
    # prepared receptor as well, and an operator who fixes only the box would
    # otherwise hit a second refusal for the first one's reason.
    assert "--receptor-pdbqt" in hint
    assert set(error.context["missing_settings"]) >= {"center_x", "size_z"}


def test_a_complete_stage_passes_and_says_which_it_checked(receptor: Path) -> None:
    registry = create_builtin_registry()
    checked = preflight_docking_target(
        [
            _stage(
                executable="/opt/unidock",
                receptor_path=str(receptor),
                receptor_pdbqt_path=str(receptor.with_suffix(".pdbqt")),
                **_box_settings(),
            )
        ],
        registry=registry,
    )
    assert checked == ("dock",)


def test_a_stage_that_does_not_dock_is_not_asked_for_a_receptor() -> None:
    """The marker is the config field, so the check cannot drift onto other tiers."""

    registry = create_builtin_registry()
    stage = StageConfig(id="props", slot="describe", plugin="rdkit.properties@1.0.0")
    assert preflight_docking_target([stage], registry=registry) == ()


def test_a_pinned_digest_that_the_bytes_do_not_match_stops_the_run(receptor: Path) -> None:
    """A score is only comparable to another score computed on the same protein.

    ``receptor_id`` in ``docking_score/v1`` is this digest, so letting a
    mismatch through would produce rows that look comparable and are not.
    """

    registry = create_builtin_registry()
    with pytest.raises(ConfigError) as caught:
        preflight_docking_target(
            [
                _stage(
                    executable="/opt/unidock",
                    receptor_path=str(receptor),
                    receptor_sha256="0" * 64,
                    receptor_pdbqt_path=str(receptor.with_suffix(".pdbqt")),
                    **_box_settings(),
                )
            ],
            registry=registry,
        )
    error = caught.value
    assert error.code == "DOCKING_RECEPTOR_DIGEST_MISMATCH"
    assert error.context["actual_sha256"] == _digest(receptor)


def test_a_matching_digest_passes(receptor: Path) -> None:
    registry = create_builtin_registry()
    checked = preflight_docking_target(
        [
            _stage(
                executable="/opt/unidock",
                receptor_path=str(receptor),
                receptor_sha256=_digest(receptor),
                receptor_pdbqt_path=str(receptor.with_suffix(".pdbqt")),
                **_box_settings(),
            )
        ],
        registry=registry,
    )
    assert checked == ("dock",)


def test_a_receptor_that_is_not_a_pdb_is_refused_by_suffix(tmp_path: Path) -> None:
    """An mmCIF handed to a PDB parser is empty, not invalid, which is worse."""

    structure = tmp_path / "receptor.cif"
    structure.write_text(_RECEPTOR_PDB, encoding="utf-8")
    registry = create_builtin_registry()
    with pytest.raises(PluginError) as caught:
        preflight_docking_target(
            [
                _stage(
                    executable="/opt/unidock",
                    receptor_path=str(structure),
                    receptor_pdbqt_path=str(tmp_path / "receptor.pdbqt"),
                    **_box_settings(),
                )
            ],
            registry=registry,
        )
    assert caught.value.code == "DOCKING_RECEPTOR_FORMAT_INVALID"


def test_a_missing_receptor_file_is_found_before_the_first_molecule(tmp_path: Path) -> None:
    registry = create_builtin_registry()
    with pytest.raises(PluginError) as caught:
        preflight_docking_target(
            [
                _stage(
                    executable="/opt/unidock",
                    receptor_path=str(tmp_path / "absent.pdb"),
                    receptor_pdbqt_path=str(tmp_path / "absent.pdbqt"),
                    **_box_settings(),
                )
            ],
            registry=registry,
        )
    assert caught.value.code == "DOCKING_RECEPTOR_UNREADABLE"


@pytest.mark.parametrize(
    ("filename", "text"),
    [
        ("reference.sdf", _LIGAND_SDF),
        ("reference.mol", _LIGAND_SDF),
        ("reference.mol2", _LIGAND_MOL2),
    ],
)
def test_a_reference_ligand_expands_to_its_extent_plus_padding(
    tmp_path: Path, filename: str, text: str
) -> None:
    """The same four atoms in three formats must give one box.

    Otherwise the engines in a consensus tier would be searching volumes that
    differ by whichever file the operator happened to have.
    """

    ligand = tmp_path / filename
    ligand.write_text(text, encoding="utf-8")
    box = box_from_structure(ligand)
    assert (box.center_x, box.center_y, box.center_z) == _EXPECTED_CENTRE
    assert (box.size_x, box.size_y, box.size_z) == _EXPECTED_SIZE


def test_a_pocket_pdb_is_read_from_its_columns_not_its_whitespace(tmp_path: Path) -> None:
    """``1000.000-999.000 999.000`` is three coordinates, not two fields.

    A whitespace split reads that line as two numbers, drops the atom, and
    reports a box around the remaining two -- a smaller, wrong site with no
    error anywhere.
    """

    pocket = tmp_path / "pocket.pdb"
    pocket.write_text(_RECEPTOR_PDB, encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        box_from_structure(pocket)
    # Which is the right answer for this fixture: an atom a thousand angstrom
    # away means the file is not a pocket. That it is *detected* is the proof
    # the third coordinate was read at all.
    assert caught.value.code == "DOCKING_SITE_TOO_LARGE"
    assert caught.value.context["size_x"] > 900.0


def test_a_v3000_molfile_is_refused_rather_than_half_read(tmp_path: Path) -> None:
    """Skipping the lines it does not recognise would report a site of size zero."""

    ligand = tmp_path / "reference.sdf"
    ligand.write_text(_LIGAND_V3000, encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        box_from_structure(ligand)
    assert caught.value.code == "DOCKING_SITE_FORMAT_UNSUPPORTED"
    assert "--box" in (caught.value.hint or "")


def test_an_empty_structure_is_not_a_site_of_size_zero(tmp_path: Path) -> None:
    ligand = tmp_path / "reference.pdb"
    ligand.write_text("HEADER    NOTHING HERE\nEND\n", encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        box_from_structure(ligand)
    assert caught.value.code == "DOCKING_SITE_STRUCTURE_EMPTY"


def test_parse_box_wants_six_numbers_and_says_which_six() -> None:
    box = parse_box("10, 20,30 , 12,10,14")
    assert box.settings == _box_settings()
    for malformed in ("10,20,30", "10,20,30,12,10", "10,20,30,12,10,14,1", "10,20,30,12,10,x"):
        with pytest.raises(ConfigError) as caught:
            parse_box(malformed)
        assert caught.value.code == "DOCKING_BOX_MALFORMED"
        assert "cx,cy,cz,sx,sy,sz" in (caught.value.hint or "")


def test_a_box_larger_than_a_pocket_is_refused() -> None:
    with pytest.raises(ConfigError) as caught:
        parse_box("0,0,0,500,10,10")
    assert caught.value.code == "DOCKING_BOX_INVALID"


def test_no_target_flags_and_no_cascade_target_is_not_an_error() -> None:
    """Most cascades have no docking tier, and must not be made to explain that."""

    assert merge_target(None) is None


def test_a_site_without_a_receptor_asks_for_the_receptor(tmp_path: Path) -> None:
    ligand = tmp_path / "reference.sdf"
    ligand.write_text(_LIGAND_SDF, encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        merge_target(None, reference_ligand=ligand)
    assert caught.value.code == "DOCKING_TARGET_RECEPTOR_REQUIRED"


def test_two_site_flags_have_no_correct_resolution(receptor: Path, tmp_path: Path) -> None:
    ligand = tmp_path / "reference.sdf"
    ligand.write_text(_LIGAND_SDF, encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        merge_target(None, receptor=receptor, reference_ligand=ligand, box="0,0,0,10,10,10")
    error = caught.value
    assert error.code == "DOCKING_TARGET_SITE_AMBIGUOUS"
    assert set(error.context["flags"]) == {"--reference-ligand", "--box"}


def test_a_receptor_flag_without_a_site_flag_names_the_three_forms(receptor: Path) -> None:
    with pytest.raises(ConfigError) as caught:
        merge_target(None, receptor=receptor)
    error = caught.value
    assert error.code == "DOCKING_TARGET_SITE_REQUIRED"
    assert "--reference-ligand" in (error.hint or "")
    assert "--pocket" in (error.hint or "")


def test_a_site_flag_replaces_the_cascade_s_site_rather_than_joining_it(
    receptor: Path, tmp_path: Path
) -> None:
    """An override that could only ever conflict would not be an override.

    A cascade written against one pocket, run against another, is the ordinary
    case; refusing it as "the site was named twice" would make ``--box``
    unusable exactly where it is wanted.
    """

    ligand = tmp_path / "reference.sdf"
    ligand.write_text(_LIGAND_SDF, encoding="utf-8")
    base = TargetConfig(
        name="stored",
        receptor_path=str(receptor),
        reference_ligand_path=str(ligand),
    )
    merged = merge_target(base, box="1,2,3,10,10,10")
    assert merged is not None
    assert merged.reference_ligand_path is None
    assert merged.box is not None
    assert merged.box.center_x == 1.0
    # The name is the cascade's, because the operator moved the site, not the
    # campaign.
    assert merged.name == "stored"


def test_naming_a_new_receptor_drops_a_digest_pinned_against_the_old_one(
    receptor: Path, tmp_path: Path
) -> None:
    """Otherwise every receptor swap would stop on a mismatch already resolved."""

    other = tmp_path / "other.pdb"
    other.write_text(_RECEPTOR_PDB.replace("TEST RECEPTOR", "OTHER RECEPTOR"), encoding="utf-8")
    base = TargetConfig(
        name="stored",
        receptor_path=str(receptor),
        receptor_sha256=_digest(receptor),
        box=BoxConfig(
            center_x=1.0, center_y=2.0, center_z=3.0, size_x=10.0, size_y=10.0, size_z=10.0
        ),
    )
    merged = merge_target(base, receptor=other)
    assert merged is not None
    assert merged.receptor_path == str(other.resolve())
    assert merged.receptor_sha256 is None


def test_resolve_derives_only_what_the_cascade_s_engines_require(
    receptor: Path, tmp_path: Path
) -> None:
    """A KarmaDock-only cascade must not be made to produce a PDBQT nothing reads.

    Deriving one needs meeko installed and takes seconds on a real protein, so
    the requirement set is what decides -- not the presence of a docking tier
    in general.
    """

    ligand = tmp_path / "reference.sdf"
    ligand.write_text(_LIGAND_SDF, encoding="utf-8")
    target = TargetConfig(
        name="site",
        receptor_path=str(receptor),
        reference_ligand_path=str(ligand),
    )
    resolved = resolve_target(
        target,
        requirements=frozenset({"receptor_path", "reference_ligand_path", "center_x"}),
        workspace=tmp_path / "workspace",
    )
    assert resolved is not None
    assert resolved.target.receptor_pdbqt_path is None
    box = resolved.target.box
    assert box is not None
    assert (box.center_x, box.center_y, box.center_z) == _EXPECTED_CENTRE
    assert (box.size_x, box.size_y, box.size_z) == _EXPECTED_SIZE
    assert any("box derived from reference.sdf" in note for note in resolved.notes)
    # Nothing was written into the workspace, because nothing needed to be.
    assert not (tmp_path / "workspace").exists()


def test_an_explicit_box_is_never_recomputed(receptor: Path, tmp_path: Path) -> None:
    """The numbers the operator typed are the numbers that reach the engine."""

    target = TargetConfig(
        name="site",
        receptor_path=str(receptor),
        box=BoxConfig(
            center_x=1.5, center_y=2.5, center_z=3.5, size_x=11.0, size_y=12.0, size_z=13.0
        ),
    )
    resolved = resolve_target(
        target,
        requirements=frozenset({"receptor_path", "center_x", "size_x"}),
        workspace=tmp_path / "workspace",
    )
    assert resolved is not None
    assert resolved.target.box == target.box
    assert resolved.notes == ()


def test_a_derived_box_travels_beside_the_ligand_it_came_from(
    receptor: Path, tmp_path: Path
) -> None:
    """Both forms reach the stages, because different engines read different ones.

    Uni-Dock and GNINA take six numbers; KarmaDock takes the ligand.  They are
    the same site said twice, which is why a resolved target is allowed to hold
    both while an authored one may not hold two *files*.
    """

    ligand = tmp_path / "reference.sdf"
    ligand.write_text(_LIGAND_SDF, encoding="utf-8")
    resolved = resolve_target(
        TargetConfig(
            name="site",
            receptor_path=str(receptor),
            reference_ligand_path=str(ligand),
        ),
        requirements=frozenset({"receptor_path", "reference_ligand_path", "center_x"}),
        workspace=tmp_path / "workspace",
    )
    assert resolved is not None
    assert resolved.target.reference_ligand_path == str(ligand)
    assert resolved.target.box is not None


def test_two_structures_both_claiming_the_site_are_refused(receptor: Path, tmp_path: Path) -> None:
    ligand = tmp_path / "reference.sdf"
    ligand.write_text(_LIGAND_SDF, encoding="utf-8")
    pocket = tmp_path / "pocket.pdb"
    pocket.write_text(_RECEPTOR_PDB, encoding="utf-8")
    with pytest.raises(ValueError, match="two different structures"):
        TargetConfig(
            name="site",
            receptor_path=str(receptor),
            reference_ligand_path=str(ligand),
            pocket_path=str(pocket),
        )


def test_a_hand_written_box_that_misses_the_hand_written_ligand_is_refused(
    receptor: Path, tmp_path: Path
) -> None:
    """The one way box-plus-file can still mean two different pockets.

    Only reachable from a cascade file -- the command line refuses two site
    flags outright -- and exactly the case where a tier would average scores
    from two places and call it a consensus.
    """

    ligand = tmp_path / "reference.sdf"
    ligand.write_text(_LIGAND_SDF, encoding="utf-8")
    target = TargetConfig(
        name="site",
        receptor_path=str(receptor),
        reference_ligand_path=str(ligand),
        # The ligand's centre is (10, 20, 30); this box reaches x = 5 at most.
        box=BoxConfig(
            center_x=0.0, center_y=20.0, center_z=30.0, size_x=10.0, size_y=10.0, size_z=10.0
        ),
    )
    with pytest.raises(ConfigError) as caught:
        resolve_target(
            target,
            requirements=frozenset({"receptor_path", "reference_ligand_path", "center_x"}),
            workspace=tmp_path / "workspace",
        )
    error = caught.value
    assert error.code == "DOCKING_SITE_DISAGREEMENT"
    assert error.context["axes_outside"] == ["x"]


def test_a_hand_written_box_around_the_same_site_is_kept(receptor: Path, tmp_path: Path) -> None:
    """Tightening the box around a reference ligand is ordinary, not a conflict."""

    ligand = tmp_path / "reference.sdf"
    ligand.write_text(_LIGAND_SDF, encoding="utf-8")
    tight = BoxConfig(
        center_x=10.0, center_y=20.0, center_z=30.0, size_x=8.0, size_y=8.0, size_z=8.0
    )
    resolved = resolve_target(
        TargetConfig(
            name="site",
            receptor_path=str(receptor),
            reference_ligand_path=str(ligand),
            box=tight,
        ),
        requirements=frozenset({"receptor_path", "reference_ligand_path", "center_x"}),
        workspace=tmp_path / "workspace",
    )
    assert resolved is not None
    assert resolved.target.box == tight
    assert any("checked against reference.sdf" in note for note in resolved.notes)


def test_an_operator_supplied_pdbqt_is_used_instead_of_deriving_one(
    receptor: Path, tmp_path: Path
) -> None:
    """The documented escape from a receptor meeko cannot prepare.

    If this were not honoured the escape would not exist, and the run would
    stop on meeko's failure with no way past it.
    """

    prepared = tmp_path / "prepared.pdbqt"
    prepared.write_text("REMARK prepared by hand\n", encoding="utf-8")
    target = TargetConfig(
        name="site",
        receptor_path=str(receptor),
        receptor_pdbqt_path=str(prepared),
        box=BoxConfig(
            center_x=1.0, center_y=2.0, center_z=3.0, size_x=10.0, size_y=10.0, size_z=10.0
        ),
    )
    resolved = resolve_target(
        target,
        requirements=frozenset({"receptor_path", "receptor_pdbqt_path", "center_x"}),
        workspace=tmp_path / "workspace",
    )
    assert resolved is not None
    assert resolved.target.receptor_pdbqt_path == str(prepared)
    assert resolved.notes == ()
