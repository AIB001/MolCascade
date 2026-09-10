"""The Lilly Medchem Rules adapter.

The engine is four C++ executables from a conda package and the rules are data
files inside ``medchem``, so neither is assumed present.  The tests split three
ways by what they actually need.

The parsers are tested against **lines captured from the real programs**, which
is where the value is: the output format is the one thing here most likely to
change underneath the adapter, and it can be pinned without the engine at all.
``bad0``/``bad1``/``bad2`` write ``<smiles> <token> (N matches to 'rule')`` and
the demerit stage writes ``<smiles> <token> : D(160) (1 matches to 'D100 rule')``
with the demerit value folded into the rule name -- two shapes, one of which has
no colon when nothing fired.

Configuration, the mode-to-bounds mapping and the unavailable path need nothing
installed either.  Only the chemistry tests need the engine, and they assert on
molecules whose verdict is not a matter of taste: an acyl halide is the paper's
own example of a reactive rejection, and an alkane lacking both oxygen and
nitrogen is its example of a druggability one.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from molcascade.errors import PluginError
from molcascade.plugins.builtin.lilly_medchem import (
    LillyMedchemConfig,
    _kekule,
    _parse_demerits,
    _parse_rejects,
    _reason_code,
    _resolve_binaries,
    _resolve_queries,
)

#: Lines taken verbatim from ``mc_first_pass`` and ``tsubstructure`` output.
_REJECT_LINES = (
    "CCCCCCCC L1 (1 matches to 'no_interesting_atoms')\n"
    "CCO L6 (1 matches to 'not_enough_atoms')\n"
    "CC(=O)OC1=CC=CC=C1C(=O)O L0 (1 matches to 'phenolic_ester_or_carbamate')\n"
)

#: One line from ``bad3`` -- two rules, each carrying its own demerit value in
#: its name, and the total in ``D(...)``.  One from the survivor file with no
#: annotation at all, which is what a clean molecule looks like.
_DEMERIT_LINES = (
    "CCCCCCCC1=CC=C(N(=O)=O)C=C1 L5 : D(160) "
    "(1 matches to 'D100 LongCChain') (1 matches to 'D60 nitro')\n"
    "CC(C)NCC(O)COC1=CC=CC2=CC=CC=C12 L4\n"
)

_ENGINE = all(shutil.which(name) for name in ("mc_first_pass", "tsubstructure", "iwdemerit"))
_needs_engine = pytest.mark.skipif(
    not _ENGINE,
    reason="the Lilly Medchem Rules executables are not on PATH",
)


def test_reject_lines_yield_the_rule_that_fired(tmp_path: Path) -> None:
    path = tmp_path / "bad1.smi"
    path.write_text(_REJECT_LINES, encoding="utf-8")

    found = _parse_rejects(path)

    assert set(found) == {"L0", "L1", "L6"}
    assert found["L1"] == [(1, "no_interesting_atoms")]
    assert found["L0"] == [(1, "phenolic_ester_or_carbamate")]


def test_a_missing_reject_file_is_empty_rather_than_an_error(tmp_path: Path) -> None:
    """A stage that rejected nothing writes no file, and that is not a failure."""

    assert _parse_rejects(tmp_path / "never-written.smi") == {}


def test_the_demerit_total_and_every_contributing_rule_are_read(tmp_path: Path) -> None:
    path = tmp_path / "bad3.smi"
    path.write_text(_DEMERIT_LINES, encoding="utf-8")

    found = _parse_demerits(path)

    total, reasons = found["L5"]
    assert total == 160
    assert reasons == [(1, "D100 LongCChain"), (1, "D60 nitro")]
    # 100 + 60 is the total, which is the whole point of a graded system: neither
    # rule alone reaches the cutoff.
    assert sum(int(name.split()[0][1:]) for _, name in reasons) == total


def test_a_molecule_with_no_annotation_reads_as_zero_not_as_missing(tmp_path: Path) -> None:
    """``<smiles> <token>`` with no colon is a clean molecule, not an absent one.

    Reading it as missing would drop it from the metric table, and a downstream
    gate with no row cannot tell "scored zero" from "never scored".
    """

    path = tmp_path / "passed.smi"
    path.write_text(_DEMERIT_LINES, encoding="utf-8")

    total, reasons = _parse_demerits(path)["L4"]
    assert total == 0
    assert reasons == []


def test_modes_map_to_the_published_bounds() -> None:
    assert LillyMedchemConfig(mode="regular").bounds() == (7, 25, 40)
    assert LillyMedchemConfig(mode="regular").cutoff() is None
    assert LillyMedchemConfig(mode="relaxed").bounds() == (7, 26, 50)
    assert LillyMedchemConfig(mode="relaxed").cutoff() == 160
    # rejections_only collapses the soft bound onto the hard one so that no
    # atom-count demerit can accrue, which is what the upstream -nodemerit does.
    lower, soft, hard = LillyMedchemConfig(mode="rejections_only").bounds()
    assert (lower, soft, hard) == (7, 39, 40)


def test_an_override_wins_over_the_mode() -> None:
    config = LillyMedchemConfig(mode="regular", soft_upper_atom_count=30, demerit_cutoff=80)
    assert config.bounds() == (7, 30, 40)
    assert config.cutoff() == 80


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "strict"},
        {"lower_atom_count": 50, "hard_upper_atom_count": 40},
        {"soft_upper_atom_count": 60, "hard_upper_atom_count": 40},
        {"binary_dir": "relative/path"},
        {"queries_dir": "also/relative"},
    ],
)
def test_incoherent_configuration_is_refused(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        LillyMedchemConfig(**kwargs)  # type: ignore[arg-type]


def test_a_named_binary_directory_without_the_engine_says_so(tmp_path: Path) -> None:
    """The error names the install, because two installs are involved."""

    with pytest.raises(PluginError) as raised:
        _resolve_binaries(LillyMedchemConfig(binary_dir=str(tmp_path)))

    assert raised.value.code == "LILLY_BACKEND_UNAVAILABLE"
    assert "lilly-medchem-rules" in (raised.value.hint or "")
    # The rules are a separate install and the hint has to say so, or an operator
    # who installs the engine and still gets nothing has no way to know why.
    assert "medchem" in (raised.value.hint or "")


def test_a_queries_directory_without_the_rules_says_which_file_is_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(PluginError) as raised:
        _resolve_queries(LillyMedchemConfig(queries_dir=str(tmp_path)))

    assert raised.value.code == "LILLY_QUERIES_UNAVAILABLE"
    assert "reject1" in str(raised.value)


def test_reason_codes_are_bounded_and_order_independent() -> None:
    """The demerit set fires in combination, so codes are hashed, not spelled.

    Order independence matters because the engine reports matches in whatever
    order its queries happen to fire, and the same two rules must not produce two
    different codes.
    """

    forward = _reason_code("demerits", [(1, "D60 nitro"), (1, "D100 LongCChain")])
    reverse = _reason_code("demerits", [(1, "D100 LongCChain"), (1, "D60 nitro")])
    assert forward == reverse
    assert len(forward) < 64
    assert _reason_code("reject1", []) == "LILLY_REJECT1"


def test_kekule_conversion_is_what_the_engine_agrees_with() -> None:
    """Aromatic input is what 89 of 2,000 molecules silently failed on."""

    kekule = _kekule("c1ccccc1")
    assert kekule is not None
    assert "c" not in kekule
    assert "C1=CC=CC=C1" in kekule
    assert _kekule("not a molecule") is None


@_needs_engine
def test_the_engine_rejects_the_paper_s_own_examples() -> None:
    """An acyl halide and a molecule with neither oxygen nor nitrogen.

    Both are named in the 2012 paper as reasons for rejection -- reactivity and
    lack of druggability -- so a run that keeps them is not applying these rules.
    """

    import tempfile

    from molcascade.plugins.builtin.lilly_medchem import _pipeline

    config = LillyMedchemConfig()
    binaries = _resolve_binaries(config)
    queries, charge_assigner = _resolve_queries(config)
    molecules = {
        "L0": _kekule("O=C(Cl)c1ccccc1"),
        "L1": _kekule("CCCCCCCC"),
        "L2": _kekule("CC(C)NCC(O)COc1cccc2ccccc12"),
    }
    with tempfile.TemporaryDirectory() as workspace:
        rejections, readings, unaccounted = _pipeline(
            binaries=binaries,
            queries=queries,
            charge_assigner=charge_assigner,
            config=config,
            tokens={token: value for token, value in molecules.items() if value},
            workspace=Path(workspace),
        )

    assert not unaccounted
    assert rejections["L0"][1] == [(1, "acid_halide")]
    assert rejections["L1"][1] == [(1, "no_interesting_atoms")]
    # Propranolol carries no liability and no demerits, so it survives with a
    # reading of zero rather than with no reading.
    assert "L2" not in rejections
    assert readings["L2"] == (0, [])


@_needs_engine
def test_demerits_accumulate_across_motifs() -> None:
    """The property no boolean catalogue has: two sub-cutoff blemishes summing.

    A heptyl chain is 100 demerits and a nitro group 60. Neither reaches the
    regular-mode cutoff alone; together they do.
    """

    import tempfile

    from molcascade.plugins.builtin.lilly_medchem import _pipeline

    config = LillyMedchemConfig(mode="regular")
    binaries = _resolve_binaries(config)
    queries, charge_assigner = _resolve_queries(config)
    nitro_heptyl = _kekule("CCCCCCCc1ccc([N+](=O)[O-])cc1")
    assert nitro_heptyl is not None
    with tempfile.TemporaryDirectory() as workspace:
        rejections, readings, _ = _pipeline(
            binaries=binaries,
            queries=queries,
            charge_assigner=charge_assigner,
            config=config,
            tokens={"L0": nitro_heptyl},
            workspace=Path(workspace),
        )

    total, reasons = readings["L0"]
    assert total >= 100
    assert len(reasons) >= 2
    assert "L0" in rejections
    assert rejections["L0"][0] == "demerits"
