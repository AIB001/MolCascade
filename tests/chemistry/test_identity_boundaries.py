"""Semantic boundary tests for parent identity reconciliation.

These tests intentionally exercise cases where graph equality alone is not a
sufficient identity comparison: coordinate-bonded metals, annotations carried
by only one input field, enhanced stereo, and policy-relative fragment ties.
"""

from __future__ import annotations

from typing import Any

import pytest
from rdkit import Chem

from molcascade.chemistry import (
    AtomLabelPolicy,
    AtomMapPolicy,
    ChargePolicy,
    IdentityPolicy,
    IsotopePolicy,
    MetalDisconnectionPolicy,
    ParentStandardizer,
    RegistrationHashVersion,
    StandardizationFailure,
    StereoPolicy,
    TautomerPolicy,
    UndefinedStereoPolicy,
    standardize_parent,
)


def _molblock(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None, f"invalid test fixture SMILES: {smiles}"
    return Chem.MolToMolBlock(molecule)


def _notice_codes(result: Any) -> set[str]:
    return {notice.code for notice in result.notices}


def _notice_snapshot(result: Any) -> tuple[tuple[str, str], ...]:
    return tuple((notice.code, notice.detail) for notice in result.notices)


def test_default_metal_disconnection_unifies_bonded_and_ionic_sodium_benzoate() -> None:
    bonded = standardize_parent(raw_smiles="[Na]OC(=O)c1ccccc1")
    disconnected = standardize_parent(raw_smiles="[Na+].[O-]C(=O)c1ccccc1")

    assert bonded.parent_smiles == disconnected.parent_smiles
    assert bonded.registration_key == disconnected.registration_key
    assert bonded.parent_id == disconnected.parent_id


def test_explicit_metal_preservation_keeps_bonded_sodium_in_parent_identity() -> None:
    preserve = IdentityPolicy(
        metal_disconnection_policy=MetalDisconnectionPolicy.PRESERVE,
        charge_policy=ChargePolicy.PRESERVE,
        normalize_functional_groups=False,
    )
    bonded = standardize_parent(
        raw_smiles="[Na]OC(=O)c1ccccc1",
        policy=preserve,
    )
    disconnected = standardize_parent(
        raw_smiles="[Na+].[O-]C(=O)c1ccccc1",
        policy=preserve,
    )

    assert "Na" in bonded.parent_smiles
    assert bonded.parent_id != disconnected.parent_id


@pytest.mark.parametrize("mapped_field", ["raw_smiles", "raw_molblock"])
def test_atom_map_policy_cannot_be_bypassed_by_the_other_structure_field(
    mapped_field: str,
) -> None:
    fields = {
        "raw_smiles": "CO",
        "raw_molblock": _molblock("CO"),
    }
    fields[mapped_field] = (
        "[CH3:1][OH:2]"
        if mapped_field == "raw_smiles"
        else _molblock("[CH3:1][OH:2]")
    )

    removed = standardize_parent(
        **fields,
        policy=IdentityPolicy(atom_map_policy=AtomMapPolicy.REMOVE),
    )
    assert removed.parent_smiles == "CO"
    assert "ATOM_MAP_REMOVED" in _notice_codes(removed)

    with pytest.raises(StandardizationFailure) as caught:
        standardize_parent(
            **fields,
            policy=IdentityPolicy(atom_map_policy=AtomMapPolicy.REJECT),
        )
    assert caught.value.reason_code == "ATOM_MAP_NOT_ALLOWED"


def test_cx_atom_labels_are_explicitly_removed_or_rejected_across_fields() -> None:
    labelled = "CO |$foo;$|"
    removed = standardize_parent(
        raw_smiles=labelled,
        raw_molblock=_molblock("CO"),
    )
    plain = standardize_parent(raw_smiles="CO")

    assert removed.parent_id == plain.parent_id
    assert removed.parent_smiles == plain.parent_smiles == "CO"
    assert "ATOM_LABEL_REMOVED" in _notice_codes(removed)

    with pytest.raises(StandardizationFailure) as caught:
        standardize_parent(
            raw_smiles=labelled,
            raw_molblock=_molblock("CO"),
            policy=IdentityPolicy(atom_label_policy=AtomLabelPolicy.REJECT),
        )
    assert caught.value.reason_code == "ATOM_LABEL_NOT_ALLOWED"


@pytest.mark.parametrize("specified_field", ["raw_smiles", "raw_molblock"])
def test_dual_structure_fields_choose_the_more_stereo_informative_encoding(
    specified_field: str,
) -> None:
    specified = "C[C@H](F)Cl"
    fields = {
        "raw_smiles": "CC(F)Cl",
        "raw_molblock": _molblock("CC(F)Cl"),
    }
    fields[specified_field] = (
        specified if specified_field == "raw_smiles" else _molblock(specified)
    )
    policy = IdentityPolicy(undefined_stereo_policy=UndefinedStereoPolicy.REJECT)

    result = standardize_parent(**fields, policy=policy)
    reference = standardize_parent(raw_smiles=specified, policy=policy)

    assert result.parent_smiles == reference.parent_smiles
    assert result.parent_id == reference.parent_id


def test_dual_structure_fields_reject_opposite_specified_stereochemistry() -> None:
    with pytest.raises(StandardizationFailure) as caught:
        standardize_parent(
            raw_smiles="C[C@H](F)Cl",
            raw_molblock=_molblock("C[C@@H](F)Cl"),
        )

    assert caught.value.reason_code == "CONFLICTING_STRUCTURE_FIELDS"


def test_stereo_insensitive_policy_accepts_opposite_dual_field_assignments() -> None:
    policy = IdentityPolicy(stereo_policy=StereoPolicy.INSENSITIVE)

    result = standardize_parent(
        raw_smiles="C[C@H](F)Cl",
        raw_molblock=_molblock("C[C@@H](F)Cl"),
        policy=policy,
    )
    achiral = standardize_parent(raw_smiles="CC(F)Cl", policy=policy)

    assert result.parent_smiles == achiral.parent_smiles
    assert result.parent_id == achiral.parent_id


def test_enhanced_and_stereo_global_inversion_has_one_canonical_cx_identity() -> None:
    clockwise = standardize_parent(
        raw_smiles="C[C@H](F)[C@H](Cl)Br |&1:1,3|",
    )
    inverted = standardize_parent(
        raw_smiles="C[C@@H](F)[C@@H](Cl)Br |&1:1,3|",
    )

    assert clockwise.parent_id == inverted.parent_id
    assert clockwise.parent_smiles == inverted.parent_smiles
    assert "|&1:" in clockwise.parent_smiles


def test_enhanced_and_and_or_stereo_remain_distinct() -> None:
    and_group = standardize_parent(
        raw_smiles="C[C@H](F)[C@H](Cl)Br |&1:1,3|",
    )
    or_group = standardize_parent(
        raw_smiles="C[C@H](F)[C@H](Cl)Br |o1:1,3|",
    )

    assert and_group.parent_smiles != or_group.parent_smiles
    assert and_group.parent_id != or_group.parent_id


def test_stereo_insensitive_equal_fragment_tie_collapses_and_is_order_invariant() -> None:
    policy = IdentityPolicy(stereo_policy=StereoPolicy.INSENSITIVE)
    clockwise = "F[C@H](Cl)Br"
    anticlockwise = "F[C@@H](Cl)Br"

    forward = standardize_parent(
        raw_smiles=f"{clockwise}.{anticlockwise}",
        policy=policy,
    )
    reverse = standardize_parent(
        raw_smiles=f"{anticlockwise}.{clockwise}",
        policy=policy,
    )
    reference = standardize_parent(raw_smiles=clockwise, policy=policy)

    assert forward.parent_smiles == reverse.parent_smiles == reference.parent_smiles
    assert forward.parent_id == reverse.parent_id == reference.parent_id
    assert _notice_snapshot(forward) == _notice_snapshot(reverse)


def test_v2_tautomer_insensitive_equal_fragment_tie_collapses_deterministically() -> None:
    policy = IdentityPolicy(
        tautomer_policy=TautomerPolicy.INSENSITIVE,
        registration_hash_version=RegistrationHashVersion.V2,
    )

    forward = standardize_parent(raw_smiles="CC=CO.CCC=O", policy=policy)
    reverse = standardize_parent(raw_smiles="CCC=O.CC=CO", policy=policy)
    enol = standardize_parent(raw_smiles="CC=CO", policy=policy)
    keto = standardize_parent(raw_smiles="CCC=O", policy=policy)

    assert forward.parent_id == reverse.parent_id == enol.parent_id == keto.parent_id
    assert forward.parent_smiles == reverse.parent_smiles
    assert _notice_snapshot(forward) == _notice_snapshot(reverse)


def test_uncharged_equal_fragment_tie_has_order_invariant_audit_notices() -> None:
    policy = IdentityPolicy(charge_policy=ChargePolicy.UNCHARGE_WHERE_POSSIBLE)

    forward = standardize_parent(raw_smiles="C[NH3+].CN", policy=policy)
    reverse = standardize_parent(raw_smiles="CN.C[NH3+]", policy=policy)

    assert forward.parent_id == reverse.parent_id
    assert _notice_snapshot(forward) == _notice_snapshot(reverse)
    assert "CHARGE_NORMALIZED" in _notice_codes(forward)


def test_v1_tautomer_insensitive_equal_fragment_tie_is_ambiguous() -> None:
    policy = IdentityPolicy(
        tautomer_policy=TautomerPolicy.INSENSITIVE,
        registration_hash_version=RegistrationHashVersion.V1,
    )

    with pytest.raises(StandardizationFailure) as caught:
        standardize_parent(raw_smiles="CC=CO.CCC=O", policy=policy)

    assert caught.value.reason_code == "AMBIGUOUS_FRAGMENTS"


def test_equal_fragment_tie_with_and_vs_or_enhanced_stereo_is_ambiguous() -> None:
    structure = (
        "C[C@H](F)[C@H](Cl)Br.C[C@H](F)[C@H](Cl)Br "
        "|&1:1,3,o1:7,9|"
    )

    with pytest.raises(StandardizationFailure) as caught:
        standardize_parent(raw_smiles=structure)

    assert caught.value.reason_code == "AMBIGUOUS_FRAGMENTS"


def test_v1_and_v2_tautomer_registration_semantics_are_explicit() -> None:
    v2 = IdentityPolicy(
        tautomer_policy=TautomerPolicy.INSENSITIVE,
        registration_hash_version=RegistrationHashVersion.V2,
    )
    assert standardize_parent(raw_smiles="CC=CO", policy=v2).parent_id == (
        standardize_parent(raw_smiles="CCC=O", policy=v2).parent_id
    )

    v1 = IdentityPolicy(
        tautomer_policy=TautomerPolicy.INSENSITIVE,
        registration_hash_version=RegistrationHashVersion.V1,
    )
    assert standardize_parent(raw_smiles="CC=CO", policy=v1).parent_id != (
        standardize_parent(raw_smiles="CCC=O", policy=v1).parent_id
    )


def test_parent_id_never_aliases_incompatible_persisted_parent_values() -> None:
    first = standardize_parent(raw_smiles="C=C1N=C(N)C(C#N)=CN1")
    second = standardize_parent(raw_smiles="C=C1N=CC(C#N)C(=N)N1")

    if first.parent_id == second.parent_id:
        assert (
            first.parent_smiles,
            first.registration_key,
            first.stereo_key,
            first.formula,
        ) == (
            second.parent_smiles,
            second.registration_key,
            second.stereo_key,
            second.formula,
        )


def test_atropisomers_are_sensitive_by_default_and_removable_by_policy() -> None:
    core = "Cc1cc2c(C(N)=O)c(N)n(-c3c(C)ccc(O)c3C)c2nc1C"
    first_input = core + " |wD:10.9|"
    second_input = core + " |wD:10.20|"

    first = standardize_parent(raw_smiles=first_input)
    second = standardize_parent(raw_smiles=second_input)
    assert first.parent_id != second.parent_id

    insensitive = IdentityPolicy(stereo_policy=StereoPolicy.INSENSITIVE)
    stripped_first = standardize_parent(raw_smiles=first_input, policy=insensitive)
    stripped_second = standardize_parent(raw_smiles=second_input, policy=insensitive)
    assert stripped_first.parent_smiles == stripped_second.parent_smiles
    assert stripped_first.parent_id == stripped_second.parent_id
    assert "STEREO_REMOVED" in _notice_codes(stripped_first)


def test_labile_deuterium_uses_audited_display_fallback_for_insensitive_identity() -> None:
    policy = IdentityPolicy(
        isotope_policy=IsotopePolicy.SENSITIVE,
        tautomer_policy=TautomerPolicy.INSENSITIVE,
    )

    # RDKit can enumerate a display tautomer for this labile O-D case which
    # subsequently fails sanitization with an O valence of three.  The
    # identity hash remains valid on the pre-display graph.
    result = standardize_parent(raw_smiles="[2H]Oc1ncccc1", policy=policy)

    assert "[2H]" in result.parent_smiles
    assert "TAUTOMER_DISPLAY_FALLBACK" in _notice_codes(result)

    reordered = standardize_parent(raw_smiles="c1cnc(O[2H])cc1", policy=policy)
    assert reordered.parent_id == result.parent_id
    assert _notice_snapshot(reordered) == _notice_snapshot(result)


def test_canonicalize_policy_reports_a_dedicated_tautomer_failure() -> None:
    class ExplodingTautomerEnumerator:
        def Canonicalize(self, molecule: Chem.Mol) -> Chem.Mol:
            raise RuntimeError("synthetic tautomer failure")

    standardizer = ParentStandardizer(
        IdentityPolicy(tautomer_policy=TautomerPolicy.CANONICALIZE),
    )
    standardizer._tautomer_enumerator = ExplodingTautomerEnumerator()

    with pytest.raises(StandardizationFailure) as caught:
        standardizer.standardize(raw_smiles="CC=CO")

    assert caught.value.reason_code == "TAUTOMER_CANONICALIZATION_ERROR"


def test_crossed_double_bond_is_unknown_stereo_and_can_be_rejected() -> None:
    molecule = Chem.MolFromSmiles("FC=CCl")
    assert molecule is not None
    double_bond = next(
        bond
        for bond in molecule.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
    )
    double_bond.SetStereo(Chem.BondStereo.STEREOANY)
    crossed_molblock = Chem.MolToMolBlock(molecule)

    with pytest.raises(StandardizationFailure) as caught:
        standardize_parent(
            raw_molblock=crossed_molblock,
            policy=IdentityPolicy(
                undefined_stereo_policy=UndefinedStereoPolicy.REJECT,
            ),
        )

    assert caught.value.reason_code == "UNDEFINED_STEREO"


def test_undefined_stereo_notices_do_not_depend_on_smiles_atom_order() -> None:
    first = standardize_parent(raw_smiles="FC(Cl)C")
    second = standardize_parent(raw_smiles="C(F)(C)Cl")

    assert first.parent_id == second.parent_id
    assert _notice_snapshot(first) == _notice_snapshot(second)
