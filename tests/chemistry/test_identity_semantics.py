from __future__ import annotations

import pytest
from pydantic import ValidationError
from rdkit import Chem

from molcascade.chemistry import (
    AtomMapPolicy,
    ChargePolicy,
    FragmentPolicy,
    IdentityPolicy,
    IsotopePolicy,
    RegistrationHashVersion,
    StandardizationFailure,
    StereoPolicy,
    TautomerPolicy,
    UndefinedStereoPolicy,
    identity_policy_id,
    parent_id_from_registration,
    standardize_parent,
)

ETHANOL_MOL2 = """@<TRIPOS>MOLECULE
ethanol
 9 8 0 0 0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
      1 C1          0.0000    0.0000    0.0000 C.3       1 ETH       0.0000
      2 C2          1.5000    0.0000    0.0000 C.3       1 ETH       0.0000
      3 O1          2.1000    1.2000    0.0000 O.3       1 ETH       0.0000
      4 H1         -0.5000    0.9000    0.0000 H         1 ETH       0.0000
      5 H2         -0.5000   -0.9000    0.0000 H         1 ETH       0.0000
      6 H3          0.0000    0.0000    1.0000 H         1 ETH       0.0000
      7 H4          2.0000   -0.8000    0.0000 H         1 ETH       0.0000
      8 H5          1.5000    0.0000    1.0000 H         1 ETH       0.0000
      9 H6          3.0000    1.2000    0.0000 H         1 ETH       0.0000
@<TRIPOS>BOND
     1    1    2 1
     2    2    3 1
     3    1    4 1
     4    1    5 1
     5    1    6 1
     6    2    7 1
     7    2    8 1
     8    3    9 1
"""


def test_identity_policy_is_strict_frozen_and_forbids_extra_fields() -> None:
    policy = IdentityPolicy()

    with pytest.raises(ValidationError, match="frozen"):
        policy.fragment_policy = FragmentPolicy.LARGEST  # type: ignore[misc]
    with pytest.raises(ValidationError):
        IdentityPolicy.model_validate({"normalize_functional_groups": 1})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        IdentityPolicy.model_validate({"remove_salts": True})


def test_different_smiles_spellings_register_as_one_parent() -> None:
    first = standardize_parent(raw_smiles="CCO")
    second = standardize_parent(raw_smiles="OCC")

    assert first.parent_smiles == second.parent_smiles == "CCO"
    assert first.registration_key == second.registration_key
    assert first.parent_id == second.parent_id


def test_formatted_smiles_and_mol2_use_the_same_identity_pipeline() -> None:
    legacy = standardize_parent(raw_smiles="CCO")
    formatted_smiles = standardize_parent(
        raw_format="SMILES",
        raw_structure="CCO",
    )
    formatted_mol2 = standardize_parent(
        raw_format="MOL2",
        raw_structure=ETHANOL_MOL2,
    )

    assert formatted_smiles.parent_id == legacy.parent_id
    assert formatted_mol2.parent_id == legacy.parent_id
    assert formatted_mol2.parent_smiles == "CCO"


def test_formatted_molblock_uses_the_same_identity_pipeline() -> None:
    molecule = Chem.MolFromSmiles("CCO")
    assert molecule is not None
    molblock = Chem.MolToMolBlock(molecule)

    formatted = standardize_parent(
        raw_format="MOLBLOCK",
        raw_structure=molblock,
    )

    assert formatted.parent_id == standardize_parent(raw_smiles="CCO").parent_id


@pytest.mark.parametrize(
    ("raw_format", "raw_structure", "reason_code"),
    [
        ("SMILES", " \n\t", "EMPTY_OR_FRAGMENT_ONLY"),
        ("PDB", "ATOM", "RAW_FORMAT_UNSUPPORTED"),
        (
            "MOL2",
            ETHANOL_MOL2 + "\n" + ETHANOL_MOL2,
            "MOL2_MULTIPLE_MOLECULES",
        ),
        ("MOL2", "@<TRIPOS>MOLECULE\nbroken", "PARSE_ERROR"),
    ],
)
def test_formatted_input_failures_have_stable_reason_codes(
    raw_format: str,
    raw_structure: str,
    reason_code: str,
) -> None:
    with pytest.raises(StandardizationFailure) as caught:
        standardize_parent(raw_format=raw_format, raw_structure=raw_structure)

    assert caught.value.reason_code == reason_code


def test_formatted_and_legacy_input_modes_cannot_be_mixed() -> None:
    with pytest.raises(StandardizationFailure) as caught:
        standardize_parent(
            raw_format="SMILES",
            raw_structure="CCO",
            raw_smiles="CCO",
        )

    assert caught.value.reason_code == "RAW_INPUT_MODE_CONFLICT"


def test_salt_selects_largest_organic_parent_and_audits_fragment_removal() -> None:
    salt = standardize_parent(raw_smiles="CC[NH3+].[Cl-]")
    parent = standardize_parent(raw_smiles="CC[NH3+]")

    assert salt.parent_id == parent.parent_id
    assert salt.parent_smiles == "CC[NH3+]"
    assert "FRAGMENTS_REMOVED" in {notice.code for notice in salt.notices}

    with pytest.raises(StandardizationFailure) as caught:
        standardize_parent(
            raw_smiles="CC[NH3+].[Cl-]",
            policy=IdentityPolicy(fragment_policy=FragmentPolicy.REJECT_MULTICOMPONENT),
        )
    assert caught.value.reason_code == "MULTICOMPONENT_NOT_ALLOWED"


def test_charge_policy_preserves_or_uncharges_explicitly() -> None:
    preserve = IdentityPolicy(charge_policy=ChargePolicy.PRESERVE)
    assert standardize_parent(raw_smiles="C[NH3+]", policy=preserve).parent_id != (
        standardize_parent(raw_smiles="CN", policy=preserve).parent_id
    )

    uncharge = IdentityPolicy(charge_policy=ChargePolicy.UNCHARGE_WHERE_POSSIBLE)
    charged = standardize_parent(raw_smiles="C[NH3+]", policy=uncharge)
    neutral = standardize_parent(raw_smiles="CN", policy=uncharge)
    assert charged.parent_smiles == neutral.parent_smiles == "CN"
    assert charged.parent_id == neutral.parent_id
    assert "CHARGE_NORMALIZED" in {notice.code for notice in charged.notices}


def test_tautomer_insensitive_policy_merges_keto_and_enol() -> None:
    policy = IdentityPolicy(tautomer_policy=TautomerPolicy.INSENSITIVE)
    keto = standardize_parent(raw_smiles="O=c1[nH]cccc1", policy=policy)
    enol = standardize_parent(raw_smiles="Oc1ncccc1", policy=policy)

    assert keto.parent_smiles == enol.parent_smiles
    assert keto.registration_key == enol.registration_key
    assert keto.parent_id == enol.parent_id


def test_preserved_tautomers_have_distinct_identity() -> None:
    policy = IdentityPolicy(tautomer_policy=TautomerPolicy.PRESERVE)
    keto = standardize_parent(raw_smiles="CC(=O)C", policy=policy)
    enol = standardize_parent(raw_smiles="C=C(O)C", policy=policy)

    assert keto.parent_smiles != enol.parent_smiles
    assert keto.parent_id != enol.parent_id


def test_stereo_sensitive_and_insensitive_policies_are_explicit() -> None:
    clockwise = "F[C@H](Cl)Br"
    anticlockwise = "F[C@@H](Cl)Br"

    sensitive = IdentityPolicy(stereo_policy=StereoPolicy.SENSITIVE)
    assert standardize_parent(raw_smiles=clockwise, policy=sensitive).parent_id != (
        standardize_parent(raw_smiles=anticlockwise, policy=sensitive).parent_id
    )

    insensitive = IdentityPolicy(stereo_policy=StereoPolicy.INSENSITIVE)
    first = standardize_parent(raw_smiles=clockwise, policy=insensitive)
    second = standardize_parent(raw_smiles=anticlockwise, policy=insensitive)
    assert first.parent_smiles == second.parent_smiles == "FC(Cl)Br"
    assert first.parent_id == second.parent_id


def test_atom_mapping_is_removed_or_rejected_but_never_silently_preserved() -> None:
    remove = IdentityPolicy(atom_map_policy=AtomMapPolicy.REMOVE)
    mapped = standardize_parent(raw_smiles="[CH3:1][OH:2]", policy=remove)
    plain = standardize_parent(raw_smiles="CO", policy=remove)

    assert mapped.parent_smiles == "CO"
    assert mapped.parent_id == plain.parent_id
    assert "ATOM_MAP_REMOVED" in {notice.code for notice in mapped.notices}

    reject = IdentityPolicy(atom_map_policy=AtomMapPolicy.REJECT)
    with pytest.raises(StandardizationFailure) as caught:
        standardize_parent(raw_smiles="[CH3:1][OH:2]", policy=reject)
    assert caught.value.reason_code == "ATOM_MAP_NOT_ALLOWED"


def test_isotope_policy_changes_identity_semantics() -> None:
    sensitive = IdentityPolicy(isotope_policy=IsotopePolicy.SENSITIVE)
    assert standardize_parent(raw_smiles="[13CH3]O", policy=sensitive).parent_id != (
        standardize_parent(raw_smiles="CO", policy=sensitive).parent_id
    )

    insensitive = IdentityPolicy(isotope_policy=IsotopePolicy.INSENSITIVE)
    labelled = standardize_parent(raw_smiles="[13CH3]O", policy=insensitive)
    plain = standardize_parent(raw_smiles="CO", policy=insensitive)
    assert labelled.parent_smiles == plain.parent_smiles == "CO"
    assert labelled.parent_id == plain.parent_id

    with pytest.raises(StandardizationFailure) as caught:
        standardize_parent(
            raw_smiles="[13CH3]O",
            policy=IdentityPolicy(isotope_policy=IsotopePolicy.REJECT),
        )
    assert caught.value.reason_code == "ISOTOPE_NOT_ALLOWED"


def test_equal_size_non_equivalent_fragments_are_structured_rejection() -> None:
    with pytest.raises(StandardizationFailure) as caught:
        standardize_parent(raw_smiles="CC.CN")

    assert caught.value.reason_code == "AMBIGUOUS_FRAGMENTS"
    assert "CC" in caught.value.detail and "CN" in caught.value.detail


def test_equal_identical_fragments_have_a_deterministic_parent() -> None:
    result = standardize_parent(raw_smiles="CC.CC")

    assert result.parent_smiles == "CC"
    assert "FRAGMENTS_REMOVED" in {notice.code for notice in result.notices}


@pytest.mark.parametrize(
    ("smiles", "reason"),
    [
        ("C1", "PARSE_ERROR"),
        ("[CH5]", "VALENCE_ERROR"),
        ("C~C", "QUERY_FEATURE_NOT_ALLOWED"),
        ("CC |Sg:n:0,1:ht|", "UNSUPPORTED_STRUCTURAL_METADATA"),
        ("c1ccccc1 |LN:1:1.3|", "UNSUPPORTED_STRUCTURAL_METADATA"),
        ("", "EMPTY_OR_FRAGMENT_ONLY"),
    ],
)
def test_invalid_or_empty_structures_have_stable_reason_codes(
    smiles: str,
    reason: str,
) -> None:
    with pytest.raises(StandardizationFailure) as caught:
        standardize_parent(raw_smiles=smiles)

    assert caught.value.reason_code == reason


def test_undefined_stereo_can_warn_or_reject() -> None:
    warned = standardize_parent(
        raw_smiles="CC(F)Cl",
        policy=IdentityPolicy(undefined_stereo_policy=UndefinedStereoPolicy.WARN),
    )
    assert "UNDEFINED_STEREO" in {notice.code for notice in warned.notices}

    with pytest.raises(StandardizationFailure) as caught:
        standardize_parent(
            raw_smiles="CC(F)Cl",
            policy=IdentityPolicy(undefined_stereo_policy=UndefinedStereoPolicy.REJECT),
        )
    assert caught.value.reason_code == "UNDEFINED_STEREO"


def test_parent_id_binds_policy_rdkit_version_and_hash_version() -> None:
    registration_key = "abc123"
    canonical_key = "canonical-parent:sha256:" + "1" * 64
    baseline = IdentityPolicy()
    changed_policy = IdentityPolicy(fragment_policy=FragmentPolicy.LARGEST)
    changed_hash = IdentityPolicy(registration_hash_version=RegistrationHashVersion.V1)

    ids = {
        parent_id_from_registration(
            registration_key,
            baseline,
            canonical_key=canonical_key,
            rdkit_version="2026.03.1",
        ),
        parent_id_from_registration(
            registration_key,
            changed_policy,
            canonical_key=canonical_key,
            rdkit_version="2026.03.1",
        ),
        parent_id_from_registration(
            registration_key,
            changed_hash,
            canonical_key=canonical_key,
            rdkit_version="2026.03.1",
        ),
        parent_id_from_registration(
            registration_key,
            baseline,
            canonical_key=canonical_key,
            rdkit_version="2027.03.1",
        ),
        parent_id_from_registration(
            registration_key,
            baseline,
            canonical_key="canonical-parent:sha256:" + "2" * 64,
            rdkit_version="2026.03.1",
        ),
        parent_id_from_registration(
            registration_key,
            baseline,
            canonical_key=canonical_key,
            rdkit_version="2026.03.1",
            scheme_name="ALL_LAYERS",
        ),
    }
    assert len(ids) == 6
    assert identity_policy_id(baseline).startswith("identity-policy:sha256:")


def test_registration_layers_are_exposed_as_an_immutable_audit_snapshot() -> None:
    result = standardize_parent(raw_smiles="CCO")

    assert "CANONICAL_SMILES" in result.registration_layers
    assert "TAUTOMER_HASH" in result.registration_layers
    with pytest.raises(TypeError):
        result.registration_layers["CANONICAL_SMILES"] = "changed"  # type: ignore[index]


#: Risperidone.  Its 4H-pyrido[1,2-a]pyrimidin-4-one makes RDKit's tautomer
#: enumerator build a candidate it cannot kekulize; the enumerator logs
#: "Can't kekulize mol" and then throws the candidate away by itself.
RISPERIDONE = "CC1=C(CCN2CCC(CC2)C2=NOC3=C2C=CC(F)=C3)C(=O)N2CCCCC2=N1"


def test_standardization_does_not_narrate_rdkits_discarded_intermediates(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """A screening run must not print chemistry warnings it has already handled.

    ``capfd`` rather than ``capsys`` because RDKit logs from C++, below Python's
    ``sys.stderr``.  The assertion that matters is the pair: silent *and*
    correct.  Silencing alone would be indistinguishable from swallowing a
    molecule that got mangled, so the parent is checked in the same test --
    RDKit discarded only a candidate, and risperidone comes back whole.
    """

    capfd.readouterr()
    result = standardize_parent(raw_smiles=RISPERIDONE)
    captured = capfd.readouterr()

    assert captured.out == ""
    assert captured.err == ""
    assert result.parent_smiles == "Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCCC2"
    assert result.formula == "C23H27FN4O2"
