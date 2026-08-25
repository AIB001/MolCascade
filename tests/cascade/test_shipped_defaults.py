"""What the shipped alert tier does to molecules that are known to be fine.

Tier 4 of the starter cascade is the only place in the funnel that deletes a
molecule outright on a substructure match, with no score to argue with and no
later stage able to rescue it.  Two of its rejects turned out to be wrong for
reasons that were invisible from reading the configuration, and both are the
kind that come back:

* Standardization rewrites neutral sulfoxides into a charge-separated form, and
  two independent alert catalogues flag the rewrite -- so omeprazole, sulindac
  and modafinil left the run over notation.  ``molcascade.chemistry.alerts``
  hands the catalogues the depiction they were authored against.
* NIBR excludes 67 compound *classes* -- steroid, peptide, nucleoside,
  glycoside -- because a physical screening deck does not want them.  Applied as
  a reject that deleted dexamethasone and prednisolone for being steroids.

So these tests run the shipped settings, read straight out of ``defaults.py``,
against a panel of approved oral drugs, and pin the exact set of molecules the
tier removes.  Pinning the set rather than a count is deliberate: a fix that
over-reaches shows up as a name that stopped being rejected, which a retention
percentage would hide.

Scope is this tier and this panel.  Whole-funnel retention is a different
measurement and needs the pipeline; retention here is a floor, never a target.
"""

from __future__ import annotations

import pytest

pytest.importorskip("medchem", reason="the medchem alert arm is an optional backend")

from fixtures.approved_drugs import APPROVED_ORAL_DRUGS, IN_WINDOW

from molcascade.cascade.defaults import default_cascade
from molcascade.cascade.models import CascadeConfig, CriterionConfig
from molcascade.chemistry import standardize_parent
from molcascade.chemistry.alerts import parse_for_alert_matching
from molcascade.plugins import create_builtin_registry
from molcascade.plugins.builtin.medchem_alerts import (
    MedchemAlertsConfig,
    _build_filters,
    _evaluate,
    _nibr_rule_index,
)
from molcascade.plugins.builtin.structural_alerts import (
    _CATALOG_NAMES,
    CatalogAction,
    StructuralAlertConfig,
)

_TIER_ID = "t4_alerts"

#: Sulfoxides, which is the whole point of the first fix.  ``omeprazole`` is in
#: the panel already; the other two are here because the class is worth more
#: than one witness and because sulindac carries a *genuine* Brenk alert
#: (stilbene) alongside the spurious one -- rescuing it must not rescue that.
_SULFOXIDES = {
    "omeprazole": APPROVED_ORAL_DRUGS["omeprazole"],
    "sulindac": "CC1=C(CC(=O)O)c2cc(F)ccc2/C1=C/c1ccc(S(C)=O)cc1",
    "modafinil": "NC(=O)CS(=O)C(c1ccccc1)c1ccccc1",
}

#: Everything the shipped tier removes from the panel, and why it is right to.
#: Three are covalent or polyiodinated matter that a non-covalent docking and
#: MM-PBSA downstream cannot score meaningfully; two are reactive groups.
_EXPECTED_REJECTS = {
    "levothyroxine",  # four aryl iodides
    "ibrutinib",  # acrylamide warhead
    "amoxicillin",  # beta-lactam
    "isoniazid",  # acylhydrazide
    "ranitidine",  # nitroalkene
}


# ------------------------------------------------------------------ fixtures


def _cascade() -> CascadeConfig:
    return default_cascade(registry=create_builtin_registry())


def _criterion(config_id: str) -> CriterionConfig:
    tier = next(tier for tier in _cascade().tiers if tier.id == _TIER_ID)
    criterion = next((c for c in tier.criteria if c.id == config_id), None)
    assert criterion is not None, f"the starter cascade always ships {config_id}"
    return criterion


@pytest.fixture(scope="module")
def rdkit_settings() -> StructuralAlertConfig:
    """The RDKit catalogue arm exactly as it ships.

    Validating through the adapter's own config model means a setting that the
    adapter would reject cannot sit unnoticed in ``defaults.py``.
    """

    return StructuralAlertConfig.model_validate(dict(_criterion("alerts_rdkit").settings))


@pytest.fixture(scope="module")
def medchem_settings() -> MedchemAlertsConfig:
    return MedchemAlertsConfig.model_validate(dict(_criterion("alerts_medchem").settings))


def _parent_smiles(smiles: str) -> str:
    """The form the tier actually sees: a registered parent, not the input."""

    return standardize_parent(raw_smiles=smiles).parent_smiles


def _rdkit_rejects(smiles: str, settings: StructuralAlertConfig) -> set[str]:
    """Catalogue names that would remove this molecule, under these settings."""

    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

    molecule = parse_for_alert_matching(_parent_smiles(smiles))
    assert molecule is not None
    actions = settings.actions()
    hits: set[str] = set()
    for name in _CATALOG_NAMES:
        if actions[name] is not CatalogAction.REJECT:
            continue
        catalog = FilterCatalog(
            FilterCatalogParams(getattr(FilterCatalogParams.FilterCatalogs, name))
        )
        if catalog.HasMatch(molecule):
            hits.add(name)
    return hits


def _medchem_findings(
    names: list[str], smiles: list[str], settings: MedchemAlertsConfig
) -> list[list[tuple[str, str, str, dict[str, object]]]]:
    """Run the adapter's own evaluation, so the test cannot drift from it."""

    molecules = [parse_for_alert_matching(_parent_smiles(s)) for s in smiles]
    assert all(molecule is not None for molecule in molecules)
    common, nibr, _ = _build_filters(settings)
    compound_class_rules, exclude_level_rules = _nibr_rule_index(nibr)
    return _evaluate(
        molecules,
        common=common,
        nibr=nibr,
        reject_at_severity=settings.nibr_reject_at_severity,
        compound_class_rules=compound_class_rules,
        exclude_level_rules=exclude_level_rules,
    )


def _tier_rejects(
    rdkit_settings: StructuralAlertConfig, medchem_settings: MedchemAlertsConfig
) -> dict[str, set[str]]:
    """``name -> reasons`` for every panel molecule the whole tier removes.

    The tier joins its two arms with ``all``, so a molecule survives only by
    passing both, and a reject from either is a reject.
    """

    names = list(APPROVED_ORAL_DRUGS)
    smiles = [APPROVED_ORAL_DRUGS[name] for name in names]
    actions = medchem_settings.actions()
    rejects: dict[str, set[str]] = {}
    for name, findings in zip(
        names, _medchem_findings(names, smiles, medchem_settings), strict=True
    ):
        for family, status, reasons, _extra in findings:
            if actions[status] is CatalogAction.REJECT:
                rejects.setdefault(name, set()).add(f"{family}/{reasons}")
    for name, structure in zip(names, smiles, strict=True):
        for catalog in _rdkit_rejects(structure, rdkit_settings):
            rejects.setdefault(name, set()).add(catalog)
    return rejects


# ------------------------------------------------- the depiction, not the tier


@pytest.mark.parametrize("name", sorted(_SULFOXIDES))
def test_standardization_really_does_charge_separate_the_sulfoxides(name: str) -> None:
    """Guard against the rest of this file passing for the wrong reason.

    If RDKit ever stops rewriting ``S(=O)`` the bug disappears on its own, and
    the tests below would keep passing while testing nothing.  This one fails
    instead, which is the signal to go and delete the workaround.
    """

    assert "[S+]" in _parent_smiles(_SULFOXIDES[name])


# ----------------------------------------------------------- what it removes


def test_sulfoxides_are_not_rejected_for_how_they_were_written(
    rdkit_settings: StructuralAlertConfig, medchem_settings: MedchemAlertsConfig
) -> None:
    names = sorted(_SULFOXIDES)
    smiles = [_SULFOXIDES[name] for name in names]
    actions = medchem_settings.actions()

    for name, structure, findings in zip(
        names, smiles, _medchem_findings(names, smiles, medchem_settings), strict=True
    ):
        rejected = [f for f in findings if actions[f[1]] is CatalogAction.REJECT]
        assert rejected == [], f"{name} rejected by medchem: {rejected}"
        assert _rdkit_rejects(structure, rdkit_settings) == set(), name


def test_the_genuine_alert_on_a_rescued_molecule_is_still_reported(
    medchem_settings: MedchemAlertsConfig,
) -> None:
    """Sulindac is a sulfoxide *and* a stilbene, and only one of those is noise.

    Reverting the charge separation must not quietly clear the molecule of
    everything else it matched.
    """

    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

    molecule = parse_for_alert_matching(_parent_smiles(_SULFOXIDES["sulindac"]))
    catalog = FilterCatalog(
        FilterCatalogParams(FilterCatalogParams.FilterCatalogs.BRENK)
    )
    descriptions = {entry.GetDescription() for entry in catalog.GetMatches(molecule)}
    assert "stilbene" in descriptions
    assert "charged_oxygen_or_sulfur_atoms" not in descriptions


def test_steroids_are_not_deleted_for_being_steroids(
    rdkit_settings: StructuralAlertConfig, medchem_settings: MedchemAlertsConfig
) -> None:
    rejects = _tier_rejects(rdkit_settings, medchem_settings)
    assert "dexamethasone" not in rejects
    assert "prednisolone" not in rejects


def test_a_compound_class_exclusion_is_still_reported_as_one(
    medchem_settings: MedchemAlertsConfig,
) -> None:
    """Downgraded is not the same as unseen.

    The molecule survives, and the decision row still says NIBR excluded it and
    which rule did -- otherwise a reader of the report has no way to know a
    whole chemotype was being judged.
    """

    findings = _medchem_findings(
        ["dexamethasone"], [APPROVED_ORAL_DRUGS["dexamethasone"]], medchem_settings
    )[0]
    nibr = [f for f in findings if f[0] == "NIBR"]
    assert len(nibr) == 1
    _family, status, reasons, extra = nibr[0]
    assert status == "compound_class"
    assert "steroid" in reasons
    # Reported at the severity that would otherwise have rejected it, so the
    # downgrade is visible as a decision rather than as an absence.
    assert extra["severity"] >= medchem_settings.nibr_reject_at_severity


def test_the_liability_rules_are_untouched_by_the_class_exemption(
    medchem_settings: MedchemAlertsConfig,
) -> None:
    """Ranitidine reaches the same severity as the steroids and must still go.

    This is the assertion that the exemption is narrow: it keys on which rules
    fired, not on how severe the result was.
    """

    findings = _medchem_findings(
        ["ranitidine"], [APPROVED_ORAL_DRUGS["ranitidine"]], medchem_settings
    )[0]
    nibr = [f for f in findings if f[0] == "NIBR"]
    assert [f[1] for f in nibr] == ["exclude"]
    assert medchem_settings.actions()["exclude"] is CatalogAction.REJECT


def test_the_tier_removes_exactly_these_approved_drugs_and_no_others(
    rdkit_settings: StructuralAlertConfig, medchem_settings: MedchemAlertsConfig
) -> None:
    """The pin.

    Every name here has a chemical argument behind it, written next to it in
    ``_EXPECTED_REJECTS``.  A name appearing is a new false positive; a name
    disappearing is an exemption that grew past what it was for.  Either way the
    right response is to measure, not to edit this list.
    """

    rejects = _tier_rejects(rdkit_settings, medchem_settings)

    assert set(rejects) == _EXPECTED_REJECTS, {
        "unexpected": {n: sorted(rejects[n]) for n in set(rejects) - _EXPECTED_REJECTS},
        "no longer rejected": sorted(_EXPECTED_REJECTS - set(rejects)),
    }


def test_the_tier_keeps_almost_all_of_the_in_window_panel(
    rdkit_settings: StructuralAlertConfig, medchem_settings: MedchemAlertsConfig
) -> None:
    """A ceiling on the false-positive rate, stated as a rate.

    Three in-window drugs go: ibrutinib and amoxicillin for covalent warheads
    and ranitidine for a nitroalkene, all of which a docking-and-MM-PBSA
    downstream cannot score honestly.  A tier that started removing twice that
    many would still pass every test above.
    """

    rejects = set(_tier_rejects(rdkit_settings, medchem_settings))
    removed = rejects & set(IN_WINDOW)

    assert len(removed) / len(IN_WINDOW) <= 0.06, sorted(removed)


# --------------------------------------------------------- the discriminator


def test_the_compound_class_split_comes_from_nibrs_own_rule_file(
    medchem_settings: MedchemAlertsConfig,
) -> None:
    """The exemption is Novartis' annotation, not a list somebody typed.

    If a medchem update changes the rule file layout, ``_nibr_rule_index``
    raises rather than silently classifying everything as a liability -- but a
    layout that still parses while meaning something else would not raise, so
    the shape of the answer is checked too.
    """

    _common, nibr, _ = _build_filters(medchem_settings)
    compound_class, exclude_level = _nibr_rule_index(nibr)

    # A class rule is a kind of exclusion, so it must be in both sets; a class
    # rule outside ``exclude_level`` could never be downgraded because it would
    # never have rejected anything.
    assert compound_class <= exclude_level
    # And it must stay a small minority: an exemption covering most of the
    # catalogue would be a way of turning NIBR off while appearing to run it.
    assert len(compound_class) < len(exclude_level) / 3
    assert any("steroid" in rule for rule in compound_class)
