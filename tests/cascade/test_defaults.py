"""The starter cascade: how it is assembled, and the defaults it overrides.

Two things here are decisions rather than description, and both are the kind
that get quietly reverted by someone tidying up.

The synthesizability tier is the one part of the default that changes shape
depending on the filesystem: SCScore needs a 6 MB weight file MolCascade will
not download, so the tier ships with one arm or two.  That is a branch with
real semantics rather than a cosmetic difference -- one arm rejects on a single
score, two arms reject only when both agree -- so both sides are pinned here.
Neither test reads the published weights.  What is under test is how the
cascade is assembled from an answer about the asset, so the answer is supplied
directly; a test that needed the bytes would be testing the download.

The standardization step then overrides the identity default for tautomers,
which is the only identity setting the starter cascade touches.  The tests at
the bottom pin both halves of that argument: what the default would do to a
real molecule, and what the override does instead.
"""

from __future__ import annotations

import pytest

from molcascade.cascade.defaults import SCSCORE_ASSET_MEMBER, default_cascade
from molcascade.cascade.models import CascadeConfig, TierConfig
from molcascade.plugins import create_builtin_registry

_TIER_ID = "t5_synthesis"


class _Answer:
    """Stands in for the verified status of an asset directory."""

    def __init__(self, ready: bool) -> None:
        self.ready = ready


def _pretend_scscore_is(ready: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    # ``_scscore_criterion`` imports this inside the function body, so patching
    # the module attribute reaches it without touching the asset root.
    monkeypatch.setattr("molcascade.assets.asset_status", lambda _spec: _Answer(ready))


def _synthesis_tier(cascade: CascadeConfig) -> TierConfig:
    tier = next((tier for tier in cascade.tiers if tier.id == _TIER_ID), None)
    assert tier is not None, "the starter cascade always offers synthesizability"
    return tier


def test_without_the_weights_the_tier_keeps_working_with_sa_score_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pretend_scscore_is(False, monkeypatch)

    tier = _synthesis_tier(default_cascade(registry=create_builtin_registry()))

    assert len(tier.criteria) == 1
    # With one arm "any" and "all" are the same rule, and "all" is the one that
    # reads honestly in a report.
    assert tier.mode == "all"


def test_a_missing_asset_names_the_command_that_would_supply_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pretend_scscore_is(False, monkeypatch)

    tier = _synthesis_tier(default_cascade(registry=create_builtin_registry()))

    # A default that silently drops a capability teaches the user it was never
    # on offer.  The note is the only place that difference surfaces.
    assert "molcascade assets fetch scscore" in tier.note


def test_with_the_weights_a_second_arm_appears_and_the_tier_becomes_a_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pretend_scscore_is(True, monkeypatch)

    tier = _synthesis_tier(default_cascade(registry=create_builtin_registry()))

    assert len(tier.criteria) == 2
    # The point of the second opinion: drop a molecule only when two scores
    # built on different evidence agree it is hard to make.
    assert tier.mode == "any"


def test_the_second_arm_is_pinned_by_asset_reference_and_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from molcascade.assets import asset_spec

    _pretend_scscore_is(True, monkeypatch)

    tier = _synthesis_tier(default_cascade(registry=create_builtin_registry()))
    scscore = next(c for c in tier.criteria if c.id == "synthesis_scscore")

    settings = dict(scscore.settings)
    declared = asset_spec("scscore").file(SCSCORE_ASSET_MEMBER).sha256
    # An absolute path would make the exported cascade mean something different
    # on the machine that runs it.
    assert settings["weights_path"] == f"asset:scscore/{SCSCORE_ASSET_MEMBER}"
    assert settings["expected_weights_sha256"] == declared
    # The 1024bool variant is the bit model; guessing from the file name is
    # exactly what the adapter refuses to do.
    assert settings["fingerprint"] == "bits"


def test_both_arms_agree_on_which_direction_is_harder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pretend_scscore_is(True, monkeypatch)

    tier = _synthesis_tier(default_cascade(registry=create_builtin_registry()))

    # SA and SCScore are unrelated scales, but both rise with difficulty.  A
    # tier that mixed directions would reject the molecules it meant to keep.
    for criterion in tier.criteria:
        # An evidence producer only filters once it carries a gate; a scoring
        # arm with no gate would silently contribute nothing to the union.
        assert criterion.gate is not None
        assert criterion.gate.settings["expected_direction"] == "HIGHER_HARDER"
        assert criterion.gate.settings["maximum"] > 0


# ------------------------------------------------------- the depiction shipped

#: Indirubin-3'-oxime, the published GSK-3beta/CDK inhibitor.  Two oxindoles
#: joined by an exocyclic alkene, with an oxime on one of them -- which is the
#: exact arrangement RDKit's canonical tautomer scoring gets wrong.
INDIRUBIN_3_OXIME = r"O=C1Nc2ccccc2/C1=C1\Nc2ccccc2/C1=N\O"

_NITROSO = "[#6][N;X2]=[O;X1]"
_OXIME = "[#6]=[N][OH]"


def _has(smiles: str, smarts: str) -> bool:
    from rdkit import Chem

    return Chem.MolFromSmiles(smiles).HasSubstructMatch(Chem.MolFromSmarts(smarts))


def test_the_canonical_tautomer_would_invent_a_nitroso() -> None:
    """Why the shipped cascade overrides the identity default.

    RDKit scores a tautomer with a large bonus per aromatic ring.  Aromatizing
    both oxindole lactams to lactims wins two of those, and paying for it moves
    the oxime's hydrogen so the group comes out as a nitroso -- which two alert
    catalogues then reject as a genuine liability.

    This test asserts the *unwanted* behaviour on purpose.  If RDKit ever
    rescores and this starts passing chemistry, the override below has stopped
    earning its place and should be reconsidered rather than left in.
    """

    from molcascade.chemistry import standardize_parent
    from molcascade.chemistry.policies import IdentityPolicy, TautomerPolicy

    canonical = standardize_parent(
        raw_smiles=INDIRUBIN_3_OXIME,
        policy=IdentityPolicy(tautomer_policy=TautomerPolicy.INSENSITIVE),
    ).parent_smiles

    assert _has(canonical, _NITROSO)
    assert not _has(canonical, _OXIME)


def test_the_shipped_cascade_registers_molecules_as_they_were_drawn() -> None:
    """The override, and the thing it protects.

    ``parent_smiles`` is not only what the alert catalogues match against --
    it is what the shortlist hands to docking.  A depiction that swaps a
    functional group is wrong in both places, and only one of them would show
    up in a funnel report.
    """

    from molcascade.chemistry import standardize_parent
    from molcascade.chemistry.policies import IdentityPolicy

    cascade = default_cascade(registry=create_builtin_registry())
    assert cascade.standardize is not None
    policy = IdentityPolicy.model_validate(
        cascade.standardize.settings["identity_policy"]
    )
    assert policy.tautomer_policy.value == "preserve"

    parent = standardize_parent(raw_smiles=INDIRUBIN_3_OXIME, policy=policy).parent_smiles

    assert _has(parent, _OXIME)
    assert not _has(parent, _NITROSO)
