"""Whether the funnel does what its own tiers say it does.

Every tier in the starter cascade carries a note explaining the standard of
evidence it applies, and the builder shows that note to whoever assembles a
screen.  A tier is therefore a promise, and two tiers can break each other's
promises without either one looking wrong on its own.

That is not hypothetical.  The drug-likeness tier put three oral-absorption
rules to a 2-of-3 vote, on the stated grounds that "no single rule set gets to
delete a molecule on its own idiosyncrasy" -- and a later tier then applied one
of those same three rules, alone, as a hard gate.  A molecule that lost that
one vote and won the other two was deleted anyway.  Caffeine was.

The reject counts hid it.  Each tier's own number was defensible; only the
composition was wrong.  So the checks here read the assembled cascade rather
than any single tier, and they are about coherence between tiers rather than
about how strict any of them is.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures.approved_drugs import APPROVED_ORAL_DRUGS

from molcascade.cascade.defaults import default_cascade
from molcascade.cascade.models import CascadeConfig
from molcascade.plugins.builtin.medchem_rules import RuleCombination

_MEDCHEM_RULES_PLUGIN = "chemistry.medchem_rules@0.1.0"

#: Molecules that pass Lipinski and Veber but fail Egan.  Each one is an
#: approved oral drug, and each was deleted by the removed tier after winning
#: the drug-likeness vote.
_VOTE_WINNERS_EGAN_REJECTS = ("caffeine", "theophylline", "metformin", "amoxicillin")


@pytest.fixture(scope="module")
def cascade() -> CascadeConfig:
    return default_cascade()


def _rule_criteria(cascade: CascadeConfig) -> list[tuple[str, tuple[str, ...], str]]:
    """Return ``(tier_id, rules, combination)`` for every medchem-rules criterion."""

    found = []
    for tier in cascade.tiers:
        for criterion in tier.criteria:
            if not criterion.enabled or criterion.backend != _MEDCHEM_RULES_PLUGIN:
                continue
            raw = criterion.settings.get("rules", ())
            rules = tuple(
                part.strip() for part in raw.split(",") if part.strip()
            ) if isinstance(raw, str) else tuple(raw)
            combination = str(
                criterion.settings.get("combination", RuleCombination.ALL.value)
            )
            found.append((tier.id, rules, combination))
    return found


def test_no_rule_is_put_to_a_vote_in_one_tier_and_enforced_in_another(
    cascade: CascadeConfig,
) -> None:
    """A quorum is only a quorum if nothing downstream overrules it.

    This is the invariant the removed ADMET tier broke.  It is stated over the
    whole cascade rather than pinned to Egan because the same mistake is
    available to any rule the catalogue offers: adding a second rules criterion
    is easy, and noticing that it re-enforces a rule three tiers up is not.
    """

    voted: dict[str, str] = {}
    enforced: dict[str, str] = {}
    for tier_id, rules, combination in _rule_criteria(cascade):
        target = voted if combination == RuleCombination.AT_LEAST.value else enforced
        for rule in rules:
            target.setdefault(rule, tier_id)

    overruled = sorted(set(voted) & set(enforced))
    assert not overruled, (
        "these rules are voted on in one tier and enforced as a verdict in "
        "another, so the vote does not decide: "
        + ", ".join(f"{rule} (voted in {voted[rule]}, enforced in {enforced[rule]})" for rule in overruled)
    )


def test_the_same_rule_is_never_evaluated_by_two_tiers(cascade: CascadeConfig) -> None:
    """Cheap to state, and it is the cost half of the same defect.

    Re-running a rule cannot change its answer, so a duplicate is either
    redundant work or a tier quietly disagreeing with another -- and at the
    scale this cascade is built for, the redundant reading is not cheap either.
    """

    seen: dict[str, str] = {}
    duplicated: list[str] = []
    for tier_id, rules, _combination in _rule_criteria(cascade):
        for rule in rules:
            if rule in seen:
                duplicated.append(f"{rule} ({seen[rule]} and {tier_id})")
            else:
                seen[rule] = tier_id
    assert not duplicated, "evaluated more than once: " + ", ".join(duplicated)


@pytest.mark.parametrize("name", _VOTE_WINNERS_EGAN_REJECTS)
def test_a_drug_that_wins_the_quorum_survives_the_whole_funnels_rule_tiers(
    cascade: CascadeConfig, name: str
) -> None:
    """The regression itself, run through medchem rather than reasoned about.

    Each of these passes Lipinski and Veber and fails Egan, so each one tests
    the composition and not any single rule.  Caffeine is the clearest: a
    default that drops it is not making a scope decision, it is making an
    arithmetic mistake.
    """

    pytest.importorskip("medchem", reason="medchem is not installed in this environment")
    from medchem.rules import RuleFilters

    smiles = APPROVED_ORAL_DRUGS[name]
    for tier_id, rules, combination in _rule_criteria(cascade):
        frame = RuleFilters(rule_list=list(rules)).__call__(
            mols=[smiles], n_jobs=1, progress=False
        )
        passes = [
            bool(frame[column][0])
            for column in frame.columns
            if column.startswith("rule_")
        ]
        if combination == RuleCombination.AT_LEAST.value:
            required = int(
                dict(
                    (tid, crit.settings.get("minimum_passes", 1))
                    for tier in cascade.tiers
                    for crit in tier.criteria
                    for tid in [tier.id]
                    if crit.backend == _MEDCHEM_RULES_PLUGIN
                )[tier_id]
            )
        elif combination == RuleCombination.ANY.value:
            required = 1
        else:
            required = len(passes)

        assert sum(passes) >= required, (
            f"{name} is removed by tier {tier_id}: it passes {sum(passes)} of "
            f"{len(passes)} rules ({', '.join(rules)}) and the tier requires {required}"
        )
