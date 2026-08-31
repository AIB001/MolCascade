"""The shipped starter cascade, minus the docking it now ships with.

``default_cascade`` gained a docking tier -- Uni-Dock and KarmaDock reporting in
parallel -- and, above it, the conformer tier that feeds them both.  That is the
right default for an operator and the wrong starting point for a test that adds
a docking block of its own and then asks what the machinery makes of it: the
shipped pair answers first, and the question never reaches the block the test
added.  It shows up three different ways -- a duplicate criterion id, an error
message naming Uni-Dock's unfilled fields instead of KarmaDock's, a dependency
that is already satisfied so the refusal under test never happens -- and in
three test modules, so the stripping lives here rather than three times.

Everything is derived rather than named.  The docking tiers are found by the
plugin namespace their blocks are drawn from, the feeder above them by the
contracts those blocks read, and the tiers below them by what those blocks
produce -- the post-docking metrics read the docking score, so stripping docking
without them leaves a cascade that cannot be lowered for a reason no caller is
asking about.  Renaming a tier in ``default_cascade`` therefore cannot quietly
turn this into a no-op, and a starter that ships no docking at all raises here
rather than handing back a cascade the caller believes it modified.
"""

from __future__ import annotations

import json
from typing import Any

from molcascade.contracts import DECISION_V1, PARENT_V1
from molcascade.plugins import create_builtin_registry


def starter_without_docking(default_cascade: dict[str, Any]) -> dict[str, Any]:
    """A deep copy of the starter cascade with its docking tiers removed.

    The argument is ``build_cascade_builder_payload()["default_cascade"]``.  The
    copy is deep because callers go on to append tiers to it, and the payload is
    rebuilt per call rather than cached -- but a caller that mutated the shipped
    dictionary would still be editing the object every other assertion in the
    module reads.
    """

    registry = create_builtin_registry()
    config: dict[str, Any] = json.loads(json.dumps(default_cascade))
    docking = [
        tier
        for tier in config["tiers"]
        if any(item["backend"].startswith("docking.") for item in tier["criteria"])
    ]
    # Only the evidence a docking block reads and does not compute: the two
    # contracts every stage carries would otherwise match every tier in the file.
    consumed = {
        contract
        for tier in docking
        for item in tier["criteria"]
        for contract in registry.entry(item["backend"]).descriptor.inputs
    } - {PARENT_V1.id, DECISION_V1.id}
    drop = {tier["id"] for tier in docking}
    drop |= {
        tier["id"]
        for tier in config["tiers"]
        for item in tier["criteria"]
        if consumed.intersection(registry.entry(item["backend"]).descriptor.outputs)
    }
    assert drop, "the starter ships no docking; this helper has become a no-op"
    drop |= _orphaned_by(config, drop, registry)
    config["tiers"] = [tier for tier in config["tiers"] if tier["id"] not in drop]
    return config


def _orphaned_by(config: dict[str, Any], drop: set[str], registry: Any) -> set[str]:
    """Tiers left reading a contract no *earlier* surviving tier produces.

    The feeder above docking is not the only neighbour: the post-docking
    metrics below it consume ``docking_score/v1``, so removing docking alone
    leaves a cascade that cannot be lowered at all -- and every caller here
    goes on to assert something about a *different* refusal.

    Two details make this a loop rather than one pass.  A tier is orphaned only
    when every producer of what it needs is going away, so a contract two
    surviving tiers emit survives losing one of them; and dropping a tier can
    orphan the one below it in turn.  Producers are counted only from tiers that
    come earlier in the funnel, which is the rule lowering applies.
    """

    orphaned: set[str] = set()
    while True:
        surviving = [tier for tier in config["tiers"] if tier["id"] not in drop | orphaned]
        available: set[str] = set()
        found: set[str] = set()
        for tier in surviving:
            needs: set[str] = set()
            for item in tier["criteria"]:
                descriptor = registry.entry(item["backend"]).descriptor
                needs |= set(descriptor.inputs) - {PARENT_V1.id, DECISION_V1.id}
            if needs - available:
                found.add(tier["id"])
            for item in tier["criteria"]:
                available |= set(registry.entry(item["backend"]).descriptor.outputs)
        if not found:
            return orphaned
        orphaned |= found
