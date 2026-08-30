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
plugin namespace their blocks are drawn from, and the feeder above them by the
contracts those blocks read.  Renaming a tier in ``default_cascade`` therefore
cannot quietly turn this into a no-op, and a starter that ships no docking at
all raises here rather than handing back a cascade the caller believes it
modified.
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
    config["tiers"] = [tier for tier in config["tiers"] if tier["id"] not in drop]
    return config
