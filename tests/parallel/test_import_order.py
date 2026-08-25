"""``molcascade.parallel`` must stay importable from a builtin plugin.

``plugins/registry.py`` imports ``plugins.builtin`` at its module bottom, so the
plugin package root is already mid-import while every adapter module is being
executed.  An adapter that reaches for something in ``parallel`` which in turn
reaches back into ``molcascade.plugins`` would close that loop, and the failure
would be a partially-initialised module rather than a clean ``ImportError`` --
which surfaces as an ``AttributeError`` somewhere unrelated.

Fresh interpreters, both orders, because an import cycle is invisible once
either package is already in ``sys.modules``.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "order",
    [
        "import molcascade.parallel, molcascade.plugins",
        "import molcascade.plugins, molcascade.parallel",
    ],
)
def test_both_import_orders_succeed_in_a_fresh_interpreter(order: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", f"{order}; print('ok')"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_the_parallel_package_does_not_pull_in_the_plugin_registry() -> None:
    """Stated as an observation, not as a promise in a docstring.

    A future edit that adds ``from molcascade.plugins.api import StageInput``
    for a type hint would still pass the cycle test above -- the cycle only
    bites in one direction -- and would fail here.
    """

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, molcascade.parallel; "
            "print(sorted(m for m in sys.modules if m.startswith('molcascade.plugins')))",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"
