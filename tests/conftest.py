"""Make every test run on the same machine: one with nothing installed.

MolCascade resolves an out-of-process engine's installation path from three
places, in order: the cascade, ``MOLCASCADE_<ENGINE>_<FIELD>``, and then the
layout ``envs/bootstrap.sh`` creates on disk.  The third is the reason this file
exists.  A developer who has actually installed KarmaDock -- which is the point
of the feature, and true of the machine this was written on -- would otherwise
run a different suite from CI: a config left deliberately blank would come back
carrying ``/home/them/anaconda3/envs/karmadock/bin/python``, and assertions
about what a blank path means would pass or fail depending on whose laptop was
running them.

So discovery is pointed at an empty directory for every test, and the engine
variables are cleared.  Both are ordinary environment settings, so a test that
wants to exercise discovery overrides them with ``monkeypatch`` and populates a
directory of its own -- which is what ``tests/plugins/test_machine_paths.py``
does.  Nothing here disables the mechanism; it only stops the mechanism from
reading a machine the test did not set up.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


def _engine_variables() -> tuple[str, ...]:
    """Every variable that could answer an installation path.

    Read from the models rather than listed, so an engine added later cannot
    quietly start reading the developer's environment again.
    """

    from molcascade.plugins.builtin.aizynthfinder import AiZynthFinderConfig
    from molcascade.plugins.builtin.docking.gnina import GninaConfig
    from molcascade.plugins.builtin.docking.karmadock import KarmaDockConfig
    from molcascade.plugins.builtin.docking.unidock import UniDockConfig

    return tuple(
        sorted(
            {
                spec.variable(model.engine_id)
                for model in (KarmaDockConfig, GninaConfig, UniDockConfig, AiZynthFinderConfig)
                for spec in model.installed_paths()
            }
        )
    )


@pytest.fixture(scope="session")
def _nothing_installed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One directory, created once, that no test ever puts anything in."""

    return tmp_path_factory.mktemp("nothing-installed")


def apply_hermetic_installation_paths(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    """Point every place discovery looks at a directory that stays empty."""

    monkeypatch.setenv("MOLCASCADE_BACKEND_ROOT", str(root / "backends"))
    monkeypatch.setenv("MOLCASCADE_CONDA_ENVS", str(root / "envs"))
    monkeypatch.setenv("MOLCASCADE_AIZYNTH_DATA", str(root / "aizynth-data"))
    for variable in _engine_variables():
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture(autouse=True)
def _hermetic_installation_paths(
    monkeypatch: pytest.MonkeyPatch,
    _nothing_installed: Path,
) -> None:
    apply_hermetic_installation_paths(monkeypatch, _nothing_installed)


@pytest.fixture(scope="module")
def hermetic_installation_paths(_nothing_installed: Path) -> Iterator[None]:
    """The same clearing, for a fixture that is built before the autouse one.

    pytest constructs higher-scoped fixtures first, so a module- or
    session-scoped fixture runs *before* ``_hermetic_installation_paths`` above.
    Measured: inside a module-scoped fixture ``MOLCASCADE_CONDA_ENVS`` is still
    unset and ``MOLCASCADE_KARMADOCK_PYTHON`` still holds whatever the developer
    exported, while inside the test body both are the hermetic values.

    Nothing in the suite was affected by that until something wanted to *run a
    pipeline* at module scope: building a cascade resolves no engine path -- the
    ``executable`` field comes back empty under either scope, because resolution
    is deferred to run start -- but a real run resolves them for real.  So a
    module-scoped fixture that executes anything must ask for this one, and its
    dependents inherit the ordering by depending on it.
    """

    with pytest.MonkeyPatch.context() as monkeypatch:
        apply_hermetic_installation_paths(monkeypatch, _nothing_installed)
        yield
