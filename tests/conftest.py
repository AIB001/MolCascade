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


@pytest.fixture(autouse=True)
def _hermetic_installation_paths(
    monkeypatch: pytest.MonkeyPatch,
    _nothing_installed: Path,
) -> None:
    """Point every place discovery looks at a directory that stays empty."""

    monkeypatch.setenv("MOLCASCADE_BACKEND_ROOT", str(_nothing_installed / "backends"))
    monkeypatch.setenv("MOLCASCADE_CONDA_ENVS", str(_nothing_installed / "envs"))
    monkeypatch.setenv("MOLCASCADE_AIZYNTH_DATA", str(_nothing_installed / "aizynth-data"))
    for variable in _engine_variables():
        monkeypatch.delenv(variable, raising=False)
