"""Where a docking engine lives is a fact about the machine, not about the run.

An operator who has installed KarmaDock has installed it once, at one path, and
that path is the same for every campaign they will ever run on that box.  Before
these tests there was nowhere to record it: the interpreter path and the
repository path were ordinary config fields, so every exported ``cascade.json``
had to be hand-edited before it could run, and a config written on one machine
could not be handed to another without editing it again.

So each engine path may now be supplied by an environment variable named after
the engine and the field -- ``MOLCASCADE_KARMADOCK_EXECUTABLE`` -- set once per
machine, or found where ``envs/bootstrap.sh`` installs things.  What is under
test here is the boundary of that idea:

* the fallback fills only *machine* paths.  The receptor and the reference
  ligand are facts about the campaign; an environment variable that quietly
  supplied a receptor would be answering the most important question in the run
  from a shell profile nobody reads.
* an explicit config value always wins, so a cascade that names a path is not
  silently redirected by the machine it lands on.
* the filled path goes through the same absolute-path check as a typed one.
  This is not the ``PATH`` probing ``resolved_executable`` refuses -- an
  explicit variable is a statement, a ``PATH`` search is a guess -- but a
  statement can still be wrong, and a relative one is refused either way.
* with no answer from anywhere, validation *passes* with the field blank, and
  the refusal moves to ``preflight_engine_paths`` on the host about to do the
  work.  It used to happen here, and that was wrong in the one case that
  matters most: the builder runs on a laptop with no engines, so its own
  exported cascade failed its own validator, and the operator was told to
  hand-edit a file a tool had just generated for them.  See
  ``tests/unit/test_backend_preflight.py`` for what now catches it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from molcascade.plugins.builtin.docking.common import environment_key
from molcascade.plugins.builtin.docking.gnina import GninaConfig
from molcascade.plugins.builtin.docking.karmadock import KarmaDockConfig
from molcascade.plugins.builtin.docking.unidock import UniDockConfig

#: Campaign facts, supplied on every construction below so that the only thing
#: varying between cases is where the engine is.  These are never touched.
_KARMADOCK_SITE = {
    "receptor_path": "/campaign/receptor.pdb",
    "reference_ligand_path": "/campaign/ligand.mol2",
}
_BOX = {
    "receptor_path": "/campaign/receptor.pdb",
    "center_x": 1.0,
    "center_y": 2.0,
    "center_z": 3.0,
    "size_x": 20.0,
    "size_y": 20.0,
    "size_z": 20.0,
}
_UNIDOCK_SITE = {**_BOX, "receptor_pdbqt_path": "/campaign/receptor.pdbqt"}


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start from a machine that has configured nothing.

    The engine variables are already cleared by the autouse fixture in
    ``tests/conftest.py``, which also points discovery at an empty directory.
    This adds the campaign fields, which are named here precisely because
    nothing should ever read them from the environment -- clearing them means a
    regression that started to would be caught by the assertion rather than
    masked by a developer's shell.
    """

    for engine in ("karmadock", "gnina", "unidock"):
        for field in ("executable", "repo_path", "receptor_path", "reference_ligand_path"):
            monkeypatch.delenv(environment_key(engine, field), raising=False)


def test_variable_names_follow_the_asset_store_precedent() -> None:
    assert environment_key("karmadock", "executable") == "MOLCASCADE_KARMADOCK_EXECUTABLE"
    assert environment_key("karmadock", "repo_path") == "MOLCASCADE_KARMADOCK_REPO_PATH"
    assert environment_key("gnina", "executable") == "MOLCASCADE_GNINA_EXECUTABLE"
    assert environment_key("unidock", "executable") == "MOLCASCADE_UNIDOCK_EXECUTABLE"


def test_engines_do_not_share_a_variable() -> None:
    """Three engines, three installations, three names.

    A single ``MOLCASCADE_DOCKING_EXECUTABLE`` would make it impossible to have
    GNINA and UniDock configured at the same time, which is the ordinary case.
    """

    ids = {KarmaDockConfig.engine_id, GninaConfig.engine_id, UniDockConfig.engine_id}
    assert ids == {"karmadock", "gnina", "unidock"}


def test_a_missing_path_validates_blank_rather_than_refusing_the_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cascade the builder exports has to survive its own validator.

    A blank engine path is not a mistake in the file; it is the file correctly
    declining to state a fact about a machine it was not written on.  Refusing
    it here is what made ``molcascade generate config`` produce something the
    operator then had to hand-edit -- and the refusal is not lost, only moved to
    the host that can actually answer.
    """

    _clear(monkeypatch)
    config = KarmaDockConfig(executable="", repo_path="", **_KARMADOCK_SITE)
    assert config.executable == ""
    assert config.repo_path == ""


def test_a_config_that_names_no_paths_at_all_still_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting the keys and writing them blank have to mean the same thing.

    A cascade written by hand leaves them out; the builder writes every field it
    shows.  If only one of those shapes validated, the fix would depend on which
    tool wrote the file.
    """

    _clear(monkeypatch)
    config = KarmaDockConfig(**_KARMADOCK_SITE)
    assert config.executable == ""
    assert config.repo_path == ""


def test_the_environment_supplies_paths_the_config_omits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("MOLCASCADE_KARMADOCK_EXECUTABLE", "/opt/envs/karmadock/bin/python")
    monkeypatch.setenv("MOLCASCADE_KARMADOCK_REPO_PATH", "/opt/backends/KarmaDock")
    config = KarmaDockConfig(**_KARMADOCK_SITE)
    assert config.executable == "/opt/envs/karmadock/bin/python"
    assert config.repo_path == "/opt/backends/KarmaDock"


def test_the_environment_also_fills_the_blanks_the_builder_exports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The builder writes every field it shows, filled or not.

    So the shape that actually arrives from an exported cascade is not a missing
    key but ``"executable": ""``.  Both have to mean "not configured", or the
    fallback would work only for configs written by hand.
    """

    _clear(monkeypatch)
    monkeypatch.setenv("MOLCASCADE_KARMADOCK_EXECUTABLE", "/opt/envs/karmadock/bin/python")
    monkeypatch.setenv("MOLCASCADE_KARMADOCK_REPO_PATH", "/opt/backends/KarmaDock")
    config = KarmaDockConfig(executable="  ", repo_path="", **_KARMADOCK_SITE)
    assert config.executable == "/opt/envs/karmadock/bin/python"
    assert config.repo_path == "/opt/backends/KarmaDock"


def test_a_configured_path_beats_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("MOLCASCADE_KARMADOCK_EXECUTABLE", "/opt/envs/karmadock/bin/python")
    config = KarmaDockConfig(
        executable="/home/alice/envs/karmadock/bin/python",
        repo_path="/home/alice/KarmaDock",
        **_KARMADOCK_SITE,
    )
    assert config.executable == "/home/alice/envs/karmadock/bin/python"


def test_a_blank_variable_is_not_a_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """``export MOLCASCADE_KARMADOCK_EXECUTABLE=`` means unset, not "".

    An empty export is what a half-finished shell profile leaves behind.
    Treating it as an answer would put ``""`` past the preflight's "is it
    blank?" test and into the "nothing exists at the configured path" branch,
    which describes a different mistake from the one that was made.
    """

    _clear(monkeypatch)
    monkeypatch.setenv("MOLCASCADE_KARMADOCK_EXECUTABLE", "   ")
    monkeypatch.setenv("MOLCASCADE_KARMADOCK_REPO_PATH", "/opt/backends/KarmaDock")
    config = KarmaDockConfig(**_KARMADOCK_SITE)
    assert config.executable == ""
    assert config.repo_path == "/opt/backends/KarmaDock"


def test_campaign_paths_are_not_filled_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The receptor is the question, not the toolchain.

    Two campaigns on one machine share an installation and share nothing else.
    A variable that supplied the receptor would let a cascade that names no
    protein run anyway, against whatever the last project left exported.
    """

    _clear(monkeypatch)
    monkeypatch.setenv("MOLCASCADE_KARMADOCK_EXECUTABLE", "/opt/envs/karmadock/bin/python")
    monkeypatch.setenv("MOLCASCADE_KARMADOCK_REPO_PATH", "/opt/backends/KarmaDock")
    monkeypatch.setenv("MOLCASCADE_KARMADOCK_RECEPTOR_PATH", "/somewhere/else.pdb")
    monkeypatch.setenv("MOLCASCADE_KARMADOCK_REFERENCE_LIGAND_PATH", "/somewhere/else.mol2")
    with pytest.raises(ValidationError) as caught:
        KarmaDockConfig()
    message = str(caught.value)
    assert "receptor_path" in message
    assert "reference_ligand_path" in message
    assert "/somewhere/else.pdb" not in message
    assert "receptor_path" not in {spec.field for spec in KarmaDockConfig.installed_paths()}


@pytest.mark.parametrize(
    ("model", "site", "variable"),
    [
        (GninaConfig, _BOX, "MOLCASCADE_GNINA_EXECUTABLE"),
        (UniDockConfig, _UNIDOCK_SITE, "MOLCASCADE_UNIDOCK_EXECUTABLE"),
    ],
)
def test_the_box_engines_take_their_executable_from_the_environment_too(
    monkeypatch: pytest.MonkeyPatch,
    model: type[GninaConfig] | type[UniDockConfig],
    site: dict[str, object],
    variable: str,
) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv(variable, "/opt/bin/engine")
    assert model(**site).executable == "/opt/bin/engine"
    monkeypatch.delenv(variable)
    assert model(**site).executable == ""


@pytest.mark.parametrize(
    ("model", "site"),
    [
        (GninaConfig, _BOX),
        (UniDockConfig, _UNIDOCK_SITE),
        (KarmaDockConfig, _KARMADOCK_SITE),
    ],
)
def test_a_relative_variable_is_refused_for_every_engine(
    monkeypatch: pytest.MonkeyPatch,
    model: type[GninaConfig] | type[UniDockConfig] | type[KarmaDockConfig],
    site: dict[str, object],
) -> None:
    """Blank means "ask the run host"; a bare name means "search PATH", which is
    the one thing these adapters will not do.

    The two have to stay distinguishable now that one of them passes, or the
    typo that turns an absolute path into a command name would be reported as an
    unresolved installation instead of as the typo it is.
    """

    _clear(monkeypatch)
    variable = environment_key(model.engine_id, "executable")
    monkeypatch.setenv(variable, "engine")
    with pytest.raises(ValidationError) as caught:
        model(**site)
    message = str(caught.value)
    assert "executable must be an absolute path, not a name to resolve on PATH" in message
    assert variable in message
