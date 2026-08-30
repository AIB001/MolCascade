"""The package half of the preflight: a missing backend costs nothing.

The behaviour under test is a timing property, not a message.  Every adapter
that needs an optional package already fails closed with an install hint, but it
did so from inside ``execute()`` -- so the failure arrived after the tiers above
it had processed the whole library, and after a run directory and a cache had
been written.  These tests pin the answer to "before any of that", which is the
same guarantee ``preflight_assets`` already gave for weights.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from molcascade.backends.catalog import BUILTIN_BACKEND_SPECS
from molcascade.backends.models import (
    BackendInterface,
    BackendSpec,
    BackendTier,
    Capability,
    LicenseClass,
)
from molcascade.backends.preflight import (
    _index_by_plugin_ref,
    preflight_backends,
    required_backends,
)
from molcascade.errors import ConfigError, PluginError
from molcascade.pipeline.compiler import CompiledStage
from molcascade.plugins import create_builtin_registry

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _normalize(requirement: str) -> str:
    """The PEP 503 form of the distribution name at the head of ``requirement``."""

    name = re.split(r"[<>=!~\[;\s]", requirement, maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def _stage(
    stage_id: str,
    plugin_key: str,
    config: dict[str, object] | None = None,
) -> CompiledStage:
    """A compiled stage stripped to the three fields the preflight reads.

    Building one through the compiler would need a whole valid pipeline for
    each case; the preflight only ever looks at ``stage_id``, ``plugin_key``
    and -- for an isolated backend -- ``config``, and a descriptor it never
    touches is not worth constructing.
    """

    descriptor = create_builtin_registry().entries()[0].descriptor
    return CompiledStage(
        stage_id=stage_id,
        slot="gate",
        plugin_key=plugin_key,
        descriptor=descriptor,
        config=config or {},
        input_contract=None,
        output_contract="molcascade.decision.v1",
        input_port=None,
    )


def test_a_stage_whose_package_is_absent_is_named_before_the_run_starts() -> None:
    absent = BackendSpec(
        id="nonexistent.tool",
        capability=Capability.GATE,
        display_name="A tool that is not installed",
        tier=BackendTier.OPTIONAL,
        interface=BackendInterface.PYTHON,
        license_spdx="MIT",
        license_class=LicenseClass.PERMISSIVE,
        module="molcascade_no_such_module",
        distribution="molcascade-no-such-distribution",
        plugin_ref="chemistry.imaginary@0.1.0",
    )
    index = _index_by_plugin_ref((*BUILTIN_BACKEND_SPECS, absent))
    assert index["chemistry.imaginary@0.1.0"] is absent


def test_probe_reports_every_missing_backend_rather_than_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One command has to fix the whole cascade.

    Reporting one package at a time turns provisioning into an
    install-restart-install loop, which is the delay this check exists to
    remove -- and it is the loop the old per-adapter failure produced, since
    each restart got a tier further before hitting the next missing package.
    """

    missing = {
        "chemistry.a@0.1.0": _spec("a", "molcascade_absent_a", extra="alerts"),
        "chemistry.b@0.1.0": _spec("b", "molcascade_absent_b", extra="admet"),
    }
    monkeypatch.setattr(
        "molcascade.backends.preflight._BY_PLUGIN_REF", missing, raising=True
    )
    stages = (_stage("tier3_a", "chemistry.a@0.1.0"), _stage("tier5_b", "chemistry.b@0.1.0"))

    with pytest.raises(PluginError) as raised:
        preflight_backends(stages)

    error = raised.value
    assert error.code == "BACKEND_PREFLIGHT_FAILED"
    assert "tier3_a" in str(error)
    assert "tier5_b" in str(error)
    # Both extras, one command, and quoted -- a bare bracket is a shell glob.
    assert error.hint is not None
    assert 'pip install "molcascade[admet,alerts]"' in error.hint


def _spec(suffix: str, module: str, *, extra: str | None = None) -> BackendSpec:
    return BackendSpec(
        id=f"absent.{suffix}",
        capability=Capability.GATE,
        display_name=f"Absent backend {suffix}",
        tier=BackendTier.OPTIONAL,
        interface=BackendInterface.PYTHON,
        license_spdx="MIT",
        license_class=LicenseClass.PERMISSIVE,
        module=module,
        distribution=f"absent-{suffix}",
        plugin_ref=f"chemistry.{suffix}@0.1.0",
        extra=extra,
    )


def test_an_uncatalogued_plugin_does_not_block_the_run() -> None:
    """A third-party plugin has dependencies this module cannot know.

    Failing closed on the unknown would mean a working custom cascade refuses to
    start, so the unknown is skipped and its adapter keeps whatever fail-closed
    behaviour it wrote for itself.
    """

    stages = (_stage("custom", "someones.own.plugin@9.9.9"),)
    assert required_backends(stages) == ()
    assert preflight_backends(stages) == ()


def test_installed_backends_are_reported_not_merely_tolerated() -> None:
    """RDKit is a core dependency, so this is the confirmed-ready path."""

    stages = (_stage("t1_validity", "chemistry.rdkit_hard_gate@0.1.0"),)
    assert preflight_backends(stages) == ("rdkit.filtercatalog",)


def test_one_plugin_ref_served_by_backends_needing_different_packages_is_skipped() -> None:
    """Which one runs would depend on settings this module does not interpret."""

    contested = (
        _spec("x", "module_one"),
        BackendSpec(
            id="absent.y",
            capability=Capability.GATE,
            display_name="Absent backend y",
            tier=BackendTier.OPTIONAL,
            interface=BackendInterface.PYTHON,
            license_spdx="MIT",
            license_class=LicenseClass.PERMISSIVE,
            module="module_two",
            distribution="absent-y",
            plugin_ref="chemistry.x@0.1.0",
        ),
    )
    assert _index_by_plugin_ref(contested) == {}


def test_every_declared_extra_exists_in_pyproject_and_installs_its_distribution() -> None:
    """The reason ``extra`` is declared on the spec instead of read at run time.

    ``importlib.metadata`` reports the extras of the installed dist-info, which
    in an editable checkout lags pyproject by however long since the last
    reinstall -- during this change it still advertised two extras that had been
    renamed away.  Declaring the pairing and testing it keeps the hint honest
    without depending on when the user last ran ``pip install -e .``.
    """

    project = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]
    extras: dict[str, list[str]] = project["optional-dependencies"]

    for spec in BUILTIN_BACKEND_SPECS:
        if spec.extra is None:
            continue
        assert spec.extra in extras, (
            f"{spec.id} names extra '{spec.extra}', which pyproject does not define"
        )
        assert spec.distribution is not None
        # PEP 503: '-', '_' and '.' are the same character in a distribution
        # name, so 'chembl-structure-pipeline' and 'chembl_structure_pipeline'
        # are one package spelled two ways, and both spellings are in the tree.
        installed = {_normalize(requirement) for requirement in extras[spec.extra]}
        assert _normalize(spec.distribution) in installed, (
            f"extra '{spec.extra}' does not install {spec.distribution}, "
            f"which {spec.id} imports as {spec.module}"
        )


def test_every_selectable_optional_backend_names_the_extra_that_installs_it() -> None:
    """An optional backend a cascade can select but nothing documents is a trap.

    ``plugin_ref`` set means the builder can put it in a cascade; no ``extra``
    then means the preflight can only offer a bare ``pip install <dist>``, which
    drops the version floor the adapter was written against.  Core dependencies
    are exempt because they cannot be missing, and copyleft backends because
    deployment policy blocks them before installability matters.
    """

    core = {"rdkit", "pyarrow", "openpyxl", "numpy", "pydantic", "PyYAML"}
    for spec in BUILTIN_BACKEND_SPECS:
        if spec.plugin_ref is None or spec.interface is not BackendInterface.PYTHON:
            continue
        if spec.distribution in core or spec.license_class is LicenseClass.COPYLEFT:
            continue
        assert spec.extra is not None, (
            f"{spec.id} is selectable in a cascade but names no extra; "
            f"add one to pyproject and declare it on the spec"
        )


def test_every_backend_spec_names_a_plugin_that_is_actually_registered() -> None:
    """A ``plugin_ref`` here pins ``id@version``, so it goes stale silently.

    Bumping a plugin's descriptor version is the sanctioned way to invalidate a
    warm stage cache, and it also renames the registry key.  A spec still naming
    the old version would then advertise a backend the builder cannot resolve --
    and because this catalogue only ever *describes* backends, nothing would fail
    until an operator's cascade did.  The parallel catalogue in
    ``cascade/catalog.py`` is covered by its own test; this is the other half.
    """

    registry = create_builtin_registry()
    for spec in BUILTIN_BACKEND_SPECS:
        if spec.plugin_ref is None:
            continue
        assert spec.plugin_ref in registry, (
            f"backend '{spec.id}' names plugin '{spec.plugin_ref}', which is not "
            f"registered -- a descriptor version was bumped without this catalogue"
        )


def test_an_isolated_backend_is_found_where_it_lives_not_on_path(tmp_path: Path) -> None:
    """PATH is the one place an isolated backend must never be looked up.

    ``aizynthfinder`` is isolated precisely because its RDKit and NumPy pins
    conflict with this environment's.  Its ``bin`` is therefore not on this
    process's ``PATH`` -- and if it were, a lookup would be finding the very
    installation that must not be imported.  What decides the verdict is the
    absolute path the stage carries.
    """

    executable = tmp_path / "aizynthcli"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    stages = (
        _stage(
            "t7_routes",
            "synthesis.aizynthfinder@0.1.0",
            {"executable": str(executable), "config_path": str(tmp_path / "config.yml")},
        ),
    )
    assert preflight_backends(stages) == ("aizynthfinder",)


def test_an_isolated_backend_with_no_configured_path_stops_the_run() -> None:
    """The stage has not been told where the other environment is."""

    stages = (_stage("t7_routes", "synthesis.aizynthfinder@0.1.0"),)
    with pytest.raises(PluginError) as raised:
        preflight_backends(stages)
    error = raised.value
    assert error.code == "BACKEND_PREFLIGHT_FAILED"
    assert "'executable' is unset" in str(error)
    # The instruction differs in kind from every other missing backend: pip
    # installing this one *here* is the thing that would break the run.
    assert error.hint is not None
    assert "environment of its own" in error.hint
    assert "pip install" not in error.hint.split("environment of its own")[0]


def test_an_isolated_backend_will_not_accept_a_bare_command_name(tmp_path: Path) -> None:
    stages = (
        _stage("t7_routes", "synthesis.aizynthfinder@0.1.0", {"executable": "aizynthcli"}),
    )
    with pytest.raises(PluginError) as raised:
        preflight_backends(stages)
    assert "absolute path" in str(raised.value)


def test_every_isolated_backend_with_an_adapter_says_where_its_executable_lives() -> None:
    """Otherwise the preflight would fall back to the PATH probe and be wrong.

    A ``BackendTier.ISOLATED`` spec that is selectable in a cascade but does not
    name its config key would be checked with ``shutil.which``, which reports
    the correctly-isolated case as missing and a wrongly-installed one as fine.
    """

    for spec in BUILTIN_BACKEND_SPECS:
        if spec.tier is not BackendTier.ISOLATED or spec.plugin_ref is None:
            continue
        assert spec.executable_config_key is not None, (
            f"{spec.id} has an adapter and is isolated, so it must declare "
            f"executable_config_key"
        )


_SIMILARITY = "applicability.rdkit_reference_similarity@0.1.0"


def test_a_named_data_file_that_is_not_there_stops_the_run(tmp_path: Path) -> None:
    """An installed package is not the same thing as a runnable stage.

    The reference-similarity backend needs a lead set only the project has, and
    a config validator cannot check for it: validation has to work on a laptop
    that is not the machine the run will happen on, so it can only check the
    shape of the path.  Existence therefore belongs here, where the check is
    running on the host that is about to do the work.
    """

    absent = tmp_path / "leads.csv"
    stages = (_stage("t8_similarity", _SIMILARITY, {"reference_path": str(absent)}),)
    with pytest.raises(PluginError) as raised:
        preflight_backends(stages)

    error = raised.value
    assert error.code == "BACKEND_PREFLIGHT_FAILED"
    detail = str(error)
    assert "t8_similarity" in detail
    assert "nothing readable at the configured 'reference_path'" in detail
    assert str(absent) in detail


def test_a_missing_data_file_is_not_reported_as_a_missing_install(tmp_path: Path) -> None:
    """RDKit is installed; telling anyone to install it is sending them nowhere.

    This is the failure mode the whole distinction exists for.  Before it, a
    lead file nobody had produced the hint ``pip install rdkit``, so the
    operator went and reinstalled a package that was already correct while the
    actual cause -- a path in their own cascade -- went unmentioned.
    """

    stages = (_stage("t8_similarity", _SIMILARITY, {"reference_path": str(tmp_path / "x.csv")}),)
    with pytest.raises(PluginError) as raised:
        preflight_backends(stages)

    hint = raised.value.hint
    assert hint is not None
    assert "that is a path in the cascade, not a missing installation" in hint
    assert "pip install" not in hint


def test_a_data_file_that_is_there_is_not_a_problem(tmp_path: Path) -> None:
    reference = tmp_path / "leads.csv"
    reference.write_text("id,smiles\nlead,CCO\n", encoding="utf-8")
    stages = (_stage("t8_similarity", _SIMILARITY, {"reference_path": str(reference)}),)
    assert "rdkit.reference-similarity" in preflight_backends(stages)


@pytest.mark.parametrize("config", [{}, {"reference_path": ""}, {"reference_path": None}])
def test_a_stage_that_names_no_file_is_left_to_the_plugin(config: dict[str, object]) -> None:
    """Absent is not the same as wrong, and this module cannot tell which.

    The same backend also accepts a small lead set embedded in the cascade, in
    which case there is no path and nothing to check.  Whether a stage may go
    without one is the plugin's judgement -- and it does make it, at config
    time.  Guessing here would fail the one shape that needs nothing from the
    machine it runs on.
    """

    assert "rdkit.reference-similarity" in preflight_backends(
        (_stage("t8_similarity", _SIMILARITY, config),)
    )


def test_a_missing_file_and_a_missing_package_each_keep_their_own_remedy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Two different problems in one cascade, two different sentences.

    Merging them would lose whichever one is not the pip command, and the pip
    command is the one that looks actionable.
    """

    catalogued = dict(_index_by_plugin_ref(BUILTIN_BACKEND_SPECS))
    catalogued["chemistry.a@0.1.0"] = _spec("a", "molcascade_absent_a", extra="alerts")
    monkeypatch.setattr("molcascade.backends.preflight._BY_PLUGIN_REF", catalogued, raising=True)
    stages = (
        _stage("t3_alerts", "chemistry.a@0.1.0"),
        _stage("t8_similarity", _SIMILARITY, {"reference_path": str(tmp_path / "x.csv")}),
    )
    with pytest.raises(PluginError) as raised:
        preflight_backends(stages)

    hint = raised.value.hint
    assert hint is not None
    assert "that is a path in the cascade, not a missing installation" in hint
    assert 'pip install "molcascade[alerts]"' in hint


def test_a_path_the_machine_supplied_says_so_when_it_does_not_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The path in the message may appear nowhere in the file being read.

    An engine path can be answered once per machine by an environment variable
    instead of written into every cascade, which is the point -- but it means an
    operator debugging "nothing exists at the configured 'executable'" could
    search their whole cascade for a path that was never in it.
    """

    missing = tmp_path / "envs" / "aizynth" / "bin" / "aizynthcli"
    monkeypatch.setenv("MOLCASCADE_AIZYNTHFINDER_EXECUTABLE", str(missing))
    stages = (_stage("t7_routes", "synthesis.aizynthfinder@0.1.0", {"executable": str(missing)}),)
    with pytest.raises(PluginError) as raised:
        preflight_backends(stages)
    assert "from MOLCASCADE_AIZYNTHFINDER_EXECUTABLE" in str(raised.value)


def test_a_path_the_cascade_carries_is_not_blamed_on_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configured value wins over the variable, so it must not be attributed to it.

    Sending someone to unset a variable that is not what their run is using is
    worse than saying nothing about where the path came from.
    """

    monkeypatch.setenv("MOLCASCADE_AIZYNTHFINDER_EXECUTABLE", str(tmp_path / "from-the-machine"))
    stages = (
        _stage(
            "t7_routes",
            "synthesis.aizynthfinder@0.1.0",
            {"executable": str(tmp_path / "from-the-cascade")},
        ),
    )
    with pytest.raises(PluginError) as raised:
        preflight_backends(stages)
    message = str(raised.value)
    assert "from-the-cascade" in message
    assert "MOLCASCADE_AIZYNTHFINDER_EXECUTABLE" not in message


def test_the_variable_the_preflight_names_is_the_one_the_engine_reads() -> None:
    """Otherwise the message would send an operator to export something inert.

    The preflight derives the name from ``spec.id``, and the adapter derives it
    from its own ``engine_id``; nothing forces those to agree, so this is where
    they are held together.  Backends with no config model of their own are left
    out -- the message says nothing about a variable it cannot verify.
    """

    from molcascade.plugins.builtin._machine_paths import environment_key

    registry = create_builtin_registry()
    checked = 0
    for spec in BUILTIN_BACKEND_SPECS:
        key = spec.executable_config_key
        if key is None or spec.plugin_ref is None:
            continue
        model = getattr(registry.get(spec.plugin_ref), "config_model", None)
        declared = getattr(model, "installed_paths", None)
        if declared is None:
            continue
        fields = {spec.field for spec in declared()}
        assert environment_key(spec.id, key) == environment_key(model.engine_id, key)
        assert key in fields, f"{spec.id} names '{key}', which its adapter does not honour"
        checked += 1
    assert checked == 4, "the three docking engines and AiZynthFinder"


# ---------------------------------------------- where the engines are installed
#
# ``preflight_backends`` above asks whether a *package* is importable and whether
# a configured path works.  What follows is the question before that one: is
# there a configured path at all, on this machine, for each engine this cascade
# needs?
#
# It is separate because it is the only preflight whose answer is allowed to be
# "no" in a perfectly good cascade.  A ``cascade.json`` is the screening policy
# -- reviewed, committed, handed to a collaborator -- and where KarmaDock lives
# is not part of a policy.  The builder that writes the file runs in a browser
# on a laptop, so it cannot fill those fields in even in principle.  Every one
# of these tests exists because the previous design refused the file instead:
# validation failed on a blank path, which meant a generated cascade did not run
# until the operator hand-edited a machine's paths into a file that was supposed
# to travel.

_KARMADOCK = "docking.karmadock@0.3.0"
_UNIDOCK = "docking.unidock@0.2.0"


def _karmadock_stage(config: dict[str, object]) -> CompiledStage:
    return _stage("t9_docking", _KARMADOCK, config)


def test_a_blank_engine_path_stops_the_run_and_names_the_variable() -> None:
    """Blank is legal in the file and fatal on the machine, which is the point.

    Two halves of one guarantee: the cascade compiles anywhere, and it refuses
    to *start* anywhere it cannot actually run -- before a run directory, a cache
    or a single standardized molecule exists.
    """

    from molcascade.backends.preflight import preflight_engine_paths

    registry = create_builtin_registry()
    stages = (_karmadock_stage({"executable": "", "repo_path": ""}),)
    with pytest.raises(ConfigError) as raised:
        preflight_engine_paths(stages, registry=registry)
    error = raised.value
    assert error.code == "ENGINE_PATH_UNRESOLVED"
    message = str(error)
    # Both gaps, not just the first: setting a machine up one refusal at a time
    # is the install-restart-install loop this check exists to remove.
    assert "MOLCASCADE_KARMADOCK_EXECUTABLE" in message
    assert "MOLCASCADE_KARMADOCK_REPO_PATH" in message
    assert "t9_docking" in message
    # And where it looked, so an operator who has installed it somewhere else can
    # see why it was not found.
    assert "Looked in:" in message
    assert error.hint is not None
    assert "envs/bootstrap.sh karmadock" in error.hint


def test_the_refusal_says_it_is_an_installation_and_not_a_screening_setting() -> None:
    """Otherwise the obvious fix is to type the path into the cascade.

    Which works, and quietly undoes the portability the blank field was for: the
    file now carries one machine's layout and fails on the next one.
    """

    from molcascade.backends.preflight import preflight_engine_paths

    registry = create_builtin_registry()
    stages = (_karmadock_stage({"executable": "", "repo_path": ""}),)
    with pytest.raises(ConfigError) as raised:
        preflight_engine_paths(stages, registry=registry)
    hint = raised.value.hint
    assert hint is not None
    assert "do not belong in the cascade" in hint
    assert "runs here unchanged" in hint


def test_an_engine_installed_where_it_is_expected_is_simply_confirmed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The case that has to be silent: everything found, nothing said.

    Discovery has already filled the compiled config by the time this runs, so
    what is checked here is the resolved value -- which is why the preflight
    reads the compiled stages rather than the cascade as written.
    """

    from molcascade.backends.preflight import preflight_engine_paths
    from molcascade.plugins.builtin.docking.karmadock import KarmaDockConfig

    envs = tmp_path / "envs"
    backends = tmp_path / "backends"
    monkeypatch.setenv("MOLCASCADE_CONDA_ENVS", str(envs))
    monkeypatch.setenv("MOLCASCADE_BACKEND_ROOT", str(backends))
    python = envs / "karmadock" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    python.chmod(0o755)
    checkout = backends / "KarmaDock"
    for relative in ("utils/virtual_screening.py", "trained_models/karmadock_screening.pkl"):
        item = checkout / relative
        item.parent.mkdir(parents=True, exist_ok=True)
        item.write_text("", encoding="utf-8")

    resolved = KarmaDockConfig(
        receptor_path="/campaign/receptor.pdb",
        reference_ligand_path="/campaign/ligand.mol2",
    )
    assert resolved.executable == str(python)
    stages = (
        _karmadock_stage({"executable": resolved.executable, "repo_path": resolved.repo_path}),
    )
    registry = create_builtin_registry()
    assert preflight_engine_paths(stages, registry=registry) == ("t9_docking",)


def test_a_checkout_missing_its_committed_checkpoint_is_not_a_checkout(
    tmp_path: Path,
) -> None:
    """A directory of the right name is not the thing KarmaDock needs.

    Its checkpoint is committed to the repository rather than fetched, so the
    clone *is* the weight download -- and a shallow clone that was interrupted,
    or a directory somebody made by hand, would otherwise pass a bare existence
    check and fail hours later inside the engine.
    """

    from molcascade.backends.preflight import preflight_engine_paths

    checkout = tmp_path / "KarmaDock"
    entry = checkout / "utils" / "virtual_screening.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("", encoding="utf-8")
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    python.chmod(0o755)

    stages = (_karmadock_stage({"executable": str(python), "repo_path": str(checkout)}),)
    with pytest.raises(ConfigError) as raised:
        preflight_engine_paths(stages, registry=create_builtin_registry())
    message = str(raised.value)
    assert "trained_models/karmadock_screening.pkl" in message
    assert "not a complete" in message


def test_a_configured_path_that_is_not_there_is_reported_as_that(tmp_path: Path) -> None:
    """Distinct from "not found on this machine", because the fix is different.

    One is an engine that was never installed; the other is a cascade or a shell
    profile naming the wrong place, and telling that operator where the search
    *would* have looked sends them after the wrong problem.
    """

    from molcascade.backends.preflight import preflight_engine_paths

    absent = tmp_path / "gone" / "python"
    stages = (_karmadock_stage({"executable": str(absent), "repo_path": str(tmp_path)}),)
    with pytest.raises(ConfigError) as raised:
        preflight_engine_paths(stages, registry=create_builtin_registry())
    message = str(raised.value)
    assert f"nothing exists at the configured 'executable': {absent}" in message


def test_a_program_nobody_made_executable_is_named_as_that(tmp_path: Path) -> None:
    """A downloaded binary with no ``chmod +x`` is a two-second fix.

    Reported as "not found" it looks like a failed install, which is not.
    """

    from molcascade.backends.preflight import preflight_engine_paths

    binary = tmp_path / "unidock"
    binary.write_text("", encoding="utf-8")
    binary.chmod(0o644)
    stages = (_stage("t9_docking", _UNIDOCK, {"executable": str(binary)}),)
    with pytest.raises(ConfigError) as raised:
        preflight_engine_paths(stages, registry=create_builtin_registry())
    assert "is not executable by this user" in str(raised.value)


def test_every_gap_across_every_stage_is_reported_at_once() -> None:
    """Two engines in one cascade, two installations, one message.

    An operator provisioning a machine should learn everything it needs in one
    reading rather than discovering the second engine after installing the
    first.
    """

    from molcascade.backends.preflight import preflight_engine_paths

    stages = (
        _stage("t9_karmadock", _KARMADOCK, {"executable": "", "repo_path": ""}),
        _stage("t10_unidock", _UNIDOCK, {"executable": ""}),
    )
    with pytest.raises(ConfigError) as raised:
        preflight_engine_paths(stages, registry=create_builtin_registry())
    error = raised.value
    message = str(error)
    assert "3 installation path(s)" in message
    assert "t9_karmadock" in message
    assert "t10_unidock" in message
    assert error.hint is not None
    assert "envs/bootstrap.sh karmadock" in error.hint
    assert "envs/bootstrap.sh unidock" in error.hint
    gaps = error.context["gaps"]
    assert [gap["field"] for gap in gaps] == ["executable", "repo_path", "executable"]


def test_a_stage_that_needs_nothing_installed_is_left_alone() -> None:
    """Most of a cascade runs in this process, and has no installation to check."""

    from molcascade.backends.preflight import preflight_engine_paths

    stages = (_stage("t1_validity", "chemistry.rdkit_hard_gate@0.1.0"),)
    assert preflight_engine_paths(stages, registry=create_builtin_registry()) == ()


def test_an_unregistered_plugin_is_the_compilers_problem_not_this_ones() -> None:
    """Guessing at an unknown plugin's requirements would refuse a working run.

    And the compiler reports it properly a moment later, with the list of what
    is registered.
    """

    from molcascade.backends.preflight import preflight_engine_paths

    stages = (_stage("custom", "someones.own.plugin@9.9.9"),)
    assert preflight_engine_paths(stages, registry=create_builtin_registry()) == ()


def test_the_gnina_note_travels_with_the_refusal() -> None:
    """Installing GNINA is not enough to run it; the run also needs a licence flag.

    Two separate refusals for one setup step, so the first mentions the second.
    """

    from molcascade.backends.preflight import preflight_engine_paths

    stages = (_stage("t10_rescore", "docking.gnina@0.3.0", {"executable": ""}),)
    with pytest.raises(ConfigError) as raised:
        preflight_engine_paths(stages, registry=create_builtin_registry())
    assert "--allow-copyleft" in str(raised.value)
