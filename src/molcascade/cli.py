"""Argparse command-line interface for local MolCascade workflows.

The two leading-flag forms are retained for the initial user journey::

    molcascade --generate config --output builder.html
    molcascade --vs --config pipeline.yaml --molecules library.smi

Canonical subcommands expose the same implementation without maintaining a
second execution path.  The CLI performs no plugin discovery, installation, or
network access: only the statically reviewed built-ins are used.
"""

from __future__ import annotations

import argparse
import contextvars
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from pydantic import ValidationError

from molcascade import __version__
from molcascade.artifacts import (
    ArtifactDatasetRef,
    ArtifactIntegrityError,
    canonical_json_bytes,
)
from molcascade.assets import (
    AssetSpec,
    PayloadTrust,
    asset_directory,
    asset_spec,
    asset_status,
    default_asset_root,
    iter_assets,
    preflight_assets,
    resolve_reference,
)
from molcascade.assets.fetch import fetch_asset, render_manifest
from molcascade.backends import Availability, ProbePolicy, create_backend_registry
from molcascade.backends.preflight import (
    copyleft_backends,
    preflight_backends,
    preflight_docking_target,
)
from molcascade.cascade import (
    LibraryConfig,
    LibraryFormat,
    LoweredCascade,
    ScreeningConfig,
    TargetConfig,
    TierMode,
    load_screening_config,
    lower_cascade,
)
from molcascade.cascade.citations import render_citations, uncited_options
from molcascade.cascade.funnel import build_funnel
from molcascade.cascade.library import reader_settings
from molcascade.cascade.target import docking_requirements, merge_target, resolve_target
from molcascade.config import PipelineConfig
from molcascade.environment import detect_environment
from molcascade.errors import (
    AssetError,
    ConfigError,
    ExecutionError,
    MolCascadeError,
    PipelineError,
    PluginError,
)
from molcascade.handoff import materialize_shortlist
from molcascade.io.atomic import DestinationExistsError, atomic_write_bytes
from molcascade.parallel import StageResources, count_completed_shards, describe_lanes, plan_lanes
from molcascade.pipeline import CompiledStage, PipelineCompiler
from molcascade.plugins import PluginKind, PluginRegistry, create_builtin_registry
from molcascade.plugins.builtin.chemprop_model import (
    MANIFEST_FILENAME as CHEMPROP_MANIFEST_FILENAME,
)
from molcascade.plugins.builtin.chemprop_model import inspect_chemprop_bundle
from molcascade.plugins.builtin.custom_model import inspect_model_bundle
from molcascade.runtime import (
    AuditEvent,
    LocalRunner,
    RunState,
    RunStatus,
    StageRunState,
    StageRunStatus,
    stage_cache_key,
)
from molcascade.runtime.models import validate_run_id
from molcascade.ui import generate_cascade_builder, generate_config_builder
from molcascade.ui.report import generate_run_report

_DEFAULT_WORKSPACE = Path(".molcascade")
_DEFAULT_BUILDER_OUTPUT = Path("molcascade-config.html")
_RUN_STATE_SIZE_LIMIT = 16 * 1024 * 1024
_AUDIT_SIZE_LIMIT = 128 * 1024 * 1024
_MOLECULE_FORMATS = (
    "auto",
    "delimited",
    "xlsx",
    "sdf",
    "parquet",
    "mol2-directory",
)
_SOURCE_PLUGIN_BY_FORMAT = {
    "delimited": "source.delimited_smiles@0.1.0",
    "xlsx": "source.xlsx@0.1.0",
    "sdf": "source.sdf@0.1.0",
    "parquet": "source.raw_molecule_parquet@0.1.0",
    "mol2-directory": "source.mol2_directory@0.1.0",
}
_JSON_ERROR_MODE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "molcascade_cli_json_error_mode",
    default=False,
)


class _ArgumentParser(argparse.ArgumentParser):
    """Emit one structured error object when the invocation requests JSON."""

    def error(self, message: str) -> None:
        if _JSON_ERROR_MODE.get():
            error = PipelineError(message, code="CLI_USAGE_ERROR")
            _json_dump({"ok": False, "error": error.as_dict()}, sys.stderr)
            self.exit(2)
        super().error(message)


def _json_dump(value: Any, stream: TextIO) -> None:
    stream.write(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    stream.write("\n")


def _render_error(error: MolCascadeError, *, as_json: bool) -> None:
    if as_json:
        _json_dump({"ok": False, "error": error.as_dict()}, sys.stderr)
        return
    print(f"molcascade: error [{error.code}]: {error.message}", file=sys.stderr)
    if error.hint:
        print(f"hint: {error.hint}", file=sys.stderr)


def _parallel_parser_options(parser: argparse.ArgumentParser) -> None:
    """Flags that decide how much of this machine a run may use.

    Deliberately not configuration.  A cascade is a screening *policy* and is
    meant to be shared; a worker count is a fact about one box.  Putting either
    of these in the config file would also change ``stage_cache_key``, so the
    same science would miss the cache on every machine it moved to.
    """

    parser.add_argument(
        "--workers",
        type=int,
        help=(
            "Parallel shards to run at once; default is one per detected GPU, "
            "or usable cores minus one on CPU"
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "auto (default), cpu, cuda, or cuda:0,2. 'auto' falls back to CPU and "
            "records why; an explicit request that cannot be honoured stops the run"
        ),
    )


def _license_parser_options(parser: argparse.ArgumentParser) -> None:
    """The one flag that changes which licences a run is allowed to link.

    Also deliberately not configuration, and for a stronger reason than the
    flags above.  A cascade travels between people and organisations; whether
    linking GPL code into a screening pipeline is acceptable is a property of
    the organisation running it, not of the science, and a config file that
    carried the answer would be answering on someone else's behalf.
    """

    parser.add_argument(
        "--allow-copyleft",
        action="store_true",
        help=(
            "Permit copyleft-licensed backends such as GNINA (GPL-2.0). Off by "
            "default; the run records that it was given"
        ),
    )


def _target_parser_options(parser: argparse.ArgumentParser) -> None:
    """Which protein this run docks into, supplied at the run rather than in it.

    A cascade is a screening policy and is meant to be reused against many
    targets; the receptor is the one thing about a docking run that changes
    every time.  A cascade may still carry a ``target`` block -- these flags
    override it -- but the shipped cascades do not, which is why a docking run
    started without them stops before it reads a molecule and says so.
    """

    parser.add_argument(
        "--receptor",
        type=Path,
        help="Protein structure to dock into, in PDB. Read and hashed, never edited",
    )
    parser.add_argument(
        "--reference-ligand",
        type=Path,
        help=(
            "A ligand bound in the site (mol2/sdf/mol). Its extent plus padding "
            "becomes the search box"
        ),
    )
    parser.add_argument(
        "--pocket",
        type=Path,
        help="A PDB holding the residues that line the site, as an alternative to a ligand",
    )
    parser.add_argument(
        "--box",
        help="The site as six numbers directly: cx,cy,cz,sx,sy,sz in angstrom",
    )
    parser.add_argument(
        "--receptor-pdbqt",
        type=Path,
        help=(
            "A receptor already prepared as PDBQT for Uni-Dock, for when the "
            "automatic meeko preparation cannot cope with the structure"
        ),
    )


@dataclass(frozen=True, slots=True)
class _TargetArguments:
    """The five target flags, carried as one value.

    They are only ever used together -- a receptor without a site is an error
    and a site without a receptor is a different one -- so threading them
    through the entry points as five parameters would be five chances to pass
    four.  The empty instance is what the forms that cannot spell them pass.
    """

    receptor: Path | None = None
    reference_ligand: Path | None = None
    pocket: Path | None = None
    box: str | None = None
    receptor_pdbqt: Path | None = None

    @classmethod
    def from_namespace(cls, arguments: argparse.Namespace) -> _TargetArguments:
        return cls(
            receptor=arguments.receptor,
            reference_ligand=arguments.reference_ligand,
            pocket=arguments.pocket,
            box=arguments.box,
            receptor_pdbqt=arguments.receptor_pdbqt,
        )

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "receptor": self.receptor,
            "reference_ligand": self.reference_ligand,
            "pocket": self.pocket,
            "box": self.box,
            "receptor_pdbqt": self.receptor_pdbqt,
        }


def _probe_policy(*, allow_copyleft: bool) -> ProbePolicy:
    """The licence policy a preflight applies, and nothing else.

    ``run_version_commands`` stays off whatever the operator asked for: probing
    a version means starting a subprocess, and a preflight on the way into a run
    has no business doing that.  ``doctor`` is the one command where running
    them is the point, and it builds its own policy.
    """

    return ProbePolicy(allow_copyleft=allow_copyleft, run_version_commands=False)


def _execution_parser_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--molecules",
        type=Path,
        help="Override path in the first enabled source stage",
    )
    parser.add_argument(
        "--molecule-format",
        "--format",
        choices=_MOLECULE_FORMATS,
        default="auto",
        help="Input format; explicit values override automatic path detection",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=_DEFAULT_WORKSPACE,
        help="Local artifacts, cache, run state, and audit directory",
    )
    parser.add_argument("--run-id", help="Portable run identifier used for resume/status")
    parser.add_argument("--resume", action="store_true", help="Resume a verified run")
    parser.add_argument("--force", action="store_true", help="Bypass reusable stage cache")
    _parallel_parser_options(parser)
    _license_parser_options(parser)
    _target_parser_options(parser)
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Write machine-readable JSON",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the complete parser without importing or discovering external plugins."""

    parser = _ArgumentParser(
        prog="molcascade",
        description="Auditable, modular hierarchical molecular screening",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "compatibility forms:\n"
            "  molcascade --generate config [--output PATH]\n"
            "  molcascade --vs --config PATH --molecules PATH [run options]"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    # Compatibility surface.  Distinct destinations keep it impossible for a
    # canonical subcommand to accidentally consume a legacy option.
    parser.add_argument(
        "--generate",
        choices=("config",),
        dest="compat_generate",
        metavar="config",
        help="Generate the offline configuration builder",
    )
    parser.add_argument("--vs", action="store_true", help="Run a vertical-screening pipeline")
    parser.add_argument("--config", "-config", type=Path, dest="compat_config")
    parser.add_argument(
        "--molecules",
        "-molecules",
        type=Path,
        dest="compat_molecules",
    )
    parser.add_argument(
        "--molecule-format",
        "--format",
        choices=_MOLECULE_FORMATS,
        dest="compat_molecule_format",
    )
    parser.add_argument("--output", type=Path, dest="compat_output")
    parser.add_argument("--workspace", type=Path, dest="compat_workspace")
    parser.add_argument("--run-id", dest="compat_run_id")
    parser.add_argument("--resume", action="store_true", dest="compat_resume")
    parser.add_argument("--force", action="store_true", dest="compat_force")
    parser.add_argument("--json", action="store_true", dest="compat_json")

    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    generate = commands.add_parser("generate", help="Generate offline user interfaces")
    generate_targets = generate.add_subparsers(
        dest="generate_target",
        metavar="TARGET",
        required=True,
    )
    generate_config = generate_targets.add_parser(
        "config",
        help="Generate a self-contained screening cascade builder",
    )
    generate_config.add_argument(
        "--output",
        "-o",
        type=Path,
        default=_DEFAULT_BUILDER_OUTPUT,
    )
    generate_config.add_argument(
        "--pipeline",
        action="store_true",
        help=(
            "Generate the flat schema-1 pipeline graph editor instead of the "
            "tier-first cascade builder"
        ),
    )

    validate = commands.add_parser(
        "validate",
        help="Strictly parse and compile a JSON/YAML pipeline",
    )
    validate.add_argument("config", type=Path)
    validate.add_argument("--json", action="store_true", dest="json_output")

    plugins = commands.add_parser("plugins", help="List reviewed built-in stage plugins")
    plugins.add_argument("--json", action="store_true", dest="json_output")

    cite = commands.add_parser(
        "cite",
        help="Print the reference for every tool the builder can put in a screen",
    )
    cite.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Write the bibliography to a file instead of standard output",
    )

    doctor = commands.add_parser(
        "doctor",
        help="Probe local backend availability without installing anything",
    )
    doctor.add_argument("--json", action="store_true", dest="json_output")
    doctor.add_argument(
        "--run-version-commands",
        action="store_true",
        help="Opt in to local executable version probes",
    )
    _license_parser_options(doctor)

    assets = commands.add_parser(
        "assets",
        help="Inspect and provision vendored model weights and rule tables",
    )
    asset_actions = assets.add_subparsers(dest="asset_action", metavar="ACTION")

    assets_list = asset_actions.add_parser(
        "list", help="Show every declared asset, its licence and its citation"
    )
    assets_list.add_argument("--json", action="store_true", dest="json_output")

    assets_status = asset_actions.add_parser(
        "status", help="Check what is on disk and whether every digest matches"
    )
    assets_status.add_argument("asset_id", nargs="?", help="Limit to one asset")
    assets_status.add_argument(
        "--deep",
        action="store_true",
        help="Re-hash every file instead of trusting the size/mtime cache",
    )
    assets_status.add_argument("--json", action="store_true", dest="json_output")

    assets_fetch = asset_actions.add_parser(
        "fetch",
        help="Download declared assets over the network (never happens during a run)",
    )
    assets_fetch.add_argument("asset_id", nargs="?", help="Asset to download")
    assets_fetch.add_argument("--all", action="store_true", help="Download every asset")
    assets_fetch.add_argument(
        "--force", action="store_true", help="Re-download files that already verify"
    )
    assets_fetch.add_argument("--json", action="store_true", dest="json_output")

    assets_where = asset_actions.add_parser(
        "where", help="Print the asset root, or the verified path of one file"
    )
    assets_where.add_argument(
        "reference",
        nargs="?",
        help="An 'asset:<asset-id>/<file>' reference to resolve and verify",
    )
    assets_where.add_argument("--json", action="store_true", dest="json_output")

    env = commands.add_parser(
        "env",
        help="Report the CPU, memory, disk and GPU this machine actually has",
    )
    env.add_argument(
        "--workspace",
        type=Path,
        default=_DEFAULT_WORKSPACE,
        help="Also report free space on the filesystem holding this workspace",
    )
    env.add_argument(
        "--no-gpu-probe",
        action="store_true",
        help="Skip nvidia-smi and rocm-smi",
    )
    env.add_argument("--json", action="store_true", dest="json_output")

    model_bundle = commands.add_parser(
        "model-bundle",
        help="Validate your own model bundle and print the digest to pin in a config",
    )
    model_bundle.add_argument(
        "bundle",
        type=Path,
        help="Directory holding molcascade_model.yaml or molcascade_chemprop.yaml",
    )
    model_bundle.add_argument("--json", action="store_true", dest="json_output")

    pins = commands.add_parser(
        "pins",
        help="Recompute the code and weight digests a pip-installed backend is pinned to",
    )
    pins.add_argument(
        "backend",
        choices=("admet-ai",),
        help="Which pip-installed backend to inspect",
    )
    pins.add_argument(
        "--models-dir",
        default=None,
        help=(
            "Override the model tree. Defaults to the copy shipped inside the "
            "installed wheel."
        ),
    )
    pins.add_argument("--json", action="store_true", dest="json_output")

    screen = commands.add_parser(
        "screen",
        help="Screen a molecule library with a configuration from the builder",
    )
    screen.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Cascade or pipeline configuration produced by the builder",
    )
    screen.add_argument(
        "--library",
        "-l",
        type=Path,
        required=True,
        help=(
            "Molecule library to screen (.csv, .tsv, .smi, .xlsx, .sdf, .parquet, "
            "or a MOL2 directory)"
        ),
    )
    screen.add_argument(
        "--format",
        dest="molecule_format",
        choices=_MOLECULE_FORMATS,
        default="auto",
        help="Input format; explicit values override automatic path detection",
    )
    # Reader options.  The configuration from the builder is a reusable
    # *screening policy*; the library changes every run, and so does its layout.
    # Without these the only way to read a spreadsheet whose column is called
    # anything but 'smiles' was to hand-edit the JSON.
    screen.add_argument(
        "--smiles-column",
        help="Name of the column holding SMILES (default: smiles; matched ignoring case)",
    )
    screen.add_argument(
        "--id-column",
        help="Column or SD property to carry through as the molecule identifier",
    )
    screen.add_argument("--sheet", help="Worksheet to read; default is the first visible sheet")
    screen.add_argument(
        "--skip-rows",
        type=int,
        help="Discard this many leading rows before the header, for files that open with a banner",
    )
    screen.add_argument(
        "--workspace",
        type=Path,
        default=_DEFAULT_WORKSPACE,
        help="Local artifacts, cache, run state, and audit directory",
    )
    screen.add_argument("--run-id", help="Portable run identifier used for resume/status")
    screen.add_argument("--resume", action="store_true", help="Resume a verified run")
    screen.add_argument("--force", action="store_true", help="Bypass reusable stage cache")
    _parallel_parser_options(screen)
    _license_parser_options(screen)
    _target_parser_options(screen)
    screen.add_argument(
        "--dry-run",
        action="store_true",
        help="Compile and print the funnel without executing it",
    )
    screen.add_argument("--json", action="store_true", dest="json_output")

    run = commands.add_parser("run", help="Execute a pipeline in a local workspace")
    run.add_argument("config", type=Path)
    _execution_parser_options(run)

    status = commands.add_parser("status", help="Inspect durable run state and audit events")
    status.add_argument("run_id")
    status.add_argument("--workspace", type=Path, default=_DEFAULT_WORKSPACE)
    status.add_argument("--events", action="store_true", help="Include the audit trail")
    status.add_argument("--json", action="store_true", dest="json_output")

    export = commands.add_parser(
        "export",
        help="Materialize a verified shortlist for downstream local software",
    )
    export.add_argument("run_id")
    export.add_argument("--workspace", type=Path, default=_DEFAULT_WORKSPACE)
    export.add_argument("--output", type=Path)
    export.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing shortlist, and a docking sidecar this tool "
            "wrote beside it. Never removes anything else."
        ),
    )
    export.add_argument("--json", action="store_true", dest="json_output")

    report = commands.add_parser("report", help="Generate a self-contained run report")
    report.add_argument("run_id")
    report.add_argument("--workspace", type=Path, default=_DEFAULT_WORKSPACE)
    report.add_argument("--output", type=Path)
    report.add_argument("--overwrite", action="store_true")
    report.add_argument("--json", action="store_true", dest="json_output")
    return parser


def _molecule_override(
    config: PipelineConfig,
    molecules: Path,
    registry: PluginRegistry,
    *,
    molecule_format: str = "auto",
    reader: Mapping[str, Any] | None = None,
) -> PipelineConfig:
    """Route and replace only the first enabled source stage.

    A source-specific configuration is retained only when its registered plugin
    already matches the selected input format.  Switching source types rebuilds
    the configuration from the target plugin model defaults, preventing fields
    from the old source from leaking into a strict replacement model.

    ``reader`` carries the layout options typed on the command line, named as
    :class:`LibraryConfig` fields.  Each is translated to whatever the chosen
    source plugin calls it and dropped if that plugin has no such field, so
    ``--sheet`` on a CSV is ignored rather than rejected by a strict model.
    """

    selected_format = _resolve_molecule_format(molecules, molecule_format)
    target_plugin = _SOURCE_PLUGIN_BY_FORMAT[selected_format]
    target_entry = registry.entry(target_plugin)

    payload = config.model_dump(mode="json")
    stages = payload["stages"]
    if not isinstance(stages, list):  # Defensive boundary around model serialization.
        raise PipelineError(
            "pipeline stages did not serialize as an array",
            code="CLI_CONFIG_SERIALIZATION_INVALID",
        )
    for stage in stages:
        if not isinstance(stage, dict) or stage.get("enabled") is not True:
            continue
        plugin_ref = stage.get("plugin")
        if not isinstance(plugin_ref, str):
            continue
        entry = registry.entry(plugin_ref)
        if entry.descriptor.kind is not PluginKind.SOURCE:
            continue
        stage_config = stage.get("config")
        if not isinstance(stage_config, dict):
            raise PipelineError(
                "source stage config did not serialize as an object",
                code="CLI_CONFIG_SERIALIZATION_INVALID",
                context={"stage_id": stage.get("id")},
            )
        source_path = str(molecules.expanduser())
        if entry.key == target_entry.key:
            stage_config["path"] = source_path
        else:
            config_model = getattr(target_entry.plugin, "config_model", None)
            if not isinstance(config_model, type) or not hasattr(
                config_model,
                "model_validate",
            ):
                raise PipelineError(
                    "target source plugin does not expose a configuration model",
                    code="CLI_SOURCE_CONFIG_MODEL_MISSING",
                    context={"plugin": target_entry.key},
                )
            target_config = config_model.model_validate({"path": source_path})
            stage["plugin"] = target_entry.key
            stage["config"] = target_config.model_dump(mode="json")
        if reader:
            stage["config"] = _apply_reader_options(
                stage["config"],
                plugin_key=target_entry.key,
                config_model=getattr(target_entry.plugin, "config_model", None),
                reader=reader,
            )
        return PipelineConfig.model_validate(payload)
    raise PipelineError(
        "cannot apply --molecules because the pipeline has no enabled source stage",
        code="CLI_SOURCE_STAGE_NOT_FOUND",
        hint="Enable a registered source plugin as the first pipeline stage.",
    )


def _resolve_molecule_format(molecules: Path, requested: str) -> str:
    if requested not in _MOLECULE_FORMATS:
        raise PipelineError(
            f"unsupported molecule format: {requested}",
            code="CLI_MOLECULE_FORMAT_INVALID",
            context={"format": requested},
        )
    path = molecules.expanduser()
    if requested == "mol2-directory" and (path.is_symlink() or not path.is_dir()):
        raise PipelineError(
            "MOL2 ingestion requires a real directory, not an individual file",
            code="CLI_MOL2_DIRECTORY_REQUIRED",
            hint="Place one or more .mol2 files in a directory and pass that directory.",
            context={"path": str(path)},
        )
    if requested != "auto":
        return requested

    if path.is_dir():
        if path.is_symlink():
            raise PipelineError(
                "MOL2 ingestion requires a real non-symlink directory",
                code="CLI_MOL2_DIRECTORY_REQUIRED",
                context={"path": str(path)},
            )
        return "mol2-directory"
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        return "xlsx"
    if suffix in {".sdf", ".sd"}:
        return "sdf"
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".mol2":
        raise PipelineError(
            "a single MOL2 file is not a supported source; MOL2 ingestion requires a directory",
            code="CLI_MOL2_DIRECTORY_REQUIRED",
            hint=(
                "Place one or more .mol2 files in a directory and pass that directory, "
                "or explicitly select another supported input format."
            ),
            context={"path": str(path)},
        )
    return "delimited"


def _generate_config(output: Path, *, pipeline: bool = False) -> int:
    """Write the offline builder a first-time user opens in their browser.

    The cascade builder is the default surface.  ``--pipeline`` still reaches
    the flat graph editor, which is not an older skin of the same thing: it
    authors a schema-1 pipeline directly, which the cascade builder cannot do.
    """

    if pipeline:
        destination = generate_config_builder(output)
        print(f"Generated pipeline graph editor: {destination}")
        return 0
    destination = generate_cascade_builder(output)
    print(f"Generated screening cascade builder: {destination}")
    print("Open it in a browser, design the cascade, then download the config file.")
    print("Screen a library with: molcascade screen --config cascade.json --library LIBRARY")
    return 0


def _validate(config_path: Path, *, as_json: bool) -> int:
    screening = load_screening_config(config_path)
    registry = create_builtin_registry()
    placeholder_library = False
    if screening.cascade is not None:
        lowered = lower_cascade(
            screening.cascade,
            registry=registry,
            allow_missing_library=True,
        )
        config = lowered.pipeline
        placeholder_library = lowered.library_is_placeholder
    else:
        assert screening.pipeline is not None  # ScreeningConfig always carries one kind
        config = screening.pipeline
    compiled = PipelineCompiler(registry).compile(config)
    # Reported, never fatal.  A configuration is valid or not on its own terms;
    # whether *this* machine holds the weights is a property of the machine, and
    # authoring a cascade on a laptop to run on the GPU box has to keep working.
    # Saying which assets it will need is still worth a line, because the answer
    # is otherwise found only by starting the run.
    try:
        provisioned = preflight_assets(compiled.stages)
        asset_note: str | None = None
    except AssetError as error:
        provisioned = ()
        asset_note = error.hint or str(error)
    # Same rule for the packages: naming the pip command here is the whole point
    # of validating on the machine that will not run the cascade.
    try:
        installed = preflight_backends(compiled.stages)
        backend_note: str | None = None
    except PluginError as error:
        installed = ()
        backend_note = error.hint or str(error)
    payload = {
        "valid": True,
        "kind": screening.kind,
        "name": config.name,
        "library_required": placeholder_library,
        "assets_ready": list(provisioned),
        "assets_note": asset_note,
        "backends_ready": list(installed),
        "backends_note": backend_note,
        "revision_id": compiled.revision_id,
        "stages": [
            {
                "id": stage.stage_id,
                "slot": stage.slot,
                "plugin": stage.plugin_key,
                "inputs": [
                    {
                        "request_port": binding.request_port,
                        "source_stage": binding.source_stage_id,
                        "source_port": binding.source_port,
                        "contract_id": binding.contract_id,
                    }
                    for binding in stage.input_bindings
                ],
                "primary_output_contract": stage.output_contract,
            }
            for stage in compiled.stages
        ],
    }
    if as_json:
        _json_dump(payload, sys.stdout)
    else:
        print(f"Valid {screening.kind}: {config.name}")
        print(f"Revision: {compiled.revision_id}")
        print(f"Enabled stages: {len(compiled.stages)}")
        if placeholder_library:
            print("This cascade names no library; pass one with: molcascade screen --library PATH")
        if provisioned:
            print(f"Vendored assets ready: {', '.join(provisioned)}")
        if asset_note is not None:
            print(f"Not runnable here yet: {asset_note}")
        if installed:
            print(f"Backends ready: {', '.join(installed)}")
        if backend_note is not None:
            print(f"Not runnable here yet: {backend_note}")
    return 0


def _plugin_rows(registry: PluginRegistry) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in registry.entries():
        descriptor = entry.descriptor
        rows.append(
            {
                "key": entry.key,
                "kind": descriptor.kind.value,
                "display_name": descriptor.display_name or descriptor.id,
                "trusted": entry.trusted,
                "origin": entry.origin,
                "distribution": entry.distribution,
                "inputs": list(descriptor.inputs),
                "outputs": dict(descriptor.output_ports),
                "determinism": descriptor.determinism.value,
            }
        )
    return rows


def _plugins(*, as_json: bool) -> int:
    rows = _plugin_rows(create_builtin_registry())
    if as_json:
        _json_dump({"plugins": rows}, sys.stdout)
    else:
        print(f"Built-in plugins ({len(rows)}):")
        for row in rows:
            print(f"  {row['key']:<48} {row['kind']:<14} {row['display_name']}")
    return 0


def _cite(output: Path | None) -> int:
    """Emit the bibliography for the whole catalogue.

    Writing up a screen means naming the tools that made the cuts, and those
    references currently live in Python literals.  This is the command that
    gets them out, and the same call regenerates ``docs/citations.md`` so the
    checked-in copy is never hand-maintained.
    """

    document = render_citations()
    missing = uncited_options()
    if output is None:
        sys.stdout.write(document)
    else:
        destination = atomic_write_bytes(output, document.encode("utf-8"))
        print(f"Wrote {destination}")
    if missing:
        # Not fatal.  An option can be legitimately uncited -- a rule table with
        # no paper behind it -- but it should never pass unremarked, because a
        # blank reference and an unused tool look identical in a methods section.
        print(f"{len(missing)} option(s) carry no reference:", file=sys.stderr)
        for criterion_id, option_id in missing:
            print(f"  {criterion_id}.{option_id}", file=sys.stderr)
    return 0


def _doctor(*, as_json: bool, run_version_commands: bool, allow_copyleft: bool) -> int:
    """Answer one question: is this machine ready to screen, and if not, why.

    Three things have to be true, and a backend can fail on any of them
    independently.  The code must be installed, the weights it reads must be
    present and match their digests, and the hardware it needs must exist.
    Reporting only the first is how a run gets to the fourth tier before
    discovering that a model file was never downloaded.
    """

    # The one command where running a version probe is the point rather than an
    # unwanted side effect, so this policy is built here rather than shared with
    # the preflights.
    policy = ProbePolicy(
        allow_copyleft=allow_copyleft,
        run_version_commands=run_version_commands,
    )
    statuses = create_backend_registry().probe_all(policy)
    rows = [status.model_dump(mode="json") for status in statuses]
    counts = Counter(status.availability.value for status in statuses)
    summary = {
        availability.value: counts.get(availability.value, 0)
        for availability in Availability
    }
    asset_specs = list(iter_assets())
    asset_states = [asset_status(spec) for spec in asset_specs]
    environment = detect_environment()
    # The same function a run uses, so 'doctor' cannot report lanes that a run
    # would then decline to use.
    lanes = plan_lanes(environment)
    if as_json:
        _json_dump(
            {
                "backends": rows,
                "summary": summary,
                "assets": [status.model_dump(mode="json") for status in asset_states],
                "asset_root": str(default_asset_root()),
                "environment": environment.model_dump(mode="json"),
                "lanes": {
                    "devices": list(lanes.devices),
                    "workers": lanes.workers,
                    "notes": list(lanes.notes),
                },
            },
            sys.stdout,
        )
        return 0
    print("Local backend health (no installation or download performed):")
    for status in statuses:
        version = f" {status.version}" if status.version else ""
        print(
            f"  {status.availability.value:<11} {status.spec.id:<30}"
            f"{version} — {status.reason}"
        )
    print("Summary: " + ", ".join(f"{key}={value}" for key, value in summary.items()))
    print()
    print(f"Vendored assets in {default_asset_root()}:")
    for spec, status in zip(asset_specs, asset_states, strict=True):
        megabytes = spec.total_bytes / (1024 * 1024)
        print(
            f"  {status.state.value:<11} {spec.id:<30} "
            f"{megabytes:>8,.1f} MiB — {spec.display_name}"
        )
    unready = [status.asset_id for status in asset_states if not status.ready]
    if unready:
        print("  Provision with: molcascade assets fetch " + " ".join(unready))
    print()
    plan = environment.plan
    print("Hardware:")
    print(
        f"  cpu {environment.cpu.usable_cores or '?'} usable core(s) · "
        f"memory {_format_bytes(environment.memory.effective_total_bytes)}"
    )
    # Every card, one line each.  A summary that says "accelerator: NVIDIA A100"
    # on a four-card node is the report a user reads just before wondering why
    # only one of them ever gets warm.
    if environment.gpus:
        for gpu in environment.gpus:
            memory = (
                f"{gpu.memory_total_mib / 1024:,.1f} GiB"
                if gpu.memory_total_mib is not None
                else "memory unknown"
            )
            capability = f" · sm_{gpu.compute_capability}" if gpu.compute_capability else ""
            print(f"  gpu {gpu.index}: {gpu.name} · {memory}{capability}")
    else:
        print("  gpu: none detected")
    print(f"  lanes: {describe_lanes(lanes.devices)}")
    for note in lanes.notes:
        print(f"    note: {note}")
    print(
        f"  plan: device={plan.device}, workers={plan.recommended_workers}, "
        f"batch={plan.recommended_batch_size:,}"
    )
    return 0


def _assets_list(*, as_json: bool) -> int:
    specs = list(iter_assets())
    if as_json:
        _json_dump({"assets": [spec.model_dump(mode="json") for spec in specs]}, sys.stdout)
        return 0
    print(f"Declared assets ({len(specs)}), asset root: {default_asset_root()}")
    for spec in specs:
        megabytes = spec.total_bytes / (1024 * 1024)
        print()
        print(f"  {spec.id}  —  {spec.display_name}")
        print(f"    kind    : {spec.kind.value} · {spec.license_spdx} · {megabytes:,.1f} MiB")
        print(f"    version : {spec.version}")
        print(f"    home    : {spec.homepage}")
        if spec.trust is PayloadTrust.EXECUTABLE:
            print("    trust   : EXECUTABLE payload — loading it runs upstream code")
        for citation in spec.citations:
            print(f"    cite    : {citation.reference}")
        for plugin in spec.used_by:
            print(f"    used by : {plugin}")
    return 0


def _asset_specs_for(asset_id: str | None) -> list[AssetSpec]:
    if asset_id is None:
        return list(iter_assets())
    try:
        return [asset_spec(asset_id)]
    except KeyError as error:
        known = ", ".join(spec.id for spec in iter_assets())
        raise AssetError(
            f"unknown asset {asset_id!r}",
            code="ASSET_UNKNOWN",
            hint=f"known assets: {known}",
        ) from error


def _assets_status(asset_id: str | None, *, deep: bool, as_json: bool) -> int:
    specs = _asset_specs_for(asset_id)
    statuses = [asset_status(spec, deep=deep) for spec in specs]
    if as_json:
        _json_dump(
            {
                "asset_root": str(default_asset_root()),
                "assets": [status.model_dump(mode="json") for status in statuses],
            },
            sys.stdout,
        )
        return 0 if all(status.ready for status in statuses) else 1
    print(f"Asset root: {default_asset_root()}")
    for spec, status in zip(specs, statuses, strict=True):
        print(f"  {status.state.value:<11} {spec.id:<24} {spec.display_name}")
        for entry in status.files:
            if entry.verified:
                continue
            print(f"      {entry.name}: {entry.detail}")
    missing = [status.asset_id for status in statuses if not status.ready]
    if missing:
        print()
        print("Not ready: " + ", ".join(missing))
        print("Provision them with: molcascade assets fetch --all")
        return 1
    return 0


def _assets_fetch(asset_id: str | None, *, fetch_all: bool, force: bool, as_json: bool) -> int:
    if not fetch_all and asset_id is None:
        raise AssetError(
            "name an asset to fetch, or pass --all",
            code="ASSET_SELECTION_REQUIRED",
            hint="see 'molcascade assets list'",
        )
    specs = list(iter_assets()) if fetch_all else _asset_specs_for(asset_id)
    root = default_asset_root()
    results: list[dict[str, Any]] = []
    if not as_json:
        total = sum(spec.total_bytes for spec in specs) / (1024 * 1024)
        print(f"Downloading into {root} ({total:,.1f} MiB declared across {len(specs)} asset(s))")
    for spec in specs:
        if not as_json:
            print(f"  {spec.id}: {spec.display_name}")
        installed = fetch_asset(spec, root=root, force=force)
        results.append({"asset_id": spec.id, "installed": list(installed)})
        if not as_json:
            if installed:
                for name in installed:
                    print(f"    fetched  {name}")
            else:
                print("    already present and verified")
            print(f"    citation {asset_directory(spec, root=root) / 'CITATION.md'}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "MANIFEST.md").write_text(render_manifest(iter_assets()), encoding="utf-8")
    if as_json:
        _json_dump({"asset_root": str(root), "results": results}, sys.stdout)
    else:
        print(f"Wrote {root / 'MANIFEST.md'}")
    return 0


def _assets_where(reference: str | None, *, as_json: bool) -> int:
    if reference is None:
        root = default_asset_root()
        if as_json:
            _json_dump({"asset_root": str(root)}, sys.stdout)
        else:
            print(root)
        return 0
    path = resolve_reference(reference)
    if as_json:
        _json_dump({"reference": reference, "path": str(path)}, sys.stdout)
    else:
        print(path)
    return 0


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    gib = value / (1024 * 1024 * 1024)
    return f"{gib:,.1f} GiB" if gib >= 1.0 else f"{value / (1024 * 1024):,.0f} MiB"


def _environment(*, workspace: Path, probe_gpus: bool, as_json: bool) -> int:
    """Measure this machine and say what it is fit to run.

    Never fails.  A host that answers nothing still gets a complete report with
    ``unknown`` in place of every number it declined to supply.
    """

    environment = detect_environment(workspace=workspace, include_gpus=probe_gpus)
    if as_json:
        _json_dump(environment.model_dump(mode="json"), sys.stdout)
        return 0

    cpu, memory, plan = environment.cpu, environment.memory, environment.plan
    print("Local execution environment:")
    print(
        f"  platform   : {environment.platform.system} {environment.platform.release or ''}"
        f" ({environment.platform.machine or 'unknown'})"
        f"{' · WSL' if environment.platform.wsl else ''}"
        f"{' · container' if environment.platform.in_container else ''}"
    )
    print(
        f"  python     : {environment.platform.python_implementation} "
        f"{environment.platform.python_version}"
    )
    print(f"  cpu        : {cpu.model_name or 'unknown model'}")
    cores = (
        f"{cpu.usable_cores or '?'} usable / {cpu.logical_cores or '?'} logical"
        f" / {cpu.physical_cores or '?'} physical"
    )
    quota = f" · cgroup quota {cpu.cgroup_quota_cores}" if cpu.cgroup_quota_cores else ""
    print(f"  cores      : {cores}{quota}")
    if cpu.features:
        print(f"  features   : {', '.join(cpu.features)}")
    limit = (
        f" · cgroup limit {_format_bytes(memory.cgroup_limit_bytes)}"
        if memory.cgroup_limit_bytes
        else ""
    )
    print(
        f"  memory     : {_format_bytes(memory.total_bytes)} total, "
        f"{_format_bytes(memory.available_bytes)} available{limit}"
    )
    for disk in environment.disks:
        print(
            f"  disk       : {disk.label:<9} {_format_bytes(disk.free_bytes)} free of "
            f"{_format_bytes(disk.total_bytes)}  ({disk.path})"
        )
    if environment.gpus:
        for gpu in environment.gpus:
            capability = f" · sm_{gpu.compute_capability}" if gpu.compute_capability else ""
            driver = f" · driver {gpu.driver_version}" if gpu.driver_version else ""
            memory_text = (
                f"{gpu.memory_total_mib:,} MiB total, {gpu.memory_free_mib:,} MiB free"
                if gpu.memory_total_mib is not None and gpu.memory_free_mib is not None
                else "memory unknown"
            )
            print(f"  gpu {gpu.index}      : {gpu.name} — {memory_text}{capability}{driver}")
    else:
        print("  gpu        : none detected")
    for framework in environment.frameworks:
        marker = "accelerated" if framework.accelerated else "cpu-only"
        print(f"  framework  : {framework.distribution} {framework.version} ({marker})")
    print()
    print("Recommended execution plan:")
    print(f"  device           : {plan.device}")
    print(f"  worker processes : {plan.recommended_workers}")
    print(f"  batch size       : {plan.recommended_batch_size:,}")
    print(f"  reason           : {plan.rationale}")
    for note in environment.notes:
        print(f"  note             : {note}")
    return 0


def _model_bundle(bundle: Path, *, as_json: bool) -> int:
    """Report what a model bundle contains and the digest that pins it.

    Two kinds of bundle exist -- an ONNX graph MolCascade featurizes for, and a
    Chemprop checkpoint that featurizes for itself -- and they are told apart by
    which manifest is present rather than by a flag, because the directory
    already knows which one it is and asking the user to repeat it is a way to
    be told the wrong answer.

    This reads and hashes; it never loads the model, so it is safe to run on a
    bundle before deciding whether to trust it.
    """

    directory = bundle.expanduser()
    if (directory / CHEMPROP_MANIFEST_FILENAME).is_file():
        report = inspect_chemprop_bundle(bundle)
    else:
        # The ONNX reader is the fallback so that a directory holding neither
        # manifest still fails with its "no molcascade_model.yaml" message,
        # which names a file and is the more common mistake.
        report = inspect_model_bundle(bundle)
    if as_json:
        _json_dump(report, sys.stdout)
        return 0
    print(f"Model bundle: {report['bundle_dir']}")
    print(f"  name       : {report['model_name']}")
    print(f"  endpoint   : {report['endpoint_id']}")
    print(f"  task       : {report['task']}")
    if "n_features" in report:
        print(f"  features   : {report['n_features']}")
    else:
        print(f"  backend    : chemprop {report['chemprop_version'] or '(not installed)'}")
        print(f"  tasks      : {report['n_tasks']} (scoring index {report['task_index']})")
    for entry in report["files"]:
        print(f"  file       : {entry['name']}  {entry['sha256']}  {entry['size_bytes']} bytes")
    if report["model_id"] is None:
        # Only reachable for a Chemprop bundle inspected without Chemprop
        # installed: the featurizer lives in the package, so the identity is not
        # settled by the bundle's bytes alone.
        print("  model_id   : unavailable until chemprop is installed")
    else:
        print(f"  model_id   : {report['model_id']}")
    print()
    print("Paste this into the criterion's settings:")
    print(f"  bundle_dir: {report['bundle_dir']}")
    print(f"  expected_bundle_sha256: {report['bundle_sha256']}")
    return 0


def _pins(backend: str, *, models_dir: str | None, as_json: bool) -> int:
    """Print the digests that pin a pip-installed backend to what is on disk.

    The catalogue ships the digests of one exact release, and a stage refuses to
    run against anything else.  That is the behaviour we want -- a screen should
    not silently change models between the laptop it was designed on and the node
    it runs on -- but it needs an escape hatch that is not "edit the source".
    Upgrade the package deliberately, run this, paste the two lines.

    Nothing here imports the backend.  Files are hashed as bytes, which is the
    whole point: the digests must be computable before deciding to trust them.
    """

    from molcascade.plugins.builtin.admet import (
        DEFAULT_MODELS_REFERENCE,
        inspect_admet_ai_installation,
        inspect_admet_ai_model_assets,
    )

    assert backend == "admet-ai"  # argparse restricts the choices
    reference = models_dir or DEFAULT_MODELS_REFERENCE
    installation = inspect_admet_ai_installation()
    assets = inspect_admet_ai_model_assets(reference)
    report = {
        "backend": backend,
        "version": installation["version"],
        "package_root": installation["package_root"],
        "models_dir": reference,
        "resolved_models_dir": assets["models_dir"],
        "expected_package_code_sha256": installation["package_code_sha256"],
        "expected_model_manifest_sha256": assets["model_manifest_sha256"],
        "checkpoint_count": assets["checkpoint_count"],
        "model_size_bytes": assets["model_size_bytes"],
        "dependency_versions": installation["dependency_versions"],
    }
    if as_json:
        _json_dump(report, sys.stdout)
        return 0
    print(f"ADMET-AI {report['version']}")
    print(f"  package     : {report['package_root']}")
    print(f"  models_dir  : {report['models_dir']}")
    print(f"  resolved to : {report['resolved_models_dir']}")
    print(f"  checkpoints : {report['checkpoint_count']} files, {report['model_size_bytes']} bytes")
    for name, version in sorted(report["dependency_versions"].items()):
        print(f"  requires    : {name} {version or '(not installed)'}")
    print()
    print("Paste this into the criterion's settings:")
    print(f"  models_dir: {report['models_dir']}")
    print(f"  expected_package_code_sha256: {report['expected_package_code_sha256']}")
    print(f"  expected_model_manifest_sha256: {report['expected_model_manifest_sha256']}")
    return 0


def _stage_resources(*, workers: int | None, device: str) -> StageResources:
    """Turn one ``--device``/``--workers`` request into this run's lanes.

    Measured once, here, rather than per stage: ``detect_environment`` shells
    out to ``nvidia-smi``, and twenty tiers asking the same machine the same
    question twenty times would be twenty subprocesses for one answer.
    """

    environment = detect_environment()
    plan = plan_lanes(environment, device=device, workers=workers)
    return StageResources(
        workers=plan.workers,
        devices=plan.devices,
        device_request=device,
        batch_size=environment.plan.recommended_batch_size,
        notes=plan.notes,
    )


def _report_lanes(resources: StageResources) -> None:
    """Say what the run is about to use, before it starts using it."""

    print(f"Execution: {describe_lanes(resources.devices)}")
    for note in resources.notes:
        print(f"  note: {note}")


def _permitted_licenses(
    stages: Sequence[CompiledStage],
    *,
    allow_copyleft: bool,
) -> dict[str, str]:
    """Backend id to SPDX identifier for every copyleft tool this run may link.

    Empty unless the operator asked for it, which is the point: an empty record
    is a positive statement that nothing copyleft ran, and it is the default.
    """

    if not allow_copyleft:
        return {}
    return {spec.id: spec.license_spdx for spec in copyleft_backends(stages)}


def _report_licenses(permitted: Mapping[str, str]) -> None:
    """Say out loud what the licence flag actually turned on.

    ``--allow-copyleft`` is a blanket permission, so the operator who passed it
    does not necessarily know which tool in a twenty-tier cascade needed it.
    Naming them is the difference between a permission and an informed one.
    """

    for backend_id, license_spdx in sorted(permitted.items()):
        print(f"Licence: {backend_id} permitted under {license_spdx} (--allow-copyleft)")


def _run_pipeline(
    config_path: Path,
    *,
    molecules: Path | None,
    molecule_format: str,
    workspace: Path,
    run_id: str | None,
    resume: bool,
    force: bool,
    workers: int | None,
    device: str,
    allow_copyleft: bool,
    target: _TargetArguments,
    as_json: bool,
) -> int:
    if resume and run_id is None:
        raise PipelineError(
            "--resume requires --run-id",
            code="CLI_RESUME_RUN_ID_REQUIRED",
        )
    if molecules is None and molecule_format != "auto":
        raise PipelineError(
            "--molecule-format/--format requires --molecules",
            code="CLI_MOLECULE_FORMAT_REQUIRES_INPUT",
        )
    screening = load_screening_config(config_path)
    registry = create_builtin_registry()
    docking_target, target_notes = _docking_target(
        screening,
        registry,
        workspace=workspace,
        **target.as_kwargs(),
    )
    if molecules is not None:
        config, _ = _prepare_screen(
            screening,
            molecules,
            registry,
            molecule_format=molecule_format,
            target=docking_target,
        )
    elif screening.cascade is not None:
        config = lower_cascade(
            screening.cascade, registry=registry, target=docking_target
        ).pipeline
    else:
        assert screening.pipeline is not None  # ScreeningConfig always carries one kind
        config = screening.pipeline
    # Ahead of the compiler on purpose.  Compilation validates each stage
    # against its plugin's config model, so a docking stage with no receptor
    # already fails -- with a Pydantic report that never mentions '--receptor'.
    preflight_docking_target(config.stages, registry=registry)
    # 'run' used to preflight neither assets nor packages, so the flat-graph
    # entry point could reach the fourth tier before discovering that a weight
    # file was never fetched -- the exact failure 'screen' refuses to ship.  The
    # extra compile is free by comparison: it reads no molecules.
    compiled = PipelineCompiler(registry).compile(config)
    preflight_assets(compiled.stages)
    preflight_backends(
        compiled.stages, policy=_probe_policy(allow_copyleft=allow_copyleft)
    )
    permitted = _permitted_licenses(compiled.stages, allow_copyleft=allow_copyleft)
    resources = _stage_resources(workers=workers, device=device)
    if not as_json:
        _report_target(docking_target, target_notes)
        _report_licenses(permitted)
        _report_lanes(resources)
    runner = LocalRunner(
        workspace,
        plugins=registry,
        resources=resources,
        permitted_licenses=permitted,
    )
    result = runner.run(
        config,
        run_id=run_id,
        resume=resume,
        force=force,
    )
    if as_json:
        _json_dump(result.model_dump(mode="json"), sys.stdout)
    else:
        _print_run_state(result)
    return 0


_CLI_LIBRARY_FORMATS = {
    "delimited": LibraryFormat.DELIMITED,
    "xlsx": LibraryFormat.XLSX,
    "sdf": LibraryFormat.SDF,
    "parquet": LibraryFormat.PARQUET,
    "mol2-directory": LibraryFormat.MOL2_DIRECTORY,
}


def _apply_reader_options(
    stage_config: dict[str, Any],
    *,
    plugin_key: str,
    config_model: object,
    reader: Mapping[str, Any],
) -> dict[str, Any]:
    """Layer command-line layout options onto one source stage's configuration.

    Used for the raw-pipeline path, where there is no ``LibraryConfig`` to merge
    into.  One is built from the overrides purely to reuse
    :func:`reader_settings`, so the translation from ``--id-column`` to the
    field each reader actually declares stays in a single place.
    """

    settings = reader_settings(LibraryConfig.model_validate(dict(reader)), plugin_key)
    fields = set(getattr(config_model, "model_fields", {}))
    updated = dict(stage_config)
    updated.update({name: value for name, value in settings.items() if name in fields})
    return updated


def _reader_overrides(arguments: argparse.Namespace) -> dict[str, Any]:
    """The reader options the user actually typed, as ``LibraryConfig`` fields.

    Only flags that were given appear.  An absent flag has to leave whatever the
    configuration says untouched: filling in argparse's ``None`` as a value
    would let the command line quietly erase a column name the builder wrote.
    """

    if arguments.skip_rows is not None and arguments.skip_rows < 0:
        raise PipelineError(
            "--skip-rows cannot be negative",
            code="CLI_SKIP_ROWS_INVALID",
            context={"skip_rows": arguments.skip_rows},
        )
    typed = {
        "smiles_column": arguments.smiles_column,
        "id_column": arguments.id_column,
        "sheet_name": arguments.sheet,
        "skip_rows": arguments.skip_rows,
    }
    return {name: value for name, value in typed.items() if value is not None}


def _docking_target(
    screening: ScreeningConfig,
    registry: PluginRegistry,
    *,
    workspace: Path,
    receptor: Path | None,
    reference_ligand: Path | None,
    pocket: Path | None,
    box: str | None,
    receptor_pdbqt: Path | None,
) -> tuple[TargetConfig | None, tuple[str, ...]]:
    """Settle what this run docks into before it compiles anything.

    Everything expensive or fragile about a target happens here, once: meeko
    turns the PDB into a PDBQT, a reference ligand becomes six numbers, and the
    receptor's bytes are hashed.  Doing it now rather than inside the docking
    stage is what stops a run from spending the whole funnel and then finding
    out that meeko cannot read the structure.
    """

    merged = merge_target(
        screening.cascade.target if screening.cascade is not None else None,
        receptor=receptor,
        reference_ligand=reference_ligand,
        pocket=pocket,
        box=box,
        receptor_pdbqt=receptor_pdbqt,
    )
    if merged is None:
        return None, ()
    if screening.cascade is None:
        raise ConfigError(
            "the target flags configure a cascade's docking tier, and this "
            "configuration is a flat pipeline",
            code="CLI_TARGET_NOT_APPLICABLE",
            hint=(
                "A flat pipeline names its receptor in the docking stage's own "
                "settings. Either set them there, or screen with a cascade."
            ),
        )
    requirements = docking_requirements(screening.cascade, registry=registry)
    if not requirements:
        raise ConfigError(
            "this cascade has no docking tier to point at a receptor",
            code="CLI_TARGET_WITHOUT_DOCKING",
            hint=(
                "Nothing in the funnel reads a protein, so the target would be "
                "silently ignored. Add a docking tier, or drop the target flags -- "
                "the usual cause is the wrong configuration file."
            ),
            context={"config_name": screening.cascade.name},
        )
    resolved = resolve_target(merged, requirements=requirements, workspace=workspace)
    assert resolved is not None  # merged is not None, so neither is this
    return resolved.target, resolved.notes


def _report_target(target: TargetConfig | None, notes: Sequence[str]) -> None:
    """Say which protein, and say which numbers were computed rather than typed."""

    if target is None:
        return
    print(f"Target: {target.name} ({target.receptor_path})")
    for note in notes:
        print(f"  {note}")


def _prepare_screen(
    config: ScreeningConfig,
    library: Path,
    registry: PluginRegistry,
    *,
    molecule_format: str,
    reader: Mapping[str, Any] | None = None,
    target: TargetConfig | None = None,
) -> tuple[PipelineConfig, LoweredCascade | None]:
    """Bind a library to whichever configuration kind the user supplied."""

    overrides = dict(reader or {})
    if config.cascade is not None:
        cascade = config.cascade
        if overrides:
            # Re-validated rather than ``model_copy``d in: the bounds on these
            # fields are the only thing standing between a typo on the command
            # line and a reader configured with nonsense.
            merged = LibraryConfig.model_validate(
                {**cascade.library.model_dump(mode="python"), **overrides}
            )
            cascade = cascade.model_copy(update={"library": merged})
        lowered = lower_cascade(
            cascade,
            registry=registry,
            library_path=str(library.expanduser()),
            library_format=_CLI_LIBRARY_FORMATS.get(molecule_format),
            target=target,
        )
        return lowered.pipeline, lowered
    assert config.pipeline is not None  # ScreeningConfig always carries one kind
    pipeline = _molecule_override(
        config.pipeline,
        library,
        registry,
        molecule_format=molecule_format,
        reader=overrides,
    )
    return pipeline, None


def _screen(
    config_path: Path,
    *,
    library: Path,
    molecule_format: str,
    reader: Mapping[str, Any],
    workspace: Path,
    run_id: str | None,
    resume: bool,
    force: bool,
    workers: int | None,
    device: str,
    allow_copyleft: bool,
    target: _TargetArguments,
    dry_run: bool,
    as_json: bool,
) -> int:
    if resume and run_id is None:
        raise PipelineError(
            "--resume requires --run-id",
            code="CLI_RESUME_RUN_ID_REQUIRED",
        )
    screening = load_screening_config(config_path)
    registry = create_builtin_registry()
    docking_target, target_notes = _docking_target(
        screening,
        registry,
        workspace=workspace,
        **target.as_kwargs(),
    )
    pipeline, lowered = _prepare_screen(
        screening,
        library,
        registry,
        molecule_format=molecule_format,
        reader=reader,
        target=docking_target,
    )
    # Ahead of the compiler on purpose.  Compilation validates each stage
    # against its plugin's config model, so a docking stage with no receptor
    # already fails -- with a Pydantic report that never mentions '--receptor'.
    preflight_docking_target(pipeline.stages, registry=registry)
    compiled = PipelineCompiler(registry).compile(pipeline)

    # Before the first molecule is read, not when execution reaches the stage
    # that opens the file.  An adapter discovering its weights are missing
    # halfway down the funnel has already spent the expensive part of the run.
    provisioned = preflight_assets(compiled.stages)
    # And the same question about the packages those adapters import, which used
    # to be asked only by ``import_medchem()`` inside ``execute()`` -- four tiers
    # too late, with a run directory and a cache already written.
    installed = preflight_backends(
        compiled.stages, policy=_probe_policy(allow_copyleft=allow_copyleft)
    )
    # Which copyleft tools the operator actually let through, by name. Computed
    # after the preflight passed, so the list describes a run that is going to
    # happen rather than one that was merely asked for.
    permitted = _permitted_licenses(compiled.stages, allow_copyleft=allow_copyleft)
    # Before the dry-run branch as well: '--dry-run --device cuda' on a machine
    # with no card is a question worth being able to ask, and the answer must be
    # the same refusal a real run would give.
    resources = _stage_resources(workers=workers, device=device)

    if dry_run:
        payload = {
            "config_kind": screening.kind,
            "library": str(library),
            "revision_id": compiled.revision_id,
            "stages": [stage.stage_id for stage in compiled.stages],
            "assets": list(provisioned),
            "backends": list(installed),
            "execution_lanes": list(resources.devices),
            "execution_notes": list(resources.notes),
            "permitted_licenses": dict(permitted),
            "target": None if docking_target is None else docking_target.model_dump(mode="json"),
            "target_notes": list(target_notes),
        }
        if as_json:
            _json_dump(payload, sys.stdout)
        else:
            print(f"Configuration: {config_path} ({screening.kind})")
            print(f"Library: {library}")
            print(f"Revision: {compiled.revision_id}")
            _print_planned_funnel(lowered, screening)
            if provisioned:
                print(f"Vendored assets ready: {', '.join(provisioned)}")
            if installed:
                print(f"Backends ready: {', '.join(installed)}")
            _report_target(docking_target, target_notes)
            _report_licenses(permitted)
            _report_lanes(resources)
        return 0

    if not as_json:
        _report_target(docking_target, target_notes)
        _report_licenses(permitted)
        _report_lanes(resources)
    runner = LocalRunner(
        workspace,
        plugins=registry,
        resources=resources,
        permitted_licenses=permitted,
    )
    result = runner.run(pipeline, run_id=run_id, resume=resume, force=force)

    funnel = None
    if lowered is not None and screening.cascade is not None:
        funnel = build_funnel(lowered, screening.cascade, result, runner.store)
    if as_json:
        payload = result.model_dump(mode="json")
        payload["config_kind"] = screening.kind
        if funnel is not None:
            payload["funnel"] = funnel.as_dict()
        _json_dump(payload, sys.stdout)
    elif funnel is not None:
        print(funnel.render())
        print()
        print(f"Run {result.run_id}: {result.status.value}")
        print(f"Export the shortlist with: molcascade export {result.run_id}")
    else:
        _print_run_state(result)
    return 0 if result.status is RunStatus.SUCCEEDED else 1


def _print_planned_funnel(lowered: LoweredCascade | None, screening: ScreeningConfig) -> None:
    """Describe the tiers a cascade will execute, before executing them."""

    cascade = screening.cascade
    if lowered is None or cascade is None:
        print(f"Stages: {len(lowered.pipeline.stages) if lowered else 0}")
        return
    for index, tier in enumerate(cascade.active_tiers, start=1):
        criteria = ", ".join(
            criterion.label or criterion.id for criterion in tier.active_criteria
        )
        mode = tier.mode.value
        if tier.mode is TierMode.AT_LEAST:
            mode = f"{mode} {tier.minimum_passes}"
        print(f"  Tier {index} · {tier.title} [{mode}]: {criteria}")
    print(f"  Shortlist target: {cascade.finalize.target_count:,}")


def _shard_counts(
    checkpoints_root: Path | None,
    stage: StageRunState,
) -> tuple[int, int] | None:
    """``(37, 40)`` for a stage that was interrupted part-way through.

    Only meaningful while a stage is unfinished: the checkpoint directory is
    removed the moment its artifact commits, so a succeeded stage correctly
    reports nothing rather than reporting zero.
    """

    if checkpoints_root is None or stage.cache_key is None:
        return None
    if stage.status in {StageRunStatus.SUCCEEDED, StageRunStatus.CACHED}:
        return None
    complete, total = count_completed_shards(checkpoints_root / stage.cache_key)
    return (complete, total) if total else None


def _print_run_state(state: RunState, *, checkpoints_root: Path | None = None) -> None:
    print(f"Run {state.run_id}: {state.status.value}")
    print(f"Revision: {state.revision_id}")
    for stage in state.stages:
        artifact = stage.output_ref.artifact_id if stage.output_ref else "-"
        counts = _shard_counts(checkpoints_root, stage)
        shards = f" shards={counts[0]}/{counts[1]}" if counts else ""
        print(
            f"  {stage.stage_id:<24} {stage.status.value:<10} "
            f"attempts={stage.attempts} artifact={artifact}{shards}"
        )
    if state.output_ref is not None:
        print(f"Output: {state.output_ref.artifact_id}#{state.output_ref.port}")


def _status(
    run_id: str,
    *,
    workspace: Path,
    include_events: bool,
    as_json: bool,
) -> int:
    workspace_root, state = _read_existing_run(workspace, run_id)
    checked_run_id = state.run_id
    events = ()
    if include_events:
        events_root = workspace_root / "events"
        event_path = events_root / f"{checked_run_id}.jsonl"
        if events_root.is_symlink() or not events_root.is_dir():
            raise ArtifactIntegrityError(
                f"audit log is missing for run {checked_run_id}",
                code="RUNTIME_AUDIT_INVALID",
                context={"run_id": checked_run_id},
            )
        if not event_path.exists() and not event_path.is_symlink():
            raise ArtifactIntegrityError(
                f"audit log is missing for run {checked_run_id}",
                code="RUNTIME_AUDIT_INVALID",
                context={"run_id": checked_run_id},
            )
        events = _read_audit_events(event_path, state)
    checkpoints_root = workspace_root / "checkpoints"
    if as_json:
        payload: dict[str, Any] = state.model_dump(mode="json")
        for entry, stage in zip(payload["stages"], state.stages, strict=True):
            counts = _shard_counts(checkpoints_root, stage)
            if counts is not None:
                entry["shards"] = {"complete": counts[0], "total": counts[1]}
        if include_events:
            payload["events"] = [event.model_dump(mode="json") for event in events]
        _json_dump(payload, sys.stdout)
    else:
        _print_run_state(state, checkpoints_root=checkpoints_root)
        if include_events:
            print("Events:")
            for event in events:
                stage = f" stage={event.stage_id}" if event.stage_id else ""
                print(f"  {event.sequence:04d} {event.event_type}{stage}")
    return 0


def _read_existing_run(workspace: Path, run_id: str) -> tuple[Path, RunState]:
    """Read a durable state without initializing or changing its workspace."""

    requested = workspace.expanduser()
    if requested.is_symlink() or not requested.is_dir():
        raise PipelineError(
            f"workspace does not exist: {workspace}",
            code="CLI_WORKSPACE_NOT_FOUND",
            context={"workspace": str(workspace)},
        )
    workspace_root = requested.resolve(strict=True)
    checked_run_id = _checked_run_id(run_id)
    runs_root = workspace_root / "runs"
    state_path = runs_root / f"{checked_run_id}.json"
    if runs_root.is_symlink() or not runs_root.is_dir():
        raise PipelineError(
            f"run does not exist: {checked_run_id}",
            code="RUN_NOT_FOUND",
            context={"run_id": checked_run_id},
        )
    state = _read_run_state(state_path, checked_run_id)
    return workspace_root, state


def _runner_for_existing_run(workspace: Path, run_id: str) -> tuple[LocalRunner, RunState]:
    """Verify an initialized workspace and exact state/revision before mutation."""

    workspace_root, state = _read_existing_run(workspace, run_id)
    required_directories = (
        workspace_root / ".locks",
        workspace_root / ".run-locks",
        workspace_root / ".staging",
        workspace_root / "artifacts",
        workspace_root / "artifacts" / "sha256",
        workspace_root / "cache",
        workspace_root / "events",
        workspace_root / "revisions",
        workspace_root / "runs",
    )
    for directory in required_directories:
        if directory.is_symlink() or not directory.is_dir():
            raise ArtifactIntegrityError(
                f"run workspace is incomplete or unsafe: {directory}",
                code="RUNTIME_WORKSPACE_INVALID",
                context={"run_id": state.run_id, "path": str(directory)},
            )

    event_path = workspace_root / "events" / f"{state.run_id}.jsonl"
    _read_audit_events(event_path, state)

    runner = LocalRunner(workspace_root, plugins=create_builtin_registry())
    revision = runner._load_revision(state.revision_id)
    compiled = runner.compile(revision)
    expected = tuple((stage.stage_id, stage.plugin_key) for stage in compiled.stages)
    observed = tuple((stage.stage_id, stage.plugin_key) for stage in state.stages)
    if observed != expected:
        raise ArtifactIntegrityError(
            f"run state stages differ from revision {state.revision_id}",
            code="RUNTIME_STATE_INVALID",
            context={"run_id": state.run_id, "revision_id": state.revision_id},
        )
    _verify_compiled_checkpoints(runner, state, compiled.stages)
    return runner, state


def _verify_compiled_checkpoints(
    runner: LocalRunner,
    state: RunState,
    compiled_stages: Sequence[CompiledStage],
) -> None:
    """Bind every completed checkpoint back to its exact compiled invocation."""

    retained: dict[tuple[str, str], ArtifactDatasetRef] = {}
    completed = {StageRunStatus.SUCCEEDED, StageRunStatus.CACHED}
    for saved, stage in zip(state.stages, compiled_stages, strict=True):
        if saved.status not in completed:
            continue
        if saved.output_ref is None or saved.cache_key is None:
            raise ArtifactIntegrityError(
                f"completed stage is missing checkpoint identity: {saved.stage_id}",
                code="RUNTIME_CHECKPOINT_INVALID",
                context={"run_id": state.run_id, "stage_id": saved.stage_id},
            )
        bound_inputs = runner._resolve_stage_inputs(stage, retained)
        expected_key = stage_cache_key(stage, bound_inputs)
        if saved.cache_key != expected_key:
            raise ArtifactIntegrityError(
                f"checkpoint cache key differs from revision for {saved.stage_id}",
                code="RUNTIME_CHECKPOINT_INVALID",
                context={"run_id": state.run_id, "stage_id": saved.stage_id},
            )
        restored = runner._verify_checkpoint(
            stage,
            expected_key,
            saved.output_ref,
            bound_inputs,
        )
        retained.update(restored)


def _export_shortlist(
    run_id: str,
    *,
    workspace: Path,
    output: Path | None,
    overwrite: bool,
    as_json: bool,
) -> int:
    runner, _ = _runner_for_existing_run(workspace, run_id)
    result = materialize_shortlist(
        runner,
        run_id,
        output,
        overwrite=overwrite,
    )
    payload: dict[str, Any] = {
        "path": str(result.path),
        "record_format": result.record_format,
        "row_count": result.row_count,
        "identified_count": result.identified_count,
        "sha256": result.sha256,
        "source_artifact_id": result.source_artifact_id,
        "export_spec_id": result.export_spec_id,
    }
    if result.docking is not None:
        # Absent, not null, when the run had no docking tier: a key that is
        # always there invites a script to treat an empty bundle and no bundle
        # as the same thing, and they are not.
        payload["docking"] = {
            "path": str(result.docking.path),
            "engines": list(result.docking.engines),
            "engine_files": [str(item) for item in result.docking.engine_files],
            "best_pose_path": str(result.docking.best_pose_path),
            "pose_count": result.docking.pose_count,
            "best_pose_count": result.docking.best_pose_count,
        }
    if as_json:
        _json_dump(payload, sys.stdout)
    else:
        print(f"Format: {result.record_format}")
        print(f"Rows: {result.row_count}")
        if result.identified_count:
            # Reported only when identifiers exist, and reported as a fraction
            # because a shortfall is the visible symptom of an id column with
            # blank cells -- worth seeing before the file is joined elsewhere.
            print(f"Named rows: {result.identified_count}/{result.row_count}")
        print(f"SHA-256: {result.sha256}")
        print(f"Path: {result.path}")
        if result.docking is not None:
            engines = ", ".join(result.docking.engines) or "none recorded"
            print(f"Docking evidence: {result.docking.path}")
            print(f"  Engines: {engines}")
            # Both counts, because the gap between them is the pose depth each
            # engine kept -- and a best-pose count below the shortlist size means
            # some shortlisted molecules were never docked, which is worth
            # noticing before the table is joined to anything.
            print(
                f"  Poses: {result.docking.pose_count}"
                f" ({result.docking.best_pose_count} best-pose rows)"
            )
    return 0


def _report(
    run_id: str,
    *,
    workspace: Path,
    output: Path | None,
    overwrite: bool,
    as_json: bool,
) -> int:
    runner, _ = _runner_for_existing_run(workspace, run_id)
    destination = (
        runner.workspace / "reports" / f"{run_id}.html"
        if output is None
        else output
    )
    try:
        generated = generate_run_report(
            runner,
            run_id,
            destination,
            overwrite=overwrite,
        )
    except DestinationExistsError as error:
        raise PipelineError(
            f"run report output already exists: {destination}",
            code="REPORT_OUTPUT_EXISTS",
            hint="Choose another path or explicitly pass --overwrite.",
            context={"path": str(destination)},
        ) from error
    if as_json:
        _json_dump({"path": str(generated), "run_id": run_id}, sys.stdout)
    else:
        print(f"Generated run report: {generated}")
    return 0


def _checked_run_id(run_id: str) -> str:
    try:
        return validate_run_id(run_id)
    except ValueError as error:
        raise PipelineError(
            f"invalid run ID: {error}",
            code="RUN_ID_INVALID",
            context={"run_id": str(run_id)},
        ) from error


def _read_run_state(path: Path, run_id: str) -> RunState:
    if path.is_symlink() or not path.is_file():
        raise PipelineError(
            f"run does not exist: {run_id}",
            code="RUN_NOT_FOUND",
            context={"run_id": run_id},
        )
    try:
        if path.stat().st_size > _RUN_STATE_SIZE_LIMIT:
            raise ValueError("run state is unreasonably large")
        content = path.read_bytes()
        if len(content) > _RUN_STATE_SIZE_LIMIT:
            raise ValueError("run state is unreasonably large")
        state = RunState.model_validate_json(content)
    except (OSError, ValueError, ValidationError) as error:
        raise ArtifactIntegrityError(
            f"run state is invalid for {run_id}: {error}",
            code="RUNTIME_STATE_INVALID",
            context={"run_id": run_id},
        ) from error
    if state.run_id != run_id or content != canonical_json_bytes(state):
        raise ArtifactIntegrityError(
            f"run state is not canonical for {run_id}",
            code="RUNTIME_STATE_INVALID",
            context={"run_id": run_id},
        )
    _validate_run_state_lifecycle(state)
    return state


def _validate_run_state_lifecycle(state: RunState) -> None:
    completed = {StageRunStatus.SUCCEEDED, StageRunStatus.CACHED}
    saw_incomplete = False
    for stage in state.stages:
        if not _stage_lifecycle_valid(stage):
            raise ArtifactIntegrityError(
                f"stage lifecycle is inconsistent for {stage.stage_id}",
                code="RUNTIME_STATE_INVALID",
                context={
                    "run_id": state.run_id,
                    "stage_id": stage.stage_id,
                    "stage_status": stage.status.value,
                },
            )
        stage_completed = stage.status in completed
        if saw_incomplete and stage_completed:
            raise ArtifactIntegrityError(
                "run state contains a completed stage after an incomplete stage",
                code="RUNTIME_STATE_INVALID",
                context={"run_id": state.run_id, "stage_id": stage.stage_id},
            )
        if not stage_completed:
            saw_incomplete = True

    if state.status is RunStatus.SUCCEEDED:
        valid = (
            state.finished_at is not None
            and state.error is None
            and state.output_ref is not None
            and bool(state.stages)
            and all(
                stage.status in completed and stage.output_ref is not None
                for stage in state.stages
            )
            and state.stages[-1].output_ref == state.output_ref
        )
    elif state.status is RunStatus.FAILED:
        valid = (
            state.finished_at is not None
            and state.error is not None
            and state.output_ref is None
            and all(stage.status is not StageRunStatus.RUNNING for stage in state.stages)
        )
    else:
        valid = (
            state.finished_at is None
            and state.error is None
            and state.output_ref is None
            and all(stage.status is not StageRunStatus.FAILED for stage in state.stages)
        )
    if not valid:
        raise ArtifactIntegrityError(
            f"run state lifecycle is inconsistent for {state.run_id}",
            code="RUNTIME_STATE_INVALID",
            context={"run_id": state.run_id, "status": state.status.value},
        )


def _stage_lifecycle_valid(stage: StageRunState) -> bool:
    if stage.status is StageRunStatus.PENDING:
        return (
            stage.cache_key is None
            and stage.output_ref is None
            and stage.started_at is None
            and stage.finished_at is None
            and stage.error is None
        )
    if stage.status is StageRunStatus.RUNNING:
        return (
            stage.cache_key is not None
            and stage.output_ref is None
            and stage.finished_at is None
            and stage.error is None
            and (stage.started_at is None or stage.attempts >= 1)
        )
    if stage.status is StageRunStatus.CACHED:
        return (
            stage.cache_key is not None
            and stage.output_ref is not None
            and stage.finished_at is not None
            and stage.error is None
        )
    if stage.status is StageRunStatus.SUCCEEDED:
        return (
            stage.attempts >= 1
            and stage.cache_key is not None
            and stage.output_ref is not None
            and stage.started_at is not None
            and stage.finished_at is not None
            and stage.error is None
        )
    return (
        stage.output_ref is None
        and stage.finished_at is not None
        and stage.error is not None
    )


def _read_audit_events(path: Path, state: RunState) -> tuple[AuditEvent, ...]:
    if path.is_symlink() or not path.is_file():
        raise ArtifactIntegrityError(
            f"audit log is missing or invalid for run {state.run_id}",
            code="RUNTIME_AUDIT_INVALID",
            context={"run_id": state.run_id},
        )
    try:
        if path.stat().st_size > _AUDIT_SIZE_LIMIT:
            raise ValueError("audit log is unreasonably large")
        content = path.read_bytes()
        if len(content) > _AUDIT_SIZE_LIMIT:
            raise ValueError("audit log is unreasonably large")
        if content and not content.endswith(b"\n"):
            raise ValueError("audit log has an incomplete final line")
        events: list[AuditEvent] = []
        for line_number, line in enumerate(content.splitlines(), start=1):
            if not line:
                raise ValueError(f"blank audit line at {line_number}")
            event = AuditEvent.model_validate_json(line)
            if line != canonical_json_bytes(event):
                raise ValueError(f"non-canonical audit line at {line_number}")
            events.append(event)
    except (OSError, ValueError, ValidationError) as error:
        raise ArtifactIntegrityError(
            f"audit log is invalid for run {state.run_id}: {error}",
            code="RUNTIME_AUDIT_INVALID",
            context={"run_id": state.run_id},
        ) from error

    for expected_sequence, event in enumerate(events, start=1):
        if (
            event.sequence != expected_sequence
            or event.run_id != state.run_id
            or event.revision_id != state.revision_id
        ):
            raise ArtifactIntegrityError(
                f"audit identity or sequence is invalid for run {state.run_id}",
                code="RUNTIME_AUDIT_INVALID",
                context={
                    "run_id": state.run_id,
                    "expected_sequence": expected_sequence,
                },
            )
    if not events or events[0].event_type != "RUN_STARTED":
        raise ArtifactIntegrityError(
            f"audit log has no initial RUN_STARTED record for run {state.run_id}",
            code="RUNTIME_AUDIT_INVALID",
            context={"run_id": state.run_id},
        )
    final_event = events[-1]
    if state.status is RunStatus.SUCCEEDED:
        valid_terminal = (
            final_event.event_type == "RUN_SUCCEEDED"
            and state.output_ref is not None
            and final_event.details.get("artifact_id") == state.output_ref.artifact_id
        )
    elif state.status is RunStatus.FAILED:
        valid_terminal = final_event.event_type == "RUN_FAILED"
    else:
        valid_terminal = final_event.event_type not in {"RUN_SUCCEEDED", "RUN_FAILED"}
    if not valid_terminal:
        raise ArtifactIntegrityError(
            f"audit terminal event disagrees with run state for {state.run_id}",
            code="RUNTIME_AUDIT_INVALID",
            context={
                "run_id": state.run_id,
                "status": state.status.value,
                "final_event": final_event.event_type,
            },
        )
    return tuple(events)


def _unexpected_cli_error(error: BaseException) -> ExecutionError:
    try:
        text = str(error)
    except BaseException:
        text = f"<unprintable {type(error).__name__}>"
    return ExecutionError(
        f"CLI operation failed: {text}",
        code="CLI_OPERATION_FAILED",
        context={"error_type": type(error).__name__},
    )


def _dispatch_compatibility(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
) -> int | None:
    if arguments.compat_generate is not None:
        incompatible = (
            arguments.vs,
            arguments.compat_config,
            arguments.compat_molecules,
            arguments.compat_molecule_format,
            arguments.compat_workspace,
            arguments.compat_run_id,
            arguments.compat_resume,
            arguments.compat_force,
            arguments.compat_json,
        )
        if any(value not in {None, False} for value in incompatible):
            parser.error("--generate config accepts only the optional --output PATH")
        output = arguments.compat_output or _DEFAULT_BUILDER_OUTPUT
        return _generate_config(output)
    if arguments.vs:
        if arguments.compat_output is not None:
            parser.error("--output is only valid with --generate config")
        if arguments.compat_config is None:
            parser.error("--vs requires --config PATH")
        if arguments.compat_molecules is None:
            parser.error("--vs requires --molecules PATH")
        return _run_pipeline(
            arguments.compat_config,
            molecules=arguments.compat_molecules,
            molecule_format=arguments.compat_molecule_format or "auto",
            workspace=arguments.compat_workspace or _DEFAULT_WORKSPACE,
            run_id=arguments.compat_run_id,
            resume=arguments.compat_resume,
            force=arguments.compat_force,
            # The compatibility form predates parallel execution and gains no
            # flags; 'auto' is what it would have asked for had it been able to.
            workers=None,
            device="auto",
            # It also predates the copyleft opt-in, and a form that cannot spell
            # the permission is not a form that can grant it: linking GPL code
            # is a decision someone has to make in the command they typed.
            allow_copyleft=False,
            # Same reasoning for the receptor: the legacy form has no flags to
            # name one, so a cascade with a docking tier stops in the preflight
            # and says which command does.
            target=_TargetArguments(),
            as_json=arguments.compat_json,
        )
    legacy_values = (
        arguments.compat_config,
        arguments.compat_molecules,
        arguments.compat_molecule_format,
        arguments.compat_output,
        arguments.compat_workspace,
        arguments.compat_run_id,
        arguments.compat_resume,
        arguments.compat_force,
        arguments.compat_json,
    )
    if arguments.command is None and any(value not in {None, False} for value in legacy_values):
        parser.error("legacy options require either --generate config or --vs")
    return None


def _dispatch_canonical(arguments: argparse.Namespace) -> int:
    if arguments.command == "generate":
        return _generate_config(arguments.output, pipeline=arguments.pipeline)
    if arguments.command == "validate":
        return _validate(arguments.config, as_json=arguments.json_output)
    if arguments.command == "plugins":
        return _plugins(as_json=arguments.json_output)
    if arguments.command == "cite":
        return _cite(arguments.output)
    if arguments.command == "doctor":
        return _doctor(
            as_json=arguments.json_output,
            run_version_commands=arguments.run_version_commands,
            allow_copyleft=arguments.allow_copyleft,
        )
    if arguments.command == "assets":
        action = getattr(arguments, "asset_action", None)
        if action == "list":
            return _assets_list(as_json=arguments.json_output)
        if action == "status":
            return _assets_status(
                arguments.asset_id, deep=arguments.deep, as_json=arguments.json_output
            )
        if action == "fetch":
            return _assets_fetch(
                arguments.asset_id,
                fetch_all=arguments.all,
                force=arguments.force,
                as_json=arguments.json_output,
            )
        if action == "where":
            return _assets_where(arguments.reference, as_json=arguments.json_output)
        raise PipelineError(
            "choose an assets action: list, status, fetch, where",
            code="CLI_COMMAND_REQUIRED",
        )
    if arguments.command == "env":
        return _environment(
            workspace=arguments.workspace,
            probe_gpus=not arguments.no_gpu_probe,
            as_json=arguments.json_output,
        )
    if arguments.command == "model-bundle":
        return _model_bundle(arguments.bundle, as_json=arguments.json_output)
    if arguments.command == "pins":
        return _pins(
            arguments.backend,
            models_dir=arguments.models_dir,
            as_json=arguments.json_output,
        )
    if arguments.command == "screen":
        return _screen(
            arguments.config,
            library=arguments.library,
            molecule_format=arguments.molecule_format,
            reader=_reader_overrides(arguments),
            workspace=arguments.workspace,
            run_id=arguments.run_id,
            resume=arguments.resume,
            force=arguments.force,
            workers=arguments.workers,
            device=arguments.device,
            allow_copyleft=arguments.allow_copyleft,
            target=_TargetArguments.from_namespace(arguments),
            dry_run=arguments.dry_run,
            as_json=arguments.json_output,
        )
    if arguments.command == "run":
        return _run_pipeline(
            arguments.config,
            molecules=arguments.molecules,
            molecule_format=arguments.molecule_format,
            workspace=arguments.workspace,
            run_id=arguments.run_id,
            resume=arguments.resume,
            force=arguments.force,
            workers=arguments.workers,
            device=arguments.device,
            allow_copyleft=arguments.allow_copyleft,
            target=_TargetArguments.from_namespace(arguments),
            as_json=arguments.json_output,
        )
    if arguments.command == "status":
        return _status(
            arguments.run_id,
            workspace=arguments.workspace,
            include_events=arguments.events,
            as_json=arguments.json_output,
        )
    if arguments.command == "export":
        return _export_shortlist(
            arguments.run_id,
            workspace=arguments.workspace,
            output=arguments.output,
            overwrite=arguments.overwrite,
            as_json=arguments.json_output,
        )
    if arguments.command == "report":
        return _report(
            arguments.run_id,
            workspace=arguments.workspace,
            output=arguments.output,
            overwrite=arguments.overwrite,
            as_json=arguments.json_output,
        )
    raise PipelineError("no command selected", code="CLI_COMMAND_REQUIRED")


def _main(argv: Sequence[str]) -> int:
    parser = build_parser()
    arguments = parser.parse_args(list(argv))
    compatibility_values = (
        arguments.compat_generate,
        arguments.vs,
        arguments.compat_config,
        arguments.compat_molecules,
        arguments.compat_molecule_format,
        arguments.compat_output,
        arguments.compat_workspace,
        arguments.compat_run_id,
        arguments.compat_resume,
        arguments.compat_force,
        arguments.compat_json,
    )
    if arguments.command is not None and any(
        value not in {None, False} for value in compatibility_values
    ):
        parser.error(
            "legacy options cannot be combined with a canonical subcommand; "
            "place command options after the subcommand"
        )
    if arguments.command is None and arguments.compat_generate is None and not arguments.vs:
        if any(value not in {None, False} for value in compatibility_values):
            parser.error("legacy options require either --generate config or --vs")
        parser.print_help()
        return 0

    as_json = bool(
        getattr(arguments, "json_output", False)
        or getattr(arguments, "compat_json", False)
    )
    try:
        compatibility_result = _dispatch_compatibility(parser, arguments)
        if compatibility_result is not None:
            return compatibility_result
        return _dispatch_canonical(arguments)
    except MolCascadeError as error:
        _render_error(error, as_json=as_json)
        return 1
    except Exception as error:
        public_error = _unexpected_cli_error(error)
        _render_error(public_error, as_json=as_json)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv``, execute one command, and return a process exit code."""

    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    token = _JSON_ERROR_MODE.set("--json" in raw_arguments)
    try:
        return _main(raw_arguments)
    finally:
        _JSON_ERROR_MODE.reset(token)


def app(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point."""

    return main(argv)


__all__ = ["app", "build_parser", "main"]
