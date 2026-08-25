"""Generate the offline, block-assembly screening configuration builder.

Two earlier surfaces failed for opposite reasons.  A node-and-wire canvas made
users draw a graph by hand, so the two things they actually change -- *which
tier a criterion belongs to* and *where its threshold sits* -- were encoded as
geometry.  Replacing it with a list of cards fixed the editing but lost the
picture: nothing on screen showed that a parallel tier fans out and rejoins.

This builder is a block workshop.  Criteria are bricks dragged out of a tray and
dropped into tiers; a tier draws its own topology, so a serial tier is a visible
chain and a parallel tier is a visible fan-out ending in the join block that the
run will really execute.  Every brick, gate chip and join block on screen
corresponds one-to-one with a stage in the lowered pipeline -- the browser
predicts that stage list itself, and a test compares its prediction against
``lower_cascade`` so the drawing cannot drift away from the execution.

The generated document is a single file with no network access: assets are
inlined and pinned by CSP hash, there is no ``connect-src``, and nothing is
fetched, evaluated, or installed when it is opened.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

from molcascade import __version__
from molcascade.cascade.availability import option_availability
from molcascade.cascade.catalog import CRITERIA, STAGE_GROUPS, BackendOption
from molcascade.cascade.defaults import (
    DEFAULT_SEED,
    DEFAULT_TARGET_COUNT,
    criterion_defaults,
    default_cascade,
)
from molcascade.cascade.lower import DECISION_JOIN_PLUGIN
from molcascade.cascade.models import LibraryFormat, TierMode
from molcascade.environment import detect_environment
from molcascade.environment.models import HostEnvironment
from molcascade.io.atomic import atomic_write_bytes
from molcascade.plugins.registry import PluginRegistry, create_builtin_registry

BUILDER_SCHEMA_VERSION = 2
DEFAULT_CONFIG_FILENAME = "cascade.json"

# Distributions whose presence in a backend's ``requires`` means the tool will
# use an accelerator when one exists.  This is a hint shown next to the tool, not
# a gate: every backend MolCascade ships also runs on CPU, only slower.
_ACCELERATED_REQUIREMENTS = frozenset({"torch", "boltz", "onnxruntime-gpu", "tensorflow"})

_TIER_MODES: tuple[dict[str, Any], ...] = (
    {
        "value": TierMode.SERIAL.value,
        "label": "Serial",
        "short": "in series",
        "summary": "Each criterion screens only what the previous one kept.",
        "detail": (
            "Cheapest first. Use this when a later criterion is slow, or when it "
            "only makes sense for molecules that already passed."
        ),
    },
    {
        "value": TierMode.ALL.value,
        "label": "Parallel · all",
        "short": "all must pass",
        "summary": "Every criterion sees the same molecules; all must pass.",
        "detail": (
            "Same survivors as running them in series, but every criterion reports "
            "on every molecule, so you can see which one is doing the filtering."
        ),
    },
    {
        "value": TierMode.ANY.value,
        "label": "Parallel · any",
        "short": "any may pass",
        "summary": "Every criterion sees the same molecules; one pass is enough.",
        "detail": (
            "A union. Use it when several routes to the same conclusion are "
            "acceptable and you do not want to insist on agreement."
        ),
    },
    {
        "value": TierMode.AT_LEAST.value,
        "label": "Parallel · at least K",
        "short": "at least K must pass",
        "summary": "Every criterion votes; a molecule survives on K passes.",
        "detail": (
            "A consensus vote across independent tools. Useful when no single "
            "predictor is trusted on its own."
        ),
    },
)

_LIBRARY_FORMATS: tuple[dict[str, Any], ...] = (
    {
        "value": LibraryFormat.AUTO.value,
        "label": "Detect from file name",
        "hint": "Recommended. The reader is chosen from the suffix at run time.",
    },
    {
        "value": LibraryFormat.DELIMITED.value,
        "label": "Delimited text (.csv, .tsv, .smi)",
        "hint": "One molecule per line with a SMILES column.",
    },
    {
        "value": LibraryFormat.XLSX.value,
        "label": "Excel workbook (.xlsx)",
        "hint": "One sheet, one molecule per row.",
    },
    {
        "value": LibraryFormat.SDF.value,
        "label": "SDF (.sdf)",
        "hint": "Structure records; properties are read from SD tags.",
    },
    {
        "value": LibraryFormat.PARQUET.value,
        "label": "Parquet (.parquet)",
        "hint": "Already-typed raw molecule records.",
    },
    {
        "value": LibraryFormat.MOL2_DIRECTORY.value,
        "label": "Directory of MOL2 files",
        "hint": "Every .mol2 file under the directory becomes one record.",
    },
)

# Fields on the closing steps that a screening user genuinely tunes.  Anything
# not listed here keeps the reviewed default rather than being surfaced as a
# knob with no scientific meaning to the person turning it.
_FINALIZE_FIELDS: tuple[dict[str, Any], ...] = (
    {
        "step": "shortlist_select",
        "name": "max_per_scaffold",
        "label": "Max molecules per scaffold",
        "kind": "integer",
        "minimum": 1,
        "step_size": 1,
        "help": (
            "Caps how much of the shortlist one chemotype may occupy, so a single "
            "over-represented series cannot fill the docking budget."
        ),
    },
    {
        "step": "shortlist",
        "name": "filename",
        "label": "Shortlist file name",
        "kind": "text",
        "help": "Written inside the run's export artifact.",
    },
)

_LIBRARY_FIELDS: tuple[dict[str, Any], ...] = (
    {
        "name": "smiles_column",
        "label": "SMILES column",
        "kind": "text",
        "formats": ["auto", "delimited", "xlsx"],
        "help": "Column holding the structure. Ignored for SDF, Parquet and MOL2.",
    },
    {
        "name": "id_column",
        "label": "Identifier column",
        "kind": "text",
        "formats": ["auto", "delimited", "xlsx", "sdf"],
        "help": "Optional. Your own molecule ID, carried through to the shortlist.",
    },
    {
        "name": "delimiter",
        "label": "Delimiter",
        "kind": "text",
        "formats": ["auto", "delimited"],
        "help": "Left blank, this follows the suffix: comma, tab, or whitespace.",
    },
    {
        "name": "has_header",
        "label": "First row is a header",
        "kind": "boolean",
        # Unset means "whatever the reader does by default", and every delimited
        # reader MolCascade ships defaults to True.  Showing an unticked box for
        # an unset value would misreport what the run will actually do.
        "default_value": True,
        "formats": ["auto", "delimited", "xlsx"],
        "help": "Untick only if the file starts straight into data.",
    },
    {
        "name": "sheet_name",
        "label": "Worksheet name",
        "kind": "text",
        "formats": ["xlsx"],
        "help": "Blank uses the first sheet.",
    },
    {
        "name": "batch_size",
        "label": "Rows per batch",
        "kind": "integer",
        "minimum": 1,
        "step_size": 1024,
        "formats": ["auto", "delimited", "xlsx", "sdf", "parquet", "mol2_directory"],
        "help": (
            "Streaming batch size. Lower it if memory is tight; it never changes "
            "which molecules survive."
        ),
    },
)


def _option_payload(
    option: BackendOption,
    payload: dict[str, Any],
    *,
    registry: PluginRegistry,
) -> dict[str, Any]:
    """Attach the exact settings a fresh criterion would start with.

    The browser must not re-derive plugin defaults: which fields a backend
    accepts (``schema_version`` among them) is a property of the installed
    plugin, so it is computed once here and copied verbatim by the UI.
    """

    payload["accelerated"] = bool(_ACCELERATED_REQUIREMENTS.intersection(option.requires))
    if payload.get("executable"):
        settings, gate_settings = criterion_defaults(option, registry=registry)
        payload["initial_settings"] = settings
        payload["initial_gate_settings"] = gate_settings
    else:
        payload["initial_settings"] = {}
        payload["initial_gate_settings"] = {}
    return payload


def _machine_payload(environment: HostEnvironment) -> dict[str, Any]:
    """Condense hardware detection into the few lines the browser shows.

    The builder is usually opened on a laptop and the cascade is usually run
    somewhere else, so this panel is a statement about *where the file was
    written*, phrased that way.  It exists to stop the most expensive mistake in
    the workflow: assembling a GPU-heavy cascade, running it on the machine that
    drew it, and discovering the difference eight hours later.
    """

    cpu = environment.cpu
    memory = environment.memory
    gpus = [
        {
            "name": gpu.name,
            "memory_gib": gpu.memory_total_gib,
            "compute_capability": gpu.compute_capability,
        }
        for gpu in environment.gpus
    ]
    total_gib = memory.total_gib
    return {
        "platform": environment.platform.system
        + (" · WSL" if environment.platform.wsl else "")
        + (" · container" if environment.platform.in_container else ""),
        "cpu_name": cpu.model_name or "unknown CPU",
        "cpu_cores": cpu.usable_cores or cpu.logical_cores,
        "memory_gib": total_gib,
        "gpus": gpus,
        "gpu_count": len(gpus),
        "total_gpu_memory_gib": environment.total_gpu_memory_gib,
        "device": environment.plan.device,
        "workers": environment.plan.recommended_workers,
        "batch_size": environment.plan.recommended_batch_size,
        "gpu_backends_runnable": environment.plan.gpu_backends_runnable,
        "rationale": environment.plan.rationale,
    }


def build_cascade_builder_payload(
    *,
    plugins: PluginRegistry | None = None,
    environment: HostEnvironment | None = None,
) -> dict[str, Any]:
    """Create the finite JSON payload the offline builder renders from."""

    registry = plugins or create_builtin_registry()
    host = environment if environment is not None else detect_environment()

    criteria: list[dict[str, Any]] = []
    for spec in CRITERIA:
        spec_payload = spec.as_json()
        options: list[dict[str, Any]] = []
        for option, option_payload in zip(spec.options, spec_payload["options"], strict=True):
            # "Executable" here means *this machine can run it*, which is a
            # stricter question than whether an adapter is registered: the
            # built-in adapters defer their third-party imports, so one can be
            # registered on a host that lacks every package it needs, or with
            # every package and none of the data files it reads.  Offering such
            # a tool as selectable would move the failure from now, where it
            # costs a sentence, to hour three of a screening run.
            availability = option_availability(option, registry)
            option_payload["executable"] = availability.runnable
            option_payload["missing_packages"] = list(availability.missing_packages)
            option_payload["missing_assets"] = list(availability.missing_assets)
            if availability.reason is not None:
                option_payload["notes"] = availability.reason
            options.append(_option_payload(option, option_payload, registry=registry))
        spec_payload["options"] = options
        executable = [option for option in options if option["executable"]]
        spec_payload["has_executable_option"] = bool(executable)
        # Which tool a freshly added criterion starts with is decided here, by
        # the same rule the Python default uses, so adding from the browser and
        # adding from ``default_cascade`` cannot drift apart.
        preferred = next(
            (option for option in executable if option["recommended"]),
            executable[0] if executable else None,
        )
        spec_payload["default_option_id"] = None if preferred is None else preferred["id"]
        criteria.append(spec_payload)

    starter = default_cascade(registry=registry, target_count=DEFAULT_TARGET_COUNT)
    return {
        "builder_schema_version": BUILDER_SCHEMA_VERSION,
        "molcascade_version": __version__,
        "stages": [group.as_json() for group in STAGE_GROUPS],
        "criteria": criteria,
        "tier_modes": [dict(mode) for mode in _TIER_MODES],
        "library_formats": [dict(entry) for entry in _LIBRARY_FORMATS],
        "library_fields": [dict(field) for field in _LIBRARY_FIELDS],
        "finalize_fields": [dict(field) for field in _FINALIZE_FIELDS],
        "default_cascade": starter.model_dump(mode="json"),
        "default_seed": DEFAULT_SEED,
        "default_target_count": DEFAULT_TARGET_COUNT,
        "default_filename": DEFAULT_CONFIG_FILENAME,
        # A parallel tier with more than one criterion needs the policy join.
        # The browser must be able to say so before the user exports a cascade
        # this installation could not lower.
        "join_plugin": DECISION_JOIN_PLUGIN,
        "join_plugin_available": DECISION_JOIN_PLUGIN in registry,
        # Written by Python so the command the browser prints cannot drift away
        # from the argument parser that has to accept it.
        "run_command": (
            f"molcascade screen --config {DEFAULT_CONFIG_FILENAME} --library <your library file>"
        ),
        "machine": _machine_payload(host),
    }


@lru_cache(maxsize=8)
def _asset(name: str) -> str:
    return files("molcascade.ui.assets").joinpath(name).read_text(encoding="utf-8")


def _csp_hash(content: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).digest()
    return "sha256-" + base64.b64encode(digest).decode("ascii")


def render_cascade_builder(payload: dict[str, Any]) -> str:
    """Render the complete offline document for ``payload``."""

    css = _asset("cascade.css")
    javascript = _asset("cascade.js")
    payload_text = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    escaped_payload = html.escape(payload_text, quote=True)
    csp = (
        "default-src 'none'; "
        f"script-src '{_csp_hash(javascript)}'; "
        f"style-src '{_csp_hash(css)}'; "
        "img-src data:; connect-src 'none'; object-src 'none'; base-uri 'none'; "
        "form-action 'none'; frame-ancestors 'none'"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="{html.escape(csp, quote=True)}">
  <title>MolCascade screening builder</title>
  <style>{css}</style>
</head>
<body>
  <a class="skip-link" href="#funnel">Skip to the cascade</a>
  <header class="topbar">
    <div class="brand">
      <span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i></span>
      <span class="brand-text">
        <strong>MolCascade</strong>
        <small>Offline screening cascade builder</small>
      </span>
    </div>
    <div class="topbar-fields">
      <label class="field inline" for="cascade-name">
        <span>Cascade name</span>
        <input id="cascade-name" type="text" autocomplete="off" spellcheck="false">
      </label>
      <label class="field inline narrow" for="target-count">
        <span>Shortlist target</span>
        <input id="target-count" type="number" min="1" step="1" inputmode="numeric">
      </label>
    </div>
    <div class="topbar-actions">
      <button id="action-reset" class="button ghost" type="button">Restore default</button>
      <button id="action-load" class="button ghost" type="button">Load config…</button>
      <button id="action-review" class="button ghost" type="button">Review plan</button>
      <button id="action-download" class="button primary" type="button">Download config</button>
      <input id="config-file" type="file" accept=".json,application/json" hidden>
    </div>
  </header>

  <p id="status-bar" class="status-bar" role="status" aria-live="polite"></p>

  <main class="workshop">
    <section class="tray-column" aria-labelledby="tray-heading">
      <div class="column-head">
        <h2 id="tray-heading">Block tray</h2>
        <p>Drag a block into a tier — or select a tier and press <kbd>Enter</kbd> on a block.</p>
      </div>
      <label class="field" for="tray-search">
        <span>Find a block</span>
        <input id="tray-search" type="search" autocomplete="off"
               placeholder="Property, liability, tool name…">
      </label>
      <div id="tray" class="tray"></div>
      <div id="machine" class="machine"></div>
    </section>

    <section class="canvas-column" aria-labelledby="funnel-heading">
      <div class="column-head">
        <h2 id="funnel-heading">Screening cascade</h2>
        <p>Molecules fall from the top. Every block drawn here becomes one stage of the run.</p>
      </div>
      <div id="funnel" class="funnel" tabindex="-1"></div>
    </section>

    <aside class="inspector-column" aria-labelledby="inspector-heading">
      <div class="column-head">
        <h2 id="inspector-heading">Details</h2>
        <p id="inspector-hint">Select a tier or block to edit its thresholds.</p>
      </div>
      <div id="inspector" class="inspector"></div>
    </aside>
  </main>

  <dialog id="picker" class="dialog" aria-labelledby="picker-title">
    <form method="dialog" class="dialog-inner">
      <header class="dialog-head">
        <h2 id="picker-title">Add a screening criterion</h2>
        <button class="button ghost" value="cancel" type="submit">Close</button>
      </header>
      <label class="field" for="picker-search">
        <span>Search</span>
        <input id="picker-search" type="search" autocomplete="off"
               placeholder="Property, liability, tool name…">
      </label>
      <div id="picker-list" class="picker-list"></div>
    </form>
  </dialog>

  <dialog id="review" class="dialog wide" aria-labelledby="review-title">
    <form method="dialog" class="dialog-inner">
      <header class="dialog-head">
        <h2 id="review-title">What this file will run</h2>
        <button class="button ghost" value="cancel" type="submit">Close</button>
      </header>
      <p class="dialog-note">
        Every block on the canvas becomes one stage below, in this order. Run it with:
        <code id="review-command"></code>
      </p>
      <ol id="review-plan" class="plan-list"></ol>
      <pre id="review-body" class="review-body" tabindex="0"></pre>
    </form>
  </dialog>

  <textarea id="builder-payload" hidden aria-hidden="true">{escaped_payload}</textarea>
  <script>{javascript}</script>
</body>
</html>
"""


def generate_cascade_builder(
    output: str | Path,
    *,
    plugins: PluginRegistry | None = None,
    environment: HostEnvironment | None = None,
    overwrite: bool = True,
) -> Path:
    """Atomically write the standalone builder produced by ``generate config``."""

    destination = Path(output).expanduser()
    if destination.suffix.casefold() not in {".html", ".htm"}:
        raise ValueError("configuration builder output must end in .html or .htm")
    payload = build_cascade_builder_payload(plugins=plugins, environment=environment)
    document = render_cascade_builder(payload).encode("utf-8")
    return atomic_write_bytes(destination, document, overwrite=overwrite)


__all__ = [
    "BUILDER_SCHEMA_VERSION",
    "build_cascade_builder_payload",
    "generate_cascade_builder",
    "render_cascade_builder",
]
