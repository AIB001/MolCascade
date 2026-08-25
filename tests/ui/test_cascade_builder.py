"""The tier-first builder, checked end to end rather than by inspection.

A configuration builder is only worth anything if the file it writes runs.  So
the central test here does not assert on markup: it drives the shipped
``cascade.js`` through a scripted session in Node, takes whatever JSON the
Download button would have written, and pushes that byte-for-byte through
``parse_cascade_config`` -> ``lower_cascade`` -> ``PipelineCompiler``.  A
browser change that produces a plausible-looking but unexecutable cascade fails
here instead of failing on a user's first run.

Node is optional.  Where it is missing the driven tests *skip*, and the
pure-Python assertions still run.  They used to return early instead, which
reported a pass for the guarantee they exist to check -- on a machine with no
Node the suite would go green while never once comparing the drawn funnel to
the executed one.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from fixtures.offline_html import assert_offline

from molcascade.cascade import (
    CASCADE_SCHEMA_VERSION,
    CriterionConfig,
    lower_cascade,
    parse_cascade_config,
)
from molcascade.cascade.catalog import CRITERIA_BY_ID
from molcascade.pipeline import PipelineCompiler
from molcascade.plugins import create_builtin_registry
from molcascade.ui.cascade_builder import (
    BUILDER_SCHEMA_VERSION,
    DEFAULT_CONFIG_FILENAME,
    build_cascade_builder_payload,
    generate_cascade_builder,
    render_cascade_builder,
)

_UI = Path(__file__).parents[2] / "src" / "molcascade" / "ui" / "assets"
_HERE = Path(__file__).parent
_ID_PATTERN = re.compile(r'\sid="([a-zA-Z0-9_-]+)"')


def _compile(config: Any) -> Any:
    """Validate, lower and compile exactly what the browser handed back."""

    registry = create_builtin_registry()
    cascade = parse_cascade_config(json.dumps(config))
    lowered = lower_cascade(cascade, registry=registry, allow_missing_library=True)
    return cascade, lowered, PipelineCompiler(registry).compile(lowered.pipeline)


def _drive(tmp_path: Path) -> dict[str, Any]:
    """Run the scripted browser session, or skip if Node is not installed.

    Skipping rather than returning ``None`` for the caller to shrug off.  These
    are the tests that hold the drawn funnel against the executed one, and a
    machine without Node used to run them to a silent pass -- reporting green
    for the one guarantee that, if it broke, would let the builder draw one
    cascade and export another.  An honest skip says the check did not happen.
    """

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the scripted browser session cannot run")
    tmp_path.mkdir(parents=True, exist_ok=True)
    payload = build_cascade_builder_payload()
    document = render_cascade_builder(payload)
    ids = sorted(set(_ID_PATTERN.findall(document)))
    ids_path = tmp_path / "ids.json"
    payload_path = tmp_path / "payload.json"
    ids_path.write_text(json.dumps(ids), encoding="utf-8")
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    completed = subprocess.run(
        [
            node,
            str(_HERE / "cascade_driver.js"),
            str(_HERE / "cascade_dom_stub.js"),
            str(_UI / "cascade.js"),
            str(ids_path),
            str(payload_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_payload_ships_one_reviewed_default_per_criterion() -> None:
    payload = build_cascade_builder_payload()
    assert payload["builder_schema_version"] == BUILDER_SCHEMA_VERSION
    assert payload["default_filename"] == DEFAULT_CONFIG_FILENAME
    assert payload["join_plugin_available"] is True

    for spec in payload["criteria"]:
        options = spec["options"]
        assert options, f"{spec['id']} offers no tool at all"
        executable = [option for option in options if option["executable"]]
        assert spec["has_executable_option"] is bool(executable)
        # The browser copies ``default_option_id`` rather than re-deriving a
        # preference, so a criterion that can run must name the tool it starts
        # with, and one that cannot must name nothing at all.
        if executable:
            assert spec["default_option_id"] in {option["id"] for option in executable}
        else:
            assert spec["default_option_id"] is None
        for option in options:
            if not option["executable"]:
                # Unavailable tools stay visible with a reason.  Hiding them
                # would make an installation look like it never offered the
                # capability, rather than like it is missing an adapter.
                assert option["notes"]
                assert option["initial_settings"] == {}


def test_default_cascade_in_the_payload_compiles_unchanged() -> None:
    payload = build_cascade_builder_payload()
    config = payload["default_cascade"]
    assert config["kind"] == "cascade"
    assert config["schema_version"] == CASCADE_SCHEMA_VERSION
    cascade, lowered, compiled = _compile(config)
    assert compiled.stages[0].plugin_key.startswith("source.")
    assert compiled.stages[-1].plugin_key == "export.native_smiles_shortlist@0.1.0"
    # Every criterion the starter ships must trace back to a catalogue entry;
    # a stale default would otherwise reach users as an unexplainable stage.
    for tier in cascade.tiers:
        for criterion in tier.criteria:
            assert criterion.criterion in CRITERIA_BY_ID
            assert lowered.origin_of(criterion.id) is not None


def test_document_is_a_single_offline_file_pinned_by_hash() -> None:
    document = render_cascade_builder(build_cascade_builder_payload())
    style = re.search(r"<style>(.*?)</style>", document, re.DOTALL)
    script = re.search(r"<script>(.*?)</script>", document, re.DOTALL)
    csp = re.search(
        r'<meta http-equiv="Content-Security-Policy" content="([^"]+)">', document
    )
    assert style and script and csp
    decoded = html.unescape(csp.group(1))
    for content in (style.group(1), script.group(1)):
        digest = base64.b64encode(hashlib.sha256(content.encode()).digest()).decode()
        assert f"sha256-{digest}" in decoded
    assert "'unsafe-inline'" not in decoded
    assert "connect-src 'none'" in decoded
    assert "default-src 'none'" in decoded
    # Nothing may be fetched when the file is opened, and no handler may be
    # written as an attribute, because neither would survive the CSP above.
    assert_offline(document)
    assert not re.search(r"\son(click|change|input|load)=", document)


def test_generated_file_round_trips_through_disk(tmp_path: Path) -> None:
    destination = generate_cascade_builder(tmp_path / "builder.html")
    document = destination.read_text(encoding="utf-8")
    assert destination.exists() and document.startswith("<!doctype html>")
    embedded = re.search(
        r'<textarea id="builder-payload"[^>]*>(.*?)</textarea>', document, re.DOTALL
    )
    assert embedded
    payload = json.loads(html.unescape(embedded.group(1)))
    _compile(payload["default_cascade"])

    try:
        generate_cascade_builder(tmp_path / "builder.txt")
    except ValueError as error:
        assert "html" in str(error)
    else:  # pragma: no cover - the guard above is the behaviour under test
        raise AssertionError("a non-HTML suffix must be refused")


def test_a_scripted_browser_session_exports_an_executable_cascade(
    tmp_path: Path,
) -> None:
    steps = _drive(tmp_path)

    # 1. It boots into a working cascade, not an empty canvas.
    boot = steps["boot"]
    assert boot["downloadDisabled"] is False
    assert len(boot["shape"]) >= 3
    assert all(tier["criteria"] for tier in boot["shape"])
    # The tray is the surface the whole redesign is built around, so an empty
    # one is a broken builder even when every other assertion here passes.
    assert boot["trayBricks"] >= 5
    # The builder is written on one machine and usually run on another, so it
    # says which machine it was written on rather than leaving that implicit.
    assert "Where this file was written" in boot["machineText"]

    # 2. The complaint that started the redesign: moving a criterion between
    #    tiers.  One click on the block's own arrow, for anyone not dragging.
    before = steps["boot"]["shape"]
    after = steps["moveAcrossTier"]["shape"]
    moved = before[1]["criteria"][0]
    assert moved not in after[1]["criteria"]
    assert after[0]["criteria"][-1] == moved
    assert steps["moveAcrossTier"]["status"]["className"] == "status-bar ok"

    # 2b. And the gesture the redesign is named for: a brick dragged out of the
    #     tray lands in the tier it was dropped on, at the end of it.
    tray_drop = steps["dragFromTray"]
    assert tray_drop["status"]["className"] == "status-bar ok"
    assert tray_drop["landed"][-1] == tray_drop["spec"]
    landed_id = tray_drop["shape"][-1]["criteria"][-1]

    # 2c. Dragging a block that already exists moves it, rather than copying it,
    #     and honours the slot it was dropped into instead of appending.
    hop = steps["dragBetweenTiers"]
    assert hop["moved"] == landed_id
    hopped = {tier["id"]: tier["criteria"] for tier in hop["shape"]}
    assert hop["moved"] not in hopped[hop["from"]]
    assert hopped[hop["to"]][0] == hop["moved"]
    assert sum(tier["criteria"].count(hop["moved"]) for tier in hop["shape"]) == 1

    # 3. A tier's mode and its consensus threshold are edited in place.
    tier = steps["consensus"]["tier"]
    assert tier["mode"] == "at_least"
    # Seeded as a strict majority, never as a K that would behave like "any".
    assert tier["minimum_passes"] == 2

    # 4. Tools are chosen from a catalogue, and ones this installation cannot
    #    run are offered disabled rather than hidden.
    offered = steps["picker"]["offered"]
    assert steps["picker"]["open"] is True
    assert any(entry["disabled"] for entry in offered)
    assert any(not entry["disabled"] for entry in offered)
    assert steps["added"]["status"]["className"] == "status-bar ok"

    # 5. Thresholds are labelled in the tool's own units.
    assert "Maximum SA score" in steps["thresholdLabels"]

    # 6. An unsatisfiable consensus blocks export and is never silently
    #    repaired into something that would have run but meant something else.
    unsatisfiable = steps["unsatisfiable"]
    assert unsatisfiable["downloadDisabled"] is True
    assert unsatisfiable["status"]["className"] == "status-bar error"
    assert "criterion is enabled" in unsatisfiable["status"]["text"]
    assert unsatisfiable["minimumPasses"] == 2
    # A disabled block stays visible -- turning it off should not make it hard
    # to find again -- but it must stop claiming a stage the run will not have.
    for criterion_id in unsatisfiable["disabledCriteria"]:
        assert criterion_id not in unsatisfiable["canvasStages"]

    # 7. What downloads is exactly what was reviewed, byte for byte.
    assert steps["download"] == {"count": 1, "matches": True}

    # 8. And it runs.  This is the assertion the whole file exists for.
    config = steps["final"]["config"]
    assert steps["final"]["downloadDisabled"] is False
    cascade, lowered, compiled = _compile(config)

    edited = cascade.tiers[0]
    assert edited.mode.value == "at_least"
    assert edited.minimum_passes == 2

    keys = [stage.plugin_key for stage in compiled.stages]
    ids = [stage.stage_id for stage in compiled.stages]
    roles = {
        stage.stage_id: origin.role
        for stage in compiled.stages
        if (origin := lowered.origin_of(stage.stage_id)) is not None
    }

    # A parallel tier of several criteria must lower to a policy join; without
    # it the tier would silently degrade into an unordered series.
    assert "policy.native_decision_join@0.1.0" in keys
    assert roles[f"{edited.id}__policy"] == "policy"

    # The criterion added through the picker survives with an id unique across
    # the whole cascade, its own score producer, and its own threshold gate.
    added: CriterionConfig = edited.criteria[-1]
    every_id = [
        criterion.id for tier_ in cascade.tiers for criterion in tier_.criteria
    ]
    assert every_id.count(added.id) == 1
    assert added.id in ids
    assert f"{added.id}__gate" in ids
    assert roles[added.id] == "criterion"
    assert roles[f"{added.id}__gate"] == "threshold"

    # The threshold typed into the browser is the one the run will apply.
    assert added.gate is not None
    assert added.gate.settings["maximum"] == 3.25

    # Two independent criteria sharing one catalogue entry keep separate
    # settings, so editing one cannot move the other's threshold.  The twin is
    # checked against the value it was seeded with rather than merely against
    # the typed one: a default that happened to coincide with what the driver
    # types would otherwise make this pass for the wrong reason.
    twin = next(
        criterion
        for tier_ in cascade.tiers
        for criterion in tier_.criteria
        if criterion.criterion == added.criterion and criterion.id != added.id
    )
    seeded = parse_cascade_config(
        json.dumps(build_cascade_builder_payload()["default_cascade"])
    )
    original = next(
        criterion
        for tier_ in seeded.tiers
        for criterion in tier_.criteria
        if criterion.id == twin.id
    )
    assert twin.gate is not None and original.gate is not None
    assert twin.gate.settings["maximum"] == original.gate.settings["maximum"]
    assert twin.gate.settings["maximum"] != added.gate.settings["maximum"]

    # 7. A block whose configuration cannot be completed in this file says so
    #    on the tier that holds it.  The receptor belongs to the campaign, not
    #    to the screening policy, so the builder's job is to make the debt
    #    visible now rather than to let the run discover it later.
    needs = steps["runTimeNeeds"]
    assert needs["offeredDisabled"] is False
    assert needs["before"] == []
    assert len(needs["after"]) == 1
    line = needs["after"][0]
    assert "At run time" in line
    assert "--receptor" in line
    # Naming only the receptor would be half the requirement: the engines also
    # need to be told where on the protein to look.
    assert "--reference-ligand" in line and "--pocket" in line and "--box" in line
    # Turning the block off withdraws the demand, because a tier that compiles
    # into no stage is not what the preflight will be looking at.
    assert needs["whenTurnedOff"] == []
    # And it is a property of the blocks in that tier, not decoration on every
    # tier card.
    assert all(other == [] for other in needs["elsewhere"])


def test_the_canvas_shows_exactly_the_stages_that_will_run(tmp_path: Path) -> None:
    """The blocks on screen are the run, in order -- not an illustration of it.

    A builder that draws one funnel and exports another is worse than no builder
    at all: the user reasons about the picture and the machine executes the
    file.  So the browser predicts its own stage list, renders the canvas and the
    review dialog from that single prediction, and this test holds the
    prediction against what ``lower_cascade`` and the compiler really produce.

    Both directions matter.  Equality as *lists* catches a stage that the canvas
    omits (a join silently dropped), a stage it invents (a gate drawn for a tool
    that has none), and a stage it merely misplaces (a criterion shown above the
    tier that will actually run first).
    """

    steps = _drive(tmp_path)

    final = steps["final"]
    _, lowered, compiled = _compile(final["config"])

    lowered_ids = [stage.id for stage in lowered.pipeline.stages]
    assert final["canvasStages"] == lowered_ids
    # The review dialog is what a user reads before committing, so it is checked
    # separately rather than trusted to agree with the canvas by construction.
    assert final["reviewPlan"] == lowered_ids
    # And the compiler must not reorder what lowering laid out, or the promise
    # would hold right up to the moment it stopped mattering.
    assert [stage.stage_id for stage in compiled.stages] == lowered_ids

    # The same guarantee has to survive the boot state, which is the only
    # cascade most users will ever look at without editing it.
    _, boot_lowered, _ = _compile(build_cascade_builder_payload()["default_cascade"])
    assert steps["boot"]["canvasStages"] == [
        stage.id for stage in boot_lowered.pipeline.stages
    ]


def test_the_same_session_is_reproducible(tmp_path: Path) -> None:
    """Two runs of the builder must produce the same file and the same run id.

    Provenance is only meaningful if authoring is deterministic: the same
    clicks have to yield the same revision, or a rerun would look like a
    different experiment.
    """

    first = _drive(tmp_path / "a")
    second = _drive(tmp_path / "b")
    assert first["final"]["config"] == second["final"]["config"]
    _, _, left = _compile(first["final"]["config"])
    _, _, right = _compile(second["final"]["config"])
    assert left.revision_id == right.revision_id
