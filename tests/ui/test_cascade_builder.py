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
from fixtures.starter_cascade import starter_without_docking
from pydantic import ValidationError

from molcascade.backends.preflight import preflight_backends, preflight_engine_paths
from molcascade.cascade import (
    CASCADE_SCHEMA_VERSION,
    CriterionConfig,
    lower_cascade,
    parse_cascade_config,
)
from molcascade.cascade.catalog import CRITERIA_BY_ID
from molcascade.contracts import DECISION_V1, PARENT_V1
from molcascade.errors import ConfigError, PipelineError
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

#: Every per-machine engine path the catalogue lets an environment variable
#: answer.  Read from the catalogue rather than listed, so a new engine cannot be
#: added with a variable the scripted session then silently inherits.
_ENGINE_VARIABLES = tuple(
    sorted(
        {
            threshold.environment_variable
            for spec in CRITERIA_BY_ID.values()
            for option in spec.options
            for threshold in option.thresholds
            if threshold.environment_variable
        }
    )
)


def _compile(config: Any, *, allow_missing_target: bool = False) -> Any:
    """Validate, lower and compile exactly what the browser handed back.

    ``allow_missing_target`` defaults to *off* even though the builder's own
    export always lacks a target, because the tests below need both answers:
    one set asks whether the shipped cascade is well formed, and the other
    asks whether a run that never names a receptor is refused.  Turning the
    placeholder on everywhere would silently delete the second question.
    """

    registry = create_builtin_registry()
    cascade = parse_cascade_config(json.dumps(config))
    lowered = lower_cascade(
        cascade,
        registry=registry,
        allow_missing_library=True,
        allow_missing_target=allow_missing_target,
    )
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
    # The engine variables are cleared even though the payload no longer reads
    # them, so that this session keeps failing if one is ever consulted again.
    # It used to be, and a developer with KarmaDock configured then drove a
    # different browser from one without it -- the difference being whether the
    # download button worked at all.
    with pytest.MonkeyPatch.context() as clean:
        for variable in _ENGINE_VARIABLES:
            clean.delenv(variable, raising=False)
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
    cascade, lowered, compiled = _compile(config, allow_missing_target=True)
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
    _compile(payload["default_cascade"], allow_missing_target=True)

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
    cascade, lowered, compiled = _compile(config, allow_missing_target=True)

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

    # 7a. Docking ships enabled, so the scripted session removes it before the
    #     next step can say anything about *adding* one.  Asserting on that
    #     removal is what keeps step 7 honest: if the starter ever stops
    #     shipping docking, this fails instead of letting the rest of the
    #     section quietly re-observe the defaults.  The tier that feeds it goes
    #     too, since a producer left behind would answer the dependency that
    #     step 9 exists to watch being provisioned.
    shipped = steps["shippedDocking"]
    assert len(shipped["removed"]) == 2
    assert all(tier not in shipped["remaining"] for tier in shipped["removed"])
    # Removing the pair leaves a cascade that still exports; a starter whose
    # docking cannot be taken out again is one an operator is stuck with.
    assert shipped["downloadDisabled"] is False
    assert shipped["status"]["className"] != "status-bar error"

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

    # 8. Where the engine is installed is the one question this page cannot
    #    answer, and the export no longer waits for it.  It used to: a blank
    #    engine path disabled Download and told the operator to export a
    #    variable "on the machine that will run this cascade" -- which is not
    #    the machine drawing the page, so the advice could not be taken and the
    #    builder refused to produce anything at all.  Now the file leaves with
    #    the field blank and the run host is what settles it.
    paths = steps["installationPaths"]
    assert paths["blankDownloadDisabled"] is False
    assert paths["blankStatus"]["className"] != "status-bar error"
    assert paths["label"] is not None
    # Shown as deferred rather than as missing, and the variable is still named
    # for the case the search cannot cover -- an engine installed somewhere odd.
    assert paths["placeholder"] == "found when the run starts"
    assert paths["variable"] in _ENGINE_VARIABLES
    assert "machine that runs the cascade" in paths["help"]
    # A file really did leave the browser with no installation in it, and it is
    # a cascade the compiler accepts.
    assert paths["downloadsSoFar"] == steps["download"]["count"] + 1
    blank = json.loads(paths["blankDownload"])
    docked = [
        criterion
        for tier in blank["tiers"]
        for criterion in tier["criteria"]
        if criterion["criterion"] == "docking_score"
    ]
    assert len(docked) == 1
    assert docked[0]["settings"]["executable"] == ""
    # Compiled the way ``molcascade validate`` compiles it -- the receptor is
    # still owed by the command line, which is what step 7 said on the tier.
    registry = create_builtin_registry()
    lowered = lower_cascade(
        parse_cascade_config(paths["blankDownload"]),
        registry=registry,
        allow_missing_library=True,
        allow_missing_target=True,
    )
    compiled = PipelineCompiler(registry).compile(lowered.pipeline)
    # And the refusal is not lost, only moved: the host about to do the work is
    # where a blank path stops being an open question.
    with pytest.raises(ConfigError) as refused:
        preflight_engine_paths(compiled.stages, registry=registry)
    message = str(refused.value)
    assert paths["variable"] in message
    assert "was not found on this machine" in message
    # Filling it in is still allowed and still lands in the settings, because an
    # engine somewhere unusual is exactly what the field is for.
    assert paths["filledDownloadDisabled"] is False
    assert paths["filledStatus"]["className"] != "status-bar error"
    assert paths["filledSettings"]["executable"] == "/opt/envs/karmadock/bin/python"

    # 9. Some engines read evidence they do not compute -- Uni-Dock and GNINA
    #    dock a structure and generate none -- and nothing on the page tells an
    #    operator which ones.  Adding one used to export a file `lower_cascade`
    #    refused outright with CASCADE_CRITERION_EVIDENCE_UNAVAILABLE, which the
    #    blank-path check above would now have let through: the dependency has to
    #    be provisioned when the block is added, not diagnosed an hour later.
    evidence = steps["evidence"]
    assert len(evidence["inserted"]) == 1
    producer = evidence["insertedTiers"][0]
    consumer = evidence["dockTier"]
    # Directly above the tier that reads it, and serial: a pure evidence producer
    # has no threshold, and a parallel tier demands a decision from everything in
    # it -- so the placement is what keeps the cascade compilable, not cosmetic.
    order = evidence["tiersAfter"]
    assert order.index(producer["id"]) == order.index(consumer["id"]) - 1
    assert producer["mode"] == "serial"
    assert [item["criterion"] for item in producer["criteria"]] == ["ligand_conformers"]
    assert producer["criteria"][0]["gate"] is None
    # Said out loud, because a tier appearing on its own is otherwise indistinguishable
    # from a misclick.
    assert "reads evidence it does not compute" in evidence["addStatus"]["text"]
    assert evidence["addStatus"]["className"] != "status-bar error"
    assert evidence["downloadDisabled"] is False
    # And it really is in the exported file, above the block that needs it: this
    # is the same download step 8 pushed through the compiler.
    order_in_file = [tier["id"] for tier in blank["tiers"]]
    assert order_in_file.index(producer["id"]) < order_in_file.index(consumer["id"])

    # 10. It is an ordinary block afterwards -- movable, editable, deletable --
    #     and a cascade loaded from disk can arrive without it, so deleting it
    #     blocks the export rather than quietly restoring the old failure mode.
    withdrawn = steps["evidenceWithdrawn"]
    assert withdrawn["removed"] == producer["id"]
    assert producer["id"] not in withdrawn["tiers"]
    assert withdrawn["downloadDisabled"] is True
    assert withdrawn["status"]["className"] == "status-bar error"
    assert "no enabled block above it produces it" in withdrawn["status"]["text"]
    # Naming the block that would fix it, since the contract id means nothing to
    # someone who did not write the plugin.
    assert producer["title"] in withdrawn["status"]["text"]


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
    _, lowered, compiled = _compile(final["config"], allow_missing_target=True)

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
    _, boot_lowered, _ = _compile(
        build_cascade_builder_payload()["default_cascade"], allow_missing_target=True
    )
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
    _, _, left = _compile(first["final"]["config"], allow_missing_target=True)
    _, _, right = _compile(second["final"]["config"], allow_missing_target=True)
    assert left.revision_id == right.revision_id


def _text_thresholds(payload: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (spec["id"], option["id"], threshold)
        for spec in payload["criteria"]
        for option in spec["options"]
        for threshold in option["thresholds"]
        if threshold["kind"] == "text"
    ]


def test_every_text_setting_the_run_needs_is_one_the_browser_will_demand() -> None:
    """The rule the browser applies has to cover the fields that actually matter.

    ``collectIssues`` treats a text field as required when it is ``required`` or
    not ``nullable``, and then exempts the installation paths -- which the run
    resolves for itself.  That is only a useful rule if what survives both tests
    is the set of fields whose absence stops a run and that nobody but the author
    can supply, so this holds the flags against the config models, which are the
    things that will refuse.
    """

    payload = build_cascade_builder_payload()
    demanded = {
        (criterion, option, threshold["name"])
        for criterion, option, threshold in _text_thresholds(payload)
        if (threshold["required"] or not threshold["nullable"]) and not threshold["installed_path"]
    }
    # The lead set, which is nullable only because the same plugin will take
    # references embedded in the cascade -- an alternative the builder does not
    # offer, so leaving the box empty here really does mean an unrunnable stage.
    reference = [
        (criterion, option)
        for criterion, option, threshold in _text_thresholds(payload)
        if threshold["name"] == "reference_path"
    ]
    assert reference, "the lead-similarity criteria should still offer the field"
    for criterion, option in reference:
        assert (criterion, option, "reference_path") in demanded
    # Optional overrides stay optional: demanding all of them would make the
    # rule useless by making every docking block un-exportable.
    assert ("docking_score", "karmadock", "weights_path") not in demanded
    assert ("docking_score", "karmadock", "pocket_pdb_path") not in demanded
    # And no engine path is demanded, which is the point of the exemption: the
    # browser is almost never open on the machine that has the engine.
    assert ("docking_score", "karmadock", "executable") not in demanded
    assert ("docking_score", "karmadock", "repo_path") not in demanded
    assert ("docking_cnn_rescore", "gnina", "executable") not in demanded
    assert ("docking_score", "unidock", "executable") not in demanded
    assert ("retrosynthesis_routes", "aizynthfinder", "executable") not in demanded


def _plausible(name: str) -> str:
    """Something the field's own validator would accept, for the given name."""

    if "sha256" in name:
        return "0" * 64
    if name.endswith(("_path", "_dir")) or name == "executable":
        return "/opt/example/" + name
    return "example"


def _complaints(model: Any, data: dict[str, Any]) -> frozenset[tuple[str, str]]:
    """What the config model objects to, as ``(location, error type)`` pairs.

    A model-level refusal -- one with no field in its location, which is how
    ``ReferenceSimilarityConfig`` reports a criterion with no reference set at
    all -- is kept under ``<model>`` rather than dropped, because for that field
    it is the entire objection.
    """

    try:
        model.model_validate(data)
    except ValidationError as exc:
        return frozenset(
            (".".join(str(part) for part in error["loc"]) or "<model>", error["type"])
            for error in exc.errors()
        )
    return frozenset()


def test_nothing_the_browser_demands_is_a_field_the_plugin_would_have_accepted_blank() -> None:
    """The other half of the rule: it must not block an export that would have run.

    A builder that refuses to download is worse than one that exports something
    broken if the thing it refuses over was fine -- there is no way past it. So
    every field ``collectIssues`` insists on is checked against the model that
    will receive it: blanking the field has to change what that model objects
    to.  A model indifferent to the field would produce the same complaint
    either way, and this would fail.

    Resolving the model through ``target`` also pins the routing.  A gate field
    looked up in the criterion's own settings would be reported missing forever,
    since filling the box writes it to the other bag -- an error the operator
    could not clear by doing what it asked.

    Installation paths are skipped for the same reason the browser skips them,
    and the models agree: a blank engine path now validates cleanly and is
    settled by ``preflight_engine_paths`` on the host that has the engine, so
    there is nothing for a refusal here to protect.
    """

    registry = create_builtin_registry()
    payload = build_cascade_builder_payload()
    checked = 0
    for spec in payload["criteria"]:
        criterion = CRITERIA_BY_ID[spec["id"]]
        for option in spec["options"]:
            catalogued = next(item for item in criterion.options if item.id == option["id"])
            for threshold in option["thresholds"]:
                if threshold["kind"] != "text":
                    continue
                if not (threshold["required"] or not threshold["nullable"]):
                    continue
                if threshold["installed_path"]:
                    continue
                gated = threshold["target"] == "gate"
                plugin_ref = catalogued.gate_plugin if gated else catalogued.plugin_ref
                assert plugin_ref is not None
                model = registry.get(plugin_ref).config_model
                seed = dict(
                    option["initial_gate_settings"] if gated else option["initial_settings"]
                )
                name = threshold["name"]
                blank = _complaints(model, {**seed, name: ""})
                filled = _complaints(model, {**seed, name: _plausible(name)})
                assert blank != filled, (
                    f"{spec['id']}/{option['id']} demands '{name}', but "
                    f"{plugin_ref} objects to exactly the same things with it "
                    f"filled in -- either the flags are wrong or the target is"
                )
                checked += 1
    # Both lead sets, both ONNX bundles with their digests and endpoint ids, and
    # the SCScore checkpoint with its digest.  The six engine paths are not here
    # by design; ``test_every_text_setting_the_run_needs_is_one_the_browser_will
    # _demand`` is what holds them out of the demanded set.
    assert checked >= 10


def test_the_generated_html_says_nothing_about_the_machine_that_generated_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The payload is frozen into a static file, so it must not measure this host.

    It used to.  Each installation path carried a flag saying whether *this*
    shell exported the matching variable, and the browser skipped its demand
    when it did -- which meant the same clicks produced a downloadable cascade on
    a workstation with the engines and an undownloadable one on a laptop without
    them.  The flag is now declared by the catalogue, so both produce the same
    file and the run host answers the question.

    The path itself still never travels, which is the original reason for the
    variable: the payload is embedded in HTML the user is invited to share.
    """

    for variable in _ENGINE_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    before = build_cascade_builder_payload()

    secret = "/opt/some-operators-private-path/karmadock/bin/python"
    monkeypatch.setenv("MOLCASCADE_KARMADOCK_EXECUTABLE", secret)
    after = build_cascade_builder_payload()

    assert before == after
    assert secret not in json.dumps(after)
    assert secret not in render_cascade_builder(after)


def test_an_installation_path_is_flagged_as_one_and_a_screening_setting_is_not() -> None:
    """The flag the browser branches on has to select exactly the machine facts.

    Too wide and the builder stops asking for a threshold the run genuinely
    cannot invent; too narrow and it blocks a download over a path only the run
    host can know.
    """

    payload = build_cascade_builder_payload()
    installed = {
        (criterion, option, threshold["name"])
        for criterion, option, threshold in _text_thresholds(payload)
        if threshold["installed_path"]
    }
    assert ("docking_score", "karmadock", "executable") in installed
    assert ("docking_score", "karmadock", "repo_path") in installed
    assert ("docking_score", "unidock", "executable") in installed
    assert ("docking_cnn_rescore", "gnina", "executable") in installed
    assert ("retrosynthesis_routes", "aizynthfinder", "executable") in installed
    assert ("retrosynthesis_routes", "aizynthfinder", "config_path") in installed
    # A lead set is the campaign, not the machine: nobody can find it by looking.
    assert not any(name == "reference_path" for _, _, name in installed)
    for _, _, threshold in _text_thresholds(payload):
        assert threshold["installed_path"] == bool(threshold["environment_variable"])


def test_every_advertised_variable_is_one_the_engine_actually_reads() -> None:
    """A name in the builder that the config ignores is worse than no name.

    It would tell an operator to export something, and then the run would still
    refuse -- with the field they were told they could leave blank still blank.
    """

    from molcascade.plugins.builtin.aizynthfinder import AiZynthFinderConfig
    from molcascade.plugins.builtin.docking.gnina import GninaConfig
    from molcascade.plugins.builtin.docking.karmadock import KarmaDockConfig
    from molcascade.plugins.builtin.docking.unidock import UniDockConfig

    honoured = {
        spec.variable(model.engine_id)
        for model in (KarmaDockConfig, GninaConfig, UniDockConfig, AiZynthFinderConfig)
        for spec in model.installed_paths()
    }
    payload = build_cascade_builder_payload()
    advertised = {
        threshold["environment_variable"]
        for _, _, threshold in _text_thresholds(payload)
        if threshold["environment_variable"]
    }
    assert advertised
    assert advertised <= honoured


def _with_criterion(
    payload: dict[str, Any],
    criterion_id: str,
    option_id: str,
    settings: dict[str, Any],
    *,
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one catalogue option to the starter cascade, as the browser would.

    Seeded from ``initial_settings`` rather than hand-written, so this is the
    same starting point an operator gets -- engine paths included, which the
    builder exports blank when the machine already answers them.

    ``base`` replaces the starter for callers that need the block they add to be
    the only one of its kind in the file; see ``_starter_without_docking``.
    """

    spec = next(entry for entry in payload["criteria"] if entry["id"] == criterion_id)
    option = next(entry for entry in spec["options"] if entry["id"] == option_id)
    config = json.loads(json.dumps(payload["default_cascade"] if base is None else base))
    config["tiers"].append(
        {
            "id": f"t9_{option_id}",
            "title": spec["label"],
            "mode": "all",
            "criteria": [
                {
                    "id": option_id,
                    "criterion": criterion_id,
                    "backend": option["plugin_ref"],
                    "label": option["label"],
                    "settings": {**option["initial_settings"], **settings},
                    "gate": {
                        "backend": option["gate_plugin"],
                        "settings": dict(option["initial_gate_settings"]),
                    },
                    "enabled": True,
                }
            ],
            "enabled": True,
        }
    )
    return config


def test_a_cascade_that_names_no_installation_runs_where_the_machine_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exported file keeps the campaign and leaves the machine to the machine.

    This is the end of the round trip the variables exist for.  Validation only
    checks the shape of a path, so a config that compiles proves nothing on its
    own; what matters is that the filled value lands in the *compiled* stage
    config, because that is the mapping the preflight stats, ``method_id`` hashes
    and the provenance record reports.  A fallback that satisfied the validator
    and stopped there would leave a run that passed its checks and then could not
    find the engine.
    """

    for variable in _ENGINE_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    executable = tmp_path / "aizynthcli"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    az_config = tmp_path / "config.yml"
    az_config.write_text("expansion: {}\n", encoding="utf-8")

    payload = build_cascade_builder_payload()
    config = _with_criterion(payload, "retrosynthesis_routes", "aizynthfinder", {})
    stage = config["tiers"][-1]["criteria"][0]["settings"]
    assert stage["executable"] == "" and stage["config_path"] == ""

    # A blank installation path is not a malformed cascade, so it compiles: the
    # file is declining to state a fact about a machine it was not written on.
    # The refusal happens on the machine that can answer, one step further in.
    registry = create_builtin_registry()
    _, _, unresolved = _compile(config, allow_missing_target=True)
    with pytest.raises(ConfigError) as refused:
        preflight_engine_paths(unresolved.stages, registry=registry)
    message = str(refused.value)
    assert refused.value.code == "ENGINE_PATH_UNRESOLVED"
    # Both gaps in one refusal.  One at a time would turn setting a machine up
    # into an install-restart-install loop.
    assert "MOLCASCADE_AIZYNTHFINDER_EXECUTABLE" in message
    assert "MOLCASCADE_AIZYNTHFINDER_CONFIG_PATH" in message
    # Naming both ways out, since an operator who has not installed it yet
    # cannot act on a variable.  The installer command travels in the hint,
    # which the CLI prints under the message.
    assert "bash envs/bootstrap.sh aizynth" in (refused.value.hint or "")

    monkeypatch.setenv("MOLCASCADE_AIZYNTHFINDER_EXECUTABLE", str(executable))
    monkeypatch.setenv("MOLCASCADE_AIZYNTHFINDER_CONFIG_PATH", str(az_config))
    _, _, compiled = _compile(config, allow_missing_target=True)
    resolved = next(
        stage for stage in compiled.stages if stage.plugin_key.startswith("synthesis.aizynth")
    )
    assert resolved.config["executable"] == str(executable)
    assert resolved.config["config_path"] == str(az_config)
    # And both preflights, which read that same mapping, are satisfied by it.
    assert preflight_engine_paths((resolved,), registry=registry) == (resolved.stage_id,)
    assert preflight_backends((resolved,)) == ("aizynthfinder",)


def test_the_campaign_still_has_to_be_written_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receptor is the question the run is asking, not part of the toolchain.

    Two campaigns on one machine share an installation and share nothing else,
    so the fallback stops at the engine.  Naming the missing receptor is the
    whole of the remaining work for an operator whose paths are exported.
    """

    for variable in _ENGINE_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("MOLCASCADE_KARMADOCK_EXECUTABLE", str(tmp_path / "python"))
    monkeypatch.setenv("MOLCASCADE_KARMADOCK_REPO_PATH", str(tmp_path / "KarmaDock"))

    payload = build_cascade_builder_payload()
    # Built on a starter with the shipped docking taken out, so the refusal read
    # below is about the KarmaDock block this test added.  Left in, the shipped
    # Uni-Dock stage is validated first and the message names *its* unfilled
    # fields -- a different tool's debt, and the assertion on
    # ``reference_ligand_path`` would be checking a field KarmaDock alone has.
    bare = starter_without_docking(payload["default_cascade"])
    config = _with_criterion(payload, "docking_score", "karmadock", {}, base=bare)
    with pytest.raises(PipelineError) as refused:
        _compile(config)
    message = str(refused.value)
    assert "receptor_path" in message
    assert "reference_ligand_path" in message

    config = _with_criterion(
        payload,
        "docking_score",
        "karmadock",
        {
            "receptor_path": str(tmp_path / "receptor.pdb"),
            "reference_ligand_path": str(tmp_path / "ligand.mol2"),
        },
        base=bare,
    )
    _, _, compiled = _compile(config, allow_missing_target=True)
    docking = next(stage for stage in compiled.stages if stage.plugin_key.startswith("docking."))
    assert docking.config["executable"] == str(tmp_path / "python")
    assert docking.config["repo_path"] == str(tmp_path / "KarmaDock")


def _producer_tier(payload: dict[str, Any], contract: str) -> dict[str, Any] | None:
    """A tier holding whatever this installation uses to compute ``contract``.

    The same choice the browser makes, made the same way -- first executable
    option that declares the contract -- so a payload the browser could not
    satisfy is a payload this cannot satisfy either.
    """

    for spec in payload["criteria"]:
        for option in spec["options"]:
            if not option["executable"]:
                continue
            if contract not in option["produces_evidence"]:
                continue
            return {
                "id": f"t8_{option['id']}",
                "title": spec["label"],
                # Serial, because a pure evidence producer has no threshold and
                # a parallel tier needs a decision from everything in it.
                "mode": "serial",
                "criteria": [
                    {
                        "id": option["id"],
                        "criterion": spec["id"],
                        "backend": option["plugin_ref"],
                        "label": option["label"],
                        "settings": dict(option["initial_settings"]),
                        "gate": (
                            {
                                "backend": option["gate_plugin"],
                                "settings": dict(option["initial_gate_settings"]),
                            }
                            if option["gate_plugin"]
                            else None
                        ),
                        "enabled": True,
                    }
                ],
                "enabled": True,
            }
    return None


def test_a_tool_that_reads_evidence_it_does_not_compute_declares_it_and_can_be_fed() -> None:
    """The dependency an operator has no way to know about.

    Uni-Dock and GNINA dock a 3D structure and generate none; KarmaDock predicts
    its own geometry and needs nothing.  Nothing on a picker entry distinguishes
    them, so choosing the wrong one used to produce a file ``lower_cascade``
    refused outright -- the operator met a dependency of a plugin as a failure of
    a tool that had just written the cascade for them.

    Two halves are checked here, both pure Python.  The payload half: the need is
    read off the installed plugin rather than listed in the browser, and every
    contract it names is one something in this installation produces, so the page
    can always say which block to add.  And the half that says why it matters:
    the refusal is real, and prepending that producer is what settles it.
    """

    payload = build_cascade_builder_payload()
    registry = create_builtin_registry()
    consumers = [
        (spec, option)
        for spec in payload["criteria"]
        for option in spec["options"]
        if option["executable"] and option["needs_evidence"]
    ]
    # Not a vacuous pass: if no installed tool reads foreign evidence, the
    # provisioning in the browser is untested and this should be deleted, not
    # quietly reported green.
    assert consumers, "no installed option declares an evidence dependency"

    bare = starter_without_docking(payload["default_cascade"])
    for spec, option in consumers:
        descriptor = registry.entry(option["plugin_ref"]).descriptor
        needs = option["needs_evidence"]
        # Declared by the plugin, not by the builder, and stripped of the two
        # contracts every stage gets for free -- listing those would make every
        # tool look like it had a dependency.
        assert set(needs) <= set(descriptor.inputs)
        assert PARENT_V1.id not in needs and DECISION_V1.id not in needs

        # On a starter with the shipped docking tiers -- and the conformer tier
        # that feeds them -- taken out, because the refusal this asks for is
        # "nothing above produces this contract" and the starter now produces
        # it.  Testing the consumer against a cascade that already satisfies it
        # would report the provisioning as working without ever exercising it.
        config = _with_criterion(payload, spec["id"], option["id"], {}, base=bare)
        with pytest.raises(ConfigError) as refused:
            _compile(config)
        assert refused.value.code == "CASCADE_CRITERION_EVIDENCE_UNAVAILABLE"
        assert any(contract in str(refused.value) for contract in needs)

        for contract in needs:
            producer = _producer_tier(payload, contract)
            assert producer is not None, f"nothing here produces {contract}"
            config["tiers"].insert(len(config["tiers"]) - 1, producer)
        # Lowering only, not compiling: this cascade still owes a receptor, which
        # is the campaign's to name and is the subject of the test above.
        lowered = lower_cascade(
            parse_cascade_config(json.dumps(config)),
            registry=registry,
            allow_missing_library=True,
        )
        assert lowered.origin_of(option["id"]) is not None
