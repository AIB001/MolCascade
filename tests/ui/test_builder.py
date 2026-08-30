from __future__ import annotations

import base64
import copy
import hashlib
import html
import itertools
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Literal

import pytest
from fixtures.offline_html import assert_offline

from molcascade.config.models import PipelineConfig, StrictFrozenModel
from molcascade.contracts import DECISION_V1, PARENT_V1
from molcascade.pipeline import PipelineCompiler
from molcascade.plugins import create_builtin_registry
from molcascade.plugins.api import StageContext, StageRequest, StageResponse
from molcascade.plugins.manifest import (
    Cardinality,
    Determinism,
    PluginDescriptor,
    PluginKind,
)
from molcascade.ui.builder import (
    build_builder_payload,
    generate_config_builder,
    render_config_builder,
)


class _JoinConfig(StrictFrozenModel):
    mode: Literal["ANY_REJECT", "ALL_REJECT"] = "ANY_REJECT"


class _JoinPlugin:
    descriptor = PluginDescriptor(
        id="policy.native_decision_join",
        version="0.1.0",
        kind=PluginKind.GATE,
        inputs=(PARENT_V1.id, DECISION_V1.id),
        outputs=(PARENT_V1.id,),
        cardinality=Cardinality.FILTER,
        determinism=Determinism.DETERMINISTIC,
        display_name="Native decision policy join",
        description="Test-only executable join descriptor.",
    )
    config_model = _JoinConfig

    def execute(self, request: StageRequest, context: StageContext) -> StageResponse:
        raise AssertionError("builder tests never execute plugins")


def test_builder_payload_contains_strict_default_and_multiple_local_tools() -> None:
    payload = build_builder_payload()
    config = PipelineConfig.model_validate(payload["default_pipeline"])
    compiled = PipelineCompiler(create_builtin_registry()).compile(config)
    assert payload["builder_schema_version"] == 3
    assert config.stages[0].slot == "source"
    assert compiled.stages[-1].plugin_key == "export.native_smiles_shortlist@0.1.0"
    # Pinned against the compiled graph, not against False.  The default screen
    # has no docking tier today, so both sides are False -- but if one is ever
    # added to the defaults, this fails instead of quietly exporting provenance
    # that says no docking happened.
    assert config.metadata["docking_included"] is any(
        stage.descriptor.kind is PluginKind.DOCK for stage in compiled.stages
    )
    assert config.metadata["builder_view"] == "node_canvas/v2"
    layout = config.metadata["builder_layout"]
    assert layout["schema_version"] == 2
    assert layout["tier_modes"]["hard_gate"] == "parallel_all"
    assert set(layout["nodes"]) == {stage.id for stage in config.stages}
    assert all(
        isinstance(position[axis], (int, float))
        for position in layout["nodes"].values()
        for axis in ("x", "y")
    )
    assert {plugin["key"] for plugin in payload["plugins"]} >= {
        "source.delimited_smiles@0.1.0",
        "source.sdf@0.1.0",
        "features.rdkit_properties@0.1.0",
        "features.openbabel_properties@0.1.0",
        "source.xlsx@0.1.0",
        "source.mol2_directory@0.1.0",
        "chemistry.rdkit_property_range_gate@0.1.0",
        "chemistry.rdkit_drug_likeness@0.1.0",
        "chemistry.rdkit_structural_alerts@0.1.0",
        "policy.native_decision_join@0.1.0",
        "synthesis.rdkit_sa_score@0.1.0",
        "applicability.rdkit_reference_similarity@0.1.0",
        "scaffold.rdkit_murcko@0.1.0",
        "cluster.native_streaming_leader@0.1.0",
        "cluster.rdkit_scaffold_groups@0.1.0",
        "select.native_hash_budget@0.1.0",
        "select.native_scaffold_round_robin@0.1.0",
        "export.native_smiles_shortlist@0.1.0",
        "export.rdkit_sdf_shortlist@0.1.0",
    }
    assert len([item for item in payload["plugins"] if item["kind"] == "source"]) >= 5
    assert len([item for item in payload["plugins"] if item["kind"] == "gate"]) >= 5
    assert len(
        [item for item in payload["plugins"] if item["kind"] == "applicability"]
    ) >= 1
    assert len([item for item in payload["plugins"] if item["kind"] == "clusterer"]) >= 2
    assert len([item for item in payload["plugins"] if item["kind"] == "selector"]) >= 2
    assert len([item for item in payload["plugins"] if item["kind"] == "exporter"]) >= 2
    openbabel = next(
        item
        for item in payload["plugins"]
        if item["key"] == "features.openbabel_properties@0.1.0"
    )
    assert openbabel["default_config"]["allow_copyleft_backend"] is False
    assert openbabel["selectable"] is False
    rdkit_properties = next(
        item
        for item in payload["plugins"]
        if item["key"] == "features.rdkit_properties@0.1.0"
    )
    assert rdkit_properties["default_config"]["include_sa_score"] is False
    assert {layer["slot"] for layer in payload["layers"]} >= {
        "hard_gate",
        "predict",
        "applicability",
        "synthesis",
        "cluster",
        "select",
    }
    assert all("input_contracts" in plugin for plugin in payload["plugins"])
    assert all("cardinality" in plugin for plugin in payload["plugins"])
    assert payload["semantics"]["join_policy"].startswith("A Policy join")
    assert payload["semantics"]["serial_tier"].startswith("Serial feeds")
    assert payload["alternatives"]
    assert all(not option["selectable"] for option in payload["alternatives"])
    assert all(
        "cannot be added or exported" in option["adapter_status"]
        for option in payload["alternatives"]
    )
    by_key = {plugin["key"]: plugin for plugin in payload["plugins"]}
    admet = by_key["prediction.admet_ai_v2@0.1.0"]
    # The digests stay required and unset: they are what proves the checkpoints
    # are the ones someone reviewed, so a blank cannot be filled in for the user.
    # The *path* is not that -- ADMET-AI v2 ships its models inside the wheel, and
    # pointing at the wrong tree fails the digest check anyway -- so it defaults to
    # a reference that resolves the same way on a laptop and on the GPU node.
    assert {
        "expected_model_manifest_sha256",
        "expected_package_code_sha256",
        "endpoints",
    } <= set(admet["config_schema"]["required"])
    assert "models_dir" not in admet["config_schema"]["required"]
    assert admet["default_config"]["models_dir"] == "package:admet_ai/resources/models"
    assert admet["default_config"]["expected_model_manifest_sha256"] == ""
    assert admet["default_config"]["expected_package_code_sha256"] == ""
    assert admet["default_config"]["allow_unsafe_model_deserialization"] is False
    assert "pin both installed package code" in admet["science_note"]
    drug_likeness = by_key["chemistry.rdkit_drug_likeness@0.1.0"]
    assert drug_likeness["display_name"] == "RDKit Rule of Five + QED"
    assert drug_likeness["science_note"] == (
        "Rule of Five + QED: independent thresholds, annotation only by default."
    )
    assert drug_likeness["default_config"]["maximum_rule_of_five_violations"] is None
    assert drug_likeness["default_config"]["minimum_qed"] is None
    assert drug_likeness["default_config"]["failure_action"] == "warn"
    assert "PAINS defaults to WARN" in by_key[
        "chemistry.rdkit_structural_alerts@0.1.0"
    ]["science_note"]
    assert by_key["chemistry.rdkit_structural_alerts@0.1.0"]["default_config"][
        "pains_action"
    ] == "warn"
    assert "not retrosynthesis steps" in by_key[
        "synthesis.rdkit_sa_score@0.1.0"
    ]["science_note"]
    similarity = by_key["applicability.rdkit_reference_similarity@0.1.0"]
    assert similarity["display_name"] == "RDKit lead similarity / uniqueness"
    # No seed config, for the same reason karmadock and aizynthfinder have none:
    # the model refuses to be built without the one fact only this operator has.
    # ``reference_path`` used to default to ``"leads.csv"``, which was worse than
    # having no default at all -- a relative name that almost never exists still
    # passes ``molcascade validate``, so the failure arrived on the first row of a
    # run that had already been paid for.  The field itself is still advertised,
    # so the form can ask for it; only the invented answer is gone.
    assert similarity["default_config"] == {}
    assert similarity["config_schema"]["properties"]["reference_path"]["default"] is None
    assert "explicit local reference set" in similarity["science_note"]


def test_default_uses_parallel_gate_branches_and_one_structural_primary_flow() -> None:
    payload = build_builder_payload()
    config = PipelineConfig.model_validate(payload["default_pipeline"])
    by_id = {stage.id: stage for stage in config.stages}
    assert not any(stage.slot == "applicability" for stage in config.stages)
    assert by_id["standardize"].inputs[0].stage == "source"
    assert by_id["fingerprint"].inputs[0].stage == "properties"
    assert by_id["synthetic_accessibility"].inputs[0].stage == "fingerprint"
    assert by_id["scaffolds"].inputs[0].stage == "synthesis_policy"
    assert {
        (binding.stage, binding.port)
        for binding in by_id["synthesis_policy"].inputs
    } == {
        ("synthetic_accessibility", "primary"),
        ("synthetic_accessibility", "synthesis_scores"),
    }
    assert by_id["diversity_groups"].inputs[0].stage == "scaffolds"
    evidence_anchor = by_id["properties"].inputs[0].stage
    assert by_id[evidence_anchor].slot == "hard_gate"
    joins = [
        plugin for plugin in payload["plugins"] if plugin["is_policy_join"]
    ]
    if joins:
        policy = by_id["chemistry_policy"]
        assert policy.plugin == joins[0]["key"]
        assert {binding.port for binding in policy.inputs} >= {"primary", "decisions"}
        assert len([binding for binding in policy.inputs if binding.port == "decisions"]) >= 2
        assert "structural_gate" in {
            binding.stage for binding in policy.inputs if binding.port == "decisions"
        }
        branch_ids = {
            "structural_gate",
            "property_gate",
            "drug_likeness",
            "structural_alerts",
        }
        assert {by_id[stage_id].inputs[0].stage for stage_id in branch_ids} == {
            "standardize"
        }
        assert {
            binding.stage for binding in policy.inputs if binding.port == "decisions"
        } == branch_ids
        assert policy.config["mode"] == "all_required"


def test_default_policy_join_is_an_executable_multi_input_node() -> None:
    registry = create_builtin_registry()
    if not any(
        entry.key.startswith("policy.native_decision_join@")
        for entry in registry.entries()
    ):
        registry.register_builtin(_JoinPlugin())
    payload = build_builder_payload(plugins=registry)
    config = PipelineConfig.model_validate(payload["default_pipeline"])
    compiled = PipelineCompiler(registry).compile(config)
    by_id = {stage.id: stage for stage in config.stages}
    policy = by_id["chemistry_policy"]
    assert by_id["structural_gate"].inputs[0].stage == "standardize"
    assert by_id["property_gate"].inputs[0].stage == "standardize"
    assert policy.inputs[0].stage == "standardize"
    assert {binding.port for binding in policy.inputs} == {"primary", "decisions"}
    compiled_policy = next(
        stage for stage in compiled.stages if stage.stage_id == "chemistry_policy"
    )
    assert len(compiled_policy.input_bindings) == 5


def test_default_tier_frames_are_separated_and_chemistry_contains_policy() -> None:
    payload = build_builder_payload()
    config = PipelineConfig.model_validate(payload["default_pipeline"])
    layout = config.metadata["builder_layout"]["nodes"]
    order = {layer["slot"]: index for index, layer in enumerate(payload["layers"])}
    by_slot = {}
    for stage in config.stages:
        by_slot.setdefault(stage.slot, []).append(layout[stage.id])

    frames: list[tuple[str, float, float]] = []
    for slot, positions in sorted(by_slot.items(), key=lambda item: order[item[0]]):
        left = min(float(position["x"]) for position in positions) - 28
        right = max(float(position["x"]) for position in positions) + 238 + 28
        frames.append((slot, left, right))
    assert all(
        current[2] + 84 <= following[1]
        for current, following in itertools.pairwise(frames)
    )

    chemistry = next(frame for frame in frames if frame[0] == "hard_gate")
    policy_x = float(layout["chemistry_policy"]["x"])
    assert chemistry[1] <= policy_x
    assert policy_x + 238 <= chemistry[2]


def test_parallel_all_and_serial_gate_graphs_are_materially_different_and_compile(
) -> None:
    registry = create_builtin_registry()
    parallel_document = build_builder_payload()["default_pipeline"]
    parallel = PipelineConfig.model_validate(parallel_document)
    parallel_compiled = PipelineCompiler(registry).compile(parallel)

    serial_document = copy.deepcopy(parallel_document)
    branch_ids = [
        "structural_gate",
        "property_gate",
        "drug_likeness",
        "structural_alerts",
    ]
    by_id = {stage["id"]: stage for stage in serial_document["stages"]}
    upstream = "standardize"
    for stage_id in branch_ids:
        by_id[stage_id]["inputs"] = [
            {"request_port": "primary", "stage": upstream, "port": "primary"}
        ]
        upstream = stage_id
    for stage in serial_document["stages"]:
        for binding in stage.get("inputs", []):
            if binding["stage"] == "chemistry_policy":
                binding["stage"] = upstream
                binding["port"] = "primary"
    serial_document["stages"] = [
        stage for stage in serial_document["stages"] if stage["id"] != "chemistry_policy"
    ]
    layout = serial_document["metadata"]["builder_layout"]
    layout["tier_modes"]["hard_gate"] = "serial"
    del layout["nodes"]["chemistry_policy"]
    for stage_id in branch_ids:
        layout["nodes"][stage_id]["mode"] = "serial"
    serial = PipelineConfig.model_validate(serial_document)
    serial_compiled = PipelineCompiler(registry).compile(serial)

    assert parallel_compiled.revision_id != serial_compiled.revision_id
    parallel_by_id = {stage.id: stage for stage in parallel.stages}
    serial_by_id = {stage.id: stage for stage in serial.stages}
    assert {parallel_by_id[stage_id].inputs[0].stage for stage_id in branch_ids} == {
        "standardize"
    }
    assert [serial_by_id[stage_id].inputs[0].stage for stage_id in branch_ids] == [
        "standardize",
        "structural_gate",
        "property_gate",
        "drug_likeness",
    ]
    assert "chemistry_policy" not in serial_by_id


def test_generated_builder_is_single_file_offline_drag_ui_with_valid_csp(
    tmp_path: Path,
) -> None:
    destination = generate_config_builder(tmp_path / "builder.html")
    document = destination.read_text(encoding="utf-8")
    assert "draggable" in document
    assert "showSaveFilePicker" in document
    assert "Pipeline canvas" in document
    assert 'lang="en"' in document
    assert "Policy join" in document
    assert "Port connections" in document
    assert "Rule of Five + QED" in document
    assert "Substructure triage flags" in document
    assert "PAINS defaults to WARN" in document
    assert "not retrosynthesis steps" in document
    assert "RDKit lead similarity / uniqueness" in document
    assert "Maximum lead similarity or uniqueness" in document
    assert (
        "molcascade --vs --config molcascade-pipeline.json --molecules molecules.csv"
        in document
    )
    assert "prefers-reduced-motion" in document
    assert "@media" in document
    assert "onclick=" not in document and "onchange=" not in document
    assert "eval(" not in document and "new Function" not in document
    assert_offline(document)
    style = re.search(r"<style>(.*?)</style>", document, re.DOTALL)
    scripts = re.findall(r"<script>(.*?)</script>", document, re.DOTALL)
    csp = re.search(
        r'<meta http-equiv="Content-Security-Policy" content="([^"]+)">',
        document,
    )
    assert style and len(scripts) == 2 and csp
    decoded_csp = html.unescape(csp.group(1))
    for content in (style.group(1), *scripts):
        digest = base64.b64encode(hashlib.sha256(content.encode()).digest()).decode()
        assert f"sha256-{digest}" in decoded_csp
    assert "'unsafe-inline'" not in decoded_csp
    assert "connect-src 'none'" in decoded_csp


def test_pure_browser_rules_reject_invalid_edges_and_allocate_disjoint_lanes() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the browser rule check cannot run")
    rules_path = (
        Path(__file__).parents[2]
        / "src"
        / "molcascade"
        / "ui"
        / "assets"
        / "builder_rules.js"
    )
    script = r"""
const rules = require(process.argv[1]);
const base = {
  sourceId: "a", targetId: "b", sourceSlot: "source", targetSlot: "gate",
  sourceOrder: 0, targetOrder: 2, contract: "parent/v1",
  acceptedContracts: ["parent/v1"], occupiedTierOrders: [0, 2, 4],
  targetTierMode: "parallel_all", sourceIsPolicy: false, targetIsPolicy: false,
  sourceIsCriterion: false, targetIsCriterion: false,
  createsCycle: false,
};
const decide = (change) => rules.connectionDecision({...base, ...change});
const decisions = {
  nextTier: decide({}),
  bypass: decide({targetSlot: "select", targetOrder: 4}),
  backward: decide({sourceOrder: 4, targetOrder: 2}),
  serial: decide({sourceSlot: "gate", targetSlot: "gate", sourceOrder: 2,
    targetOrder: 2, targetTierMode: "serial"}),
  parallelChain: decide({sourceSlot: "gate", targetSlot: "gate", sourceOrder: 2,
    targetOrder: 2, targetTierMode: "parallel_all", sourceIsCriterion: true,
    targetIsCriterion: true}),
  parallelSharedParent: decide({sourceSlot: "predict", targetSlot: "predict",
    sourceOrder: 2, targetOrder: 2, targetTierMode: "parallel_all",
    sourceIsCriterion: false, targetIsCriterion: true}),
  parallelBranchBypass: decide({sourceSlot: "predict", targetSlot: "predict",
    sourceOrder: 2, targetOrder: 2, targetTierMode: "parallel_all",
    sourceIsCriterion: true, targetIsCriterion: false}),
  decisionJoin: decide({sourceSlot: "gate", targetSlot: "gate", sourceOrder: 2,
    targetOrder: 2, contract: "decision/v1", acceptedContracts: ["decision/v1"],
    targetIsPolicy: true}),
  crossTierDecision: decide({sourceSlot: "gate", targetSlot: "select", sourceOrder: 2,
    targetOrder: 4, contract: "decision/v1", acceptedContracts: ["decision/v1"],
    targetIsPolicy: true}),
  rawV1: decide({targetOrder: 2, contract: "raw_molecule/v1",
    acceptedContracts: ["raw_molecule/v1"]}),
  rawV2: decide({targetOrder: 2, contract: "raw_molecule/v2",
    acceptedContracts: ["raw_molecule/v2"]}),
  rawBypass: decide({targetOrder: 4, contract: "raw_molecule/v2",
    acceptedContracts: ["raw_molecule/v2"]}),
  evidence: decide({targetOrder: 2, contract: "prediction/v1",
    acceptedContracts: ["prediction/v1"]}),
  farEvidence: decide({targetOrder: 4, contract: "prediction/v1",
    acceptedContracts: ["prediction/v1"]}),
  sameTierGateEvidence: decide({sourceSlot: "predict", targetSlot: "predict",
    sourceOrder: 4, targetOrder: 4, contract: "prediction/v1",
    acceptedContracts: ["prediction/v1"], requireSameTierEvidence: true}),
  crossTierGateEvidence: decide({sourceSlot: "predict", targetSlot: "synthesis",
    sourceOrder: 4, targetOrder: 5, contract: "prediction/v1",
    acceptedContracts: ["prediction/v1"], requireSameTierEvidence: true}),
  wrongEvidence: decide({targetOrder: 4, contract: "prediction/v1",
    acceptedContracts: ["property/v1"]}),
};
const coProduced = {
  valid: rules.coProducedInputsDecision({
    requiredContracts: ["parent/v1", "prediction/v1"],
    bindings: [
      {sourceId: "predictor", contract: "parent/v1"},
      {sourceId: "predictor", contract: "prediction/v1"},
    ],
  }),
  mixed: rules.coProducedInputsDecision({
    requiredContracts: ["parent/v1", "prediction/v1"],
    bindings: [
      {sourceId: "predictor_a", contract: "parent/v1"},
      {sourceId: "predictor_b", contract: "prediction/v1"},
    ],
  }),
  missing: rules.coProducedInputsDecision({
    requiredContracts: ["parent/v1", "prediction/v1"],
    bindings: [{sourceId: "predictor", contract: "parent/v1"}],
  }),
};
const modes = {
  zero: rules.effectiveTierMode("parallel_all", 0),
  one: rules.effectiveTierMode("parallel_all", 1),
  two: rules.effectiveTierMode("parallel_all", 2),
  serial: rules.effectiveTierMode("serial", 4),
};
const lanes = rules.allocateNonOverlappingLanes([
  {slot: "a", boxes: [{id: "a1", x: 100, y: 100, width: 238, height: 150}]},
  {slot: "b", boxes: [{id: "b1", x: 180, y: 80, width: 238, height: 240}]},
  {slot: "c", boxes: [{id: "c1", x: 190, y: 120, width: 300, height: 150}]},
], {paddingX: 28, paddingY: 52, gap: 84});
process.stdout.write(JSON.stringify({decisions, coProduced, modes, lanes}));
"""
    completed = subprocess.run(
        [node, "-e", script, str(rules_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    decisions = result["decisions"]
    assert decisions["nextTier"]["allowed"] is True
    assert decisions["serial"]["allowed"] is True
    assert decisions["decisionJoin"]["allowed"] is True
    assert decisions["parallelSharedParent"]["allowed"] is True
    assert decisions["parallelSharedParent"]["edgeClass"] == "primary"
    assert decisions["rawV1"]["allowed"] is True
    assert decisions["rawV1"]["edgeClass"] == "primary"
    assert decisions["rawV2"]["allowed"] is True
    assert decisions["rawV2"]["edgeClass"] == "primary"
    assert decisions["rawBypass"]["allowed"] is False
    assert decisions["evidence"]["allowed"] is True
    assert decisions["evidence"]["edgeClass"] == "evidence"
    assert decisions["farEvidence"]["allowed"] is False
    assert decisions["sameTierGateEvidence"]["allowed"] is True
    assert decisions["crossTierGateEvidence"]["allowed"] is False
    assert decisions["bypass"]["allowed"] is False
    assert "bypass" in decisions["bypass"]["reason"]
    assert decisions["backward"]["allowed"] is False
    assert decisions["parallelChain"]["allowed"] is False
    assert decisions["parallelBranchBypass"]["allowed"] is False
    assert decisions["crossTierDecision"]["allowed"] is False
    assert decisions["wrongEvidence"]["allowed"] is False
    assert result["coProduced"]["valid"]["allowed"] is True
    assert result["coProduced"]["mixed"]["allowed"] is False
    assert result["coProduced"]["missing"]["allowed"] is False
    assert result["modes"] == {
        "zero": "serial",
        "one": "serial",
        "two": "parallel_all",
        "serial": "serial",
    }
    lanes = result["lanes"]
    assert all(
        current["bounds"]["right"] + 84 <= following["bounds"]["left"]
        for current, following in itertools.pairwise(lanes)
    )


def test_builder_script_exposes_true_node_canvas_and_accessible_non_drag_controls(
    tmp_path: Path,
) -> None:
    document = generate_config_builder(tmp_path / "builder.html").read_text(
        encoding="utf-8"
    )
    assert 'id="graph-scene"' in document
    assert 'id="graph-edges"' in document
    assert 'id="node-layer"' in document
    assert 'id="connection-preview"' in document
    assert "input-port" in document
    assert "output-port" in document
    assert "edge-path" in document
    assert "edge-hit" in document
    assert "function renderNode(" in document
    assert "function renderScene(" in document
    assert "function renderInputBindings(" in document
    assert "function connectOutput(" in document
    assert "function stageConnectionDecision(" in document
    assert "sourceIsCriterion: isDecisionCriterion(source)" in document
    assert "targetIsCriterion: isDecisionCriterion(target)" in document
    assert "function bindingsFromOneSource(" in document
    assert "coProducedInputsDecision" in document
    assert "function showConnectorToast(" in document
    assert document.count("stageConnectionDecision(source, binding.port, stage)") >= 1
    assert "is not allowed: ${permission.reason}" in document
    assert "function applyTierMode(" in document
    assert "function arrangeAllNodes(" in document
    assert "function normalizeTierLanes(" in document
    assert "allocateNonOverlappingLanes" in document
    assert "function orthogonalPath(" in document
    assert "edge-bus" in document
    assert "evidence-edge" in document
    assert ".edge-path.rejected-edge" in document
    assert ".edge-label.invalid-label" in document
    assert '? "invalid"' in document
    assert "Primary parent flow may enter only the next occupied tier" in document
    assert "Policy decision joins accept decisions from their own tier only" in document
    assert "function fitView(" in document
    assert "function resetView(" in document
    assert "function screenToWorld(" in document
    assert "function setFinalTarget(" in document
    assert "selector.config.target_count = target" in document
    assert "stage.inputs" in document
    assert "source_stage" not in document  # exported names match StageInputBinding exactly
    assert "Parallel (ALL)" in document
    assert "criterion) => setParentInput(criterion, anchor)" in document
    assert "setParentInput(criterion, upstream)" in document
    assert "removeJoinStages(slot, last)" in document
    assert 'mode: "all_required"' in document
    assert "Wheel to zoom" in document
    assert "drag empty space to pan" in document
    assert "Auto-connect input" in document
    assert "Rebuild ALL join" in document
    assert "styleSheet.insertRule" in document
    assert "Adapter pending" in document
    assert "Research option only — cannot be added or exported" in document
    assert "parameter-grid" in document
    assert 'plugin.key === "chemistry.rdkit_drug_likeness@0.1.0"' in document
    assert "inspect_admet_ai_installation()" in document
    assert "inspect_admet_ai_model_assets()" in document
    assert 'plugin?.key === "prediction.admet_ai_v2@0.1.0"' in document
    assert "File System Access" not in document  # no unsupported API claim in visible copy


def test_every_builder_owned_visible_string_is_english(tmp_path: Path) -> None:
    document = generate_config_builder(tmp_path / "builder.html").read_text(
        encoding="utf-8"
    )
    assert re.search(r"[\u3400-\u9fff]", document) is None


def test_embedded_payload_round_trips_without_executable_data_context(tmp_path: Path) -> None:
    document = generate_config_builder(tmp_path / "builder.html").read_text(encoding="utf-8")
    match = re.search(
        r'<textarea id="builder-payload" hidden>(.*?)</textarea>',
        document,
        re.DOTALL,
    )
    assert match
    payload = json.loads(html.unescape(match.group(1)))
    PipelineConfig.model_validate(payload["default_pipeline"])
    assert "</script>" not in match.group(1)


def test_untrusted_plugin_text_cannot_break_out_of_payload_container() -> None:
    payload = build_builder_payload()
    attack = '</textarea><script>globalThis.compromised=true</script><svg onload="x">'
    payload["plugins"][0]["description"] = attack
    document = render_config_builder(payload)
    assert attack not in document
    assert "&lt;/textarea&gt;&lt;script&gt;globalThis.compromised" in document
    match = re.search(
        r'<textarea id="builder-payload" hidden>(.*?)</textarea>',
        document,
        re.DOTALL,
    )
    assert match
    round_tripped = json.loads(html.unescape(match.group(1)))
    assert round_tripped["plugins"][0]["description"] == attack
