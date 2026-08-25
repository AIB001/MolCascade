"""The medchem-backed alert and rule adapters.

``medchem`` is an optional dependency and is not installed here, so a stub
package is injected into ``sys.modules`` for the tests that exercise scoring.
The stub answers exactly as the real package does -- the column names, the
four-level status vocabulary, the NIBR severity arithmetic and the
``pass_all``/``pass_any`` columns are all taken from datamol-io/medchem -- so
everything on the MolCascade side of the boundary is real code: config
validation, name checking, batching, decision emission, row alignment and every
fail-closed path.
"""

from __future__ import annotations

import ast
import inspect
import json
import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import DECISION_V1, PARENT_V1
from molcascade.errors import PluginError
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin import medchem_alerts as alerts_module
from molcascade.plugins.builtin import medchem_rules as rules_module
from molcascade.plugins.builtin.medchem_alerts import MedchemAlertsConfig, MedchemAlertsPlugin
from molcascade.plugins.builtin.medchem_rules import MedchemRulesConfig, MedchemRulesPlugin
from molcascade.plugins.builtin.medchem_support import import_medchem, result_column
from molcascade.plugins.manifest import Determinism
from molcascade.plugins.registry import create_builtin_registry

#: Ethanol, phenol, aspirin, eicosane.  Chosen so the stub can separate them on
#: honest chemistry -- an aromatic hydroxyl and a heavy-atom count -- rather
#: than on canonical-SMILES string matching, which would break silently if
#: RDKit ever changed its canonical output.
_SMILES = ["CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "CCCCCCCCCCCCCCCCCCCC"]

_ALERT_SET_NAMES = ("BMS", "Dundee", "Glaxo", "PAINS", "SureChEMBL")
_RULE_NAMES = ("rule_of_five", "rule_of_veber", "rule_of_egan")

#: Severity 4 sits below medchem's own reject line of 10 and above a project
#: that tightens it, which is exactly the distinction the severity test needs.
_NIBR_SEVERITIES = (0, 4, 0, 4)


# --------------------------------------------------------------- test doubles


class _Frame:
    """The slice of a pandas DataFrame these adapters actually use."""

    def __init__(self, columns: dict[str, list[Any]]) -> None:
        self._columns = columns

    def __contains__(self, name: object) -> bool:
        return name in self._columns

    def __getitem__(self, name: str) -> list[Any]:
        return self._columns[name]


class _NameTable:
    """Stands in for the DataFrame ``list_default_available_alerts`` returns."""

    def __init__(self, names: tuple[str, ...]) -> None:
        self._names = list(names)

    def __getitem__(self, _column: str) -> _NameTable:
        return self

    def tolist(self) -> list[str]:
        return list(self._names)


def _heavy_atoms(mol: Any) -> int:
    return int(mol.GetNumHeavyAtoms())


class _CommonAlertsFilters:
    """Excludes aromatic hydroxyls and flags anything large and greasy."""

    def __init__(self, alerts_set: Any = None, alerts_db_path: Any = None) -> None:
        self.alerts_set = list(alerts_set or [])
        self.alerts_db_path = alerts_db_path
        # The real object exposes the selected rule table, and the adapter
        # refuses an empty one, so the stub sizes it from the request.
        self.alerts_df = ["rule"] * (10 * len(self.alerts_set))

    @staticmethod
    def list_default_available_alerts() -> _NameTable:
        return _NameTable(_ALERT_SET_NAMES)

    def __call__(self, mols: list[Any], **_kwargs: Any) -> _Frame:
        from rdkit import Chem

        phenol = Chem.MolFromSmarts("[OX2H]c")
        statuses, reasons = [], []
        for mol in mols:
            if mol.HasSubstructMatch(phenol):
                statuses.append("exclude")
                reasons.append("aromatic hydroxyl")
            elif _heavy_atoms(mol) > 15:
                statuses.append("flag")
                reasons.append("long aliphatic chain")
            else:
                statuses.append("ok")
                reasons.append(None)
        return _Frame({"status": statuses, "reasons": reasons, "pass_filter": [True] * len(mols)})


#: ``NIBR||<rule>||<severity code>||<covalent>||<compound class>``, which is the
#: layout Novartis publishes and medchem loads verbatim into an RDKit catalog.
#: Two rules is enough to exercise the split the adapter reads out of it: one
#: exclusion that describes a liability, one that only describes a class.
_NIBR_RULES = (
    "NIBR||nitroalkane_min(1)||2||0||0",
    "NIBR||steroid_non_arom_min(1)||2||0||1",
    "NIBR||thiazolidinediones_min(1)||1||0||0",
)


class _NIBRCatalogEntry:
    def __init__(self, description: str) -> None:
        self._description = description

    def GetDescription(self) -> str:
        return self._description


class _NIBRCatalog:
    #: A class attribute so a test can hand the adapter a rule file it cannot
    #: read, which is the case that has to fail loudly rather than be guessed.
    rules: tuple[str, ...] = _NIBR_RULES

    def GetNumEntries(self) -> int:
        return len(self.rules)

    def GetEntryWithIdx(self, index: int) -> _NIBRCatalogEntry:
        return _NIBRCatalogEntry(self.rules[index])


class _NIBRFilters:
    #: Reasons the stub reports, which the adapter matches back against the
    #: catalog above.  ``None`` where the severity is 0.  Class-level on
    #: purpose: individual tests replace it wholesale with
    #: ``monkeypatch.setattr``, which restores it afterwards.
    reasons_for_severity: ClassVar[dict[int, str]] = {}

    def __init__(self) -> None:
        self.catalog = _NIBRCatalog()

    def __call__(self, mols: list[Any], **_kwargs: Any) -> _Frame:
        severities = list(_NIBR_SEVERITIES[: len(mols)])
        return _Frame(
            {
                "status": ["ok" if value == 0 else "flag" for value in severities],
                "reasons": [
                    None
                    if value == 0
                    else self.reasons_for_severity.get(value, "screening-deck flag")
                    for value in severities
                ],
                "severity": severities,
                "n_covalent_motif": [0] * len(mols),
                "special_mol": [0] * len(mols),
            }
        )


class _RuleFilters:
    """Three rules with deliberately different strictness.

    Against the four fixtures above they partition cleanly: every molecule
    passes the weight rule, the first three pass the size rule, and only the
    two smallest pass the strict weight rule.  That makes all / any /
    at-least-k give three different answers.
    """

    def __init__(self, rule_list: list[str]) -> None:
        self.rule_list = list(rule_list)

    @staticmethod
    def list_available_rules_names() -> list[str]:
        return list(_RULE_NAMES)

    def __call__(self, mols: list[Any], **_kwargs: Any) -> _Frame:
        from rdkit.Chem import Descriptors

        columns: dict[str, list[Any]] = {name: [] for name in self.rule_list}
        for mol in mols:
            verdicts = {
                "rule_of_five": Descriptors.MolWt(mol) <= 500,
                "rule_of_veber": _heavy_atoms(mol) <= 15,
                "rule_of_egan": Descriptors.MolWt(mol) <= 150,
            }
            for name in self.rule_list:
                columns[name].append(verdicts[name])
        columns["pass_all"] = [
            all(columns[name][index] for name in self.rule_list) for index in range(len(mols))
        ]
        columns["pass_any"] = [
            any(columns[name][index] for name in self.rule_list) for index in range(len(mols))
        ]
        return _Frame(columns)


@pytest.fixture
def fake_medchem(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a stub ``medchem`` package for the duration of one test."""

    package = types.ModuleType("medchem")
    package.__version__ = "2.1.0-stub"
    structural = types.ModuleType("medchem.structural")
    structural.CommonAlertsFilters = _CommonAlertsFilters
    structural.NIBRFilters = _NIBRFilters
    rules = types.ModuleType("medchem.rules")
    rules.RuleFilters = _RuleFilters
    utils = types.ModuleType("medchem.utils")
    loader = types.ModuleType("medchem.utils.loader")
    package.structural = structural
    package.rules = rules
    package.utils = utils
    utils.loader = loader
    for name, module in (
        ("medchem", package),
        ("medchem.structural", structural),
        ("medchem.rules", rules),
        ("medchem.utils", utils),
        ("medchem.utils.loader", loader),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return package


@pytest.fixture
def alerts_db(tmp_path: Path, fake_medchem: types.ModuleType) -> Path:
    """A stand-in for the SMARTS table that ships inside medchem."""

    path = tmp_path / "common_alerts_collection.csv"
    path.write_text("rule_set_name,smarts\nBMS,[N+]\n", encoding="utf-8")
    sys.modules["medchem.utils.loader"].get_data_path = lambda filename, **_: str(path)
    return path


# ------------------------------------------------------------------- harness


def _parent(parent_id: str, smiles: str) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "identity_policy_id": "identity-policy:test",
        "parent_smiles": smiles,
        "registration_key": f"registration-{parent_id}",
        "stereo_key": None,
        "formula": None,
        "duplicate_count": 1,
    }


def _input(root: Path, smiles: list[str]) -> StageInput:
    input_root = root / "input"
    input_root.mkdir(parents=True, exist_ok=True)
    rows = [_parent(f"m{index}", entry) for index, entry in enumerate(smiles)]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=PARENT_V1.schema),
        input_root / "parents.parquet",
    )
    return StageInput(
        ref=ArtifactRef(artifact_id="artifact:sha256:" + "b" * 64, kind=PARENT_V1.id),
        root=input_root,
        contract_id=PARENT_V1.id,
    )


def _run(
    plugin: Any,
    root: Path,
    config: dict[str, object],
    *,
    smiles: list[str] | None = None,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    staging = root / "out"
    response = plugin.execute(
        StageRequest(
            stage_id="medchem",
            inputs={"parents": _input(root, smiles or _SMILES)},
            config=config,
        ),
        StageContext(staging),
    )
    parents = pq.read_table(staging / response.outputs["primary"].file_paths[0])
    decisions = pq.read_table(staging / response.outputs["decisions"].file_paths[0])
    PARENT_V1.validate(parents)
    DECISION_V1.validate(decisions)
    return response, parents.to_pylist(), decisions.to_pylist()


# ------------------------------------------------------------------- contract


def _module_scope_imports(body: list[ast.stmt]) -> list[str]:
    """Names imported when the module is first loaded.

    Descends into ``if``/``try`` blocks, which still run at import time, but not
    into functions or classes -- an import deferred into a function body is the
    behaviour under test, not a violation of it.
    """

    found: list[str] = []
    for node in body:
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append(node.module or "")
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        else:
            for field in ("body", "orelse", "finalbody", "handlers"):
                nested = getattr(node, field, None)
                if isinstance(nested, list):
                    found.extend(_module_scope_imports(nested))
    return found


def test_neither_adapter_imports_medchem_until_it_is_used() -> None:
    """Listing plugins must work on a machine that has no medchem."""

    for module in (alerts_module, rules_module):
        tree = ast.parse(inspect.getsource(module))
        roots = {name.split(".")[0] for name in _module_scope_imports(tree.body)}
        assert "medchem" not in roots, f"{module.__name__} imports medchem at module scope"
        # The check is only meaningful if it is looking at real imports.
        assert "molcascade" in roots


def test_both_adapters_register_as_deterministic_filters() -> None:
    registry = create_builtin_registry()
    alerts = registry.get("chemistry.medchem_alerts@0.1.0")
    rules = registry.get("chemistry.medchem_rules@0.1.0")
    assert isinstance(alerts, MedchemAlertsPlugin)
    assert isinstance(rules, MedchemRulesPlugin)
    for plugin in (alerts, rules):
        assert plugin.descriptor.determinism is Determinism.DETERMINISTIC
        assert plugin.descriptor.output_ports == {
            "primary": PARENT_V1.id,
            "decisions": DECISION_V1.id,
        }


def test_a_missing_medchem_fails_closed_with_an_install_hint() -> None:
    try:
        import medchem  # noqa: F401
    except ImportError:
        pass
    else:  # pragma: no cover - only reachable where medchem is installed
        pytest.skip("medchem is installed, so the unavailable path cannot run here")
    with pytest.raises(PluginError) as captured:
        import_medchem()
    assert captured.value.code == "MEDCHEM_BACKEND_UNAVAILABLE"
    assert "never installs or downloads" in (captured.value.hint or "")


def test_a_result_table_that_does_not_match_the_input_is_refused() -> None:
    """Losing a row would attach one molecule's verdict to another."""

    frame = _Frame({"status": ["ok", "ok"]})
    with pytest.raises(PluginError) as captured:
        result_column(frame, "status", 3, "common")
    assert captured.value.code == "MEDCHEM_RESULT_COUNT_MISMATCH"

    with pytest.raises(PluginError) as captured:
        result_column(frame, "severity", 2, "common")
    assert captured.value.code == "MEDCHEM_RESULT_SHAPE_UNEXPECTED"


# --------------------------------------------------------------------- alerts


def test_alert_sets_accept_a_yaml_list_or_the_builder_comma_string() -> None:
    from_browser = MedchemAlertsConfig.model_validate({"alert_sets": "BMS, Dundee ,Glaxo"})
    from_yaml = MedchemAlertsConfig.model_validate({"alert_sets": ["BMS", "Dundee", "Glaxo"]})
    assert from_browser.alert_sets == from_yaml.alert_sets == ("BMS", "Dundee", "Glaxo")

    with pytest.raises(ValueError, match="same collection twice"):
        MedchemAlertsConfig.model_validate({"alert_sets": ["BMS", "bms"]})


def test_a_stage_configured_to_apply_no_rules_at_all_is_refused(
    tmp_path: Path,
    alerts_db: Path,
) -> None:
    with pytest.raises(PluginError) as captured:
        _run(MedchemAlertsPlugin(), tmp_path, {"alert_sets": "", "use_nibr": False})
    assert captured.value.code == "MEDCHEM_ALERTS_NO_RULES"


def test_a_misspelled_alert_collection_stops_the_run(tmp_path: Path, alerts_db: Path) -> None:
    """medchem would select nothing and pass everything; that must not happen."""

    with pytest.raises(PluginError) as captured:
        _run(MedchemAlertsPlugin(), tmp_path, {"alert_sets": "BMS,Dundeee"})
    assert captured.value.code == "MEDCHEM_ALERT_SET_UNKNOWN"
    assert captured.value.context["unknown"] == ["Dundeee"]
    # The message has to be actionable, so it names what is actually there.
    assert "SureChEMBL" in (captured.value.hint or "")


def test_alert_names_are_matched_without_regard_to_case(tmp_path: Path, alerts_db: Path) -> None:
    response, _, _ = _run(MedchemAlertsPlugin(), tmp_path, {"alert_sets": "bms,DUNDEE"})
    # medchem's own spelling is what lands in the provenance record.
    assert response.metadata["alert_sets"] == ["BMS", "Dundee"]


def test_an_exclude_match_rejects_and_a_flag_only_warns(tmp_path: Path, alerts_db: Path) -> None:
    response, parents, decisions = _run(MedchemAlertsPlugin(), tmp_path, {"alert_sets": "BMS"})

    # m1 is the phenol the stub excludes; m3 is the long chain it flags.
    assert [row["parent_id"] for row in parents] == ["m0", "m2", "m3"]
    assert response.metadata["reject_count"] == 1
    assert response.metadata["warning_decision_count"] == 1
    assert response.metadata["reject_decision_count"] == 1

    by_parent = {row["entity_id"]: row for row in decisions}
    assert by_parent["m1"]["outcome"] == "REJECT"
    assert by_parent["m3"]["outcome"] == "WARN"
    assert by_parent["m0"]["outcome"] == "PASS"
    assert by_parent["m0"]["reason_code"] == "MEDCHEM_ALERTS_CLEAR"
    assert by_parent["m1"]["rule_id"] == response.metadata["policy_id"]

    detail = json.loads(by_parent["m1"]["detail"])
    assert detail["status"] == "exclude"
    assert detail["reasons"] == "aromatic hydroxyl"
    # An alert is a triage hypothesis, and the record has to say so.
    assert "not experimental proof" in detail["interpretation"]


def test_the_project_decides_what_each_status_means(tmp_path: Path, alerts_db: Path) -> None:
    """medchem's own pass_filter is deliberately not adopted as the verdict."""

    response, parents, _ = _run(
        MedchemAlertsPlugin(),
        tmp_path,
        {"alert_sets": "BMS", "exclude_action": "warn", "flag_action": "reject"},
    )
    assert [row["parent_id"] for row in parents] == ["m0", "m1", "m2"]
    assert response.metadata["reject_count"] == 1


def test_ignored_statuses_produce_a_clear_pass_not_an_empty_record(
    tmp_path: Path,
    alerts_db: Path,
) -> None:
    _, parents, decisions = _run(
        MedchemAlertsPlugin(),
        tmp_path,
        {"alert_sets": "BMS", "exclude_action": "ignore", "flag_action": "ignore"},
    )
    assert len(parents) == len(_SMILES)
    assert len(decisions) == len(_SMILES)
    assert {row["outcome"] for row in decisions} == {"PASS"}


def test_nibr_severity_is_its_own_axis(tmp_path: Path, alerts_db: Path) -> None:
    """Enough accumulated soft flags can be worse than one hard exclusion."""

    lenient, _, _ = _run(
        MedchemAlertsPlugin(), tmp_path / "a", {"alert_sets": "BMS", "use_nibr": True}
    )
    # At medchem's own line of 10, severity 4 is only a warning, so the single
    # rejection is still the common-alert exclusion.
    assert lenient.metadata["reject_count"] == 1

    strict, parents, decisions = _run(
        MedchemAlertsPlugin(),
        tmp_path / "b",
        {"alert_sets": "BMS", "use_nibr": True, "nibr_reject_at_severity": 4},
    )
    assert strict.metadata["reject_count"] == 2
    assert [row["parent_id"] for row in parents] == ["m0", "m2"]

    nibr = [row for row in decisions if row["reason_code"].startswith("MEDCHEM_NIBR_")]
    assert [row["outcome"] for row in nibr] == ["REJECT", "REJECT"]
    details = [json.loads(row["detail"]) for row in nibr]
    assert {detail["severity"] for detail in details} == {4}
    # The threshold that produced the verdict travels with the verdict.
    assert {detail["reject_at_severity"] for detail in details} == {4}


def test_an_exclusion_that_is_only_a_compound_class_is_reported_as_one(
    tmp_path: Path, alerts_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NIBR excludes some molecules for what they *are*, not for a liability.

    67 of its rules fire on membership of a class -- steroid, peptide,
    nucleoside -- because a physical screening deck does not want promiscuous
    matter in it.  That argument does not carry over to a virtual cascade aimed
    at a named target, so the two kinds of exclusion have to be separable, and
    which is which comes from Novartis' own annotation in the rule file rather
    than from a list in this repository.
    """

    monkeypatch.setattr(
        _NIBRFilters, "reasons_for_severity", {4: "steroid_non_arom_min(1)"}
    )
    config = {"alert_sets": "BMS", "use_nibr": True, "nibr_reject_at_severity": 4}

    kept, parents, decisions = _run(
        MedchemAlertsPlugin(),
        tmp_path / "kept",
        {**config, "nibr_compound_class_action": "warn"},
    )
    nibr = [row for row in decisions if row["reason_code"].startswith("MEDCHEM_NIBR_")]
    assert [row["outcome"] for row in nibr] == ["WARN", "WARN"]
    # Downgraded, not hidden: the severity that would have rejected it and the
    # rule that produced it both still travel with the decision.
    details = [json.loads(row["detail"]) for row in nibr]
    assert {detail["status"] for detail in details} == {"compound_class"}
    assert {detail["severity"] for detail in details} == {4}
    # m3 is the witness: it reaches the NIBR threshold and has nothing else
    # against it, so it survives only because of the downgrade.  (m1 also
    # reaches it, but the common-alert arm excludes m1 on its own.)
    assert "m3" in {row["parent_id"] for row in parents}

    # And the adapter still applies NIBR as published when asked to.
    verbatim, _, _ = _run(MedchemAlertsPlugin(), tmp_path / "verbatim", config)
    assert verbatim.metadata["reject_count"] > kept.metadata["reject_count"]


def test_a_compound_class_cannot_speak_for_a_molecule_with_a_real_liability(
    tmp_path: Path, alerts_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One genuine exclusion in the mix and the exemption does not apply.

    This is the direction the split has to fail in: a steroid that also carries
    a nitroalkane is not rescued by being a steroid.
    """

    monkeypatch.setattr(
        _NIBRFilters,
        "reasons_for_severity",
        {4: "steroid_non_arom_min(1); nitroalkane_min(1)"},
    )

    _response, _parents, decisions = _run(
        MedchemAlertsPlugin(),
        tmp_path,
        {
            "alert_sets": "BMS",
            "use_nibr": True,
            "nibr_reject_at_severity": 4,
            "nibr_compound_class_action": "warn",
        },
    )

    nibr = [row for row in decisions if row["reason_code"].startswith("MEDCHEM_NIBR_")]
    assert [row["outcome"] for row in nibr] == ["REJECT", "REJECT"]


def test_a_nibr_rule_file_this_adapter_cannot_read_stops_the_run(
    tmp_path: Path, alerts_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed rule layout must not be guessed at.

    Reading the wrong field would silently reclassify every exclusion, which is
    the failure mode this whole mechanism exists to prevent.
    """

    monkeypatch.setattr(_NIBRCatalog, "rules", ("NIBR||only||two",))

    with pytest.raises(PluginError) as excinfo:
        _run(MedchemAlertsPlugin(), tmp_path, {"alert_sets": "BMS", "use_nibr": True})
    assert excinfo.value.code == "MEDCHEM_NIBR_RULE_UNREADABLE"


def test_the_run_is_pinned_to_the_rule_table_that_produced_it(
    tmp_path: Path,
    alerts_db: Path,
) -> None:
    first, _, _ = _run(MedchemAlertsPlugin(), tmp_path / "a", {"alert_sets": "BMS"})
    alerts_db.write_text("rule_set_name,smarts\nBMS,[N+]\nBMS,[O-]\n", encoding="utf-8")
    second, _, _ = _run(MedchemAlertsPlugin(), tmp_path / "b", {"alert_sets": "BMS"})

    assert first.metadata["rules_sha256"] != second.metadata["rules_sha256"]
    # A changed rule table is a changed policy, so the identity moves with it.
    assert first.metadata["policy_id"] != second.metadata["policy_id"]
    assert first.metadata["network_or_download_invoked_by_adapter"] is False


def test_a_user_supplied_alert_table_must_be_a_real_local_file(
    tmp_path: Path,
    alerts_db: Path,
) -> None:
    for index, path in enumerate(("relative/alerts.csv", str(tmp_path / "absent.csv"))):
        with pytest.raises(PluginError) as captured:
            _run(
                MedchemAlertsPlugin(),
                tmp_path / f"case{index}",
                {"alert_sets": "BMS", "alerts_db_path": path},
            )
        assert captured.value.code == "MEDCHEM_ALERTS_DB_INVALID"


def test_medchem_never_gets_to_spawn_its_own_worker_pool(
    tmp_path: Path,
    alerts_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested pools fight over cores and make row order depend on scheduling."""

    seen: list[dict[str, Any]] = []
    original = _CommonAlertsFilters.__call__

    def record(self: Any, mols: list[Any], **kwargs: Any) -> Any:
        seen.append(kwargs)
        return original(self, mols, **kwargs)

    monkeypatch.setattr(_CommonAlertsFilters, "__call__", record)
    _run(MedchemAlertsPlugin(), tmp_path, {"alert_sets": "BMS", "batch_size": 2})
    assert len(seen) == 2
    assert all(call["n_jobs"] == 1 for call in seen)
    assert all(call["progress"] is False for call in seen)


def test_alert_batching_does_not_change_which_molecules_survive(
    tmp_path: Path,
    alerts_db: Path,
) -> None:
    whole, one_pass, _ = _run(MedchemAlertsPlugin(), tmp_path / "a", {"alert_sets": "BMS"})
    split, chunked, _ = _run(
        MedchemAlertsPlugin(), tmp_path / "b", {"alert_sets": "BMS", "batch_size": 1}
    )
    assert one_pass == chunked
    assert whole.metadata["reject_count"] == split.metadata["reject_count"]


def test_an_unparseable_parent_is_an_integrity_failure(tmp_path: Path, alerts_db: Path) -> None:
    with pytest.raises(PluginError) as captured:
        _run(
            MedchemAlertsPlugin(),
            tmp_path,
            {"alert_sets": "BMS"},
            smiles=["CCO", "not a molecule"],
        )
    assert captured.value.code == "MEDCHEM_PARENT_INVALID"
    assert captured.value.context["parent_id"] == "m1"
    # Nothing half-written is left behind for a later stage to pick up.
    assert not (tmp_path / "out/datasets/passed_parents/part-00000.parquet").exists()
    assert not (tmp_path / "out/datasets/decisions/part-00000.parquet").exists()


def test_an_unknown_status_word_stops_rather_than_guessing(
    tmp_path: Path,
    alerts_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _CommonAlertsFilters,
        "__call__",
        lambda self, mols, **_: _Frame(
            {"status": ["quarantine"] * len(mols), "reasons": [None] * len(mols)}
        ),
    )
    with pytest.raises(PluginError) as captured:
        _run(MedchemAlertsPlugin(), tmp_path, {"alert_sets": "BMS"})
    assert captured.value.code == "MEDCHEM_STATUS_UNKNOWN"
    assert captured.value.context["status"] == "quarantine"


# ---------------------------------------------------------------------- rules


def test_a_quorum_larger_than_the_rule_list_is_unreachable_and_refused() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        MedchemRulesConfig.model_validate(
            {"rules": "rule_of_five,rule_of_veber", "combination": "at_least", "minimum_passes": 3}
        )


def test_each_combination_resolves_to_the_number_of_passes_it_requires() -> None:
    assert MedchemRulesConfig.model_validate({"rules": "rule_of_five"}).required_passes() == 1
    both = MedchemRulesConfig.model_validate({"rules": "rule_of_five,rule_of_veber"})
    assert both.required_passes() == 2
    either = MedchemRulesConfig.model_validate(
        {"rules": "rule_of_five,rule_of_veber", "combination": "any"}
    )
    assert either.required_passes() == 1


def test_an_unknown_rule_set_stops_the_run(tmp_path: Path, fake_medchem: Any) -> None:
    with pytest.raises(PluginError) as captured:
        _run(MedchemRulesPlugin(), tmp_path, {"rules": "rule_of_six"})
    assert captured.value.code == "MEDCHEM_RULE_UNKNOWN"
    assert captured.value.context["unknown"] == ["rule_of_six"]
    assert "rule_of_veber" in (captured.value.hint or "")


def test_series_and_parallel_combination_are_two_different_filters(
    tmp_path: Path,
    fake_medchem: Any,
) -> None:
    """The same two rules, combined two ways, keep different molecules."""

    rules = "rule_of_five,rule_of_egan"
    every, strict_parents, decisions = _run(
        MedchemRulesPlugin(), tmp_path / "a", {"rules": rules, "combination": "all"}
    )
    _, loose_parents, _ = _run(
        MedchemRulesPlugin(), tmp_path / "b", {"rules": rules, "combination": "any"}
    )

    assert [row["parent_id"] for row in strict_parents] == ["m0", "m1"]
    assert [row["parent_id"] for row in loose_parents] == ["m0", "m1", "m2", "m3"]
    assert every.metadata["combination"] == "all"
    assert every.metadata["required_rule_count"] == 2

    rejected = next(row for row in decisions if row["entity_id"] == "m2")
    assert rejected["outcome"] == "REJECT"
    detail = json.loads(rejected["detail"])
    assert detail["rules"] == {"rule_of_five": True, "rule_of_egan": False}
    assert detail["passed_rule_count"] == 1
    assert detail["required_rule_count"] == 2


def test_at_least_k_sits_between_all_and_any(tmp_path: Path, fake_medchem: Any) -> None:
    rules = "rule_of_five,rule_of_veber,rule_of_egan"
    quorum, majority, _ = _run(
        MedchemRulesPlugin(),
        tmp_path / "a",
        {"rules": rules, "combination": "at_least", "minimum_passes": 2},
    )
    _, unanimous, _ = _run(
        MedchemRulesPlugin(), tmp_path / "b", {"rules": rules, "combination": "all"}
    )

    assert quorum.metadata["required_rule_count"] == 2
    assert [row["parent_id"] for row in majority] == ["m0", "m1", "m2"]
    assert [row["parent_id"] for row in unanimous] == ["m0", "m1"]


def test_the_run_reports_which_rule_did_the_rejecting(tmp_path: Path, fake_medchem: Any) -> None:
    """The first question asked of a narrow funnel is which rule narrowed it."""

    response, _, _ = _run(
        MedchemRulesPlugin(),
        tmp_path,
        {"rules": "rule_of_five,rule_of_egan", "combination": "all"},
    )
    assert response.metadata["per_rule_failure_count"] == {"rule_of_five": 0, "rule_of_egan": 2}


def test_rule_batching_does_not_change_the_outcome(tmp_path: Path, fake_medchem: Any) -> None:
    _, whole, _ = _run(MedchemRulesPlugin(), tmp_path / "a", {"rules": "rule_of_veber"})
    _, chunked, _ = _run(
        MedchemRulesPlugin(), tmp_path / "b", {"rules": "rule_of_veber", "batch_size": 1}
    )
    assert [row["parent_id"] for row in whole] == ["m0", "m1", "m2"]
    assert whole == chunked


def test_every_molecule_gets_exactly_one_rule_decision(tmp_path: Path, fake_medchem: Any) -> None:
    """Per-rule rows would multiply the decision table by the rule count."""

    response, _, decisions = _run(MedchemRulesPlugin(), tmp_path, {"rules": _RULE_NAMES[0]})
    assert response.metadata["decision_count"] == len(decisions) == len(_SMILES)
    assert len({row["entity_id"] for row in decisions}) == len(_SMILES)
