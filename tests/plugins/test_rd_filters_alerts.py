"""The rd_filters alert adapter, checked against the vendored table itself.

Unlike ``medchem``, nothing here is stubbed.  The alert collection is a
digest-pinned asset, RDKit is a hard dependency, and the whole point of the
adapter is that it reproduces a specific published rule set -- so a test that
matched against invented SMARTS would check the plumbing and miss the only
thing users care about.  The tests therefore run the real 1,251 rules where the
asset is present, and fall back to a small hand-written table (in the upstream
format) where it is not, so a fresh checkout without ``molcascade assets fetch``
still exercises every code path except the digest of the real file.
"""

from __future__ import annotations

import ast
import csv
import inspect
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molcascade.artifacts import ArtifactRef
from molcascade.assets import asset_spec, asset_status
from molcascade.contracts import DECISION_V1, PARENT_V1
from molcascade.errors import PluginError
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin import rd_filters_alerts as module
from molcascade.plugins.builtin.rd_filters_alerts import (
    DEFAULT_ALERTS_REFERENCE,
    RULE_SETS,
    AlertAction,
    RdFiltersAlertConfig,
    RdFiltersAlertsPlugin,
)
from molcascade.plugins.manifest import Determinism
from molcascade.plugins.registry import create_builtin_registry

#: Aspirin (a phenol ester -- Dundee/MLSMR/SureChEMBL), caffeine, a phenacyl
#: bromide (a Glaxo reactive alkyl halide) and an aryl azo dye, which is the
#: textbook multi-set hit.  Chosen so the assertions turn on chemistry rather
#: than on which rule happens to sort first.
_ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
_CAFFEINE = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
_PHENACYL_BROMIDE = "C1=CC=C(C=C1)C(=O)CBr"
_AZO_DYE = "c1ccc2c(c1)nc(s2)N=Nc1ccc(cc1)N(C)C"
_SMILES = [_ASPIRIN, _CAFFEINE, _PHENACYL_BROMIDE, _AZO_DYE]

#: A miniature table in the upstream column order.  ``max`` is exercised with a
#: non-zero value, which the shipped file never uses, so the "max + 1" match
#: threshold is tested rather than assumed.
_MINI_ROWS = (
    ("1", "1", "Reactive alkyl halides", "[Br,Cl,I][CX4;CH,CH2]", "Glaxo", "8", "0"),
    ("2", "2", "azo group", "[N;R0]=[N;R0]", "PAINS", "6", "0"),
    ("3", "3", "more than one ester", "[CX3](=O)[OX2H0][#6]", "MLSMR", "3", "1"),
)

_HEADER = ["rule_id", "rule_set", "description", "smarts", "rule_set_name", "priority", "max"]


def _write_table(path: Path, rows: tuple[tuple[str, ...], ...]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_HEADER)
        writer.writerows(rows)
    return path


def _mini_table(root: Path) -> Path:
    return _write_table(root / "alert_collection.csv", _MINI_ROWS)


def _config(alerts_path: Path | str, **actions: str) -> dict[str, object]:
    """Every set named explicitly, so a test says exactly what it enabled.

    Defaulting the unmentioned sets would make the assertions on ``rule_counts``
    depend on the shipped defaults, which are the thing several of these tests
    exist to check.
    """

    config: dict[str, object] = {f"{name.lower()}_action": "ignore" for name in RULE_SETS}
    config.update({f"{key.lower()}": value for key, value in actions.items()})
    config["alerts_path"] = str(alerts_path)
    return config


def _vendored_alerts() -> str | None:
    """The asset reference, when the real table is present and verifies."""

    status = asset_status(asset_spec("rd_filters"))
    return DEFAULT_ALERTS_REFERENCE if status.state.value == "ready" else None


def _requires_vendored_alerts() -> str:
    reference = _vendored_alerts()
    if reference is None:
        pytest.skip("run 'molcascade assets fetch rd_filters' to exercise the real table")
    return reference


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
        ref=ArtifactRef(artifact_id="artifact:sha256:" + "c" * 64, kind=PARENT_V1.id),
        root=input_root,
        contract_id=PARENT_V1.id,
    )


def _run(
    root: Path,
    config: dict[str, object],
    *,
    smiles: list[str] | None = None,
    staging_name: str = "out",
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    staging = root / staging_name
    response = RdFiltersAlertsPlugin().execute(
        StageRequest(
            stage_id="alerts",
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


def _alerts_for(decisions: list[dict[str, Any]], parent_id: str) -> dict[str, str]:
    """``rule_set -> outcome`` for one molecule, from the decision rows."""

    found: dict[str, str] = {}
    for row in decisions:
        if row["entity_id"] != parent_id or row["outcome"] == "PASS":
            continue
        found[json.loads(row["detail"])["rule_set"]] = row["outcome"]
    return found


# ------------------------------------------------------------------ contract


def test_the_adapter_is_registered_and_declares_what_it_does() -> None:
    registry = create_builtin_registry()
    assert "chemistry.rd_filters_alerts@0.1.0" in registry
    descriptor = RdFiltersAlertsPlugin.descriptor
    assert descriptor.determinism is Determinism.DETERMINISTIC
    assert descriptor.inputs == (PARENT_V1.id,)
    assert set(descriptor.output_ports) == {"primary", "decisions"}


def test_rdkit_is_imported_lazily() -> None:
    """No third-party import may run when the module is merely loaded.

    Import order is a real constraint here: ``molcascade.plugins.builtin``
    imports every adapter in order to register it, so a module-scope ``from
    rdkit import ...`` would make listing the plugins as expensive as running
    them.
    """

    tree = ast.parse(inspect.getsource(module))
    imported: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not [name for name in imported if name.split(".")[0] == "rdkit"]


def test_every_upstream_rule_set_has_an_action() -> None:
    """A set present in the table but absent from the config would do nothing.

    The adapter refuses such a table at run time; this catches the same mistake
    at review time, when the fix is one line rather than a failed screen.
    """

    actions = RdFiltersAlertConfig().actions()
    assert set(actions) == set(RULE_SETS)
    assert all(isinstance(action, AlertAction) for action in actions.values())


# -------------------------------------------------------------------- mini table


def test_a_matched_set_rejects_and_an_ignored_set_is_not_even_compiled(
    tmp_path: Path,
) -> None:
    response, parents, decisions = _run(
        tmp_path,
        _config(_mini_table(tmp_path), glaxo_action="reject", pains_action="warn"),
    )
    survivors = {row["parent_smiles"] for row in parents}
    assert _PHENACYL_BROMIDE not in survivors  # Glaxo reactive alkyl halide
    assert _AZO_DYE in survivors  # PAINS only warns
    assert _ASPIRIN in survivors and _CAFFEINE in survivors

    assert response.metadata["reject_count"] == 1
    assert response.metadata["output_count"] == 3
    # The ignored MLSMR ester rule contributes nothing at all, so the counts
    # prove the catalog was built from the enabled sets rather than being
    # filtered after the fact.
    assert response.metadata["rule_counts"] == {"Glaxo": 1, "PAINS": 1}
    assert response.metadata["rule_total"] == 2

    azo = _alerts_for(decisions, "m3")
    assert azo == {"PAINS": "WARN"}
    assert _alerts_for(decisions, "m0") == {}  # aspirin's only hit was the ignored set


def test_the_max_column_means_more_than_max_matches(tmp_path: Path) -> None:
    """``max = 1`` fires on the second occurrence, not the first.

    Every rule in the shipped table uses ``max = 0``, so this is the one
    behaviour the real file cannot exercise -- and getting it wrong would make a
    private alert table silently stricter than its author intended.
    """

    one_ester = "CC(=O)OC"
    two_esters = "CC(=O)OCCOC(C)=O"
    _, parents, decisions = _run(
        tmp_path,
        _config(_mini_table(tmp_path), mlsmr_action="reject"),
        smiles=[one_ester, two_esters],
    )
    survivors = {row["parent_smiles"] for row in parents}
    assert one_ester in survivors
    assert two_esters not in survivors
    assert _alerts_for(decisions, "m1") == {"MLSMR": "REJECT"}


def test_a_clear_molecule_still_records_a_decision(tmp_path: Path) -> None:
    """Silence is not evidence.  A molecule that matched nothing says so."""

    _, _, decisions = _run(
        tmp_path,
        _config(_mini_table(tmp_path), glaxo_action="reject", pains_action="warn"),
        smiles=[_CAFFEINE],
    )
    assert [row["outcome"] for row in decisions] == ["PASS"]
    assert decisions[0]["reason_code"] == "RD_FILTERS_ALERTS_CLEAR"


def test_every_molecule_is_accounted_for_in_the_decision_log(tmp_path: Path) -> None:
    """No molecule may pass through the stage without leaving a record."""

    response, _, decisions = _run(
        tmp_path,
        _config(_mini_table(tmp_path), glaxo_action="reject", pains_action="warn"),
    )
    assert {row["entity_id"] for row in decisions} == {"m0", "m1", "m2", "m3"}
    assert len(decisions) == response.metadata["decision_count"]
    assert response.metadata["input_count"] == len(_SMILES)
    assert (
        response.metadata["output_count"] + response.metadata["reject_count"]
        == response.metadata["input_count"]
    )
    assert all(row["rule_id"] == response.metadata["policy_id"] for row in decisions)
    assert all(row["stage_id"] == "alerts" for row in decisions)


def test_the_policy_id_changes_when_the_actions_change(tmp_path: Path) -> None:
    """Two runs that filtered differently must not look identical in provenance."""

    table = _mini_table(tmp_path)
    strict, _, _ = _run(
        tmp_path, _config(table, pains_action="reject"), staging_name="strict"
    )
    lenient, _, _ = _run(
        tmp_path, _config(table, pains_action="warn"), staging_name="lenient"
    )
    assert strict.metadata["policy_id"] != lenient.metadata["policy_id"]
    # ...and the same settings must produce the same id, or nothing downstream
    # could tell a rerun from a re-parameterisation.
    repeat, _, _ = _run(
        tmp_path, _config(table, pains_action="reject"), staging_name="repeat"
    )
    assert repeat.metadata["policy_id"] == strict.metadata["policy_id"]


# ---------------------------------------------------------------- fail closed


def test_ignoring_every_set_is_refused_rather_than_run(tmp_path: Path) -> None:
    with pytest.raises(PluginError) as error:
        _run(tmp_path, _config(_mini_table(tmp_path)))
    assert error.value.code == "RD_FILTERS_ALERTS_NO_RULE_SET_ENABLED"


def test_a_relative_path_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PluginError) as error:
        _run(tmp_path, _config("alert_collection.csv", glaxo_action="reject"))
    assert error.value.code == "RD_FILTERS_ALERTS_PATH_INVALID"


def test_a_missing_file_fails_closed_without_reaching_for_the_network(
    tmp_path: Path,
) -> None:
    with pytest.raises(PluginError) as error:
        _run(tmp_path, _config(tmp_path / "absent.csv", glaxo_action="reject"))
    assert error.value.code == "RD_FILTERS_ALERTS_UNREADABLE"
    assert "fetch" in (error.value.hint or "")


def test_a_wrong_digest_stops_the_stage(tmp_path: Path) -> None:
    config = _config(_mini_table(tmp_path), glaxo_action="reject")
    config["expected_alerts_sha256"] = "0" * 64
    with pytest.raises(PluginError) as error:
        _run(tmp_path, config)
    assert error.value.code == "RD_FILTERS_ALERTS_DIGEST_MISMATCH"


def test_an_uncompilable_smarts_stops_the_stage_rather_than_being_skipped(
    tmp_path: Path,
) -> None:
    """Upstream warns and continues; here that would make the run irreproducible.

    If a bad rule were skipped, the number of rules applied would depend on the
    RDKit build, and two runs pinned to the same alert digest could then
    disagree about which molecules survived.
    """

    path = _write_table(
        tmp_path / "alert_collection.csv",
        (("1", "1", "not smarts", "[[[", "Glaxo", "8", "0"),),
    )
    with pytest.raises(PluginError) as error:
        _run(tmp_path, _config(path, glaxo_action="reject"))
    assert error.value.code == "RD_FILTERS_ALERTS_SMARTS_INVALID"


def test_a_repeated_rule_id_is_refused(tmp_path: Path) -> None:
    """Duplicate ids would make the per-set counts silently wrong."""

    path = _write_table(
        tmp_path / "alert_collection.csv",
        (
            ("1", "1", "nitro", "[N+](=O)[O-]", "Glaxo", "8", "0"),
            ("1", "1", "azide", "[N-]=[N+]=N", "Glaxo", "8", "0"),
        ),
    )
    with pytest.raises(PluginError) as error:
        _run(tmp_path, _config(path, glaxo_action="reject"))
    assert error.value.code == "RD_FILTERS_ALERTS_UNPARSEABLE"


def test_an_unknown_rule_set_is_refused(tmp_path: Path) -> None:
    """A set with no configured action would be applied with no action at all."""

    path = _write_table(
        tmp_path / "alert_collection.csv",
        (("1", "1", "nitro", "[N+](=O)[O-]", "HouseRules", "8", "0"),),
    )
    with pytest.raises(PluginError) as error:
        _run(tmp_path, _config(path, glaxo_action="reject"))
    assert error.value.code == "RD_FILTERS_ALERTS_UNKNOWN_RULE_SET"


def test_a_table_without_the_upstream_columns_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "alert_collection.csv"
    path.write_text("id,pattern\n1,[N+](=O)[O-]\n", encoding="utf-8")
    with pytest.raises(PluginError) as error:
        _run(tmp_path, _config(path, glaxo_action="reject"))
    assert error.value.code == "RD_FILTERS_ALERTS_UNPARSEABLE"


def test_a_misspelt_action_is_refused_at_configuration_time(tmp_path: Path) -> None:
    with pytest.raises(PluginError) as error:
        _run(tmp_path, _config(_mini_table(tmp_path), glaxo_action="rejct"))
    assert error.value.code == "PLUGIN_CONFIG_INVALID"


# ------------------------------------------------------- the real vendored table


def test_the_vendored_collection_matches_what_the_papers_describe(tmp_path: Path) -> None:
    """The published rules, applied to molecules with known liabilities.

    This is the test the adapter exists for.  Anyone reproducing a paper that
    says "we used rd_filters" needs these 1,251 rules, so the counts per set are
    asserted against the shipped file rather than against whatever happens to be
    on disk.
    """

    reference = _requires_vendored_alerts()
    response, parents, decisions = _run(
        tmp_path,
        # Every set on and every set warning, so the decision rows show the
        # whole picture and nothing is removed before it can be counted.
        _config(reference, **{f"{name.lower()}_action": "warn" for name in RULE_SETS}),
    )
    assert response.metadata["rule_total"] == 1251
    assert response.metadata["rule_counts"] == {
        "BMS": 180,
        "Dundee": 105,
        "Glaxo": 55,
        "Inpharmatica": 91,
        "LINT": 57,
        "MLSMR": 116,
        "PAINS": 481,
        "SureChEMBL": 166,
    }
    assert len(parents) == len(_SMILES)  # nothing rejected while everything warns

    # An aryl azo dye is the canonical multi-set liability: PAINS calls it an
    # interference motif, and four independent reactive-group lists agree.
    azo = _alerts_for(decisions, "m3")
    assert "PAINS" in azo
    assert {"BMS", "Dundee", "MLSMR", "SureChEMBL"} <= set(azo)

    # An alpha-halo ketone is a Glaxo hard filter and not a PAINS compound.
    bromide = _alerts_for(decisions, "m2")
    assert "Glaxo" in bromide
    assert "PAINS" not in bromide

    # Aspirin's liability is its ester, which is exactly what Dundee sees.
    assert "Dundee" in _alerts_for(decisions, "m0")


def test_the_shipped_defaults_do_not_delete_approved_drugs(tmp_path: Path) -> None:
    """The calibration behind the default actions, kept where it can fail.

    Alert sets differ by an order of magnitude in how much drug-like matter
    they reject, and the difference is invisible unless it is measured.  Setting
    Dundee or MLSMR to ``reject`` -- both superficially reasonable choices --
    would delete a third and a half of this panel respectively.  If someone
    changes the defaults, this test tells them what it costs.
    """

    reference = _requires_vendored_alerts()
    panel = {
        "aspirin": _ASPIRIN,
        "acetaminophen": "CC(=O)Nc1ccc(O)cc1",
        "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "caffeine": _CAFFEINE,
        "metformin": "CN(C)C(=N)NC(N)=N",
        "warfarin": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
        "omeprazole": "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
        "fluoxetine": "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",
        "ciprofloxacin": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
        "losartan": "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1",
        "gefitinib": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
        "imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
        "celecoxib": "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
        "metoprolol": "COCCc1ccc(OCC(O)CNC(C)C)cc1",
        "sertraline": "CN[C@H]1CC[C@@H](c2ccc(Cl)c(Cl)c2)c2ccccc21",
        "tamoxifen": r"CC/C(=C(\c1ccccc1)c1ccccc1)c1ccc(OCCN(C)C)cc1",
    }
    names = list(panel)
    # The shipped defaults, unmodified: no _config() here on purpose.
    _, parents, _ = _run(
        tmp_path, {"alerts_path": reference}, smiles=[panel[name] for name in names]
    )
    survivors = {row["parent_smiles"] for row in parents}
    deleted = [name for name in names if panel[name] not in survivors]
    assert deleted == [], f"the default actions rejected approved drugs: {deleted}"


def test_the_shipped_defaults_still_delete_reactive_chemistry(tmp_path: Path) -> None:
    """Passing every drug is only impressive if the filter rejects anything.

    A beta-lactam and an alpha-halo ketone are the two cases the reject-listed
    sets exist for.  Both are real drugs or drug-like, and both are things a
    screening deck should exclude, which is the distinction the defaults draw.
    """

    reference = _requires_vendored_alerts()
    amoxicillin = "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O"
    _, parents, decisions = _run(
        tmp_path,
        {"alerts_path": reference},
        smiles=[_PHENACYL_BROMIDE, amoxicillin, _CAFFEINE],
    )
    survivors = {row["parent_smiles"] for row in parents}
    assert _PHENACYL_BROMIDE not in survivors
    assert amoxicillin not in survivors
    assert _CAFFEINE in survivors

    # ...and the reason is recorded, not merely the removal.
    assert _alerts_for(decisions, "m0")["Glaxo"] == "REJECT"
    assert _alerts_for(decisions, "m1")["Glaxo"] == "REJECT"


def test_the_same_input_produces_byte_identical_decisions(tmp_path: Path) -> None:
    """A molecule matching seven sets must log them in one fixed order."""

    reference = _requires_vendored_alerts()
    config = _config(reference, **{f"{name.lower()}_action": "warn" for name in RULE_SETS})
    _, _, first = _run(tmp_path, config, smiles=[_AZO_DYE], staging_name="first")
    _, _, second = _run(tmp_path, config, smiles=[_AZO_DYE], staging_name="second")
    assert first == second
    assert len(first) == 7  # every set except Glaxo flags this dye
