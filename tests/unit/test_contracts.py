from __future__ import annotations

import pyarrow as pa
import pytest

from molcascade.contracts import (
    APPLICABILITY_V1,
    CLUSTER_ASSIGNMENT_V1,
    CONTRACTS,
    DECISION_V1,
    DERIVED_METRIC_V1,
    DRUG_LIKENESS_V1,
    FINGERPRINT_V1,
    PARENT_SOURCE_MAP_V1,
    PARENT_V1,
    PREDICTION_V1,
    PROPERTY_V1,
    RAW_MOLECULE_V1,
    RAW_MOLECULE_V2,
    SCAFFOLD_ASSIGNMENT_V1,
    SELECTION_DECISION_V1,
    SYNTHESIS_SCORE_V1,
    ContractRegistry,
    DataContract,
    get_contract,
    validate_table,
)
from molcascade.errors import ContractError


def test_screening_contracts_are_registered_and_toolkit_neutral() -> None:
    contracts = (
        PROPERTY_V1,
        DRUG_LIKENESS_V1,
        FINGERPRINT_V1,
        PREDICTION_V1,
        APPLICABILITY_V1,
        SYNTHESIS_SCORE_V1,
        SCAFFOLD_ASSIGNMENT_V1,
        CLUSTER_ASSIGNMENT_V1,
        SELECTION_DECISION_V1,
    )
    for contract in contracts:
        assert CONTRACTS[contract.id] is contract
        assert all("pickle" not in str(field.type).lower() for field in contract.schema)


def _table(contract: DataContract, values: dict[str, list[object]]) -> pa.Table:
    schema = pa.schema(
        [contract.schema.field(name) for name in values],
        metadata=contract.schema.metadata,
    )
    return pa.Table.from_pydict(values, schema=schema)


def test_builtin_contracts_are_exact_versioned_and_toolkit_neutral() -> None:
    assert set(CONTRACTS) == {
        "applicability/v1",
        "cluster_assignment/v1",
        "raw_molecule/v1",
        "raw_molecule/v2",
        "parent/v1",
        "parent_source_map/v1",
        "decision/v1",
        "derived_metric/v1",
        "docking_score/v1",
        "drug_likeness/v1",
        "fingerprint/v1",
        "ligand_conformer/v1",
        "md_system_input/v1",
        "prediction/v1",
        "property/v1",
        "scaffold_assignment/v1",
        "selection_decision/v1",
        "shortlist_export/v1",
        "synthesis_score/v1",
    }
    assert get_contract("raw_molecule/v1") is RAW_MOLECULE_V1
    assert get_contract("raw_molecule/v2") is RAW_MOLECULE_V2
    assert RAW_MOLECULE_V1.primary_key == ("source_record_id",)
    assert PARENT_V1.required_columns == (
        "parent_id",
        "identity_policy_id",
        "parent_smiles",
        "registration_key",
        "duplicate_count",
    )
    assert DECISION_V1.primary_key == (
        "entity_id",
        "entity_kind",
        "stage_id",
        "reason_code",
    )


def test_raw_molecule_v2_replaces_only_the_structure_payload_columns() -> None:
    provenance_columns = (
        "source_record_id",
        "source_kind",
        "source_uri",
        "source_index",
        "generator_id",
        "batch_id",
        "source_candidate_id",
        "source_metadata_json",
    )

    assert tuple(
        name for name in RAW_MOLECULE_V2.schema.names if name in provenance_columns
    ) == provenance_columns
    assert "raw_smiles" not in RAW_MOLECULE_V2.schema.names
    assert "raw_molblock" not in RAW_MOLECULE_V2.schema.names
    assert RAW_MOLECULE_V2.schema.field("raw_format") == pa.field(
        "raw_format", pa.string(), nullable=False
    )
    assert RAW_MOLECULE_V2.schema.field("raw_structure") == pa.field(
        "raw_structure", pa.large_string(), nullable=False
    )
    assert RAW_MOLECULE_V2.enum_values["raw_format"] == {
        "SMILES",
        "MOLBLOCK",
        "MOL2",
    }


def test_raw_molecule_v2_validates_format_and_non_null_structure() -> None:
    valid = _table(
        RAW_MOLECULE_V2,
        {
            "source_record_id": ["batch:0"],
            "source_kind": ["mol2"],
            "source_index": [0],
            "raw_format": ["MOL2"],
            "raw_structure": ["@<TRIPOS>MOLECULE\nexample"],
        },
    )
    assert validate_table(valid, RAW_MOLECULE_V2).row_count == 1

    invalid_format = _table(
        RAW_MOLECULE_V2,
        {
            "source_record_id": ["batch:1"],
            "source_kind": ["pdb"],
            "source_index": [1],
            "raw_format": ["PDB"],
            "raw_structure": ["ATOM"],
        },
    )
    with pytest.raises(ContractError) as format_error:
        validate_table(invalid_format, RAW_MOLECULE_V2)
    assert format_error.value.code == "CONTRACT_ENUM_INVALID"

    null_structure = _table(
        RAW_MOLECULE_V2,
        {
            "source_record_id": ["batch:2"],
            "source_kind": ["mol2"],
            "source_index": [2],
            "raw_format": ["MOL2"],
            "raw_structure": [None],
        },
    )
    with pytest.raises(ContractError) as null_error:
        validate_table(null_structure, RAW_MOLECULE_V2)
    assert null_error.value.code == "CONTRACT_NULL_VALUE"


def test_valid_table_may_omit_declared_optional_columns() -> None:
    table = _table(
        RAW_MOLECULE_V1,
        {
            "source_record_id": ["batch:0", "batch:1"],
            "source_kind": ["delimited", "delimited"],
            "source_index": [0, 1],
            "raw_smiles": ["C", "CC"],
        },
    )

    report = validate_table(table, RAW_MOLECULE_V1)

    assert report.row_count == 2
    assert report.contract_id == "raw_molecule/v1"


def test_record_batch_is_accepted() -> None:
    table = _table(
        RAW_MOLECULE_V1,
        {
            "source_record_id": ["batch:0"],
            "source_kind": ["delimited"],
            "source_index": [0],
            "raw_smiles": ["C"],
        },
    )

    assert RAW_MOLECULE_V1.validate(table.to_batches()[0]).row_count == 1


def test_missing_unknown_and_wrong_type_columns_are_structured() -> None:
    schema = pa.schema(
        [
            pa.field("source_record_id", pa.string(), nullable=False),
            pa.field("source_index", pa.string(), nullable=False),
            pa.field("unexpected", pa.string(), nullable=True),
        ]
    )

    with pytest.raises(ContractError) as caught:
        RAW_MOLECULE_V1.validate_schema(schema)

    assert caught.value.code == "CONTRACT_SCHEMA_MISMATCH"
    assert caught.value.context["missing_required_columns"] == ["source_kind"]
    assert caught.value.context["unexpected_columns"] == ["unexpected"]
    assert caught.value.context["type_mismatches"][0]["column"] == "source_index"


def test_weaker_nullable_schema_is_rejected() -> None:
    nullable_schema = pa.schema(
        [
            pa.field("source_record_id", pa.string()),
            pa.field("source_kind", pa.string()),
            pa.field("source_index", pa.int64()),
        ]
    )

    with pytest.raises(ContractError) as caught:
        RAW_MOLECULE_V1.validate_schema(nullable_schema)

    assert caught.value.code == "CONTRACT_SCHEMA_MISMATCH"
    assert set(caught.value.context["nullable_required_columns"]) == {
        "source_record_id",
        "source_kind",
        "source_index",
    }


def test_raw_structure_requires_nonempty_smiles_or_molblock() -> None:
    table = _table(
        RAW_MOLECULE_V1,
        {
            "source_record_id": ["batch:0", "batch:1"],
            "source_kind": ["delimited", "sdf"],
            "source_index": [0, 1],
            "raw_smiles": [None, "   "],
            "raw_molblock": [None, "\n\t"],
        },
    )

    with pytest.raises(ContractError) as caught:
        validate_table(table, RAW_MOLECULE_V1)

    assert caught.value.code == "CONTRACT_INVARIANT_VIOLATION"
    assert caught.value.context["columns"] == ["raw_smiles", "raw_molblock"]
    assert caught.value.context["invalid_row_count"] == 2


def test_raw_molblock_alone_satisfies_structure_invariant() -> None:
    table = _table(
        RAW_MOLECULE_V1,
        {
            "source_record_id": ["batch:0"],
            "source_kind": ["sdf"],
            "source_index": [0],
            "raw_molblock": ["example molblock"],
        },
    )

    assert validate_table(table, RAW_MOLECULE_V1).row_count == 1


def test_primary_key_duplicates_are_rejected_with_examples() -> None:
    table = _table(
        RAW_MOLECULE_V1,
        {
            "source_record_id": ["same", "same"],
            "source_kind": ["delimited", "delimited"],
            "source_index": [0, 1],
            "raw_smiles": ["C", "CC"],
        },
    )

    with pytest.raises(ContractError) as caught:
        validate_table(table, RAW_MOLECULE_V1)

    assert caught.value.code == "CONTRACT_PRIMARY_KEY_DUPLICATE"
    assert caught.value.context["examples"] == [{"source_record_id": "same"}]


def test_composite_primary_key_is_checked_as_a_tuple() -> None:
    contract = DataContract(
        id="composite/v1",
        schema=pa.schema(
            [
                pa.field("left", pa.string(), nullable=False),
                pa.field("right", pa.int64(), nullable=False),
            ]
        ),
        primary_key=("left", "right"),
    )
    table = _table(contract, {"left": ["x", "x"], "right": [1, 1]})

    with pytest.raises(ContractError, match="not unique") as caught:
        contract.validate(table)

    assert caught.value.context["primary_key"] == ["left", "right"]


def test_enum_values_are_checked_in_batch() -> None:
    table = _table(
        DECISION_V1,
        {
            "entity_id": ["p1", "p2"],
            "entity_kind": ["PARENT", "PARENT"],
            "stage_id": ["gate", "gate"],
            "outcome": ["PASS", "MAYBE"],
            "reason_code": ["OK", "UNKNOWN"],
        },
    )

    with pytest.raises(ContractError) as caught:
        validate_table(table, DECISION_V1)

    assert caught.value.code == "CONTRACT_ENUM_INVALID"
    assert caught.value.context["column"] == "outcome"
    assert caught.value.context["invalid"] == ["MAYBE"]


def _derived_metric_row(**overrides: object) -> dict[str, list[object]]:
    """One valid ``derived_metric/v1`` row, so each test states only its own point.

    Every column is spelled out because the evidence contracts declare no
    optional columns: a nullable column may be null, but it may not be absent.
    """

    row: dict[str, object] = {
        "parent_id": "p1",
        "metric_id": "ligand_efficiency",
        "method_id": "le:hac",
        "value": 0.34,
        "units": "KCAL_PER_MOL_PER_HEAVY_ATOM",
        "direction": "HIGHER_BETTER",
        "status": "OK",
        "status_detail": None,
        "source_json": None,
    }
    row.update(overrides)
    return {name: [value] for name, value in row.items()}


def _derived_metric_rows(*rows: dict[str, list[object]]) -> dict[str, list[object]]:
    columns = rows[0].keys()
    assert all(row.keys() == columns for row in rows)
    return {name: [value for row in rows for value in row[name]] for name in columns}


def test_derived_metric_keeps_one_row_per_metric_and_method() -> None:
    """Several metrics on one molecule coexist; the same metric twice does not.

    The primary key is what lets a strain number, a ligand efficiency and a
    specificity delta share a table without a column per metric, and what stops
    a re-run under different parameters from quietly overwriting the earlier
    measurement instead of sitting beside it.
    """

    distinct = _table(
        DERIVED_METRIC_V1,
        _derived_metric_rows(
            _derived_metric_row(),
            _derived_metric_row(metric_id="pose_strain", method_id="mmff94s:12", value=1.84),
            _derived_metric_row(method_id="le:hac^2", value=0.013),
        ),
    )
    validate_table(distinct, DERIVED_METRIC_V1)

    repeated = _table(
        DERIVED_METRIC_V1,
        _derived_metric_rows(_derived_metric_row(), _derived_metric_row(value=0.41)),
    )
    with pytest.raises(ContractError) as caught:
        validate_table(repeated, DERIVED_METRIC_V1)

    assert caught.value.code == "CONTRACT_PRIMARY_KEY_DUPLICATE"
    assert caught.value.context["primary_key"] == ["parent_id", "metric_id", "method_id"]


def test_derived_metric_records_a_failure_without_inventing_a_number() -> None:
    """A metric that could not be computed keeps its row, with a null value."""

    table = _table(
        DERIVED_METRIC_V1,
        _derived_metric_row(
            metric_id="pose_strain",
            method_id="mmff94s:12",
            value=None,
            units="KCAL_PER_MOL",
            direction="LOWER_BETTER",
            status="BACKEND_FAILED",
            status_detail="MMFF typing failed",
        ),
    )

    validate_table(table, DERIVED_METRIC_V1)


def test_derived_metric_out_of_domain_keeps_the_number_it_measured() -> None:
    """Negative strain is evidence the reference search failed, not a null.

    Nulling it would erase the only signal that says so, which is why
    OUT_OF_DOMAIN is a status rather than a reason to drop the value.
    """

    table = _table(
        DERIVED_METRIC_V1,
        _derived_metric_row(
            metric_id="pose_strain",
            method_id="mmff94s:12",
            value=-19.42,
            units="KCAL_PER_MOL",
            direction="LOWER_BETTER",
            status="OUT_OF_DOMAIN",
            status_detail="ensemble minimum above the relaxed pose",
        ),
    )

    validate_table(table, DERIVED_METRIC_V1)


def test_derived_metric_requires_units_even_where_the_value_is_missing() -> None:
    """``units`` is the anchor a gate checks its threshold against.

    A row that carries no scale cannot be gated at all, so it is refused at the
    contract rather than left for a gate to interpret.
    """

    table = _table(
        DERIVED_METRIC_V1,
        _derived_metric_row(value=None, units=None, status="BACKEND_FAILED"),
    )

    with pytest.raises(ContractError) as caught:
        validate_table(table, DERIVED_METRIC_V1)

    assert caught.value.code == "CONTRACT_NULL_VALUE"
    assert caught.value.context["null_counts"] == {"units": 1}


def test_derived_metric_units_and_direction_are_closed_vocabularies() -> None:
    """Spelling a unit freely is what the enum exists to catch.

    ``8`` in kcal/mol and ``8`` in torsion energy units are different filter
    strengths, and both are small positive reals -- nothing in the row itself
    would reveal the mix-up later.
    """

    misspelled = _table(DERIVED_METRIC_V1, _derived_metric_row(units="kcal/mol/heavy_atom"))
    with pytest.raises(ContractError) as units_caught:
        validate_table(misspelled, DERIVED_METRIC_V1)
    assert units_caught.value.code == "CONTRACT_ENUM_INVALID"
    assert units_caught.value.context["column"] == "units"
    assert units_caught.value.context["invalid"] == ["kcal/mol/heavy_atom"]

    house_style = _table(DERIVED_METRIC_V1, _derived_metric_row(direction="HIGHER_IS_BETTER"))
    with pytest.raises(ContractError) as direction_caught:
        validate_table(house_style, DERIVED_METRIC_V1)
    assert direction_caught.value.code == "CONTRACT_ENUM_INVALID"
    assert direction_caught.value.context["column"] == "direction"

    unknown_status = _table(DERIVED_METRIC_V1, _derived_metric_row(status="UNSCORABLE"))
    with pytest.raises(ContractError) as status_caught:
        validate_table(unknown_status, DERIVED_METRIC_V1)
    assert status_caught.value.code == "CONTRACT_ENUM_INVALID"
    assert status_caught.value.context["column"] == "status"


def test_mapping_role_enum_is_exposed_as_immutable_metadata() -> None:
    assert PARENT_SOURCE_MAP_V1.enum_values["relation"] == {
        "PRIMARY",
        "DUPLICATE",
    }
    with pytest.raises(TypeError):
        PARENT_SOURCE_MAP_V1.enums["relation"] = frozenset({"OTHER"})  # type: ignore[index]
    with pytest.raises(TypeError):
        PARENT_SOURCE_MAP_V1.invariants["new"] = "mutable"  # type: ignore[index]


def test_contract_registry_never_overwrites_duplicate_or_hides_unknown() -> None:
    registry = ContractRegistry([RAW_MOLECULE_V1])

    with pytest.raises(ContractError) as duplicate:
        registry.register(RAW_MOLECULE_V1)
    assert duplicate.value.code == "CONTRACT_DUPLICATE"

    with pytest.raises(ContractError) as missing:
        registry.get("missing/v1")
    assert missing.value.code == "CONTRACT_NOT_FOUND"


def test_data_contract_definition_rejects_primary_key_outside_required_columns() -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("value", pa.string(), nullable=False),
        ]
    )
    with pytest.raises(ValueError, match="primary-key columns must also be required"):
        DataContract(
            id="bad/v1",
            schema=schema,
            primary_key=("id",),
            required_columns=("value",),
        )

    with pytest.raises(ValueError, match="must be declared non-nullable"):
        DataContract(
            id="nullable_key/v1",
            schema=pa.schema([pa.field("id", pa.string(), nullable=True)]),
            primary_key=("id",),
        )
