from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from molcascade.chemistry import (
    HardGateEvaluator,
    HardGatePolicy,
    HardSmartsRule,
    hard_gate_policy_id,
)


def test_hard_gate_policy_is_strict_and_frozen() -> None:
    policy = HardGatePolicy()

    with pytest.raises(ValidationError, match="frozen"):
        policy.flag_pains = False  # type: ignore[misc]
    with pytest.raises(ValidationError):
        HardGatePolicy.model_validate({"flag_pains": 1})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        HardGatePolicy.model_validate({"reject_pains": True})
    assert hard_gate_policy_id(policy).startswith("hard-gate-policy:sha256:")


def test_disallowed_element_is_a_hard_rejection() -> None:
    result = HardGateEvaluator().evaluate("C[Na]")

    assert not result.passed
    assert {finding.reason_code for finding in result.rejects} == {"DISALLOWED_ELEMENT"}
    assert '"atomic_number":11' in result.rejects[0].detail


def test_project_smarts_is_a_hard_rejection() -> None:
    policy = HardGatePolicy(
        hard_smarts=(
            HardSmartsRule(
                id="peroxide",
                smarts="[O]-[O]",
                reason_code="SEVERE_REACTIVITY",
                description="project excludes peroxide bonds",
            ),
        )
    )
    result = HardGateEvaluator(policy).evaluate("COO")

    assert not result.passed
    assert result.rejects[0].reason_code == "SEVERE_REACTIVITY"
    assert result.rejects[0].rule_id == "peroxide"
    assert '"atom_matches":[[1,2]]' in result.rejects[0].detail


def test_project_smarts_defaults_to_chirality_sensitive_matching() -> None:
    rule = HardSmartsRule(
        id="configured_enantiomer",
        smarts="F[C@H](Cl)Br",
        reason_code="CONFIGURED_ENANTIOMER",
        description="project excludes only the configured enantiomer",
    )
    evaluator = HardGateEvaluator(
        HardGatePolicy(hard_smarts=(rule,), flag_pains=False)
    )

    assert not evaluator.evaluate("F[C@H](Cl)Br").passed
    assert evaluator.evaluate("F[C@@H](Cl)Br").passed
    assert evaluator.evaluate("FC(Cl)Br").passed


def test_project_can_explicitly_disable_smarts_chirality() -> None:
    rule = HardSmartsRule(
        id="either_enantiomer",
        smarts="F[C@H](Cl)Br",
        reason_code="EITHER_ENANTIOMER",
        description="project intentionally ignores stereochemistry",
        use_chirality=False,
    )
    evaluator = HardGateEvaluator(
        HardGatePolicy(hard_smarts=(rule,), flag_pains=False)
    )

    assert not evaluator.evaluate("F[C@H](Cl)Br").passed
    assert not evaluator.evaluate("F[C@@H](Cl)Br").passed
    assert not evaluator.evaluate("FC(Cl)Br").passed


def test_smarts_chirality_is_strict_frozen_and_part_of_policy_identity() -> None:
    rule = HardSmartsRule(
        id="stereo_rule",
        smarts="F[C@H](Cl)Br",
        description="test chirality policy identity",
    )
    sensitive = HardGatePolicy(hard_smarts=(rule,))
    insensitive = HardGatePolicy(
        hard_smarts=(rule.model_copy(update={"use_chirality": False}),)
    )

    assert hard_gate_policy_id(sensitive) != hard_gate_policy_id(insensitive)
    with pytest.raises(ValidationError, match="frozen"):
        rule.use_chirality = False  # type: ignore[misc]
    with pytest.raises(ValidationError):
        HardSmartsRule.model_validate(
            {
                "id": "bad_bool",
                "smarts": "C",
                "description": "strict booleans only",
                "use_chirality": 1,
            }
        )


def test_smarts_match_detail_reports_truncation() -> None:
    evaluator = HardGateEvaluator(
        HardGatePolicy(
            hard_smarts=(
                HardSmartsRule(
                    id="carbon",
                    smarts="C",
                    reason_code="CARBON_MATCH",
                    description="test bounded match diagnostics",
                ),
            ),
            flag_pains=False,
        )
    )

    result = evaluator.evaluate("C" * 20)
    detail = json.loads(result.rejects[0].detail)["matched_rules"][0]

    assert detail["atom_match_limit"] == 16
    assert len(detail["atom_matches"]) == 16
    assert detail["atom_matches_truncated"] is True


def test_multicomponent_parent_is_a_hard_rejection() -> None:
    result = HardGateEvaluator().evaluate("CCO.[Na+]")

    assert not result.passed
    assert result.rejects[0].reason_code == "MULTICOMPONENT_PARENT"


def test_query_bond_is_not_accepted_as_a_concrete_parent() -> None:
    result = HardGateEvaluator().evaluate("C~C")

    assert not result.passed
    assert result.rejects[0].reason_code == "QUERY_FEATURE_NOT_ALLOWED"


@pytest.mark.parametrize(
    "smiles",
    ["CC |Sg:n:0,1:ht|", "c1ccccc1 |LN:1:1.3|"],
)
def test_polymer_structural_metadata_is_not_accepted_by_small_molecule_gate(
    smiles: str,
) -> None:
    result = HardGateEvaluator().evaluate(smiles)

    assert not result.passed
    assert result.rejects[0].reason_code == "UNSUPPORTED_STRUCTURAL_METADATA"


def test_pains_is_warning_and_never_a_builtin_hard_failure() -> None:
    result = HardGateEvaluator().evaluate("O=C1C=CC(=O)C=C1")

    assert result.passed
    assert not result.rejects
    assert {finding.reason_code for finding in result.warnings} == {"PAINS_ALERT"}


def test_invalid_smarts_is_rejected_during_policy_validation() -> None:
    with pytest.raises(ValidationError, match="not parseable"):
        HardSmartsRule(
            id="bad",
            smarts="[",
            description="invalid test query",
        )


def test_custom_rule_cannot_collide_with_builtin_decision_codes() -> None:
    with pytest.raises(ValidationError, match="reserved"):
        HardSmartsRule(
            id="not_really_pains",
            smarts="C",
            reason_code="PAINS_ALERT",
            description="reserved-code collision",
        )
