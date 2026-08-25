from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import rdBase

from molcascade.chemistry import IdentityPolicy, identity_policy_id, standardize_parent

FIXTURE = Path(__file__).parent / "fixtures" / "identity_rdkit_2026_03_1.json"


def test_rdkit_2026_03_1_exact_identity_lock() -> None:
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if rdBase.rdkitVersion != expected["rdkit_version"]:
        pytest.skip(
            "exact identity golden is intentionally locked to RDKit "
            f"{expected['rdkit_version']}; installed {rdBase.rdkitVersion}"
        )

    policy = IdentityPolicy()
    assert identity_policy_id(policy) == expected["identity_policy_id"]
    for case in expected["cases"].values():
        result = standardize_parent(raw_smiles=case["input_smiles"], policy=policy)
        assert {
            "formula": result.formula,
            "parent_id": result.parent_id,
            "parent_smiles": result.parent_smiles,
            "registration_key": result.registration_key,
            "stereo_key": result.stereo_key,
        } == {key: value for key, value in case.items() if key != "input_smiles"}
