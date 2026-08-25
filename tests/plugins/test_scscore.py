"""The SCScore synthetic-complexity adapter.

The published weight file is a 40 MB download that no test should depend on, so
these tests build small networks of the same *shape* and check the parts that
can actually go wrong: the file-trust rules, the shape validation the reference
implementation does not do, the featurization, and the arithmetic.

Two of those deserve a note.

*The forward pass is checked against a transcription of the reference*, not
against a fixture of expected numbers.  A fixture would only prove the code
still does what it did yesterday; comparing against ``standalone_model_numpy``'s
own loop proves it does what SCScore does.  MolCascade evaluates a whole batch
at once, so this is a real difference in code, not a restatement.

*The counts fold is checked bit-for-bit against RDKit's legacy call.*  The
reference folds an unfolded sparse fingerprint with ``k % FP_len`` into
``uint8``, which is not what a folded count fingerprint of that width contains.
Getting this subtly wrong would still produce plausible 1-5 scores.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import PARENT_V1, SYNTHESIS_SCORE_V1
from molcascade.errors import PluginError
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.scscore import (
    ScScoreConfig,
    ScScorePlugin,
    _fingerprints,
    _forward,
    _parse_layers,
)
from molcascade.plugins.manifest import Cardinality, Determinism, PluginKind
from molcascade.plugins.registry import create_builtin_registry

#: Ethanol, aspirin, caffeine, a steroid.  Ordered by rough complexity so a
#: sanity check on ranking has something to say.
_SMILES = [
    "CCO",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C(=O)N(C)c2ncn(C)c2C1=O",
    "C[C@@H]1CC[C@H]2[C@@H]3CC[C@H]4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
]

_FP_BITS = 1024
_HIDDEN = 24


# ------------------------------------------------------------------- fixtures


def _network(
    *,
    bits: int = _FP_BITS,
    hidden: int = _HIDDEN,
    seed: int = 20_260_824,
) -> list[np.ndarray]:
    """A flat ``[W0, b0, W1, b1, W2, b2]`` list shaped like the real model."""

    rng = np.random.default_rng(seed)
    return [
        rng.normal(0.0, 0.05, (bits, hidden)),
        rng.normal(0.0, 0.05, hidden),
        rng.normal(0.0, 0.05, (hidden, hidden)),
        rng.normal(0.0, 0.05, hidden),
        rng.normal(0.0, 0.5, (hidden, 1)),
        rng.normal(0.0, 0.5, 1),
    ]


def _write_weights(path: Path, arrays: list[Any]) -> str:
    """Serialize as the scscore repository does and return the sha256."""

    payload = gzip.compress(
        json.dumps([np.asarray(entry).tolist() for entry in arrays]).encode("utf-8")
    )
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def weights(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "model.ckpt-10654.as_numpy.json.gz"
    return path, _write_weights(path, _network())


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
    response = ScScorePlugin().execute(
        StageRequest(
            stage_id="scscore",
            # `smiles or _SMILES` would quietly substitute the default for the
            # empty-input case, which is one of the things under test.
            inputs={"parents": _input(root, _SMILES if smiles is None else smiles)},
            config=config,
        ),
        StageContext(staging),
    )
    parents = pq.read_table(staging / response.outputs["primary"].file_paths[0])
    scores = pq.read_table(staging / response.outputs["synthesis_scores"].file_paths[0])
    PARENT_V1.validate(parents)
    SYNTHESIS_SCORE_V1.validate(scores)
    return response, parents.to_pylist(), scores.to_pylist()


def _config(weights: tuple[Path, str], **overrides: object) -> dict[str, object]:
    path, digest = weights
    settings: dict[str, object] = {
        "schema_version": 1,
        "weights_path": str(path),
        "expected_weights_sha256": digest,
    }
    settings.update(overrides)
    return settings


# ------------------------------------------------------------------- contract


def test_the_plugin_registers_as_a_deterministic_one_to_one_synthesis_stage() -> None:
    registry = create_builtin_registry()
    descriptor = registry.get("synthesis.scscore@0.1.0").descriptor
    assert descriptor.kind is PluginKind.SYNTHESIS
    assert descriptor.determinism is Determinism.DETERMINISTIC
    # Scoring is evidence, not a verdict: every molecule comes out again, and
    # a separate gate decides what the number means.
    assert descriptor.cardinality is Cardinality.ONE_TO_ONE
    assert descriptor.output_ports == {
        "primary": PARENT_V1.id,
        "synthesis_scores": SYNTHESIS_SCORE_V1.id,
    }


def test_it_needs_no_optional_python_package() -> None:
    """SCScore's whole appeal is that the weights are plain numpy.

    If this adapter ever grows a real import of tensorflow or scscore, the
    catalogue entry's promise -- runs locally with nothing extra installed --
    becomes false, and the option should be marked ``requires`` instead.
    """

    from molcascade.cascade.catalog import CRITERIA

    option = next(
        entry
        for criterion in CRITERIA
        if criterion.id == "synthesizability"
        for entry in criterion.options
        if entry.id == "scscore"
    )
    assert option.plugin_ref == "synthesis.scscore@0.1.0"
    assert option.requires == ()
    assert option.gate_plugin == "synthesis.numeric_evidence_gate@0.1.0"


# ------------------------------------------------------------- weight loading


def test_a_relative_weights_path_is_refused(tmp_path: Path, weights: tuple[Path, str]) -> None:
    """A relative path resolves against whatever directory the worker happens
    to be in, which is not a property of the configuration."""

    with pytest.raises(PluginError) as error:
        _run(tmp_path, _config(weights, weights_path="model.json.gz"))
    assert error.value.code == "SCSCORE_WEIGHTS_PATH_INVALID"


def test_a_missing_weight_file_says_where_to_get_it(tmp_path: Path) -> None:
    with pytest.raises(PluginError) as error:
        _run(
            tmp_path,
            {
                "schema_version": 1,
                "weights_path": str(tmp_path / "absent.json.gz"),
                "expected_weights_sha256": "0" * 64,
            },
        )
    assert error.value.code == "SCSCORE_WEIGHTS_UNREADABLE"
    assert "connorcoley/scscore" in (error.value.hint or "")


def test_a_symlinked_weight_file_is_refused(tmp_path: Path, weights: tuple[Path, str]) -> None:
    """The digest is only meaningful if it names the bytes that were read."""

    path, _ = weights
    link = tmp_path / "link.json.gz"
    link.symlink_to(path)
    with pytest.raises(PluginError) as error:
        _run(tmp_path, _config(weights, weights_path=str(link)))
    assert error.value.code == "SCSCORE_WEIGHTS_UNREADABLE"


def test_a_directory_is_not_a_weight_file(tmp_path: Path, weights: tuple[Path, str]) -> None:
    directory = tmp_path / "model_dir"
    directory.mkdir()
    with pytest.raises(PluginError) as error:
        _run(tmp_path, _config(weights, weights_path=str(directory)))
    assert error.value.code == "SCSCORE_WEIGHTS_UNREADABLE"


def test_the_run_is_pinned_to_the_exact_weight_bytes(
    tmp_path: Path, weights: tuple[Path, str]
) -> None:
    """A different model at the same path is a different experiment."""

    path, digest = weights
    _write_weights(path, _network(seed=7))
    with pytest.raises(PluginError) as error:
        _run(tmp_path, _config(weights))
    assert error.value.code == "SCSCORE_WEIGHTS_DIGEST_MISMATCH"
    assert error.value.context["expected"] == digest
    assert error.value.context["actual"] != digest


def test_the_pinned_digest_must_look_like_a_digest() -> None:
    # Note "A1" * 32 is absent on purpose: it is a valid digest in upper case.
    for bad in ("", "abc", "Z" * 64, "a" * 63, "a" * 64 + "b", " " + "a" * 63):
        with pytest.raises(ValueError):
            ScScoreConfig(
                weights_path="/tmp/model.json.gz",
                expected_weights_sha256=bad,
            )


def test_an_uppercase_digest_is_accepted_because_sha256sum_output_varies() -> None:
    digest = "a" * 64
    config = ScScoreConfig(
        weights_path="/tmp/model.json.gz",
        expected_weights_sha256=digest.upper(),
    )
    assert config.expected_weights_sha256 == digest


def test_a_file_that_is_not_gzip_stops_the_run(tmp_path: Path) -> None:
    path = tmp_path / "model.json.gz"
    path.write_bytes(b"[[0.0]]")
    with pytest.raises(PluginError) as error:
        _run(
            tmp_path,
            {
                "schema_version": 1,
                "weights_path": str(path),
                "expected_weights_sha256": hashlib.sha256(b"[[0.0]]").hexdigest(),
            },
        )
    assert error.value.code == "SCSCORE_WEIGHTS_MALFORMED"


def test_gzip_that_is_not_json_stops_the_run(tmp_path: Path) -> None:
    payload = gzip.compress(b"\x80\x03}q\x00.")
    path = tmp_path / "model.json.gz"
    path.write_bytes(payload)
    with pytest.raises(PluginError) as error:
        _run(
            tmp_path,
            {
                "schema_version": 1,
                "weights_path": str(path),
                "expected_weights_sha256": hashlib.sha256(payload).hexdigest(),
            },
        )
    assert error.value.code == "SCSCORE_WEIGHTS_MALFORMED"


# --------------------------------------------------------- shape validation


def _malformed() -> list[tuple[str, list[Any]]]:
    """Weight files that are wrong in ways the reference would not notice."""

    good = _network(bits=64, hidden=8)
    dropped_bias = [good[0], good[1], good[2], good[3], good[4]]
    unchained = [good[0], good[1], np.zeros((7, 1)), np.zeros(1)]
    mismatched_bias = [good[0], np.zeros(9), good[4], good[5]]
    wide_output = [good[0], good[1], np.zeros((8, 3)), np.zeros(3)]
    transposed = [good[0].T, np.zeros(64), np.zeros((64, 1)), np.zeros(1)]
    return [
        ("odd length", dropped_bias),
        ("layers do not chain", unchained),
        ("bias width disagrees", mismatched_bias),
        ("more than one output", wide_output),
        ("input width implausible", transposed),
        ("empty", []),
        ("not a list", {"W0": [[0.0]]}),
        ("bias is 2-D", [good[0], np.zeros((8, 1)), good[4], good[5]]),
    ]


@pytest.mark.parametrize(("label", "arrays"), _malformed(), ids=[n for n, _ in _malformed()])
def test_a_weight_file_of_the_wrong_shape_stops_rather_than_scoring(
    label: str, arrays: Any
) -> None:
    """The reference multiplies whatever it is given.

    Every one of these files would either raise a bare numpy error deep inside
    a batch or, worse, broadcast into a number that looks like a score.
    """

    raw = json.dumps(
        arrays
        if isinstance(arrays, dict)
        else [np.asarray(entry).tolist() for entry in arrays]
    ).encode("utf-8")
    with pytest.raises(PluginError) as error:
        _parse_layers(raw)
    assert error.value.code == "SCSCORE_WEIGHTS_MALFORMED"


def test_a_non_finite_weight_stops_rather_than_poisoning_a_batch() -> None:
    arrays = _network(bits=64, hidden=8)
    arrays[0][0][0] = np.nan
    raw = json.dumps([np.asarray(entry).tolist() for entry in arrays]).encode("utf-8")
    # json.dumps writes bare NaN, which is not valid JSON; either failure mode
    # is correct, but it must be this error and not a silent NaN column.
    with pytest.raises(PluginError) as error:
        _parse_layers(raw)
    assert error.value.code == "SCSCORE_WEIGHTS_MALFORMED"


def test_the_fingerprint_width_comes_from_the_weights_and_is_not_configurable(
    tmp_path: Path,
) -> None:
    """The reference takes FP_len as an argument and will happily featurize a
    2048-input model into 1024 bits."""

    assert "fingerprint_length" not in ScScoreConfig.model_fields
    assert "fingerprint_bits" not in ScScoreConfig.model_fields

    path = tmp_path / "wide.json.gz"
    digest = _write_weights(path, _network(bits=2048, hidden=8))
    response, _, scores = _run(
        tmp_path,
        {
            "schema_version": 1,
            "weights_path": str(path),
            "expected_weights_sha256": digest,
        },
    )
    assert response.metadata["fingerprint_bits"] == 2048
    assert len(scores) == len(_SMILES)


# ------------------------------------------------------------------ scoring


def _reference_score(fingerprint: np.ndarray, arrays: list[np.ndarray]) -> float:
    """``standalone_model_numpy.py``'s forward pass, one molecule at a time."""

    x = np.asarray(fingerprint, dtype=np.float64)
    for index in range(0, len(arrays), 2):
        last_layer = index == len(arrays) - 2
        x = np.matmul(x, arrays[index]) + arrays[index + 1]
        if not last_layer:
            x = x * (x > 0)
    x = 1 + (5.0 - 1) * (1 / (1 + np.exp(-x)))
    return float(np.asarray(x).reshape(-1)[0])


@pytest.mark.parametrize("kind", ["bits", "counts"])
def test_the_batched_forward_pass_reproduces_the_published_one(kind: str) -> None:
    from rdkit import Chem

    arrays = _network()
    layers = _parse_layers(
        json.dumps([np.asarray(entry).tolist() for entry in arrays]).encode("utf-8")
    )
    molecules = [Chem.MolFromSmiles(entry) for entry in _SMILES]
    features = _fingerprints(molecules, bits=_FP_BITS, counts=kind == "counts")

    batched = _forward(features, layers)
    one_at_a_time = np.array([_reference_score(row, arrays) for row in features])
    assert np.allclose(batched, one_at_a_time, rtol=0.0, atol=1e-12)


def test_the_counts_variant_folds_exactly_as_the_reference_does() -> None:
    """A folded count fingerprint of width N is *not* an unfolded fingerprint
    folded by ``k % N``, and only the second is what the model was trained on.
    """

    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    molecules = [Chem.MolFromSmiles(entry) for entry in _SMILES]
    mine = _fingerprints(molecules, bits=_FP_BITS, counts=True)
    for row, molecule in enumerate(molecules):
        expected = np.zeros(_FP_BITS, dtype=np.uint8)
        sparse = rdMolDescriptors.GetMorganFingerprint(molecule, 2, useChirality=True)
        for key, value in sparse.GetNonzeroElements().items():
            expected[key % _FP_BITS] += value
        assert np.array_equal(mine[row], expected.astype(np.float64))


def test_the_bit_variant_matches_the_legacy_call_the_model_was_trained_with() -> None:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    molecules = [Chem.MolFromSmiles(entry) for entry in _SMILES]
    mine = _fingerprints(molecules, bits=_FP_BITS, counts=False)
    for row, molecule in enumerate(molecules):
        legacy = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            molecule, 2, nBits=_FP_BITS, useChirality=True
        )
        assert sorted(np.flatnonzero(mine[row]).tolist()) == sorted(legacy.GetOnBits())


def test_bits_and_counts_are_genuinely_different_featurizations() -> None:
    """If these agreed, the config field would be decoration."""

    from rdkit import Chem

    molecules = [Chem.MolFromSmiles(entry) for entry in _SMILES]
    bits = _fingerprints(molecules, bits=_FP_BITS, counts=False)
    counts = _fingerprints(molecules, bits=_FP_BITS, counts=True)
    assert not np.array_equal(bits, counts)
    # Bits are the counts thresholded, so counts dominates everywhere.
    assert np.all(counts >= bits)


def test_chirality_is_part_of_the_featurization() -> None:
    """The published models were trained with useChirality=True, so two
    enantiomers must not collapse onto one score."""

    from rdkit import Chem

    pair = [Chem.MolFromSmiles(s) for s in ("C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O")]
    features = _fingerprints(pair, bits=_FP_BITS, counts=False)
    assert not np.array_equal(features[0], features[1])


def test_every_score_lands_inside_the_published_range(
    tmp_path: Path, weights: tuple[Path, str]
) -> None:
    """1 + 4*sigmoid(x) cannot leave 1-5, so a value outside it means the
    scale or the final layer was misread."""

    _, _, scores = _run(tmp_path, _config(weights))
    assert all(1.0 <= row["score"] <= 5.0 for row in scores)


def test_an_extreme_logit_saturates_instead_of_overflowing(tmp_path: Path) -> None:
    """exp(-x) for a large negative x overflows to inf and then to nan."""

    arrays = _network(bits=64, hidden=4)
    arrays[4] = np.full((4, 1), -1e6)
    arrays[5] = np.array([-1e6])
    layers = _parse_layers(
        json.dumps([np.asarray(entry).tolist() for entry in arrays]).encode("utf-8")
    )
    scores = _forward(np.ones((3, 64)), layers)
    assert np.all(np.isfinite(scores))
    assert np.allclose(scores, 1.0)


# ------------------------------------------------------------------- outputs


def test_evidence_is_emitted_for_every_molecule_and_none_are_dropped(
    tmp_path: Path, weights: tuple[Path, str]
) -> None:
    response, parents, scores = _run(tmp_path, _config(weights))
    assert [row["parent_id"] for row in parents] == [f"m{i}" for i in range(len(_SMILES))]
    assert [row["parent_id"] for row in scores] == [f"m{i}" for i in range(len(_SMILES))]
    assert response.metadata["input_count"] == response.metadata["output_count"]


def test_the_direction_is_recorded_on_every_row_not_just_in_metadata(
    tmp_path: Path, weights: tuple[Path, str]
) -> None:
    """A score with no direction is unusable evidence; the gate reads this
    column to know which side of the threshold passes."""

    response, _, scores = _run(tmp_path, _config(weights))
    assert {row["direction"] for row in scores} == {"HIGHER_HARDER"}
    assert response.metadata["direction"] == "HIGHER_HARDER"


def test_every_row_carries_the_not_a_route_warning(
    tmp_path: Path, weights: tuple[Path, str]
) -> None:
    _, _, scores = _run(tmp_path, _config(weights))
    for row in scores:
        assert json.loads(row["warning_codes_json"]) == ["SYNTHESIS_PROXY_NOT_ROUTE"]


def test_the_method_id_distinguishes_sc_from_sa(
    tmp_path: Path, weights: tuple[Path, str]
) -> None:
    """SYNTHESIS_SCORE_V1 keys on (parent_id, method_id), so two synthesis
    stages in one cascade collide unless their identities differ."""

    _, _, scores = _run(tmp_path, _config(weights))
    method_id = scores[0]["method_id"]
    assert method_id.startswith("synthesis-sc:sha256:")
    assert len({row["method_id"] for row in scores}) == 1


def test_the_method_id_changes_when_the_weights_change(tmp_path: Path) -> None:
    """Otherwise two runs of different models would look like one method."""

    identities = []
    for index, seed in enumerate((1, 2)):
        path = tmp_path / f"model{index}.json.gz"
        digest = _write_weights(path, _network(bits=64, hidden=8, seed=seed))
        _, _, scores = _run(
            tmp_path,
            {
                "schema_version": 1,
                "weights_path": str(path),
                "expected_weights_sha256": digest,
            },
            staging_name=f"out{index}",
        )
        identities.append(scores[0]["method_id"])
    assert identities[0] != identities[1]


def test_the_method_id_changes_between_bits_and_counts(
    tmp_path: Path, weights: tuple[Path, str]
) -> None:
    first = _run(tmp_path, _config(weights, fingerprint="bits"), staging_name="a")[2]
    second = _run(tmp_path, _config(weights, fingerprint="counts"), staging_name="b")[2]
    assert first[0]["method_id"] != second[0]["method_id"]
    assert [row["score"] for row in first] != [row["score"] for row in second]


def test_the_response_reports_the_digest_a_user_can_check(
    tmp_path: Path, weights: tuple[Path, str]
) -> None:
    _, expected = weights
    response, _, _ = _run(tmp_path, _config(weights))
    assert response.metadata["weights_sha256"] == expected
    assert response.metadata["network_or_download_invoked_by_adapter"] is False


def test_the_domain_names_the_model_that_produced_the_number(
    tmp_path: Path, weights: tuple[Path, str]
) -> None:
    _, _, scores = _run(tmp_path, _config(weights, fingerprint="counts"))
    domain = scores[0]["domain"]
    assert "1024" in domain
    assert "counts" in domain


# --------------------------------------------------------------- fail closed


def test_batching_does_not_change_a_single_score(
    tmp_path: Path, weights: tuple[Path, str]
) -> None:
    """Scores must be a property of the molecule, not of the batch it landed
    in -- otherwise a rerun with different memory settings reshuffles a
    shortlist."""

    one = _run(tmp_path, _config(weights, batch_size=1), staging_name="small")[2]
    many = _run(tmp_path, _config(weights, batch_size=10_000), staging_name="large")[2]
    assert [row["parent_id"] for row in one] == [row["parent_id"] for row in many]
    for left, right in zip(one, many, strict=True):
        assert left["score"] == pytest.approx(right["score"], abs=1e-12)


def test_a_parent_that_cannot_be_parsed_is_an_integrity_failure(
    tmp_path: Path, weights: tuple[Path, str]
) -> None:
    """Reaching this stage means the identity stage already accepted it."""

    with pytest.raises(PluginError) as error:
        _run(tmp_path, _config(weights), smiles=["CCO", "not-a-molecule"])
    assert error.value.code == "SCSCORE_PARENT_INVALID"
    assert error.value.context["parent_id"] == "m1"


def test_an_atomless_parent_stops_instead_of_scoring_zero(
    tmp_path: Path, weights: tuple[Path, str]
) -> None:
    """The reference returns 0.0 for an empty fingerprint.

    Zero is outside the 1-5 range and, on a HIGHER_HARDER axis, reads as
    "trivially easy to make" -- so a molecule the model could not look at
    would sail through a maximum-score gate.
    """

    with pytest.raises(PluginError) as error:
        _run(tmp_path, _config(weights), smiles=["CCO", ""])
    assert error.value.code == "SCSCORE_FINGERPRINT_EMPTY"
    assert error.value.context["parent_id"] == "m1"


def test_a_failed_run_leaves_no_half_written_outputs(
    tmp_path: Path, weights: tuple[Path, str]
) -> None:
    with pytest.raises(PluginError):
        _run(tmp_path, _config(weights), smiles=["CCO", "not-a-molecule"])
    assert not list((tmp_path / "out").rglob("*.parquet"))


def test_an_empty_input_stops_rather_than_producing_an_empty_shortlist(
    tmp_path: Path, weights: tuple[Path, str]
) -> None:
    with pytest.raises(PluginError) as error:
        _run(tmp_path, _config(weights), smiles=[])
    assert error.value.code == "SCSCORE_EMPTY_INPUT"


def test_an_unknown_fingerprint_name_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(ValueError):
        ScScoreConfig(
            weights_path="/tmp/model.json.gz",
            expected_weights_sha256="a" * 64,
            fingerprint="uint8",
        )


def test_unknown_configuration_keys_are_refused() -> None:
    """``score_scale`` is deliberately not a setting: every published model
    uses 5.0, and a knob with no scientific meaning is worse than none."""

    with pytest.raises(ValueError):
        ScScoreConfig(
            weights_path="/tmp/model.json.gz",
            expected_weights_sha256="a" * 64,
            score_scale=10.0,
        )
