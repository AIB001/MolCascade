"""The ring topology gate, and the two claims it is built on.

Two things here are load-bearing and would be silent if they broke.  The first
is the counting convention: this gate exists because ``medchem`` and Mordred
return 1 for pentacene, and a reimplementation that drifted back to counting
*systems* would still pass every functional test while doing nothing.  The
second is the calibration: the three shipped defaults are asserted against a
panel of approved drugs, so moving one does not quietly cost a class of real
chemistry -- the drugs it costs come back named in a failing assertion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molcascade.artifacts import ArtifactRef
from molcascade.contracts import DECISION_V1, PARENT_V1
from molcascade.errors import PluginError
from molcascade.plugins.api import StageContext, StageInput, StageRequest
from molcascade.plugins.builtin.ring_topology_gate import (
    RDKitRingTopologyGatePlugin,
    RingTopologyGateConfig,
    _measure,
)

#: Thirty-three approved drugs spanning the ring shapes a filter like this can
#: get wrong: flat heteroaromatics, steroids, a macrolide, and the pentacyclic
#: natural products that sit just past the fused-ring cap.
DRUGS: dict[str, str] = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "metformin": "CN(C)C(=N)NC(=N)N",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "warfarin": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "diazepam": "CN1c2ccc(Cl)cc2C(=NCC1=O)c1ccccc1",
    "omeprazole": "COc1ccc2[nH]c(nc2c1)S(=O)Cc1ncc(C)c(OC)c1C",
    "sildenafil": "CCCc1nn(C)c2c1nc([nH]c2=O)-c1cc(ccc1OCC)S(=O)(=O)N1CCN(C)CC1",
    "atorvastatin": (
        "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"
    ),
    "montelukast": ("CC(C)(O)c1ccccc1CC[C@@H](SCC1(CC(=O)O)CC1)c1cccc(/C=C/c2ccc3ccc(Cl)cc3n2)c1"),
    "ciprofloxacin": "OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "chlorpromazine": "CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21",
    "estradiol": "C[C@]12CC[C@H]3[C@@H](CCc4cc(O)ccc34)[C@@H]1CC[C@@H]2O",
    "dexamethasone": (
        "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@]2(C)[C@@]1(O)C(=O)CO"
    ),
    "morphine": "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
    "quinine": "COc1ccc2nccc([C@@H](O)[C@@H]3C[C@@H]4CCN3C[C@@H]4C=C)c2c1",
    "camptothecin": "CC[C@@]1(O)C(=O)OCc2c1cc1n(c2=O)Cc2cc3ccccc3nc2-1",
    "artemisinin": "C[C@@H]1CC[C@H]2[C@@H](C)C(=O)O[C@@H]3O[C@@]4(C)CC[C@@H]1[C@@]23OO4",
    "tetracycline": (
        "C[C@]1(O)[C@@H]2C[C@H]3[C@@H](N(C)C)C(=O)C(C(N)=O)=C(O)[C@@]3(O)C(=O)C2=C(O)c2c(O)cccc21"
    ),
    "erlotinib": "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
    "gefitinib": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "dasatinib": "Cc1nc(Nc2ncc(s2)C(=O)Nc2c(C)cccc2Cl)cc(n1)N1CCN(CCO)CC1",
    "sunitinib": "CCN(CC)CCNC(=O)c1c(C)[nH]c(/C=C2\\C(=O)Nc3ccc(F)cc32)c1C",
    "palbociclib": "CC(=O)c1c(C)c2cnc(Nc3ccc(cn3)N3CCNCC3)nc2n(C2CCCC2)c1=O",
    "erythromycin": (
        "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)"
        "[C@@H](O[C@@H]2O[C@H](C)C[C@@H]([C@H]2O)N(C)C)[C@](C)(O)C[C@@H](C)C(=O)[C@H](C)"
        "[C@@H](O)[C@]1(C)O"
    ),
    "olanzapine": "Cc1cc2c(s1)Nc1ccccc1N=C2N1CCN(C)CC1",
    "colchicine": "CC(=O)N[C@H]1CCc2cc(OC)c(OC)c(OC)c2-c2ccc(OC)c(=O)cc21",
    "risperidone": "CC1=C(C(=O)N2CCCCC2=N1)CCN1CCC(CC1)c1noc2cc(F)ccc12",
    "ibrutinib": "C=CC(=O)N1CCC[C@@H](C1)n1nc(-c2ccc(Oc3ccccc3)cc2)c2c(N)ncnc12",
    "celecoxib": "Cc1ccc(cc1)-c1cc(nn1-c1ccc(cc1)S(N)(=O)=O)C(F)(F)F",
    "lovastatin": (
        "CC[C@H](C)C(=O)O[C@H]1C[C@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@@H]12"
    ),
}


def _mol(smiles: str):
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None, smiles
    return molecule


def _row(parent_id: str, smiles: str) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "identity_policy_id": "identity:test",
        "parent_smiles": smiles,
        "registration_key": f"key:{parent_id}",
        "stereo_key": None,
        "formula": None,
        "duplicate_count": 1,
    }


def _input(tmp_path: Path, molecules: dict[str, str]) -> StageInput:
    root = tmp_path / "input"
    root.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [_row(name, smiles) for name, smiles in molecules.items()],
            schema=PARENT_V1.schema,
        ),
        root / "parents.parquet",
    )
    return StageInput(
        ref=ArtifactRef(artifact_id="artifact:sha256:" + "7" * 64, kind=PARENT_V1.id),
        root=root,
        contract_id=PARENT_V1.id,
    )


# --------------------------------------------------------------------------
# The counting convention this gate exists to provide
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("smiles", "expected", "description"),
    [
        ("c1ccccc1", 1, "benzene"),
        ("c1ccc2ccccc2c1", 2, "naphthalene"),
        ("c1ccc2cc3ccccc3cc2c1", 3, "anthracene"),
        ("c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1", 5, "pentacene"),
        ("c1cc2ccc3cccc4ccc(c1)c2c34", 4, "pyrene, peri-fused"),
        ("c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61", 7, "coronene"),
    ],
)
def test_fused_rings_counts_rings_not_systems(smiles: str, expected: int, description: str) -> None:
    """The whole reason this module exists.

    ``medchem.rules._utils.n_fused_aromatic_rings`` and Mordred's ``nFRing``
    both return 1 for every molecule in this table: they count fused *systems*.
    A pentacene that scores 1 passes any cap a user can set, which is how a
    library of polycyclic sheets gets through a filter that looks like it
    should stop them.
    """

    assert _measure(_mol(smiles))["fused_rings"] == expected, description


def test_a_rotatable_bond_between_two_rings_is_not_fusion() -> None:
    """Biphenyl is two rings and one system, but it is not a rigid sheet."""

    measured = _measure(_mol("c1ccccc1-c1ccccc1"))
    assert measured["rings"] == 2
    assert measured["fused_rings"] == 1


def test_a_spiro_junction_is_not_fusion() -> None:
    """Sharing one atom leaves the rings free to rotate about it.

    Fusion means a shared *bond*, so the merge threshold is two shared atoms.
    A spiro centre shares exactly one, and off-by-one here would silently
    reclassify every spirocycle as a fused system.
    """

    measured = _measure(_mol("C1CCC2(CC1)CCCCC2"))
    assert measured["rings"] == 2
    assert measured["fused_rings"] == 1


def test_separate_ring_systems_do_not_pool() -> None:
    """Two naphthalenes are four rings in two systems of two, not one of four."""

    measured = _measure(_mol("c1ccc2ccccc2c1.c1ccc2ccccc2c1"))
    assert measured["rings"] == 4
    assert measured["fused_rings"] == 2
    assert measured["ring_system_size"] == 10


def test_the_total_sees_the_fused_content_the_largest_system_cannot() -> None:
    """The quantity ``max_fused_rings`` is structurally unable to report.

    Three separate naphthalenes are six fused rings of flat, unrotatable
    surface, and every one of the three systems is a blameless 2.  A cap on the
    largest system reads that molecule as no worse than naphthalene itself.
    """

    measured = _measure(_mol("c1ccc2ccccc2c1.c1ccc2ccccc2c1.c1ccc2ccccc2c1"))
    assert measured["fused_rings"] == 2
    assert measured["fused_ring_total"] == 6


def test_the_total_counts_fused_content_and_not_rings_in_general() -> None:
    """Otherwise it would be ``max_rings`` under a second name.

    Benzene fused to nothing contributes nothing: a lone ring is a ring, not
    fused-ring content, and three of them are still three rotatable joints.
    """

    assert _measure(_mol("c1ccccc1.c1ccccc1.c1ccccc1"))["fused_ring_total"] == 0
    both = _measure(_mol("c1ccc2ccccc2c1.c1ccccc1"))
    assert both["rings"] == 3
    assert both["fused_ring_total"] == 2


def test_a_symmetric_cage_counts_more_rings_than_its_cyclomatic_number() -> None:
    """Pinned because it decides acceptances, not because it is elegant.

    RDKit's symmetrised SSSR keeps every smallest ring of a tied size instead of
    choosing a basis, so bicyclo[2.2.2]octane comes back as 3 rings against a
    cyclomatic 2.  The consequence is worth stating plainly: ``max_fused_rings``
    set to 2 rejects quinuclidine, and therefore quinine, even though a chemist
    would call it a bridged bicyclic.  The error runs toward counting more, so
    no cage slips under a cap by being symmetric.
    """

    assert _measure(_mol("C1CC2CCC1C2"))["fused_rings"] == 2  # norbornane, untied
    assert _measure(_mol("C1CN2CCC1CC2"))["fused_rings"] == 3  # quinuclidine
    assert _measure(_mol("C1C2CC3CC1CC(C2)C3"))["fused_rings"] == 4  # adamantane


def test_an_acyclic_molecule_measures_zero_rather_than_dividing_by_nothing() -> None:
    measured = _measure(_mol("CCCCCCO"))
    assert measured["rings"] == 0
    assert measured["fused_rings"] == 0
    assert measured["ring_system_size"] == 0
    assert measured["ring_atom_fraction"] == 0.0


# --------------------------------------------------------------------------
# Calibration: what the shipped defaults cost in real drugs
# --------------------------------------------------------------------------


def _rejected(config: RingTopologyGateConfig) -> set[str]:
    from molcascade.plugins.builtin.ring_topology_gate import _findings

    return {name for name, smiles in DRUGS.items() if _findings(_measure(_mol(smiles)), config)}


def test_the_shipped_defaults_cost_exactly_three_of_thirty_three_drugs() -> None:
    """The calibration claim in the module docstring, made falsifiable.

    FAF-Drugs4 chose ``rings <= 6`` and ``ring system <= 18 atoms`` so that up
    to 90% of 916 approved oral drugs pass; adding the fused-ring cap keeps 30
    of these 33, or 91%.  If a default moves, this test names the drugs it
    started costing instead of merely going red.
    """

    rejected = _rejected(RingTopologyGateConfig())

    assert rejected == {"morphine", "camptothecin", "artemisinin"}, sorted(rejected)
    assert len(DRUGS) - len(rejected) == 30


def test_four_fused_rings_is_the_step_that_buys_the_steroids() -> None:
    """Why the default is 4 and not Toxtree's 3.

    Toxtree's SA_18 alerts at three fused aromatic rings.  Adopting 3 here
    would be defensible on paper and would reject the steroid nucleus, which is
    four fused rings in every glucocorticoid and every sex hormone.
    """

    for steroid in ("estradiol", "dexamethasone"):
        assert _measure(_mol(DRUGS[steroid]))["fused_rings"] == 4

    at_three = _rejected(RingTopologyGateConfig(max_fused_rings=3))
    assert {"estradiol", "dexamethasone"} <= at_three


def test_the_stricter_pair_is_a_choice_the_panel_prices_at_ten_drugs() -> None:
    """A rule of "at most two bicyclics", and what the panel charges for it.

    ``max_fused_rings=2`` with ``max_fused_ring_total=4`` is the configuration
    the total exists to make possible: neither half is enough alone, since a cap
    of 2 on the largest system admits any number of naphthalenes and a total of
    4 admits one tetracyclic.  Ten of the thirty-three go, including every
    steroid and every tricyclic antipsychotic, which is why this is not shipped
    as a default -- but it is a defensible filter for a generated library, where
    the fused-ring tail is the problem being solved.
    """

    strict = _rejected(RingTopologyGateConfig(max_fused_rings=2, max_fused_ring_total=4))

    assert strict == {
        "artemisinin",
        "camptothecin",
        "chlorpromazine",
        "colchicine",
        "dexamethasone",
        "estradiol",
        "morphine",
        "olanzapine",
        "quinine",
        "tetracycline",
    }


def test_the_total_alone_takes_nothing_the_shipped_cap_has_not_taken() -> None:
    """The reason it ships off rather than at four.

    A default has to earn its place by rejecting something the other defaults
    miss.  Against ``max_fused_rings=4`` a total of 4 finds only quinine, and
    only because a symmetric cage counts 3 -- no molecule here carries two
    fused systems the pair of them would catch.  Off is the honest setting.
    """

    added = _rejected(RingTopologyGateConfig(max_fused_ring_total=4)) - _rejected(
        RingTopologyGateConfig()
    )

    assert added == {"quinine"}


def test_the_macrolide_survives_because_the_ring_size_cap_ships_empty() -> None:
    """Erythromycin's 14-membered ring is the reason ``max_ring_size`` is off."""

    assert _measure(_mol(DRUGS["erythromycin"]))["ring_size"] == 14
    assert "erythromycin" not in _rejected(RingTopologyGateConfig())
    assert "erythromycin" in _rejected(RingTopologyGateConfig(max_ring_size=12))


@pytest.mark.parametrize(
    ("field", "value", "casualties"),
    [
        ("min_rotatable_bonds", 3, {"caffeine", "estradiol", "olanzapine", "dexamethasone"}),
        ("max_ring_atom_fraction", 0.75, {"olanzapine", "imatinib", "diazepam"}),
        ("min_fraction_csp3", 0.25, {"imatinib", "celecoxib", "aspirin"}),
    ],
)
def test_the_optional_bounds_ship_empty_because_they_cost_approved_drugs(
    field: str, value: float, casualties: set[str]
) -> None:
    """The evidence for every ``None`` default in the config.

    Each of these is a reasonable thing to want on a generated library and an
    unreasonable thing to impose on everyone, and the difference is exactly the
    set of drugs it removes.  Switching one on by default means accepting this
    list, so the list is written down.
    """

    assert not (casualties & _rejected(RingTopologyGateConfig()))
    assert casualties <= _rejected(RingTopologyGateConfig(**{field: value}))


# --------------------------------------------------------------------------
# The stage itself
# --------------------------------------------------------------------------


def test_the_gate_filters_and_explains_every_decision(tmp_path: Path) -> None:
    plugin = RDKitRingTopologyGatePlugin()
    response = plugin.execute(
        StageRequest(
            stage_id="ring_topology",
            inputs={
                "parents": _input(
                    tmp_path,
                    {
                        "caffeine": DRUGS["caffeine"],
                        "estradiol": DRUGS["estradiol"],
                        "coronene": "c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61",
                    },
                )
            },
            config={},
        ),
        StageContext(tmp_path / "out"),
    )

    parents = pq.read_table(tmp_path / "out" / response.outputs["primary"].file_paths[0])
    decisions = pq.read_table(tmp_path / "out" / response.outputs["decisions"].file_paths[0])
    PARENT_V1.validate(parents)
    DECISION_V1.validate(decisions)

    assert {row["parent_id"] for row in parents.to_pylist()} == {"caffeine", "estradiol"}
    rejects = {row["reason_code"] for row in decisions.to_pylist() if row["outcome"] == "REJECT"}
    assert rejects == {
        "RING_TOPOLOGY_RINGS_ABOVE_MAX",
        "RING_TOPOLOGY_FUSED_RINGS_ABOVE_MAX",
        "RING_TOPOLOGY_RING_SYSTEM_SIZE_ABOVE_MAX",
    }
    assert response.metadata["input_count"] == 3
    assert response.metadata["output_count"] == 2
    assert response.metadata["reject_count"] == 1


def test_a_pass_carries_the_numbers_that_earned_it(tmp_path: Path) -> None:
    """The trace CSV is where a threshold gets tuned, so it needs the values.

    A PASS row that said only "passed" would make choosing the next cap a
    matter of rerunning the stage.  Every measured quantity rides along, bound
    or not, so the distribution is readable straight from the decisions table.
    """

    response = RDKitRingTopologyGatePlugin().execute(
        StageRequest(
            stage_id="ring_topology",
            inputs={"parents": _input(tmp_path, {"estradiol": DRUGS["estradiol"]})},
            config={},
        ),
        StageContext(tmp_path / "out"),
    )
    decisions = pq.read_table(
        tmp_path / "out" / response.outputs["decisions"].file_paths[0]
    ).to_pylist()

    assert decisions[0]["outcome"] == "PASS"
    detail = json.loads(decisions[0]["detail"])
    assert detail["fused_rings"] == 4
    assert detail["rotatable_bonds"] == 0
    # Reported although nothing bounds it by default -- that is the point.
    assert detail["fraction_csp3"] == pytest.approx(0.667, abs=0.01)


def test_every_bound_is_adjustable_from_the_stage_config(tmp_path: Path) -> None:
    """Loosening the defaults has to actually admit the molecule they rejected."""

    molecules = {"cage": "c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61"}
    response = RDKitRingTopologyGatePlugin().execute(
        StageRequest(
            stage_id="ring_topology",
            inputs={"parents": _input(tmp_path, molecules)},
            config={
                "max_rings": 10,
                "max_fused_rings": 10,
                "max_fused_ring_total": 10,
                "max_ring_system_size": 40,
            },
        ),
        StageContext(tmp_path / "out"),
    )
    parents = pq.read_table(tmp_path / "out" / response.outputs["primary"].file_paths[0])
    assert parents.num_rows == 1


def test_clearing_every_bound_is_refused_rather_than_passing_everything() -> None:
    """A gate that bounds nothing reports 100% forever and looks like it works.

    Emptying all nine fields is a plausible way to try to disable the
    criterion.  Removing the criterion is the way to do that; this is the
    configuration that only looks like it.
    """

    with pytest.raises(ValueError, match="at least one bound"):
        RingTopologyGateConfig(
            max_rings=None,
            max_fused_rings=None,
            max_fused_ring_total=None,
            max_ring_system_size=None,
            max_ring_size=None,
            min_rotatable_bonds=None,
            max_ring_atom_fraction=None,
            min_fraction_csp3=None,
            max_bridgehead_atoms=None,
        )


def test_an_unparseable_parent_stops_the_stage(tmp_path: Path) -> None:
    with pytest.raises(PluginError) as caught:
        RDKitRingTopologyGatePlugin().execute(
            StageRequest(
                stage_id="ring_topology",
                inputs={"parents": _input(tmp_path, {"broken": "c1ccccc"})},
                config={},
            ),
            StageContext(tmp_path / "out"),
        )
    assert caught.value.code == "RING_TOPOLOGY_PARENT_INVALID"
    assert caught.value.context["parent_id"] == "broken"


def test_the_policy_id_tracks_the_bounds_and_ignores_the_batch_size() -> None:
    """Two runs with the same policy have to be comparable across batch sizes."""

    from molcascade.plugins.builtin.ring_topology_gate import _policy_id

    baseline = _policy_id(RingTopologyGateConfig(), "2024.03.5")
    assert baseline == _policy_id(RingTopologyGateConfig(batch_size=64), "2024.03.5")
    assert baseline != _policy_id(RingTopologyGateConfig(max_fused_rings=5), "2024.03.5")
    assert baseline != _policy_id(RingTopologyGateConfig(), "2025.03.1")
