"""Following one molecule through a finished run.

This is the command a chemist reaches for first -- they take a molecule they
know and ask why it is not in the shortlist -- and the ways it can lie are
specific and quiet.

It can attribute a removal to the wrong stage. It can conflate "this molecule
was never in your library" with "this molecule was rejected", which are opposite
diagnoses. And, worst, it can recompute the parent id under the wrong identity
policy: the id then matches nothing in the run and the command answers "never in
your library" about a molecule that is sitting in the shortlist. That last one
is why the resolution path is tested directly rather than only through its
happy-path output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molcascade.chemistry import IdentityPolicy, standardize_parent
from molcascade.config import PipelineConfig
from molcascade.errors import InputError, PipelineError
from molcascade.explain import (
    VERDICT_ABSENT,
    VERDICT_REJECTED,
    VERDICT_SELECTED,
    explain_molecule,
)
from molcascade.plugins import create_builtin_registry
from molcascade.runtime import LocalRunner

#: Ethanol is below the mass floor, the hydrochloride is multi-component *and*
#: below it, caffeine passes everything, and pyridine is deliberately absent so
#: "not in the library" has a case of its own.
_LIBRARY = (
    "id,smiles\n"
    "benzene,c1ccccc1\n"
    "ethanol,CCO\n"
    "alanine,CC(N)C(=O)O\n"
    "ethylamine_hcl,CC[NH3+].[Cl-]\n"
    "caffeine,Cn1cnc2c1c(=O)n(C)c(=O)n2C\n"
)

_MW_FLOOR = 50.0
_ABSENT = "c1ccncc1"


def _run(tmp_path: Path, *, run_id: str = "explain-run") -> tuple[LocalRunner, str]:
    source = tmp_path / "molecules.csv"
    source.write_text(_LIBRARY, encoding="utf-8")
    config = PipelineConfig.model_validate(
        {
            "schema_version": 1,
            "name": "explain-test",
            "stages": [
                {
                    "id": "source",
                    "slot": "source",
                    "plugin": "source.delimited_smiles@0.1.0",
                    "config": {
                        "path": str(source),
                        "smiles_column": "smiles",
                        "candidate_id_column": "id",
                    },
                },
                {
                    "id": "standardize",
                    "slot": "standardize",
                    "plugin": "chemistry.rdkit_standardize@0.1.0",
                    "inputs": [
                        {"request_port": "primary", "stage": "source", "port": "primary"}
                    ],
                },
                {
                    "id": "mass_gate",
                    "slot": "hard_gate",
                    "plugin": "chemistry.rdkit_property_range_gate@0.1.0",
                    "inputs": [
                        {"request_port": "primary", "stage": "standardize", "port": "primary"}
                    ],
                    "config": {"mw_min": _MW_FLOOR},
                },
                {
                    "id": "scaffolds",
                    "slot": "scaffold",
                    "plugin": "scaffold.rdkit_murcko@0.1.0",
                    "inputs": [
                        {"request_port": "primary", "stage": "mass_gate", "port": "primary"}
                    ],
                },
                {
                    "id": "shortlist_select",
                    "slot": "select",
                    "plugin": "select.native_hash_budget@0.1.0",
                    "inputs": [
                        {"request_port": "primary", "stage": "scaffolds", "port": "primary"}
                    ],
                    "config": {"target_count": 3, "seed": 7},
                },
            ],
        }
    )
    runner = LocalRunner(tmp_path / "workspace", plugins=create_builtin_registry())
    return runner, runner.run(config, run_id=run_id).run_id


def test_a_rejected_molecule_names_the_stage_and_the_reason(tmp_path: Path) -> None:
    runner, run_id = _run(tmp_path)
    verdict = explain_molecule(runner, run_id, smiles="CCO")

    assert verdict.verdict == VERDICT_REJECTED
    assert verdict.removed_by == "mass_gate"
    assert verdict.reason_code == "PROPERTY_MW_BELOW_MIN"
    # The number, not just the code: a reason a reader cannot act on is half an
    # answer, and the gate already wrote the measurement into ``detail``.
    detail = next(e.detail for e in verdict.events if e.stage_id == "mass_gate")
    assert detail is not None and "46" in detail


def test_absent_is_not_the_same_answer_as_rejected(tmp_path: Path) -> None:
    """The two diagnoses point at opposite fixes, so they must not be merged."""

    runner, run_id = _run(tmp_path)
    absent = explain_molecule(runner, run_id, smiles=_ABSENT)

    assert absent.verdict == VERDICT_ABSENT
    assert absent.removed_by is None
    assert absent.reason_code is None
    assert any("identity policy" in note for note in absent.notes)


def test_a_surviving_molecule_is_traced_to_the_shortlist(tmp_path: Path) -> None:
    runner, run_id = _run(tmp_path)
    verdict = explain_molecule(runner, run_id, smiles="Cn1cnc2c1c(=O)n(C)c(=O)n2C")

    assert verdict.verdict == VERDICT_SELECTED
    kept = {event.stage_id for event in verdict.events if event.present}
    assert {"standardize", "mass_gate", "scaffolds", "shortlist_select"} <= kept


def test_the_registered_structure_is_reported_when_it_differs(tmp_path: Path) -> None:
    """The most useful thing this command says: that is not the molecule you typed.

    Registration rewrites structures -- it picks one fragment out of a salt and,
    under the default policy, canonicalises tautomers. A funnel then reasons
    about the rewritten form, and until now nothing told the person who supplied
    the original.
    """

    runner, run_id = _run(tmp_path)
    salt = explain_molecule(runner, run_id, smiles="CC[NH3+].[Cl-]")

    assert salt.parent_smiles == "CC[NH3+]"
    assert salt.parent_smiles != salt.query
    standardize = next(e for e in salt.events if e.stage_id == "standardize")
    assert standardize.reason_code == "FRAGMENTS_REMOVED"


def test_the_warning_on_the_source_row_reaches_the_molecule(tmp_path: Path) -> None:
    """Standardizer warnings are keyed on the library row, not on the parent.

    ``FRAGMENTS_REMOVED`` and ``UNDEFINED_STEREO`` are recorded against the
    ``SOURCE_RECORD``. Without the parent_source_map join they would never be
    shown against the molecule they are about, which is exactly the pair of
    warnings most likely to explain a surprising structure.
    """

    runner, run_id = _run(tmp_path)
    verdict = explain_molecule(runner, run_id, smiles="CC(N)C(=O)O")

    assert verdict.source_record_ids
    standardize = next(e for e in verdict.events if e.stage_id == "standardize")
    assert standardize.outcome == "WARN"
    assert standardize.reason_code == "UNDEFINED_STEREO"


def test_the_parent_id_is_recomputed_under_the_runs_own_policy(tmp_path: Path) -> None:
    """The resolution step, checked against the run rather than against itself.

    A parent id computed under a different policy is absent from every table in
    the run, and the command would then report a present molecule as missing.
    """

    runner, run_id = _run(tmp_path)
    verdict = explain_molecule(runner, run_id, smiles="Cn1cnc2c1c(=O)n(C)c(=O)n2C")

    state = runner.load_run(run_id)
    manifest = runner.store.get_manifest(
        next(s for s in state.stages if s.stage_id == "standardize").output_ref.artifact_id
    )
    recorded = manifest.metadata["response_metadata"]
    assert verdict.identity_policy_id == recorded["identity_policy_id"]

    policy = IdentityPolicy.model_validate(recorded["identity_policy"])
    expected = standardize_parent(raw_smiles="Cn1cnc2c1c(=O)n(C)c(=O)n2C", policy=policy)
    assert verdict.parent_id == expected.parent_id


def test_a_parent_id_may_be_given_directly(tmp_path: Path) -> None:
    runner, run_id = _run(tmp_path)
    resolved = explain_molecule(runner, run_id, smiles="CCO")
    direct = explain_molecule(runner, run_id, parent_id=resolved.parent_id)

    assert direct.verdict == resolved.verdict
    assert direct.removed_by == resolved.removed_by
    assert direct.identity_policy_id is None


def test_exactly_one_query_is_required(tmp_path: Path) -> None:
    runner, run_id = _run(tmp_path)
    with pytest.raises(InputError):
        explain_molecule(runner, run_id)
    with pytest.raises(InputError):
        explain_molecule(runner, run_id, smiles="CCO", parent_id="parent:sha256:x")


def test_an_unparseable_structure_is_refused_not_guessed(tmp_path: Path) -> None:
    runner, run_id = _run(tmp_path)
    with pytest.raises((InputError, PipelineError)):
        explain_molecule(runner, run_id, smiles="not a molecule")


def test_a_json_detail_is_read_out_rather_than_cut_through(tmp_path: Path) -> None:
    """Two gates write two shapes of detail, and truncation only works on one.

    The physicochemical and ring gates write a sentence -- ``mw=129.167 is below
    150.0``, 25 characters. The drug-likeness gates write their whole metric
    panel as JSON: measured on the approved-oral panel, famotidine's
    ``DRUG_LIKENESS_QED_POLICY_FAIL`` detail is 758 characters and sertraline's
    is 738, so a fixed-width cut lands inside a JSON object and produces a line
    that is neither JSON nor a sentence.

    Those objects already contain the sentence, so the terminal reads it out.
    The full detail is still carried verbatim by ``--json``.
    """

    from molcascade.explain import _detail

    qed = (
        '{"message":"weighted QED=0.286612438474 is below configured minimum=0.3",'
        '"metrics":{"aromatic_ring_count":3,"clogp":-0.5,"hba":9,"hbd":4}}'
    )
    assert _detail(qed) == "weighted QED=0.286612438474 is below configured minimum=0.3"

    # The alert adapters write no message; the catalogue and the pattern are
    # the same information in their vocabulary.
    alert = (
        '{"action":"reject","alert":"betalactam","catalog":"NIH",'
        '"interpretation":"substructure triage flag; not experimental proof"}'
    )
    assert _detail(alert) == "NIH: betalactam"

    # A sentence is left alone, and something that only looks like JSON falls
    # back to truncation rather than raising.
    assert _detail("mw=129.167 is below 150.0") == "mw=129.167 is below 150.0"
    assert _detail('{"not json at all') == '{"not json at all'


def test_the_full_detail_survives_in_the_machine_readable_output(
    tmp_path: Path,
) -> None:
    """Reading a line out for a person must not cost the record.

    ``as_dict`` is what a script consumes and what a bug report carries, so the
    verbatim detail belongs there whatever the terminal chose to show.
    """

    runner, run_id = _run(tmp_path)
    verdict = explain_molecule(runner, run_id, smiles="CCO")

    event = next(e for e in verdict.events if e.stage_id == verdict.removed_by)
    payload = next(
        row for row in verdict.as_dict()["events"] if row["stage_id"] == verdict.removed_by
    )
    assert payload["detail"] == event.detail


#: Pyrantel pamoate is the textbook case: pamoate has 29 heavy atoms and
#: pyrantel 14, so the largest-organic heuristic keeps the counter-ion and
#: throws away the drug. Metformin hydrochloride is the ordinary case, where the
#: heuristic is right. ``CC.CC`` is the trap: two components, one parent, and
#: nothing was actually discarded.
_PYRANTEL = "CN1CCCN=C1/C=C/c1cccs1"
_DESALT_LIBRARY = (
    "id,smiles\n"
    "caffeine,Cn1cnc2c1c(=O)n(C)c(=O)n2C\n"
    "pyrantel_pamoate,"
    "CN1CCCN=C1/C=C/c1cccs1.O=C(O)c1cc2ccccc2c(Cc2c(C(=O)O)c(O)c3ccccc3c2)c1O\n"
    "metformin_hcl,CN(C)C(=N)NC(=N)N.Cl\n"
    "doubled,CC.CC\n"
    "benzene,c1ccccc1\n"
)


def _desalt_run(tmp_path: Path) -> tuple[LocalRunner, str]:
    from molcascade.config import PipelineConfig

    source = tmp_path / "salts.csv"
    source.write_text(_DESALT_LIBRARY, encoding="utf-8")
    config = PipelineConfig.model_validate(
        {
            "schema_version": 1,
            "name": "desalting",
            "stages": [
                {
                    "id": "source",
                    "slot": "source",
                    "plugin": "source.delimited_smiles@0.1.0",
                    "config": {
                        "path": str(source),
                        "smiles_column": "smiles",
                        "candidate_id_column": "id",
                    },
                },
                {
                    "id": "standardize",
                    "slot": "standardize",
                    "plugin": "chemistry.rdkit_standardize@0.1.0",
                    "inputs": [
                        {"request_port": "primary", "stage": "source", "port": "primary"}
                    ],
                    # The shipped cascade's own override; a canonicalising policy
                    # would rewrite these structures before they were compared.
                    "config": {
                        "identity_policy": {
                            "schema_version": 1,
                            "tautomer_policy": "preserve",
                        }
                    },
                },
            ],
        }
    )
    runner = LocalRunner(tmp_path / "workspace", plugins=create_builtin_registry())
    return runner, runner.run(config, run_id="desalting-run").run_id


def test_a_molecule_de_salting_threw_away_is_found_not_reported_absent(
    tmp_path: Path,
) -> None:
    """The question this feature exists for, and the answer it used to give.

    De-salting keeps the largest organic fragment, so pamoate (29 heavy atoms)
    beats pyrantel (14) and the drug becomes the discard. Before this, asking
    about pyrantel returned ABSENT -- which a reader can only understand as "it
    was never in your library", the opposite of the truth.

    A discarded fragment's parent id never reaches ``parent_source_map``, so the
    search cannot start from the query. It starts from the rows the run recorded
    splitting.
    """

    from molcascade.explain import VERDICT_DESALTED

    runner, run_id = _desalt_run(tmp_path)
    verdict = explain_molecule(runner, run_id, smiles=_PYRANTEL)

    assert verdict.verdict == VERDICT_DESALTED
    discarded = [f for r in verdict.desalting for f in r.discarded if f.is_query]
    assert len(discarded) == 1
    assert discarded[0].heavy_atoms == 14
    # And the answer names the row, because several rows can map to one parent
    # and their fragment composition need not be the same.
    assert all(report.source_record_id for report in verdict.desalting)


def test_the_desalted_verdict_carries_no_contradictory_note_or_event_wall(
    tmp_path: Path,
) -> None:
    """A verdict and the prose under it must not disagree.

    ABSENT prints no per-stage trail, because a wall of "gone" says nothing the
    verdict has not. DESALTED must do the same -- and must not keep ABSENT's
    note, which says the opposite of what was just concluded.
    """

    runner, run_id = _desalt_run(tmp_path)
    rendered = explain_molecule(runner, run_id, smiles=_PYRANTEL).render()

    assert "gone" not in rendered
    assert "No stage in this run holds that parent id" not in rendered
    assert "the molecule you asked about" in rendered


def test_a_component_equivalent_to_the_kept_one_is_not_called_discarded(
    tmp_path: Path,
) -> None:
    """``CC.CC`` is two components and one parent; nothing was lost.

    Reporting either half as discarded would describe a loss that did not
    happen -- and the run did write FRAGMENTS_REMOVED for this row, so the
    warning alone cannot be the trigger.
    """

    runner, run_id = _desalt_run(tmp_path)
    verdict = explain_molecule(runner, run_id, smiles="CC")

    assert all(not report.discarded for report in verdict.desalting)


def test_the_row_a_kept_molecule_came_from_reports_what_it_lost(
    tmp_path: Path,
) -> None:
    """The other half: my molecule is here, what else was on its row?

    Metformin's hydrochloride loses a chloride, which cannot be registered on
    its own -- and that failure is itself the proof it was not what the run
    kept.
    """

    runner, run_id = _desalt_run(tmp_path)
    verdict = explain_molecule(runner, run_id, smiles="CN(C)C(=N)NC(=N)N")

    lost = [f for report in verdict.desalting for f in report.discarded]
    assert [f.smiles for f in lost] == ["Cl"]
    assert lost[0].unregistrable
    assert not lost[0].is_query


def test_a_genuinely_absent_molecule_stays_absent_and_says_how_hard_it_looked(
    tmp_path: Path,
) -> None:
    """The negative has to be readable as a negative.

    "None of the split rows discarded this" is only worth saying when every
    split row was actually examined, so the claim is gated on the scan being
    exhaustive rather than merely finishing.
    """

    runner, run_id = _desalt_run(tmp_path)
    verdict = explain_molecule(runner, run_id, smiles="c1ccncc1")

    assert verdict.verdict == VERDICT_ABSENT
    assert verdict.desalting_scan is not None
    assert verdict.desalting_scan.exhaustive
    assert any("none of them discarded" in note for note in verdict.notes)


def test_the_search_can_be_turned_off_and_then_the_old_answer_returns(
    tmp_path: Path,
) -> None:
    """The budget is real, and its floor is the previous behaviour."""

    runner, run_id = _desalt_run(tmp_path)
    verdict = explain_molecule(runner, run_id, smiles=_PYRANTEL, desalt_limit=0)

    assert verdict.verdict == VERDICT_ABSENT
    assert verdict.desalting == ()
    assert verdict.desalting_scan is None


def test_the_desalting_report_survives_json(tmp_path: Path) -> None:
    """``--json`` passes no ``default=``, so a bare dataclass would raise here."""

    runner, run_id = _desalt_run(tmp_path)
    payload = json.dumps(explain_molecule(runner, run_id, smiles=_PYRANTEL).as_dict())

    assert "DESALTED" in payload
    assert "is_query" in payload
