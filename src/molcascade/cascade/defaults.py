"""The starter cascade offered to a first-time user.

The job this cascade is designed for is the one MolCascade sits in the middle
of: a generator such as PRISM proposes millions of molecules, and something has
to decide which twenty or thirty thousand are worth docking, rescoring and
running free-energy calculations on.  Everything below is arranged around that.

**Cheapest and most decisive first.**  Staged screening funnels work because
each tier is allowed to be cruder than the next, provided it is also far
cheaper: run the fast, coarse filters on everything and reserve the expensive,
accurate ones for what survives.  So the order here is parse -> descriptor
arithmetic -> substructure matching -> learned synthesizability, with anything
requiring a neural network or a route search placed after all of them.

The ordering is measured, not assumed.  On one core: RDKit parse and sanitize
~850 molecules/s, the RDKit property panel ~270, the alert catalogues ~85, the
ChEMBL structure checker ~100.  The spread is an order of magnitude, so tier
position is worth more than any amount of tuning inside a tier -- the same
check is three hours per million at the top of the funnel and minutes at the
bottom.  Those numbers are recorded on each catalogue option as
``throughput_per_second``, and the builder warns when a slow one is dropped
into the first tier.

**Two toolkits, not one, at the points where being wrong is expensive.**  A
cascade built entirely on one library cannot tell a molecule that is fine from
a molecule that library happens to be wrong about.  The alert tier already
takes a union of independently curated catalogues; the last tier here adds the
ChEMBL curation checker, whose verdicts come from the IUPAC InChI library, so
what leaves for docking has been read by two toolkits rather than by RDKit
twice.

**Terminal rejects only where the chemistry is actually wrong.**  The recurring
failure of screening funnels is over-strict early filtering: if the filters are
too aggressive, real hits are eliminated before anything has looked at them, and
nothing downstream can recover them.  Tier one therefore rejects only molecules
that are broken -- unparseable, bad valence, disallowed elements -- and every
heuristic further down is either a soft window or an explicit, editable gate.

**PAINS is a hypothesis, not a verdict.**  Baell and Holloway's filters were
derived from one assay technology and one library, and the follow-up literature
(Capuzzi and colleagues' "Phantom PAINS", among others) documents how badly they
travel.  The default warns on PAINS and rejects on Brenk, whose list is about
reactive, unstable and toxicophoric groups -- chemistry that is wrong rather
than chemistry that is suspicious.

**Synthesizability heuristics are not route searches.**  Thakkar and colleagues
(Chem. Sci. 2021) checked directly whether SAscore, SCScore or SYBA predict
whether AiZynthFinder can find a route, and found no threshold on any of them
that separates solvable from unsolvable; they warn explicitly about misuse when
filtering large virtual libraries.  The honest response is not to drop the
scores -- they are the only thing that runs at this scale -- but to stop using
one of them as a hard cut.  The synthesizability tier here is therefore a
*union*: a molecule survives if either an occurrence-frequency score (SA) or a
reaction-trained score (SCScore) calls it tractable, and is dropped only when
two methods that disagree by construction both say it is hard.

**What is deliberately absent.**  Similarity to reference leads needs a lead
file only the project has, so it cannot ship as a default; ADMET endpoints and
retrosynthesis route searches wait for their backends to be provisioned.  All
three are one click away in the builder, and belong at the bottom of the funnel
where the population is small enough to afford them.

References

- Scior, T. et al. Recognizing pitfalls in virtual screening.
  *J. Chem. Inf. Model.* 2012. DOI: 10.1021/ci200528d
- Kimber, T. B. et al. Deep learning in virtual screening: principles and
  architectures. *Int. J. Mol. Sci.* 2021. DOI: 10.3390/ijms22094435
- Baell, J. B.; Holloway, G. A. New substructure filters for removal of pan
  assay interference compounds. *J. Med. Chem.* 2010. DOI: 10.1021/jm901137j
- Capuzzi, S. J. et al. Phantom PAINS: problems with the utility of alerts for
  pan-assay interference compounds. *J. Chem. Inf. Model.* 2017.
  DOI: 10.1021/acs.jcim.7b00465
- Brenk, R. et al. Lessons learnt from assembling screening libraries for drug
  discovery for neglected diseases. *ChemMedChem* 2008.
  DOI: 10.1002/cmdc.200700139
- Ertl, P.; Schuffenhauer, A. Estimation of synthetic accessibility score.
  *J. Cheminform.* 2009. DOI: 10.1186/1758-2946-1-8
- Coley, C. W. et al. SCScore: synthetic complexity learned from a reaction
  corpus. *J. Chem. Inf. Model.* 2018. DOI: 10.1021/acs.jcim.7b00622
- Thakkar, A. et al. Retrosynthetic accessibility score (RAscore).
  *Chem. Sci.* 2021. DOI: 10.1039/D0SC05401A
- Bickerton, G. R. et al. Quantifying the chemical beauty of drugs.
  *Nat. Chem.* 2012. DOI: 10.1038/nchem.1243
- Veber, D. F. et al. Molecular properties that influence oral bioavailability.
  *J. Med. Chem.* 2002. DOI: 10.1021/jm020017n
"""

from __future__ import annotations

from typing import Any

from molcascade.cascade.availability import option_availability
from molcascade.cascade.catalog import CRITERIA_BY_ID, BackendOption, CriterionSpec
from molcascade.cascade.introspect import with_schema_version
from molcascade.cascade.models import (
    CascadeConfig,
    CriterionConfig,
    FinalizeConfig,
    GateConfig,
    LibraryConfig,
    StepConfig,
    TierConfig,
)
from molcascade.plugins.registry import PluginRegistry, create_builtin_registry

DEFAULT_TARGET_COUNT = 20_000
DEFAULT_SEED = 20_260_823

#: Which SCScore variant the default uses when the asset has been provisioned.
#: The 1024-bit binary model is the one the paper's figures were produced with.
SCSCORE_ASSET_MEMBER = "models/full_reaxys_model_1024bool/model.ckpt-10654.as_numpy.json.gz"


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    """Write ``value`` at a dotted path, creating intermediate objects."""

    keys = path.split(".")
    cursor = target
    for key in keys[:-1]:
        nested = cursor.get(key)
        if not isinstance(nested, dict):
            nested = {}
            cursor[key] = nested
        cursor = nested
    cursor[keys[-1]] = value


def criterion_defaults(
    option: BackendOption,
    *,
    registry: PluginRegistry | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the producer and gate settings a fresh criterion starts with."""

    if option.plugin_ref is None:
        return dict(option.defaults), dict(option.gate_defaults)
    settings = with_schema_version(option.plugin_ref, option.defaults, registry=registry)
    if option.gate_plugin is None:
        return settings, dict(option.gate_defaults)
    gate_settings = with_schema_version(
        option.gate_plugin, option.gate_defaults, registry=registry
    )
    return settings, gate_settings


def build_criterion(
    spec: CriterionSpec,
    option: BackendOption,
    *,
    criterion_id: str | None = None,
    label: str | None = None,
    settings: dict[str, Any] | None = None,
    gate_settings: dict[str, Any] | None = None,
    registry: PluginRegistry | None = None,
) -> CriterionConfig:
    """Create a criterion bound to one executable option."""

    if option.plugin_ref is None:
        raise ValueError(
            f"option {option.id!r} for criterion {spec.id!r} has no executable adapter"
        )
    base_settings, base_gate = criterion_defaults(option, registry=registry)
    if settings:
        for key, value in settings.items():
            _set_path(base_settings, key, value)
    if gate_settings:
        for key, value in gate_settings.items():
            _set_path(base_gate, key, value)
    gate = (
        GateConfig(backend=option.gate_plugin, settings=base_gate)
        if option.gate_plugin is not None
        else None
    )
    return CriterionConfig(
        id=criterion_id or spec.id,
        criterion=spec.id,
        backend=option.plugin_ref,
        label=label or f"{spec.label} · {option.engine}",
        settings=base_settings,
        gate=gate,
    )


def _runnable(option: BackendOption | None, registry: PluginRegistry) -> bool:
    """Whether this installation can really execute ``option`` end to end.

    Adapter, threshold gate *and* the third-party packages the option declares.
    The last of those is easy to forget because the adapters register
    unconditionally and import lazily, so a missing package looks exactly like
    an installed one until the stage runs.
    """

    if option is None or option.plugin_ref is None:
        return False
    return option_availability(option, registry).runnable


def _criterion_if_available(
    criterion_id: str,
    registry: PluginRegistry,
    *,
    option_id: str | None = None,
    config_id: str | None = None,
    label: str | None = None,
    settings: dict[str, Any] | None = None,
    gate_settings: dict[str, Any] | None = None,
) -> CriterionConfig | None:
    """Build one criterion, or return ``None`` if this machine cannot run it.

    The default cascade is offered to whoever is sitting in front of the
    installation, so it may only contain steps that installation can actually
    execute.  A cascade that names a missing adapter would fail at compile time
    with an error the user did not cause.
    """

    spec = CRITERIA_BY_ID.get(criterion_id)
    if spec is None:
        return None
    if option_id is None:
        option = spec.default_option
    else:
        option = next((entry for entry in spec.options if entry.id == option_id), None)
    if not _runnable(option, registry):
        return None
    assert option is not None  # narrowed by _runnable
    return build_criterion(
        spec,
        option,
        criterion_id=config_id,
        label=label,
        settings=settings,
        gate_settings=gate_settings,
        registry=registry,
    )


def _step_if_available(
    step_id: str,
    backend: str,
    registry: PluginRegistry,
    *,
    settings: dict[str, Any] | None = None,
) -> StepConfig | None:
    if backend not in registry:
        return None
    return StepConfig(
        id=step_id,
        backend=backend,
        settings=with_schema_version(backend, settings or {}, registry=registry),
    )


def _scscore_criterion(registry: PluginRegistry) -> CriterionConfig | None:
    """Add the SCScore arm only when its published weights are on this machine.

    SCScore is worth having in the default because it disagrees with SA score
    for principled reasons -- one counts fragment frequency, the other learns
    from reaction records -- and the union of two such scores is a far more
    defensible filter than either alone.  But it reads a 6 MB weight file that
    ``molcascade assets fetch scscore`` provides, and offering a default that
    stops on a missing file would be a worse first experience than offering one
    tier with a single arm.
    """

    try:
        from molcascade.assets import asset_spec, asset_status
    except ImportError:  # pragma: no cover - assets ship with the package
        return None
    try:
        spec = asset_spec("scscore")
        if not asset_status(spec).ready:
            return None
        digest = spec.file(SCSCORE_ASSET_MEMBER).sha256
    except (KeyError, OSError):
        return None
    return _criterion_if_available(
        "synthesizability",
        registry,
        option_id="scscore",
        config_id="synthesis_scscore",
        label="Synthesizability · SCScore (reaction-trained)",
        settings={
            "weights_path": f"asset:{spec.id}/{SCSCORE_ASSET_MEMBER}",
            "expected_weights_sha256": digest,
            "fingerprint": "bits",
        },
        # SCScore is a *relative* score -- it was trained to rank a reaction's
        # product above its reactants, not to place an absolute line -- so a
        # threshold on it has to be read off the population it will judge.  On
        # 71 approved oral drugs the distribution is median 3.52, p75 4.19,
        # p90 4.61.  The 3.5 this shipped with was sitting on the median: it
        # rejected 51% of marketed drugs, a coin flip dressed as a filter.
        gate_settings={"expected_direction": "HIGHER_HARDER", "maximum": 4.5},
    )


def _chemistry_tier(registry: PluginRegistry) -> TierConfig | None:
    criteria = [
        criterion
        for criterion in (
            _criterion_if_available(
                "structure_validity",
                registry,
                # PAINS is handled two tiers down, where the verdict is a warning
                # the user can read rather than a rejection nothing can undo.
                settings={"policy.flag_pains": False},
            ),
        )
        if criterion is not None
    ]
    if not criteria:
        return None
    return TierConfig(
        id="t1_chemistry",
        title="Chemistry triage",
        mode="all",
        criteria=tuple(criteria),
        note=(
            "Terminal rejects: unparseable strings, valence errors, fragment-only "
            "records and disallowed elements. Nothing removed here can be recovered "
            "by a later score, which is why only broken chemistry is removed here."
        ),
    )


def _property_tier(registry: PluginRegistry) -> TierConfig | None:
    criteria = [
        criterion
        for criterion in (
            _criterion_if_available(
                "physchem_window",
                registry,
                config_id="physchem_window",
                settings={
                    # Deliberately an *envelope*, not a preference.  These bounds
                    # were set by running them against 71 approved oral drugs
                    # (tests/fixtures/approved_drugs.py) and widening until the
                    # only rejections left were ones with a chemical argument
                    # behind them.  A tighter window costs real chemistry for no
                    # gain the later tiers cannot deliver better: this tier sees
                    # every molecule, so its mistakes are unrecoverable, while
                    # ADMET, alerts and synthesizability all discriminate harder
                    # on the population that reaches them.
                    #
                    # 200-550 rejected 21% of that panel -- aspirin at 180 Da,
                    # caffeine at 194, theophylline at cLogP -1.04 -- which is
                    # arithmetic, not chemistry.  The bounds below reject 5.6%,
                    # and still reject every one of the six drugs the panel marks
                    # as genuinely outside oral small-molecule space.
                    "mw_min": 150.0,  # below Ghose's 160; keeps aspirin, caffeine
                    "mw_max": 650.0,  # atorvastatin is 559 and sells by the tonne
                    "clogp_min": -3.0,  # metformin and the nucleosides live here
                    "clogp_max": 6.0,  # Egan's ceiling is 5.88; montelukast (8.9) still goes
                    "tpsa_max": 180.0,  # Veber's 140 is a soft optimum, not a cliff
                    "hbd_max": 5,  # Lipinski, unmodified: nothing on the panel hits it
                    "hba_max": 10,  # likewise
                    "rotatable_bonds_max": 12,  # Veber's 10 costs atorvastatin and montelukast
                    "absolute_formal_charge_max": 2,
                },
            ),
            _criterion_if_available(
                "physchem_window",
                registry,
                option_id="mordred_descriptor_window",
                config_id="aromatic_ring_count",
                label="Aromatic ring count",
                settings={
                    # The one thing in this tier that is not a restatement of
                    # size.  Ritchie and Macdonald showed developability falling
                    # off above three aromatic rings across ~3,000 oral drugs and
                    # candidates; four leaves a ring of slack, because the
                    # boundary is a trend and this tier's mistakes cannot be
                    # recovered later.
                    #
                    # Measured before shipping, on the same standard as the
                    # window above: of the approved oral panel it rejects one
                    # molecule -- lapatinib, at five rings, which that panel
                    # already carries as a deliberate edge case.  Ritchie's own
                    # boundary of three would reject six.
                    "descriptor": "naRing",
                    "maximum": 4,
                },
            ),
        )
        if criterion is not None
    ]
    if not criteria:
        return None
    return TierConfig(
        id="t2_physchem",
        title="Physicochemical window",
        mode="all",
        criteria=tuple(criteria),
        note=(
            "Descriptor arithmetic on every surviving molecule: cheap, and the first "
            "place a funnel can lose something it will never get back. The window is "
            "wide on purpose -- it is meant to exclude molecules that are not oral "
            "small molecules at all, not to express a preference among ones that are. "
            "Measured against the approved oral panel the size window rejects 4 and "
            "the aromatic ring cap rejects 1. Narrow it to a project window once you "
            "know what you are looking for."
        ),
    )


def _druglikeness_tier(registry: PluginRegistry) -> TierConfig | None:
    criteria = [
        criterion
        for criterion in (
            _criterion_if_available(
                "drug_likeness",
                registry,
                config_id="druglike_ro5_qed",
                label="Rule of Five and QED · RDKit",
                settings={
                    "maximum_rule_of_five_violations": 1,
                    # QED's own paper puts approved oral drugs around 0.5; a floor of
                    # 0.3 removes the clearly undesirable without pretending the
                    # score is sharp enough to rank the survivors.
                    "minimum_qed": 0.3,
                    "failure_action": "reject",
                },
            ),
            _criterion_if_available(
                "drug_likeness",
                registry,
                option_id="medchem_rules",
                config_id="druglike_oral_consensus",
                label="Oral-absorption consensus · medchem",
                settings={
                    # Three independent published rules for oral absorption, and a
                    # molecule needs two of them.  The quorum is the point.  Run
                    # alone against 71 approved oral drugs these reject 9.9%
                    # (Lipinski), 5.6% (Veber) and 8.5% (Egan); requiring two of
                    # the three rejects 5.6%, because the drugs each rule dislikes
                    # are largely different drugs.  No single rule set gets to
                    # delete a molecule on its own idiosyncrasy.
                    #
                    # medchem ships 22 rules and the tempting ones are traps.
                    # rule_of_generative_design_strict -- which this tier used to
                    # apply as a hard reject -- removes 49% of that panel,
                    # including ibuprofen, imatinib, morphine and olanzapine: it
                    # asks for a ring-heavy, chain-light, TPSA >= 40 shape that
                    # much of pharmacology simply does not have. It is a fine
                    # *preference* for generated matter and a bad gate. Select it
                    # in the builder if you want it; it is not a default.
                    "rules": "rule_of_five,rule_of_veber,rule_of_egan",
                    "combination": "at_least",
                    "minimum_passes": 2,
                },
            ),
        )
        if criterion is not None
    ]
    if not criteria:
        return None
    return TierConfig(
        id="t3_druglike",
        title="Drug-likeness",
        mode="all",
        criteria=tuple(criteria),
        note=(
            "Drug-likeness is a fuzzy judgement, so both arms here are consensus "
            "rather than verdict: an aggregate score with a low floor, and a vote "
            "among three oral-absorption rules. That is the opposite of how the "
            "alerts tier is built, and deliberately -- a rule disliking a molecule "
            "is an opinion, whereas a reactive group is a fact. Switch the tier to "
            "Serial if you would rather chain them."
        ),
    )


def _alerts_tier(registry: PluginRegistry) -> TierConfig | None:
    criteria = [
        criterion
        for criterion in (
            _criterion_if_available(
                "structural_alerts",
                registry,
                config_id="alerts_rdkit",
                label="Reactivity and interference · RDKit catalogues",
                settings={
                    # Which catalogue deletes and which annotates was decided by
                    # measurement against 71 approved oral drugs, and the answer
                    # was not the one this tier originally shipped.
                    #
                    # NIH flags 3 molecules in the whole 77-drug panel and is right
                    # about all three: ibrutinib's acrylamide, amoxicillin's
                    # beta-lactam, levothyroxine's four iodines. Two covalent
                    # warheads and a polyiodide -- exactly the matter that should
                    # not enter a funnel whose downstream is non-covalent docking
                    # and MM-PBSA, where a scored pose for a covalent binder is
                    # meaningless. 2.8% false-positive rate, every hit defensible.
                    "nih_action": "reject",
                    # Brenk flags 31% of the same panel, including aspirin,
                    # paracetamol, morphine, warfarin, gefitinib and omeprazole.
                    # Nine of those hits are "Aliphatic_long_chain" -- four sp3
                    # atoms in a row. Rejecting on this was deleting a third of
                    # known drug space to catch a handful of genuine liabilities
                    # the NIH set catches anyway.
                    #
                    # Measuring it also turned up a separate bug, which is fixed
                    # rather than tolerated: RDKit's standardizer rewrites every
                    # neutral sulfoxide S(=O) into the charge-separated [S+][O-]
                    # form, and RDKit's own Brenk catalogue then flags that form
                    # as "charged_oxygen_or_sulfur_atoms" -- so every
                    # proton-pump inhibitor, sulindac and modafinil left the run
                    # for a reason that was purely how the molecule got written
                    # down. molcascade.chemistry.alerts now hands the catalogues
                    # the neutral depiction they were authored against, and
                    # test_shipped_defaults.py holds that shut.
                    "brenk_action": "warn",
                    # PAINS is an assay-interference hypothesis, and this funnel
                    # ends in physics rather than a biochemical assay. Low false
                    # positives here (2.8%), but the claim does not apply, so the
                    # match is a note for whoever reads the report.
                    "pains_action": "warn",
                    # ZINC's hits on the panel are all "Non-Hydrogen_atoms", a size
                    # cap the physicochemical tier already enforces with a number
                    # the user can see and edit.
                    "zinc_action": "ignore",
                },
            ),
            _criterion_if_available(
                "structural_alerts",
                registry,
                option_id="medchem_common_alerts",
                config_id="alerts_medchem",
                label="Screening-deck liabilities · medchem",
                settings={
                    # Dundee used to be in this list and is now not. It is the same
                    # rule set as the Brenk catalogue in the arm above -- Brenk et
                    # al. published it from Dundee -- so listing it here rejected a
                    # third of the approved-drug panel a second time, under a
                    # different name, from a tier note that claimed two independent
                    # opinions were being taken. Two arms agreeing because they are
                    # the same data is not agreement.
                    #
                    # BMS and Glaxo are the reactive-chemistry sets, measured at 8%
                    # and 4% on that panel, and the molecules they flag there are
                    # ones a screening deck should genuinely exclude.
                    "alert_sets": "BMS,Glaxo",
                    "use_nibr": True,
                    "nibr_reject_at_severity": 10,
                    "exclude_action": "reject",
                    "flag_action": "warn",
                    "annotation_action": "ignore",
                    # NIBR's exclusions come in two kinds and only one of them
                    # is about the molecule.  67 of its 444 rules fire on
                    # compound-class membership -- steroid, peptide,
                    # nucleoside, glycoside, retinoid, fatty acid -- because a
                    # physical screening deck does not want promiscuous or
                    # assay-confounding matter taking up wells.  That is an
                    # argument about a plate, not about whether a molecule can
                    # bind the target this cascade is aimed at, and applied as
                    # a reject it costs dexamethasone and prednisolone off the
                    # approved-drug panel and every steroid PRISM generates.
                    # The liability rules are untouched: ranitidine still
                    # rejects on its nitroalkane, at the same severity.
                    "nibr_compound_class_action": "warn",
                },
            ),
        )
        if criterion is not None
    ]
    if not criteria:
        return None
    return TierConfig(
        id="t4_alerts",
        title="Alerts and liabilities",
        mode="all",
        criteria=tuple(criteria),
        note=(
            "Only reactive chemistry deletes a molecule here: covalent warheads, "
            "unstable groups, known assay-destroying functionality. Everything else "
            "-- interference hypotheses, screening-deck hygiene, medicinal-chemistry "
            "taste -- is recorded as a warning and travels with the molecule to the "
            "report. Both arms use union rather than a vote, which is the right way "
            "round for hazards: one catalogue seeing a beta-lactam is enough, and "
            "waiting for a second opinion would only let it through."
        ),
    )


def _synthesis_tier(registry: PluginRegistry) -> TierConfig | None:
    criteria = [
        criterion
        for criterion in (
            _criterion_if_available(
                "synthesizability",
                registry,
                config_id="synthesis_sa_score",
                label="Synthesizability · SA score (fragment frequency)",
                # SA score on 71 approved oral drugs: median 2.65, p95 4.53,
                # p99 5.20.  A cutoff of 4.5 sat right on the 95th percentile and
                # the four drugs above it were morphine, codeine, simvastatin and
                # dexamethasone -- i.e. the loss was not spread evenly, it fell
                # entirely on natural-product-derived polycyclics, which is a
                # chemotype bias rather than a difficulty threshold.  5.0 keeps
                # them and still rejects the genuinely intractable.
                gate_settings={"expected_direction": "HIGHER_HARDER", "maximum": 5.0},
            ),
            _scscore_criterion(registry),
        )
        if criterion is not None
    ]
    if not criteria:
        return None
    union = len(criteria) > 1
    return TierConfig(
        id="t5_synthesis",
        title="Synthesizability",
        # "any" is the point: drop a molecule only when two scores built on
        # different evidence agree that it is hard.  With one arm available the
        # mode is immaterial, and "all" reads more honestly in the report.
        mode="any" if union else "all",
        criteria=tuple(criteria),
        note=(
            "Two scores that disagree by construction -- SA counts how common a "
            "molecule's fragments are, SCScore learns complexity from twelve "
            "million reactions -- and a molecule needs only one of them to call it "
            "tractable. Neither predicts whether a route exists: no threshold on "
            "either cleanly separates molecules a retrosynthesis planner can solve "
            "from ones it cannot, so treat this as coarse triage and put a route "
            "search below it once the population is small enough."
            if union
            else "SA score is a fragment-frequency proxy. It is not a route and it "
            "is not a step count. Run 'molcascade assets fetch scscore' to add a "
            "second, reaction-trained opinion alongside it."
        ),
    )


# There is deliberately no rules-based ADMET tier in the starter cascade.
#
# There used to be one, applying Egan's egg alone as a hard gate.  It was
# removed because Egan is already one of the three rules the drug-likeness tier
# votes on, and a tier that re-applies a vote's losing member as a verdict
# silently overrides the vote.  Measured on the approved-oral-drug panel the
# damage was exact: caffeine, theophylline, metformin and amoxicillin each pass
# Lipinski and Veber, win the 2-of-3 quorum, and were then deleted downstream on
# the single rule they lost.  Panel rejection was 10/77 with the tier and is
# 6/77 without it, and the vote's stated promise -- that no single rule set gets
# to delete a molecule on its own idiosyncrasy -- is only true without it.
#
# The wider point is that this backend has no warning outcome, so any rule put
# here is a rule the run deletes on, and no rule medchem ships is permissive
# enough for that job: measured alone on the same panel the most lenient are
# Veber 6/77 and Lipinski 10/77, and the rest run from Ghose at 28/77 to
# druglike_soft at 51/77.  Absorption is asked once, as a quorum, one tier up.
#
# A predictive ADMET tier is a different thing and remains available in the
# builder -- ADMET-AI or a Chemprop bundle, with an explicit numeric gate.  It
# is not a default because both require provisioning and a trust acknowledgement
# first, and a starter cascade that fails closed on a fresh install is worse
# than one that does not claim the tier.


def _structure_qc_tier(registry: PluginRegistry) -> TierConfig | None:
    """A second toolkit's opinion on the structures, taken at the handoff.

    The reason this is a tier of its own at the bottom rather than a second arm
    of tier one is cost.  Measured on one core: RDKit parse and sanitize runs at
    ~850 molecules/s, the property panel at ~270, the alert catalogues at ~85
    and this checker at ~100 -- so against two million molecules it is roughly
    six hours in tier one and a few minutes here, and threading does not change
    that because the InChI call holds the GIL for its whole duration.

    Placing it last costs nothing scientifically, because nothing between here
    and there can rescue a molecule it rejects, and the alternative -- leaving
    it switched off by default because it is too slow to run first -- would mean
    the structures handed to docking never get a second toolkit's opinion at
    all.  It is also how the check is used upstream: EBI runs it when compounds
    are *registered* into ChEMBL, not as a bulk pre-filter.
    """

    criterion = _criterion_if_available(
        "structure_validity",
        registry,
        option_id="chembl_structure_pipeline",
        # Tier one already claims the bare criterion id, and every stage id in a
        # cascade has to be unique.
        config_id="structure_qc_chembl",
        label="Structure quality · ChEMBL checker (InChI second opinion)",
    )
    if criterion is None:
        return None
    return TierConfig(
        id="t7_structure_qc",
        title="Structure quality",
        mode="all",
        criteria=(criterion,),
        note=(
            "The last thing that happens before the shortlist leaves for docking. "
            "RDKit decided in tier one that these strings parse; this asks whether "
            "the IUPAC InChI library agrees they are molecules somebody meant to "
            "draw, and rejects at penalty 6 -- radicals off the known list, "
            "polymers, embedded 3D. Undefined stereochemistry scores 2 and is kept "
            "and counted, because in a generated library it describes almost "
            "everything. Around 100 molecules/s, which is why it is here and not "
            "at the top."
        ),
    )


def default_cascade(
    *,
    registry: PluginRegistry | None = None,
    target_count: int = DEFAULT_TARGET_COUNT,
) -> CascadeConfig:
    """Build the starter cascade against the plugins this installation has."""

    active = registry or create_builtin_registry()

    tiers = [
        tier
        for tier in (
            _chemistry_tier(active),
            _property_tier(active),
            _druglikeness_tier(active),
            _alerts_tier(active),
            _synthesis_tier(active),
            _structure_qc_tier(active),
        )
        if tier is not None
    ]

    finalize_steps = [
        step
        for step in (
            _step_if_available(
                "properties",
                "features.rdkit_properties@0.1.0",
                active,
                settings={"include_sa_score": True},
            ),
            _step_if_available("scaffolds", "scaffold.rdkit_murcko@0.1.0", active),
            _step_if_available(
                "diversity_groups", "cluster.rdkit_scaffold_groups@0.1.0", active
            ),
            _step_if_available(
                "shortlist_select",
                "select.native_scaffold_round_robin@0.1.0",
                active,
                # A generated library is full of near-neighbours, and docking
                # budget spent on the twenty-sixth analogue of one scaffold buys
                # nothing that the first twenty-five did not.
                settings={"max_per_scaffold": 25},
            ),
            _step_if_available(
                "shortlist",
                "export.native_smiles_shortlist@0.1.0",
                active,
                settings={"filename": "shortlist.smi"},
            ),
        )
        if step is not None
    ]

    ingest = StepConfig(
        id="library",
        backend="source.delimited_smiles@0.1.0",
        settings=with_schema_version("source.delimited_smiles@0.1.0", {}, registry=active),
    )
    standardize = _step_if_available(
        "standardize",
        "chemistry.rdkit_standardize@0.1.0",
        active,
        settings={
            "identity_policy": {
                "schema_version": 1,
                # Not RDKit's default of "insensitive", and this is the one
                # place in the starter cascade that overrides an identity
                # default, so it is worth saying why.
                #
                # Under "insensitive" two tautomers register as one parent, and
                # that parent is depicted as RDKit's canonical tautomer.  The
                # canonical form is chosen by a score that pays a large bonus
                # per aromatic ring, so wherever a lactam sits next to an
                # aromatic ring the lactim wins, and every substituent that has
                # to move to pay for it moves.
                #
                # On the approved-oral-drug panel that transform rewrites 15 of
                # 77 structures, and the rewrites are not cosmetic:
                #
                #   * clopidogrel, valsartan, levothyroxine and methotrexate
                #     each lose a defined stereocentre; amoxicillin loses three;
                #   * sunitinib's oxindole aromatises to the lactim and takes
                #     the defined Z-alkene with it;
                #   * ranitidine's nitroalkene becomes a nitroalkane, which is
                #     a functional group two alert catalogues reject on sight;
                #   * dexamethasone and prednisolone have the A-ring enone
                #     shifted out of conjugation.
                #
                # The alert hits are the visible damage.  The real damage is
                # that the shortlist handed to docking carries the invented
                # structure, stereochemistry and all.
                #
                # What "insensitive" buys back is cross-tautomer deduplication,
                # and for a generated library -- one writer, one spelling
                # convention, millions of distinct molecules -- that is close to
                # nothing, while the depiction is what every downstream stage
                # and the docking handoff actually consume.  "preserve" keeps
                # the structure as drawn, and is just as deterministic.
                #
                # A curated deck assembled from several vendors is the opposite
                # trade, so this stays a per-cascade setting the builder
                # exposes rather than something hard-coded.
                "tautomer_policy": "preserve",
            }
        },
    )

    return CascadeConfig(
        name="molcascade_screen",
        description=(
            "Cost-ordered triage of a generated library: chemistry, physicochemical "
            "window, drug-likeness, liability alerts, synthesizability and a "
            "second-toolkit structure check, then a scaffold-diverse shortlist "
            "sized for docking."
        ),
        library=LibraryConfig(smiles_column="smiles"),
        ingest=ingest,
        standardize=standardize,
        tiers=tuple(tiers),
        finalize=FinalizeConfig(
            target_count=target_count,
            seed=DEFAULT_SEED,
            steps=tuple(finalize_steps),
        ),
        metadata={"created_by": "MolCascade cascade builder"},
    )


__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_TARGET_COUNT",
    "SCSCORE_ASSET_MEMBER",
    "build_criterion",
    "criterion_defaults",
    "default_cascade",
]
