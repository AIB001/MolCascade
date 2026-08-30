"""The screening-criterion catalogue presented by the cascade builder.

The old builder asked users to pick a *plugin*.  That forces a scientist to
already know which package answers which question.  This catalogue inverts it:
a **criterion** is a scientific question ("is this molecule likely to be a
promiscuous binder?"), and a **backend option** is one tool that answers it.
Swapping RDKit for ``medchem``, ``rd_filters`` or a project's own model is then
a one-click change that keeps the tier structure and thresholds intact.

Every option declares whether it is executable today.  An option without a
``plugin_ref`` is a reviewed research candidate: it appears so the landscape is
visible, and it can never be selected or exported.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

ThresholdKind = Literal["number", "integer", "boolean", "choice", "text"]
ThresholdTarget = Literal["criterion", "gate"]


@dataclass(frozen=True, slots=True)
class ThresholdField:
    """One user-editable number that decides who survives this criterion.

    ``target`` says which settings object the value belongs to: the producer's
    own configuration, or the threshold gate placed after an evidence producer.
    """

    name: str
    label: str
    kind: ThresholdKind = "number"
    target: ThresholdTarget = "criterion"
    unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[tuple[str, str], ...] = ()
    nullable: bool = False
    #: The cascade cannot run without a value here, even though the plugin's own
    #: field accepts ``None``.
    #:
    #: ``nullable=False`` already means "required", and covers every engine path
    #: in this catalogue.  This flag is for the other case: a field the plugin
    #: makes optional because it has an in-config alternative the *builder* does
    #: not offer.  ``reference_path`` is the one -- the plugin will also take
    #: reference records embedded in the settings, but nothing in this UI writes
    #: them, so leaving the box empty produces a criterion that cannot run.
    required: bool = False
    #: A machine-level environment variable that can supply this value instead.
    #:
    #: Where an engine is installed is a fact about a host, not about a campaign.
    #: Naming the variable here lets the builder say so in the field's help, and
    #: lets it leave the box empty -- without baking one host's paths into a
    #: config meant to travel.
    environment_variable: str | None = None
    help: str = ""

    @property
    def installed_path(self) -> bool:
        """Whether this field says where software lives, not how to screen.

        An installation path is a property of the host that will run the cascade,
        and the builder is almost never open on that host.  So an empty box is
        the portable answer rather than an omission, and the browser must not
        block a download over one: :func:`preflight_engine_paths` settles the
        path at run start -- from the config, from this variable, or from the
        layout ``envs/bootstrap.sh`` creates -- and refuses there, naming the
        command that installs it, if none of the three answered.

        Declared by the presence of a variable rather than measured from the
        environment.  The builder used to ask whether *this* shell exported the
        path, which made the generated HTML a statement about the laptop it was
        drawn on: a cascade assembled where nothing was installed could not be
        downloaded at all, even when every engine was present on the machine
        that would run it.
        """

        return self.environment_variable is not None

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "target": self.target,
            "unit": self.unit,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
            "choices": [{"value": value, "label": label} for value, label in self.choices],
            "nullable": self.nullable,
            "required": self.required,
            "environment_variable": self.environment_variable,
            "installed_path": self.installed_path,
            "help": self.help,
        }


@dataclass(frozen=True, slots=True)
class BackendOption:
    """One concrete tool that can answer a criterion."""

    id: str
    label: str
    engine: str
    summary: str
    plugin_ref: str | None = None
    gate_plugin: str | None = None
    license_spdx: str = "unknown"
    defaults: dict[str, Any] = field(default_factory=dict)
    gate_defaults: dict[str, Any] = field(default_factory=dict)
    thresholds: tuple[ThresholdField, ...] = ()
    #: Third-party Python packages that must be importable.
    requires: tuple[str, ...] = ()
    #: Ids from the asset catalogue that must be fetched and digest-verified.
    #: An option can need these and no packages at all: an alert table or a
    #: weight file read by MolCascade's own code is a dependency just as real
    #: as a pip install, and just as fatal several hours into a run.
    requires_assets: tuple[str, ...] = ()
    notes: str = ""
    #: The paper to cite when this option decided which molecules survived.
    #: Vendored assets carry their citation in the generated ``CITATION.md``
    #: next to the bytes, but a pip-installed backend has no such folder, and
    #: a method used in a screening campaign has to be citable either way.
    #: Kept as one formatted reference ending in a DOI or URL: the value is
    #: read by people writing a methods section, not parsed.
    citation: str = ""
    #: Throughput in molecules per second on one lane, or ``None`` when nobody
    #: has a figure.  ``None`` means unknown, *not* fast.
    #:
    #: A lane is one core for the tools that run in this environment and one
    #: card for the isolated GPU engines, which is the unit the pool hands out
    #: in either case.  Those engine numbers are the ones their own papers
    #: report rather than something measured here -- measuring them needs the
    #: hardware and the licence, and the ``notes`` say which figure came from
    #: where.  Everything else was timed on one core.
    #:
    #: This is here because tier position is the most consequential decision the
    #: builder lets someone make and the cost of getting it wrong is invisible
    #: until the run is hours old: the first tier sees the entire library, so a
    #: hundred-molecules-per-second check placed there is three hours per
    #: million while the same check below three cheap tiers is minutes.  The
    #: numbers are order-of-magnitude and machine-dependent, which is all the
    #: decision needs -- it turns on ratios between tools, not on absolutes.
    #:
    #: Fractional on purpose: a CNN rescoring pass is seconds per ligand, and
    #: rounding it to a whole number would round it to zero, which every
    #: consumer here reads as "unknown" -- silencing the warning on the most
    #: expensive tool in the catalogue.
    throughput_per_second: float | None = None
    recommended: bool = False
    #: Whether running this tool requires the operator to accept its licence
    #: terms explicitly, with ``--allow-copyleft``.
    #:
    #: A copyleft backend is not blocked because it is unavailable or unproven.
    #: It is blocked because linking a screening pipeline to GPL code is a
    #: decision about the pipeline, and one an organisation may have already
    #: made either way.  MolCascade cannot make it, so it defaults to the
    #: reversible answer and says which flag reverses it.
    requires_license_optin: bool = False

    @property
    def executable(self) -> bool:
        return self.plugin_ref is not None

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "engine": self.engine,
            "summary": self.summary,
            "plugin_ref": self.plugin_ref,
            "gate_plugin": self.gate_plugin,
            "license_spdx": self.license_spdx,
            "defaults": dict(self.defaults),
            "gate_defaults": dict(self.gate_defaults),
            "thresholds": [threshold.as_json() for threshold in self.thresholds],
            "requires": list(self.requires),
            "requires_assets": list(self.requires_assets),
            "notes": self.notes,
            "citation": self.citation,
            "throughput_per_second": self.throughput_per_second,
            "recommended": self.recommended,
            "requires_license_optin": self.requires_license_optin,
            "executable": self.executable,
        }


@dataclass(frozen=True, slots=True)
class CriterionSpec:
    """A scientific question plus every tool that can answer it."""

    id: str
    label: str
    question: str
    stage: str
    evidence: str
    options: tuple[BackendOption, ...]
    summary: str = ""
    filters: bool = True

    @property
    def executable_options(self) -> tuple[BackendOption, ...]:
        return tuple(option for option in self.options if option.executable)

    @property
    def default_option(self) -> BackendOption | None:
        executable = self.executable_options
        if not executable:
            return None
        return next((option for option in executable if option.recommended), executable[0])

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "question": self.question,
            "stage": self.stage,
            "evidence": self.evidence,
            "summary": self.summary,
            "filters": self.filters,
            "options": [option.as_json() for option in self.options],
        }


@dataclass(frozen=True, slots=True)
class StageGroup:
    """A funnel level shown as one tier card in the builder."""

    id: str
    title: str
    subtitle: str
    purpose: str
    default_mode: str = "serial"
    # Every shipped group names its own hue; this is the fallback a third-party
    # group inherits, and it is the builder's neutral periwinkle so an
    # unstyled group still belongs to the same drawing.
    accent: str = "#6785b9"
    #: Something the run needs that this file cannot hold.
    #:
    #: A receptor belongs to a campaign, not to a screening policy -- the whole
    #: point of a cascade is that it is reused against the next target -- so the
    #: docking groups are configured here and completed at the command line.
    #: ``preflight_docking_target`` refuses to start without it, and this string
    #: is what puts that refusal in front of the person writing the cascade
    #: rather than the person running it an hour later.
    run_time_requirement: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "subtitle": self.subtitle,
            "purpose": self.purpose,
            "default_mode": self.default_mode,
            "accent": self.accent,
            "run_time_requirement": self.run_time_requirement,
        }


STAGE_GROUPS: tuple[StageGroup, ...] = (
    StageGroup(
        id="identity",
        title="Identity",
        subtitle="Read and register",
        purpose=(
            "Parse the generated library, register one parent structure per unique "
            "molecule, and remove exact duplicates before anything is measured."
        ),
        accent="#6785b9",
    ),
    StageGroup(
        id="chemistry",
        title="Chemistry triage",
        subtitle="Valid, allowed, not obviously reactive",
        purpose=(
            "Terminal structural rejects. A molecule removed here can never be "
            "rescued by a later score."
        ),
        default_mode="all",
        accent="#d24763",
    ),
    StageGroup(
        id="physchem",
        title="Drug-likeness",
        subtitle="Property windows, Rule of Five, QED",
        purpose=(
            "Project property policy. These are windows a team chooses, not laws of "
            "chemistry, so they belong in an editable tier."
        ),
        default_mode="all",
        accent="#f6c666",
    ),
    StageGroup(
        id="alerts",
        title="Alerts and liabilities",
        subtitle="PAINS, aggregators, unstable groups",
        purpose=(
            "Substructure alert catalogues. Alerts are triage hypotheses, not "
            "measured instability."
        ),
        default_mode="all",
        accent="#d7684b",
    ),
    StageGroup(
        id="admet",
        title="ADMET and activity",
        subtitle="Local prediction models",
        purpose=(
            "Model predictions with explicit endpoints and thresholds. Every model "
            "runs locally and every threshold is visible."
        ),
        default_mode="all",
        accent="#48418c",
    ),
    StageGroup(
        id="synthesis",
        title="Synthesizability",
        subtitle="Score proxies and route search",
        purpose=(
            "How hard the molecule is to make. A score is a proxy; only route search "
            "gives real step counts."
        ),
        default_mode="serial",
        accent="#9bceb8",
    ),
    StageGroup(
        id="ligand_prep",
        title="Ligand 3D preparation",
        subtitle="One geometry, shared by every engine",
        purpose=(
            "Embed 3D conformers once, above the docking tier, so the engines below "
            "score the same molecule rather than three separate guesses at it."
        ),
        default_mode="serial",
        accent="#5fb0c9",
    ),
    StageGroup(
        id="docking",
        title="Structure-based docking",
        subtitle="Pose and score against your receptor",
        purpose=(
            "Two engines in parallel with 'at least K' is consensus docking: a "
            "molecule survives only where independent methods agree, which is "
            "worth more than either engine's own number."
        ),
        default_mode="all",
        accent="#12808c",
        run_time_requirement=(
            "A receptor, named when the run starts: --receptor protein.pdb, plus "
            "--reference-ligand, --pocket or --box to say where on it."
        ),
    ),
    StageGroup(
        id="docking_rescore",
        title="CNN rescoring",
        subtitle="A network's opinion on what survived",
        purpose=(
            "Seconds per ligand rather than tenths. It belongs after a docking "
            "threshold has already cut the population, never over a library."
        ),
        default_mode="serial",
        accent="#8c5a2b",
        run_time_requirement=(
            "The same receptor as the docking tier above it, and GNINA's licence "
            "accepted: --allow-copyleft."
        ),
    ),
    StageGroup(
        id="similarity",
        title="Similarity and novelty",
        subtitle="Closeness to leads, or distance from them",
        purpose=(
            "Similarity to a reference set and novelty are opposing objectives. "
            "Choose one deliberately per project."
        ),
        default_mode="serial",
        accent="#e76399",
    ),
    StageGroup(
        id="custom",
        title="Custom models",
        subtitle="Your database, your algorithm",
        purpose=(
            "Bring a model trained on your own data. MolCascade computes the "
            "features and runs the exported graph locally."
        ),
        default_mode="serial",
        accent="#66407c",
    ),
)


_ALERT_ACTIONS: tuple[tuple[str, str], ...] = (
    ("reject", "Reject on a match"),
    ("warn", "Warn but keep"),
    ("ignore", "Do not evaluate"),
)

# The eight sources inside the rd_filters collection, ordered as the builder
# should present them: the two that reject by default first, then the rest.
# Each help string carries the rule count and the fraction of a panel of
# approved oral drugs the set flags, because that ratio is the whole basis on
# which a user should decide whether to promote a set to "reject".
_RD_FILTERS_SETS: tuple[tuple[str, str, str], ...] = (
    (
        "Glaxo",
        "Glaxo hard filters",
        "55 rules. Reactive and unstable chemistry. Flags 4% of approved oral "
        "drugs, and what it flags there is genuinely reactive.",
    ),
    (
        "BMS",
        "BMS (Bruns & Watson)",
        "180 rules, calibrated against a large screening analysis. Flags 8% of "
        "approved oral drugs.",
    ),
    (
        "Dundee",
        "Dundee (Brenk unwanted groups)",
        "105 rules. Broad, and blunt: flags a third of approved oral drugs, "
        "including aspirin and paracetamol. Best left as a warning.",
    ),
    (
        "PAINS",
        "PAINS (Baell & Holloway)",
        "481 rules. A hypothesis of assay interference, not a property of the "
        "molecule. Baell has written at length against applying it as a hard "
        "filter.",
    ),
    (
        "SureChEMBL",
        "SureChEMBL",
        "166 rules drawn from patent-literature curation.",
    ),
    (
        "Inpharmatica",
        "Inpharmatica",
        "91 rules. The only set upstream rd_filters enables by default.",
    ),
    ("LINT", "LINT", "57 rules from the Pfizer LINT filters."),
    (
        "MLSMR",
        "MLSMR",
        "116 rules from the NIH molecular libraries deck. The widest net here: "
        "flags half of approved oral drugs, so rejecting on it is rarely right.",
    ),
)

# medchem selects alert collections by name and rule sets by name.  Both are
# lists, and the builder has no list widget, so each preset is one comma-
# separated string the adapter splits.  Presets rather than free text because a
# name that matches nothing would silently stop filtering.
#
# Ordered narrow to wide, because that is the order someone picks in: a single
# collection they can defend in a methods section first, then the combinations.
# medchem carries fifteen collections that rd_filters does not -- the assay
# interference and toxicophore families below -- and those are the reason to
# reach for this option over the other two rather than a second opinion on the
# same patterns.
_MEDCHEM_ALERT_PRESETS: tuple[tuple[str, str], ...] = (
    ("BMS", "BMS only (180 patterns)"),
    ("Glaxo", "Glaxo only - reactive and unstable chemistry"),
    ("Dundee", "Dundee (Brenk) only - broad; better as a warning than a reject"),
    ("PAINS", "PAINS only (Baell and Holloway)"),
    ("SureChEMBL", "SureChEMBL only - patent-literature curation"),
    ("BMS,Dundee,Glaxo", "BMS + Dundee + Glaxo (340 patterns)"),
    (
        "BMS,Dundee,Glaxo,PAINS,SureChEMBL",
        "Industry standard + PAINS + SureChEMBL (987 patterns)",
    ),
    ("Inpharmatica,LINT,MLSMR", "Screening-deck collections: Inpharmatica + LINT + MLSMR"),
    (
        "Reactive-Unstable-Toxic,Electrophilic",
        "Reactive, unstable and electrophilic chemistry",
    ),
    (
        "Frequent-Hitter,AlphaScreen-Hitters,LuciferaseInhibitor,GST-Hitters,"
        "HIS-Hitters,DNABinder,Alarm-NMR,Chelator",
        "Assay interference - frequent hitters and readout artefacts",
    ),
    (
        "Genotoxic-Carcinogenicity,Non-Genotoxic-Carcinogenicity,LD50-Oral,Skin,Toxicophore",
        "Toxicophores - carcinogenicity, acute oral, skin sensitisation",
    ),
    (
        "BMS,Dundee,Glaxo,PAINS,SureChEMBL,Inpharmatica,MLSMR,LINT,"
        "Reactive-Unstable-Toxic,Electrophilic,Chelator,Frequent-Hitter",
        "Broad liability sweep (about 1,700 patterns)",
    ),
)

# Rule sets published as named filters.  Every drug-likeness rule medchem ships
# is offered, plus the combinations that are normally cited together, because
# "drug-like" is a different window in a fragment campaign, a PPI campaign and a
# CNS campaign, and the right one is a project decision rather than a default.
# A project wanting a mix that is not here can write the comma-separated list
# straight into the configuration file.
#
# The bounds in each label are the rule's own, so the choice can be made from
# the list instead of from the medchem documentation.
_MEDCHEM_RULE_PRESETS: tuple[tuple[str, str], ...] = (
    # Oral small molecules -- the mainstream of the collection.
    ("rule_of_five", "Lipinski rule of five: MW<=500, logP<=5, HBD<=5, HBA<=10"),
    ("rule_of_five,rule_of_veber", "Lipinski + Veber (oral bioavailability)"),
    (
        "rule_of_five,rule_of_veber,rule_of_egan,rule_of_ghose",
        "Lipinski + Veber + Egan + Ghose",
    ),
    ("rule_of_veber", "Veber alone: rotors <=10, TPSA <140"),
    ("rule_of_ghose", "Ghose: MW 160-480, logP -0.4 to 5.6, 20-70 atoms"),
    ("rule_of_druglike_soft", "Drug-like (soft): a wide fifteen-property window"),
    (
        "rule_of_chemaxon_druglikeness",
        "ChemAxon drug-likeness: MW <400, rotors <5, at least one ring",
    ),
    ("rule_of_zinc", "ZINC drug-like: the bounds the ZINC subsets are cut on"),
    ("rule_of_xu", "Xu: ring, rotor and heavy-atom counts, with no MW or logP term"),
    ("rule_of_reos", "REOS: the HTS deck-cleanup window, formal charge included"),
    # Lead-like and fragment starting points.
    ("rule_of_leadlike_soft", "Lead-like (soft), for hit-to-lead starting points"),
    ("rule_of_oprea", "Oprea lead-like: donor, acceptor, rotor and ring counts"),
    ("rule_of_three", "Rule of three, for fragment libraries"),
    ("rule_of_three_extended", "Rule of three, extended: adds a logP floor and a TPSA cap"),
    ("rule_of_two", "Rule of two: reagents and building blocks"),
    # Deliberately outside the small-molecule window.
    ("rule_of_five_beyond", "Beyond rule of five (bRo5): MW to 1000, rotors to 20"),
    ("rule_of_four", "Rule of four: protein-protein interaction inhibitors"),
    # Route- and compartment-specific.
    ("rule_of_cns", "CNS penetration"),
    ("rule_of_respiratory", "Respiratory / inhaled route"),
    # Machine-generated libraries.
    (
        "rule_of_generative_design",
        "Generative-design rules (permissive), for machine-generated libraries",
    ),
    (
        "rule_of_generative_design_strict",
        "Generative-design rules (strict), adds stereocentre and side-chain limits",
    ),
)

#: The ADMET-tagged subset of the same rule collection, kept separate because it
#: answers a different question.  The presets under ``drug_likeness`` ask whether
#: a molecule looks like a drug; these ask whether it is likely to be absorbed,
#: to reach the brain, or to cause trouble in an animal -- properties that a
#: neural ADMET model also predicts, far more expensively.
#:
#: Egan alone is the default, and the reason is that this adapter has no warning
#: outcome: every rule it is given either passes a molecule or deletes it.  Egan's
#: egg survives that treatment because it was fitted as an absorption *boundary*
#: and the great majority of oral drugs sit inside it.  Pfizer 3/75 does not:
#: Hughes et al. reported it as an increased *likelihood* of toxicity across 245
#: compounds, not a disqualifier, and lipophilic low-polarity chemistry is what a
#: CNS project is deliberately aiming at.  As a hard reject it would quietly
#: delete the target chemotype, so it is one click away rather than on by default.
#: GSK 4/400 is held back for that reason plus one more: at MW <= 400 and
#: logP <= 4 it mostly restates Lipinski, and counting one constraint twice makes
#: a funnel look more discriminating than it is.
_MEDCHEM_ADMET_PRESETS: tuple[tuple[str, str], ...] = (
    ("rule_of_egan", "Egan egg (passive intestinal absorption)"),
    (
        "rule_of_egan,rule_of_veber",
        "Egan absorption + Veber oral bioavailability",
    ),
    (
        "rule_of_egan,rule_of_pfizer_3_75",
        "Egan absorption + Pfizer 3/75 toxicity risk",
    ),
    ("rule_of_pfizer_3_75", "Pfizer 3/75 only (in vivo toleration risk)"),
    ("rule_of_gsk_4_400", "GSK 4/400 (ADMET rules of thumb)"),
    (
        "rule_of_gsk_4_400,rule_of_pfizer_3_75",
        "GSK 4/400 + Pfizer 3/75",
    ),
    (
        "rule_of_egan,rule_of_pfizer_3_75,rule_of_gsk_4_400",
        "All three property-based ADMET rule sets",
    ),
    ("rule_of_cns", "CNS penetration, for brain-targeted projects"),
    ("rule_of_respiratory", "Respiratory/inhaled route"),
)

_RULE_COMBINATIONS: tuple[tuple[str, str], ...] = (
    ("all", "Every rule set must pass (series)"),
    ("any", "Any one rule set is enough (parallel)"),
    ("at_least", "At least this many rule sets must pass"),
)


@dataclass(frozen=True, slots=True)
class _ADMETHead:
    """One ADMET-AI output column, with the skill its authors measured for it.

    The skill is in the builder's dropdown for a reason.  ADMET-AI is a single
    multitask ensemble, so every head costs the same to run and they look alike
    in a menu -- but they are not alike.  ``HIA_Hou`` reaches 0.99 AUROC and
    ``Half_Life_Obach`` reaches an R^2 of -2.39, which means predicting the
    training mean for every molecule would have been *better*.  A menu that
    hides that difference invites someone to delete a third of their library on
    the strength of a model that is worse than a constant.
    """

    column: str
    endpoint_id: str
    label: str
    #: Whether this head is fit to decide who survives.  Heads that are not are
    #: still predicted and recorded -- the evidence is worth having -- they just
    #: do not appear in the gate's endpoint list.
    gateable: bool = True


#: The recorded panel.  Chemprop predicts every task in one forward pass, so the
#: only cost of a wider panel is prediction rows on disk, and the ones here are
#: the questions a medicinal chemist actually asks of a kinase series.  Metrics
#: are ADMET-AI's own held-out numbers, read from ``admet_ai/resources/data``.
_ADMET_AI_HEADS: tuple[_ADMETHead, ...] = (
    # --- Absorption -------------------------------------------------------
    _ADMETHead(
        "HIA_Hou", "human_intestinal_absorption", "Human intestinal absorption · AUROC 0.99"
    ),
    _ADMETHead("Pgp_Broccatelli", "pgp_inhibition", "P-glycoprotein inhibition · AUROC 0.95"),
    _ADMETHead(
        "Solubility_AqSolDB",
        "aqueous_solubility",
        "Aqueous solubility, log(mol/L) · R² 0.82",
    ),
    _ADMETHead(
        "Lipophilicity_AstraZeneca", "lipophilicity_logd", "Lipophilicity logD7.4 · R² 0.77"
    ),
    _ADMETHead("Caco2_Wang", "caco2_permeability", "Caco-2 permeability, log(10⁻⁶ cm/s) · R² 0.71"),
    _ADMETHead("PAMPA_NCATS", "pampa_permeability", "PAMPA permeability · AUROC 0.79"),
    _ADMETHead(
        "Bioavailability_Ma", "oral_bioavailability", "Oral bioavailability · AUROC 0.72 (weak)"
    ),
    # --- Distribution -----------------------------------------------------
    _ADMETHead(
        "BBB_Martins", "blood_brain_barrier", "Blood-brain barrier penetration · AUROC 0.90"
    ),
    _ADMETHead("PPBR_AZ", "plasma_protein_binding", "Plasma protein binding, % · R² 0.59 (weak)"),
    _ADMETHead(
        "VDss_Lombardo",
        "volume_of_distribution",
        "Volume of distribution · R² -1.21 — worse than the training mean",
        gateable=False,
    ),
    # --- Metabolism -------------------------------------------------------
    _ADMETHead("CYP1A2_Veith", "cyp1a2_inhibition", "CYP1A2 inhibition · AUROC 0.94"),
    _ADMETHead("CYP2C19_Veith", "cyp2c19_inhibition", "CYP2C19 inhibition · AUROC 0.91"),
    _ADMETHead("CYP2C9_Veith", "cyp2c9_inhibition", "CYP2C9 inhibition · AUROC 0.91"),
    _ADMETHead("CYP2D6_Veith", "cyp2d6_inhibition", "CYP2D6 inhibition · AUROC 0.89"),
    _ADMETHead("CYP3A4_Veith", "cyp3a4_inhibition", "CYP3A4 inhibition · AUROC 0.91"),
    # --- Excretion --------------------------------------------------------
    _ADMETHead(
        "Clearance_Hepatocyte_AZ",
        "hepatocyte_clearance",
        "Hepatocyte clearance · R² 0.26 (weak)",
    ),
    _ADMETHead(
        "Clearance_Microsome_AZ", "microsome_clearance", "Microsomal clearance · R² 0.28 (weak)"
    ),
    _ADMETHead(
        "Half_Life_Obach",
        "half_life",
        "Half-life · R² -2.39 — worse than the training mean",
        gateable=False,
    ),
    # --- Toxicity ---------------------------------------------------------
    _ADMETHead("hERG", "herg_blocking", "hERG blocking · AUROC 0.84"),
    _ADMETHead("ClinTox", "clinical_toxicity", "Clinical trial toxicity · AUROC 0.93"),
    _ADMETHead("AMES", "ames_mutagenicity", "Ames mutagenicity · AUROC 0.88"),
    _ADMETHead("DILI", "drug_induced_liver_injury", "Drug-induced liver injury · AUROC 0.88"),
    _ADMETHead("Carcinogens_Lagunin", "carcinogenicity", "Carcinogenicity · AUROC 0.77"),
    _ADMETHead("LD50_Zhu", "acute_toxicity_ld50", "Acute toxicity LD50 · R² 0.60"),
)

_ADMET_AI_ENDPOINT_DEFAULTS: tuple[dict[str, str], ...] = tuple(
    {"output_column": head.column, "endpoint_id": head.endpoint_id} for head in _ADMET_AI_HEADS
)

_ADMET_AI_GATE_CHOICES: tuple[tuple[str, str], ...] = tuple(
    (head.endpoint_id, head.label) for head in _ADMET_AI_HEADS if head.gateable
)

#: The slice of Mordred's 1,613 two-dimensional descriptors this builder offers.
#:
#: A text box over the full registry would be a trap rather than a feature.
#: ``naRing`` counts aromatic rings and ``nARing`` counts aliphatic ones; they
#: differ by the case of a single letter, both are valid, and picking the wrong
#: one applies the opposite filter to the whole library without erroring.  So the
#: names are offered as a labelled list, and every label says what the number
#: actually means rather than repeating Mordred's abbreviation.
#:
#: The selection favours what RDKit's own range gate cannot already express --
#: ring topology, saturation, framework fraction, complexity.  The overlapping
#: entries at the top are kept so a project can put the two engines side by side
#: on the same property if it wants to.
#:
#: ``tests/plugins/test_mordred_gate.py`` asserts every name here still resolves
#: against the installed release, because a drifted name would otherwise surface
#: only after someone assembled a flow and started a screen.
_MORDRED_DESCRIPTOR_CHOICES: tuple[tuple[str, str], ...] = (
    # --- Size and composition --------------------------------------------
    ("MW", "Molecular weight (Da)"),
    ("AMW", "Average atomic mass per atom"),
    ("nHeavyAtom", "Heavy atom count"),
    ("nHetero", "Heteroatom count"),
    ("nX", "Halogen atom count"),
    ("nN", "Nitrogen atom count"),
    ("nO", "Oxygen atom count"),
    ("nS", "Sulfur atom count"),
    ("Vabc", "Van der Waals volume, Å³ — undefined for salts"),
    # --- Lipophilicity and polarity ---------------------------------------
    ("SLogP", "Wildman-Crippen logP"),
    ("SMR", "Wildman-Crippen molar refractivity"),
    ("TopoPSA", "Topological polar surface area, Å²"),
    ("TopoPSA(NO)", "Polar surface area counting only N and O, Å²"),
    # --- Hydrogen bonding and flexibility ---------------------------------
    ("nHBDon", "Hydrogen bond donors"),
    ("nHBAcc", "Hydrogen bond acceptors"),
    ("nRot", "Rotatable bonds"),
    ("RotRatio", "Rotatable bonds as a fraction of all bonds"),
    # --- Ring topology ------------------------------------------------------
    ("nRing", "Rings, all kinds"),
    ("naRing", "Aromatic rings"),
    ("nARing", "Aliphatic rings"),
    ("nHRing", "Rings containing a heteroatom"),
    ("naHRing", "Aromatic heterocycles"),
    ("nFRing", "Fused rings"),
    ("nBridgehead", "Bridgehead atoms"),
    ("nSpiro", "Spiro atoms"),
    # --- Shape, saturation and complexity ---------------------------------
    ("FCSP3", "Fraction of carbons that are sp3"),
    ("fMF", "Fraction of heavy atoms in the Murcko framework"),
    ("BertzCT", "Bertz structural complexity"),
    ("Kier1", "Kier first-order shape index — undefined for salts"),
    ("Kier2", "Kier second-order shape index — undefined for salts"),
    ("Kier3", "Kier third-order shape index — undefined for small molecules"),
    ("Zagreb1", "First Zagreb connectivity index"),
    # --- Ionisation ---------------------------------------------------------
    ("nAcid", "Acidic groups"),
    ("nBase", "Basic groups"),
)

_BATCH = ThresholdField(
    name="batch_size",
    label="Batch size",
    kind="integer",
    minimum=1,
    maximum=250_000,
    step=1_024,
    help="Rows processed per chunk. Lower this if memory is tight.",
)


def _window(
    *,
    unit: str | None = None,
    minimum_label: str = "Minimum",
    maximum_label: str = "Maximum",
    lower: float | None = None,
    upper: float | None = None,
    step: float | None = None,
    target: ThresholdTarget = "gate",
    help_text: str = "",
) -> tuple[ThresholdField, ...]:
    """Build the inclusive lower/upper pair used by every numeric evidence gate."""

    return (
        ThresholdField(
            name="minimum",
            label=minimum_label,
            target=target,
            unit=unit,
            minimum=lower,
            maximum=upper,
            step=step,
            nullable=True,
            help=help_text,
        ),
        ThresholdField(
            name="maximum",
            label=maximum_label,
            target=target,
            unit=unit,
            minimum=lower,
            maximum=upper,
            step=step,
            nullable=True,
            help=help_text,
        ),
    )


CRITERIA: tuple[CriterionSpec, ...] = (
    CriterionSpec(
        id="structure_validity",
        label="Structural validity",
        question="Can this string be parsed into a chemically sensible single molecule?",
        stage="chemistry",
        evidence="decision",
        summary=(
            "Terminal rejects for unparseable strings, valence errors, empty or "
            "fragment-only records, disallowed elements and severe reactivity."
        ),
        options=(
            BackendOption(
                id="rdkit_hard_gate",
                label="RDKit hard rules",
                engine="RDKit",
                summary=(
                    "Sanitization, valence, fragment, element allow-list and severe "
                    "reactivity checks in one auditable pass."
                ),
                plugin_ref="chemistry.rdkit_hard_gate@0.1.0",
                license_spdx="BSD-3-Clause",
                citation=(
                    "RDKit: open-source cheminformatics. https://www.rdkit.org — the "
                    "sanitization, valence and FilterCatalog machinery this gate drives."
                ),
                throughput_per_second=850,
                recommended=True,
                defaults={"policy": {"schema_version": 1}},
                thresholds=(
                    ThresholdField(
                        name="policy.flag_pains",
                        label="Treat PAINS matches as a hard reject",
                        kind="boolean",
                        help=(
                            "Leave off if you prefer to handle PAINS in the alerts tier, "
                            "where the decision is reversible."
                        ),
                    ),
                    _BATCH,
                ),
            ),
            BackendOption(
                id="chembl_structure_pipeline",
                label="ChEMBL Structure Pipeline checker",
                engine="ChEMBL",
                summary=(
                    "The curation checker EBI runs over ChEMBL itself. Its verdicts "
                    "come from InChI as well as RDKit, so agreement here is two "
                    "toolkits agreeing rather than one toolkit repeating itself."
                ),
                plugin_ref="chemistry.chembl_structure_check@0.1.0",
                license_spdx="MIT",
                requires=("chembl_structure_pipeline",),
                # Measured at ~100 molecules/s: writing a mol block and running
                # the InChI library costs eight times an RDKit sanitize, and
                # threads do not help because the GIL is held throughout.  That
                # is why the starter cascade runs this last rather than first,
                # not because the check is any less worth making.
                throughput_per_second=100,
                recommended=True,
                citation=(
                    "Bento AP, Hersey A, Félix E, et al. An open source chemical "
                    "structure curation pipeline using RDKit. J Cheminform. "
                    "2020;12:51. doi:10.1186/s13321-020-00456-1"
                ),
                notes=(
                    "Scores each finding on the published 0-9 penalty scale. The "
                    "default rejects at 6, the band upstream uses for structures "
                    "nobody meant to draw; undefined stereo scores 2 and is kept, "
                    "because in a generated library that describes almost everything. "
                    "Around 100 molecules/s, so it earns its place at the bottom of a "
                    "funnel rather than the top."
                ),
                thresholds=(
                    ThresholdField(
                        name="reject_at_penalty",
                        label="Reject at penalty",
                        kind="integer",
                        minimum=1,
                        maximum=9,
                        step=1,
                        help=(
                            "6 = structural disqualifiers only (radicals off the known "
                            "list, polymers, embedded 3D). 5 also rejects drawing and "
                            "bond-stereo faults. 2 additionally rejects any molecule "
                            "whose stereochemistry is left undefined."
                        ),
                    ),
                    _BATCH,
                ),
            ),
        ),
    ),
    CriterionSpec(
        id="physchem_window",
        label="Physicochemical window",
        question="Do molecular weight, logP, TPSA and rotors sit inside the project window?",
        stage="physchem",
        evidence="decision",
        summary="Editable MW/cLogP/TPSA/HBD/HBA/rotor/charge ranges with complete decisions.",
        options=(
            BackendOption(
                id="rdkit_property_range",
                label="RDKit property ranges",
                engine="RDKit",
                summary="Crippen logP, TPSA and Lipinski counts calculated per molecule.",
                plugin_ref="chemistry.rdkit_property_range_gate@0.1.0",
                license_spdx="BSD-3-Clause",
                citation=(
                    "Wildman SA, Crippen GM. Prediction of physicochemical parameters by "
                    "atomic contributions. J Chem Inf Comput Sci. 1999;39(5):868-873. "
                    "doi:10.1021/ci990307l (cLogP); Ertl P, Rohde B, Selzer P. Fast "
                    "calculation of molecular polar surface area. J Med Chem. "
                    "2000;43(20):3714-3717. doi:10.1021/jm000942e (TPSA)"
                ),
                throughput_per_second=270,
                recommended=True,
                thresholds=(
                    ThresholdField(
                        name="mw_min",
                        label="MW minimum",
                        unit="Da",
                        minimum=0,
                        maximum=10_000,
                        step=10,
                        nullable=True,
                    ),
                    ThresholdField(
                        name="mw_max",
                        label="MW maximum",
                        unit="Da",
                        minimum=0,
                        maximum=10_000,
                        step=10,
                        nullable=True,
                    ),
                    ThresholdField(
                        name="clogp_min",
                        label="cLogP minimum",
                        minimum=-100,
                        maximum=100,
                        step=0.5,
                        nullable=True,
                    ),
                    ThresholdField(
                        name="clogp_max",
                        label="cLogP maximum",
                        minimum=-100,
                        maximum=100,
                        step=0.5,
                        nullable=True,
                    ),
                    ThresholdField(
                        name="tpsa_max",
                        label="TPSA maximum",
                        unit="Å²",
                        minimum=0,
                        maximum=10_000,
                        step=10,
                        nullable=True,
                    ),
                    ThresholdField(
                        name="hbd_max",
                        label="H-bond donors maximum",
                        kind="integer",
                        minimum=0,
                        maximum=1_000,
                        step=1,
                        nullable=True,
                    ),
                    ThresholdField(
                        name="hba_max",
                        label="H-bond acceptors maximum",
                        kind="integer",
                        minimum=0,
                        maximum=1_000,
                        step=1,
                        nullable=True,
                    ),
                    ThresholdField(
                        name="rotatable_bonds_max",
                        label="Rotatable bonds maximum",
                        kind="integer",
                        minimum=0,
                        maximum=10_000,
                        step=1,
                        nullable=True,
                    ),
                    ThresholdField(
                        name="absolute_formal_charge_max",
                        label="Absolute formal charge maximum",
                        kind="integer",
                        minimum=0,
                        maximum=100,
                        step=1,
                        nullable=True,
                    ),
                ),
            ),
            BackendOption(
                id="mordred_descriptor_window",
                label="Mordred descriptor window",
                engine="Mordred",
                summary=(
                    "A window on any one of Mordred's 1,613 two-dimensional "
                    "descriptors -- ring topology, shape, framework fraction."
                ),
                plugin_ref="chemistry.mordred_descriptor_gate@0.1.0",
                license_spdx="BSD-3-Clause",
                citation=(
                    "Moriwaki H, Tian Y-S, Kawashita N, Takagi T. Mordred: a molecular "
                    "descriptor calculator. J Cheminform. 2018;10:4. "
                    "doi:10.1186/s13321-018-0258-y; Ritchie TJ, Macdonald SJF. The "
                    "impact of aromatic ring count on compound developability. Drug "
                    "Discov Today. 2009;14(21-22):1011-1020. "
                    "doi:10.1016/j.drudis.2009.07.014 (the shipped default window)"
                ),
                requires=("mordred",),
                # Measured on this machine: 197 mol/s for eight descriptors and
                # 65 mol/s for thirty-nine, so roughly 0.4 ms per descriptor per
                # molecule. One descriptor per block is the shape the builder
                # produces, hence the single-descriptor figure.
                throughput_per_second=400,
                defaults={
                    "schema_version": 1,
                    # Aromatic ring count, capped at four. Ritchie and Macdonald
                    # found developability falling off sharply above three rings
                    # across roughly 3,000 oral drugs and candidates, so four is
                    # one ring of slack on a well-evidenced boundary. It is also
                    # the rare structural filter that does not simply restate
                    # molecular weight, and it leaves normal kinase chemistry --
                    # two to four aromatic rings -- untouched.
                    "descriptor": "naRing",
                    "maximum": 4,
                },
                thresholds=(
                    ThresholdField(
                        name="descriptor",
                        label="Descriptor",
                        kind="choice",
                        choices=_MORDRED_DESCRIPTOR_CHOICES,
                        help=(
                            "One descriptor per block. Drag in another block to add "
                            "another window; the tier's serial or parallel join is "
                            "what combines them."
                        ),
                    ),
                    ThresholdField(
                        name="minimum",
                        label="Minimum",
                        minimum=-1e9,
                        maximum=1e9,
                        step=1,
                        nullable=True,
                        help="Inclusive lower bound. Leave empty for no lower bound.",
                    ),
                    ThresholdField(
                        name="maximum",
                        label="Maximum",
                        minimum=-1e9,
                        maximum=1e9,
                        step=1,
                        nullable=True,
                        help="Inclusive upper bound. Leave empty for no upper bound.",
                    ),
                    _BATCH,
                ),
                notes=(
                    "Mordred is a second descriptor library, not a second opinion: it "
                    "uses RDKit for molecule handling, and where the two overlap "
                    "(SLogP, TopoPSA, donor and acceptor counts) they agree. Its value "
                    "here is the 1,600 descriptors RDKit's range gate does not "
                    "expose.\n\n"
                    "A molecule whose descriptor cannot be computed is rejected, not "
                    "waved through. Mordred reports 'missing' rather than raising, and "
                    "several descriptors -- van der Waals volume and every Kier shape "
                    "index among them -- are undefined across disconnected fragments. "
                    "Desalt upstream, or a window on those will remove every salt for "
                    "a reason that has nothing to do with its chemistry.\n\n"
                    "Aromatic and aliphatic ring counts differ only by the case of one "
                    "letter in Mordred's own naming, which is why this block offers a "
                    "labelled list instead of a text box."
                ),
            ),
        ),
    ),
    CriterionSpec(
        id="ring_topology",
        label="Ring topology and rigidity",
        question="Is the ring system a shape a drug could have?",
        stage="physchem",
        evidence="decision",
        summary=(
            "Editable caps on ring count, fused-system size, ring size, rotors, "
            "ring-atom fraction, sp3 fraction and bridgeheads."
        ),
        options=(
            BackendOption(
                id="rdkit_ring_topology",
                label="RDKit ring topology",
                engine="RDKit",
                summary=(
                    "Counts the rings fused into the largest system -- the number "
                    "medchem and Mordred do not report."
                ),
                plugin_ref="chemistry.rdkit_ring_topology_gate@0.1.0",
                license_spdx="BSD-3-Clause",
                citation=(
                    "Lagorce D, Bouslama L, Becot J, Miteva MA, Villoutreix BO. "
                    "FAF-Drugs4: free ADME-tox filtering computations for chemical "
                    "biology and early stages drug discovery. Bioinformatics. "
                    "2017;33(22):3658-3660. doi:10.1093/bioinformatics/btx491 (rings "
                    "and ring-system size); Benigni R, Bossa C, Jeliazkova N, Netzeva "
                    "T, Worth A. The Benigni/Bossa rulebase for mutagenicity and "
                    "carcinogenicity. JRC Report EUR 23241 EN. 2008. "
                    "doi:10.2788/60246 (fused polycyclic aromatics, SA_18); Shearer J, "
                    "Castro JL, Lawson ADG, MacCoss M, Taylor RD. Rings in clinical "
                    "trials and drugs: present and future. J Med Chem. "
                    "2022;65(13):8699-8712. doi:10.1021/acs.jmedchem.2c00473 "
                    "(ring systems in drugs)"
                ),
                notes=(
                    "Aimed at three-dimensional generative libraries, where fused "
                    "polycyclic sheets with no rotatable bond pass Lipinski, QED, "
                    "PAINS and ADMET and then dock better than the flexible "
                    "molecules around them -- a rigid surface makes many contacts "
                    "and pays no entropy for it. Over one 2,000-molecule run here, "
                    "molecules with four or more fused rings were 71.5% of the "
                    "input and 83.3% of the final shortlist. The three defaults "
                    "leave 30 of 33 approved drugs standing, costing only morphine, "
                    "camptothecin and artemisinin. The rotor floor, ring-atom "
                    "fraction, sp3 floor, bridgehead cap and all-systems fused "
                    "total ship empty on purpose: "
                    "each has a defensible use on a generated library and none has "
                    "a threshold the literature will support against approved "
                    "drugs, so a project that wants one sets it deliberately."
                ),
                # Ring perception plus seven RDKit descriptors, all graph-only:
                # no conformer, no fingerprint. Comfortably faster than the
                # property panel's 270 mol/s, which pays for Crippen and TPSA.
                throughput_per_second=600,
                recommended=True,
                defaults={
                    "schema_version": 1,
                    # FAF-Drugs4's own drug-like values, chosen by its authors so
                    # that up to 90% of 916 approved oral drugs pass. Taken
                    # unchanged so the default strictness here is a published
                    # calibration rather than a preference.
                    "max_rings": 6,
                    "max_ring_system_size": 18,
                    # No filter to inherit from: nothing else counts rings within
                    # a fused system. Toxtree flags three or more fused aromatic
                    # rings; four is one step looser, which is what passes the
                    # steroids.
                    "max_fused_rings": 4,
                },
                thresholds=(
                    ThresholdField(
                        name="max_rings",
                        label="Rings maximum",
                        kind="integer",
                        minimum=0,
                        maximum=1_000,
                        step=1,
                        nullable=True,
                        help=(
                            "Total rings (SSSR). FAF-Drugs4 drug-like uses 6. "
                            "Leave empty for no cap."
                        ),
                    ),
                    ThresholdField(
                        name="max_fused_rings",
                        label="Fused rings maximum",
                        kind="integer",
                        minimum=1,
                        maximum=1_000,
                        step=1,
                        nullable=True,
                        help=(
                            "Rings fused into the largest single system, ignoring "
                            "aromaticity. Pentacene is 5 and pyrene is 4; both are "
                            "1 to medchem and Mordred. Spiro junctions do not "
                            "count as fusion."
                        ),
                    ),
                    ThresholdField(
                        name="max_fused_ring_total",
                        label="Fused rings maximum, all systems",
                        kind="integer",
                        minimum=0,
                        maximum=1_000,
                        step=1,
                        nullable=True,
                        help=(
                            "Rings summed over every fused system, so three "
                            "separate naphthalenes total 6 while the largest is "
                            "only 2. Empty by default; pair it with a tight "
                            "fused-rings cap, e.g. 2 and 4 for 'at most two "
                            "bicyclics'."
                        ),
                    ),
                    ThresholdField(
                        name="max_ring_system_size",
                        label="Fused system atoms maximum",
                        kind="integer",
                        unit="atoms",
                        minimum=3,
                        maximum=1_000,
                        step=1,
                        nullable=True,
                        help=(
                            "Heavy atoms in the largest fused system. A useful "
                            "second opinion, but peri-fusion slips past it: pyrene "
                            "packs 4 rings into 16 atoms."
                        ),
                    ),
                    ThresholdField(
                        name="max_ring_size",
                        label="Largest ring maximum",
                        kind="integer",
                        unit="atoms",
                        minimum=3,
                        maximum=1_000,
                        step=1,
                        nullable=True,
                        help=(
                            "Atoms in the largest single ring, for excluding "
                            "macrocycles. 12 is the conventional threshold; "
                            "erythromycin has a 14-membered ring. Empty by default."
                        ),
                    ),
                    ThresholdField(
                        name="min_rotatable_bonds",
                        label="Rotatable bonds minimum",
                        kind="integer",
                        minimum=0,
                        maximum=1_000,
                        step=1,
                        nullable=True,
                        help=(
                            "A floor, not Veber's ceiling. Empty by default: a "
                            "floor of 3 would reject caffeine, estradiol, morphine "
                            "and olanzapine, which have none."
                        ),
                    ),
                    ThresholdField(
                        name="max_ring_atom_fraction",
                        label="Ring atom fraction maximum",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        nullable=True,
                        help=(
                            "Ring atoms over heavy atoms; 1.00 means no "
                            "substituent chain at all. Empty by default: "
                            "olanzapine is 0.91."
                        ),
                    ),
                    ThresholdField(
                        name="min_fraction_csp3",
                        label="Fraction sp3 carbon minimum",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        nullable=True,
                        help=(
                            "Escape-from-flatland saturation. Empty by default: "
                            "Lovering's 0.36-0.47 are cohort means, and a floor of "
                            "0.25 rejects imatinib at 0.24."
                        ),
                    ),
                    ThresholdField(
                        name="max_bridgehead_atoms",
                        label="Bridgehead atoms maximum",
                        kind="integer",
                        minimum=0,
                        maximum=1_000,
                        step=1,
                        nullable=True,
                        help=(
                            "Atoms shared by three rings -- the signature of a "
                            "cage. Empty by default: no citable threshold, and "
                            "bridged bicyclics are ordinary chemistry."
                        ),
                    ),
                    _BATCH,
                ),
            ),
        ),
    ),
    CriterionSpec(
        id="drug_likeness",
        label="Rule of Five and QED",
        question="How many Lipinski violations does it have, and what is its QED?",
        stage="physchem",
        evidence="decision",
        summary=(
            "Rule of Five violation counting and QED are reported separately because "
            "they measure different things."
        ),
        options=(
            BackendOption(
                id="rdkit_drug_likeness",
                label="RDKit Rule of Five + QED",
                engine="RDKit",
                summary="Lipinski violation count and RDKit QED with independent thresholds.",
                plugin_ref="chemistry.rdkit_drug_likeness@0.1.0",
                license_spdx="BSD-3-Clause",
                citation=(
                    "Bickerton GR, Paolini GV, Besnard J, Muresan S, Hopkins AL. "
                    "Quantifying the chemical beauty of drugs. Nat Chem. 2012;4(2):90-98. "
                    "doi:10.1038/nchem.1243 (QED); Lipinski CA, Lombardo F, Dominy BW, "
                    "Feeney PJ. Adv Drug Deliv Rev. 1997;23(1-3):3-25. "
                    "doi:10.1016/S0169-409X(96)00423-1 (Rule of Five)"
                ),
                # Measured through the adapter at about 113 molecules/s -- less
                # than half the plain property panel's 270, because QED evaluates
                # eight desirability functions per molecule on top of the
                # descriptors the Rule of Five already needed.
                throughput_per_second=113,
                recommended=True,
                defaults={"schema_version": 1, "failure_action": "reject"},
                thresholds=(
                    ThresholdField(
                        name="maximum_rule_of_five_violations",
                        label="Maximum Rule of Five violations",
                        kind="integer",
                        minimum=0,
                        maximum=4,
                        step=1,
                        nullable=True,
                        help="0 keeps only fully compliant molecules; 1 is the usual triage.",
                    ),
                    ThresholdField(
                        name="minimum_qed",
                        label="Minimum QED",
                        minimum=0,
                        maximum=1,
                        step=0.05,
                        nullable=True,
                        help="QED is a weighted desirability score, not a Lipinski restatement.",
                    ),
                    ThresholdField(
                        name="failure_action",
                        label="When a threshold is breached",
                        kind="choice",
                        choices=(("reject", "Reject the molecule"), ("warn", "Warn only")),
                    ),
                ),
            ),
            BackendOption(
                id="medchem_rules",
                label="medchem published rule sets",
                engine="medchem",
                summary=(
                    "Twenty-two named rule sets -- Veber, Ghose, Egan, Oprea, Xu, REOS, "
                    "Pfizer 3/75, GSK 4/400, ZINC, lead-like, CNS, fragment and "
                    "generative-design rules -- combined in series or in parallel."
                ),
                plugin_ref="chemistry.medchem_rules@0.1.0",
                license_spdx="Apache-2.0",
                citation=(
                    "medchem, Datamol.io. https://github.com/datamol-io/medchem — every "
                    "rule set additionally carries the reference of the paper that "
                    "published it, and that reference is what belongs in a methods section "
                    "alongside this one."
                ),
                requires=("medchem",),
                # Measured at about 45 molecules/s on one core for a two-rule set,
                # and process parallelism only returns about 2.3x before spawn
                # overhead reverses it. That is six hours per million molecules,
                # against RDKit's own Rule-of-Five gate at 270/s; the twenty-two
                # rule sets are what you are paying for, so take them where the
                # population is already small.
                throughput_per_second=45,
                defaults={
                    "schema_version": 1,
                    "rules": "rule_of_five,rule_of_veber",
                    "combination": "all",
                },
                thresholds=(
                    ThresholdField(
                        name="rules",
                        label="Rule sets",
                        kind="choice",
                        choices=_MEDCHEM_RULE_PRESETS,
                        help=(
                            "Which published rule sets to apply. Generated libraries "
                            "usually want the generative-design rules alongside a "
                            "classical set."
                        ),
                    ),
                    ThresholdField(
                        name="combination",
                        label="How the rule sets combine",
                        kind="choice",
                        choices=_RULE_COMBINATIONS,
                        help=(
                            "The same series/parallel choice a tier offers, applied "
                            "within this one criterion."
                        ),
                    ),
                    ThresholdField(
                        name="minimum_passes",
                        label="Rule sets that must pass",
                        kind="integer",
                        minimum=1,
                        maximum=64,
                        step=1,
                        help="Used only when the combination above is 'at least this many'.",
                    ),
                ),
            ),
        ),
    ),
    CriterionSpec(
        id="structural_alerts",
        label="PAINS and reactivity alerts",
        question="Does it match a known interference, aggregation or instability substructure?",
        stage="alerts",
        evidence="decision",
        summary=(
            "Substructure alert catalogues. These are triage hypotheses about assay "
            "behaviour, not measured chemical stability."
        ),
        options=(
            BackendOption(
                id="rdkit_structural_alerts",
                label="RDKit PAINS and reactivity catalogues",
                engine="RDKit",
                summary="RDKit FilterCatalog families with per-family severity policy.",
                plugin_ref="chemistry.rdkit_structural_alerts@0.1.0",
                license_spdx="BSD-3-Clause",
                citation=(
                    "Baell JB, Holloway GA. New substructure filters for removal of "
                    "pan assay interference compounds (PAINS). J Med Chem. "
                    "2010;53(7):2719-2740. doi:10.1021/jm901137j; Brenk R, Schipani "
                    "A, James D, et al. Lessons learnt from assembling screening "
                    "libraries for drug discovery for neglected diseases. "
                    "ChemMedChem. 2008;3(3):435-444. doi:10.1002/cmdc.200700139"
                ),
                # Substructure matching against hundreds of SMARTS is the
                # slowest thing in the starter cascade -- slower per molecule
                # than the ChEMBL checker -- which is exactly why the alerts
                # tier sits below three cheaper ones.
                throughput_per_second=85,
                recommended=True,
                defaults={"schema_version": 1, "pains_action": "reject"},
                thresholds=(
                    ThresholdField(
                        name="pains_action",
                        label="PAINS catalogue",
                        kind="choice",
                        choices=_ALERT_ACTIONS,
                        help="Pan-assay interference substructures collated by Baell and Holloway.",
                    ),
                    ThresholdField(
                        name="brenk_action",
                        label="Brenk unwanted-groups catalogue",
                        kind="choice",
                        choices=_ALERT_ACTIONS,
                        help="Reactive, unstable and toxicophore groups from Brenk et al.",
                    ),
                    ThresholdField(
                        name="nih_action",
                        label="NIH catalogue",
                        kind="choice",
                        choices=_ALERT_ACTIONS,
                    ),
                    ThresholdField(
                        name="zinc_action",
                        label="ZINC catalogue",
                        kind="choice",
                        choices=_ALERT_ACTIONS,
                    ),
                ),
            ),
            BackendOption(
                id="medchem_common_alerts",
                label="medchem alert collections and NIBR rules",
                engine="medchem",
                summary=(
                    "Twenty-three curated collections (BMS, Dundee, Glaxo, Inpharmatica, "
                    "MLSMR, SureChEMBL, PAINS and more) plus the Novartis screening-deck "
                    "rules, from the datamol-io medchem package."
                ),
                plugin_ref="chemistry.medchem_alerts@0.1.0",
                license_spdx="Apache-2.0",
                citation=(
                    "Baell JB, Holloway GA. New substructure filters for removal of pan "
                    "assay interference compounds (PAINS). J Med Chem. "
                    "2010;53(7):2719-2740. doi:10.1021/jm901137j; Schuffenhauer A, "
                    "Schneider N, Hintermann S, et al. Evolution of Novartis' small "
                    "molecule screening deck design. J Med Chem. 2020;63(23):14425-14447. "
                    "doi:10.1021/acs.jmedchem.0c01332 (NIBR)"
                ),
                requires=("medchem",),
                # Measured through the adapter at about 28 molecules/s -- the
                # slowest option in the catalogue, and roughly three times slower
                # than RDKit's own alert catalogues. The extra time buys coverage:
                # about 2,400 curated SMARTS across 23 published collections
                # against RDKit's four. That is a trade worth making on a
                # shortlist and not worth making on a raw generated library.
                throughput_per_second=28,
                defaults={
                    "schema_version": 1,
                    "alert_sets": "BMS,Dundee,Glaxo",
                    "use_nibr": False,
                },
                thresholds=(
                    ThresholdField(
                        name="alert_sets",
                        label="Alert collections",
                        kind="choice",
                        choices=_MEDCHEM_ALERT_PRESETS,
                        help=(
                            "Which curated SMARTS collections to match. Wider is stricter: "
                            "the full set carries about 2,400 patterns."
                        ),
                    ),
                    ThresholdField(
                        name="use_nibr",
                        label="Add the Novartis screening-deck rules",
                        kind="boolean",
                        help=(
                            "NIBR rules carry their own severity arithmetic and are "
                            "applied alongside the collections above."
                        ),
                    ),
                    ThresholdField(
                        name="nibr_reject_at_severity",
                        label="Reject at NIBR severity",
                        kind="integer",
                        minimum=1,
                        maximum=100,
                        step=1,
                        help=(
                            "Accumulated severity at which a molecule is rejected. NIBR "
                            "scores a single hard exclusion as 10, so a lower number also "
                            "rejects molecules that merely collect many soft flags."
                        ),
                    ),
                    ThresholdField(
                        name="nibr_compound_class_action",
                        label="NIBR exclusions that are only a compound class",
                        kind="choice",
                        choices=_ALERT_ACTIONS,
                        help=(
                            "67 NIBR rules exclude a molecule for being a steroid, "
                            "peptide, nucleoside, glycoside or similar rather than for "
                            "any liability. Rejecting these removes whole chemotypes; "
                            "keep it at reject only if that is what you want."
                        ),
                    ),
                    ThresholdField(
                        name="exclude_action",
                        label="Molecules medchem marks 'exclude'",
                        kind="choice",
                        choices=_ALERT_ACTIONS,
                        help="The strongest match class. Rejecting these is the usual choice.",
                    ),
                    ThresholdField(
                        name="flag_action",
                        label="Molecules medchem marks 'flag'",
                        kind="choice",
                        choices=_ALERT_ACTIONS,
                        help="A match worth a chemist's attention rather than automatic removal.",
                    ),
                    ThresholdField(
                        name="annotation_action",
                        label="Molecules medchem marks 'annotations'",
                        kind="choice",
                        choices=_ALERT_ACTIONS,
                        help="Informational matches. Usually ignored.",
                    ),
                ),
            ),
            BackendOption(
                id="rd_filters",
                label="rd_filters alert collection (Walters)",
                engine="MolCascade + RDKit",
                summary=(
                    "The exact 1,251 SMARTS that published triage work means when it "
                    "says 'we used rd_filters', across eight sources. Each source "
                    "carries its own action, so reactive chemistry can be rejected "
                    "while PAINS is only flagged."
                ),
                plugin_ref="chemistry.rd_filters_alerts@0.1.0",
                license_spdx="MIT",
                citation=(
                    "Walters WP. rd_filters. https://github.com/PatWalters/rd_filters — "
                    "the alert definitions are the ChEMBL structural-alert sets, and the "
                    "table shipped with a run is pinned by digest."
                ),
                # Nothing to install: the rules are a vendored data file and the
                # matching is RDKit's own, so the only dependency is the table.
                requires=(),
                requires_assets=("rd_filters",),
                # Measured through the adapter at about 31 molecules/s. Same shape
                # of trade as the medchem collections: a much larger SMARTS table
                # scanned per molecule, so cost scales with the table and not with
                # the toolkit.
                throughput_per_second=31,
                defaults={
                    "schema_version": 1,
                    "alerts_path": "asset:rd_filters/data/alert_collection.csv",
                    "glaxo_action": "reject",
                    "bms_action": "reject",
                    "dundee_action": "warn",
                    "pains_action": "warn",
                    "surechembl_action": "warn",
                    "inpharmatica_action": "warn",
                    "lint_action": "warn",
                    "mlsmr_action": "warn",
                },
                notes=(
                    "Run 'molcascade assets fetch rd_filters' once. The table is "
                    "verified against a pinned digest on every run, and the stage "
                    "refuses to start if it does not match.\n\n"
                    "The default actions were set by measurement: against a panel of "
                    "approved oral drugs, Glaxo flags 4% and BMS 8% -- a beta-lactam "
                    "and a polyiodinated aryl -- while Dundee flags 33% and MLSMR "
                    "50%. Rejecting on the latter two would delete aspirin, warfarin "
                    "and gefitinib, so they warn instead."
                ),
                thresholds=tuple(
                    ThresholdField(
                        name=f"{name.lower()}_action",
                        label=label,
                        kind="choice",
                        choices=_ALERT_ACTIONS,
                        help=help_text,
                    )
                    for name, label, help_text in _RD_FILTERS_SETS
                ),
            ),
        ),
    ),
    CriterionSpec(
        id="admet_rules",
        label="ADMET property rules",
        question="Do the published absorption and toxicity-risk rules pass?",
        stage="admet",
        evidence="decision",
        summary=(
            "Fixed equations over molecular properties, published between 2000 and "
            "2008 and in continuous use since. No model, no weights, no download, "
            "and no GPU -- which is what makes them affordable above a neural "
            "ensemble rather than instead of one."
        ),
        options=(
            BackendOption(
                id="medchem_admet_rules",
                label="medchem ADMET rule sets",
                engine="medchem",
                summary=(
                    "Egan's absorption egg, Pfizer's 3/75 toxicity-risk rule, GSK's "
                    "4/400 and the CNS and respiratory sets, from datamol's curated "
                    "implementation."
                ),
                plugin_ref="chemistry.medchem_rules@0.1.0",
                license_spdx="Apache-2.0",
                citation=(
                    "Egan WJ, Merz KM Jr, Baldwin JJ. Prediction of drug absorption "
                    "using multivariate statistics. J Med Chem. 2000;43(21):3867-3877. "
                    "doi:10.1021/jm000292e (Egan); Hughes JD, Blagg J, Price DA, et al. "
                    "Physiochemical drug properties associated with in vivo "
                    "toxicological outcomes. Bioorg Med Chem Lett. 2008;18(17):4872-4875. "
                    "doi:10.1016/j.bmcl.2008.07.071 (Pfizer 3/75); Gleeson MP. Generation "
                    "of a set of simple, interpretable ADMET rules of thumb. J Med Chem. "
                    "2008;51(4):817-834. doi:10.1021/jm701122q (GSK 4/400); implemented "
                    "in medchem, Datamol.io. https://github.com/datamol-io/medchem"
                ),
                requires=("medchem",),
                # Measured on this machine: about 45 molecules/s on one core, which
                # makes this the slowest block in the catalogue -- slower than the
                # alert catalogues and slower than the ChEMBL checker. It earns its
                # place by being the only ADMET evidence that runs without a GPU,
                # a weight file or a trust decision, not by being cheap.
                throughput_per_second=45,
                recommended=True,
                defaults={
                    "schema_version": 1,
                    "rules": "rule_of_egan",
                    "combination": "all",
                },
                thresholds=(
                    ThresholdField(
                        name="rules",
                        label="Rule sets",
                        kind="choice",
                        choices=_MEDCHEM_ADMET_PRESETS,
                        help=(
                            "Which published ADMET rule sets to apply. These are "
                            "property rules, not models: they say a molecule sits "
                            "outside the region where absorbed, well-tolerated drugs "
                            "have historically been found, not what its clearance is."
                        ),
                    ),
                    ThresholdField(
                        name="combination",
                        label="How the rule sets combine",
                        kind="choice",
                        choices=_RULE_COMBINATIONS,
                        help=(
                            "Absorption and toxicity risk are different questions, so "
                            "'any one is enough' is rarely what a project means here."
                        ),
                    ),
                    ThresholdField(
                        name="minimum_passes",
                        label="Rule sets that must pass",
                        kind="integer",
                        minimum=1,
                        maximum=64,
                        step=1,
                        help="Used only when the combination above is 'at least this many'.",
                    ),
                ),
                notes=(
                    "This backend deletes what fails, so only add a rule set you are "
                    "willing to reject on. Egan is a fitted absorption boundary and "
                    "behaves that way; Pfizer 3/75 and GSK 4/400 were published as "
                    "risk trends, and marketed drugs -- CNS drugs especially -- sit "
                    "outside them. Put this above a model, not in place of one."
                ),
            ),
        ),
    ),
    CriterionSpec(
        id="admet_endpoint",
        label="ADMET endpoint",
        question="What does a local model predict for one named ADMET endpoint?",
        stage="admet",
        evidence="prediction",
        summary=(
            "Runs a local model, then applies one explicit numeric window to one "
            "exact endpoint. Missing evidence is an explicit reject, never a pass."
        ),
        options=(
            BackendOption(
                id="admet_ai_v2",
                label="ADMET-AI v2",
                engine="ADMET-AI",
                summary=(
                    "Chemprop-RDKit ensemble covering the TDC ADMET panel. Requires a "
                    "locally provisioned, hash-pinned model tree."
                ),
                plugin_ref="prediction.admet_ai_v2@0.1.0",
                gate_plugin="prediction.numeric_evidence_gate@0.1.0",
                license_spdx="MIT",
                citation=(
                    "Swanson K, Walther P, Leitz J, et al. ADMET-AI: a machine learning "
                    "ADMET platform for evaluation of large-scale chemical libraries. "
                    "Bioinformatics. 2024;40(7):btae416. "
                    "doi:10.1093/bioinformatics/btae416"
                ),
                requires=("admet_ai", "chemprop", "lightning", "torch"),
                # Measured through the adapter at about 7 molecules/s using every
                # core of a 16-core CPU with no GPU: 40 hours per million, which
                # makes this the most expensive block in the catalogue by a factor
                # of four. That is not an argument against it, it is an argument
                # about where it goes -- on a shortlist that cheap tiers have
                # already cut, never near the top. A 4090 moves the number by
                # roughly an order of magnitude; the CPU figure is the one
                # recorded because it is the one that was measured.
                throughput_per_second=7,
                recommended=True,
                defaults={
                    "schema_version": 1,
                    # Resolved against the installed wheel rather than written as
                    # an absolute path, so a cascade designed here still runs on
                    # the GPU node. ADMET-AI v2 ships its checkpoints inside the
                    # distribution, so there is nothing to download.
                    "models_dir": "package:admet_ai/resources/models",
                    # Pinned to admet-ai 2.0.1. A different release fails closed
                    # with a digest mismatch; 'molcascade pins admet-ai' prints
                    # the replacements.
                    "expected_package_code_sha256": (
                        "42ea2147b497042fa36b767a6535bafc90830ac08cc89aad8a16d6de7d33fb24"
                    ),
                    "expected_model_manifest_sha256": (
                        "29d22892a60ee553ed366ecd64a4f08c005049e4b1c19547f740ba80f4a65418"
                    ),
                    "batch_size": 256,
                    "endpoints": [dict(item) for item in _ADMET_AI_ENDPOINT_DEFAULTS],
                    # Deliberately false. See the trust field below.
                    "allow_unsafe_model_deserialization": False,
                },
                gate_defaults={
                    "schema_version": 1,
                    "endpoint_id": "herg_blocking",
                    # 0.95, not the 0.5 an unexamined "it's a probability"
                    # reading suggests, and not a number calibrated on any one
                    # target's actives.
                    #
                    # Measured on the approved-oral-drug panel -- the same
                    # positive control every other default in this file answers
                    # to -- the predicted distribution sits at p25 0.155 /
                    # median 0.518 / p75 0.780.  Half of the drugs already on
                    # the market score above 0.5.  That is not the model being
                    # wrong: it predicts hERG *binding*, and verapamil,
                    # diltiazem and imatinib genuinely bind hERG.  They were
                    # approved anyway because the margin between that affinity
                    # and their exposure is acceptable, which is a comparison
                    # this stage has no access to.
                    #
                    # So the cut is chosen where sensitivity stops paying for
                    # itself.  Against astemizole (0.995), cisapride (0.977),
                    # terfenadine (0.970) and dofetilide (0.956) -- withdrawn or
                    # restricted for QT -- every threshold from 0.80 to 0.95
                    # catches the same four, while the cost to approved drugs
                    # falls from 18/77 to 6/77.  0.80 is strictly dominated:
                    # identical detection, three times the collateral.  Above
                    # 0.95 detection starts to go (0.97 loses terfenadine and
                    # dofetilide), so 0.95 is the knee.
                    #
                    # It is still a filter that deletes imatinib, lapatinib,
                    # gefitinib, donepezil, verapamil and olanzapine, and it
                    # misses grepafloxacin (0.165) which was withdrawn for QT.
                    # A predicted probability is a triage signal; it is not a
                    # cardiac safety assessment, and this default is not one
                    # either.
                    "maximum": 0.95,
                    "semantics_label": (
                        "Predicted probability that the molecule blocks hERG; higher is worse."
                    ),
                },
                thresholds=(
                    ThresholdField(
                        name="allow_unsafe_model_deserialization",
                        label="I accept that loading these checkpoints runs code",
                        kind="boolean",
                        help=(
                            "Torch checkpoints are executable, so the digests above prove "
                            "provenance but not safety. The stage refuses to start until "
                            "this is ticked."
                        ),
                    ),
                    ThresholdField(
                        name="endpoint_id",
                        label="Endpoint to gate on",
                        kind="choice",
                        target="gate",
                        choices=_ADMET_AI_GATE_CHOICES,
                        help=(
                            "One forward pass predicts the whole panel; this chooses which "
                            "of those numbers decides. The score beside each name is what "
                            "ADMET-AI's authors measured on held-out data."
                        ),
                    ),
                    *_window(
                        lower=-1e9,
                        upper=1e9,
                        step=0.05,
                        help_text=(
                            "Inclusive window on the predicted value. Classifier heads "
                            "emit a probability in 0-1; regression heads emit the unit "
                            "named beside the endpoint."
                        ),
                    ),
                    _BATCH,
                ),
                notes=(
                    "Deserializing a Torch checkpoint executes code. The adapter "
                    "requires package and model digests plus an explicit trust flag, "
                    "and the pins ship set to admet-ai 2.0.1 as installed. Upgrade the "
                    "package deliberately, then run 'molcascade pins admet-ai' and "
                    "paste the two digests back in.\n\n"
                    "Cost: about 7 molecules/s on CPU, roughly 40 hours per million. "
                    "Put this below the cheap structural tiers, not above them.\n\n"
                    "Two of the panel's heads -- volume of distribution and half-life -- "
                    "score a negative R² on their own held-out sets, meaning they do "
                    "worse than predicting the training mean. They are still recorded "
                    "so the numbers are in the artifact, but they are deliberately "
                    "absent from the endpoint list: nothing should be deleted on their "
                    "say-so.\n\n"
                    "The hERG cut ships at 0.8, not 0.5. Across 608 published kinase "
                    "binders the predicted median is 0.70, so 0.5 removes about 69% of "
                    "them -- that is the chemotype being cut, not the liability. 0.8 "
                    "still rejects astemizole, cisapride, terfenadine and dofetilide. "
                    "Tighten it if your series can afford it, but check what a change "
                    "costs on your own molecules first."
                ),
            ),
            BackendOption(
                id="chemprop_checkpoint",
                label="Chemprop 2.x project checkpoint",
                engine="Chemprop",
                summary="A message-passing model trained on your own endpoint data.",
                license_spdx="MIT",
                citation=(
                    "Heid E, Greenman KP, Chung Y, et al. Chemprop: a machine learning "
                    "package for chemical property prediction. J Chem Inf Model. "
                    "2024;64(1):9-17. doi:10.1021/acs.jcim.3c01250; Yang K, Swanson K, Jin "
                    "W, et al. Analyzing learned molecular representations for property "
                    "prediction. J Chem Inf Model. 2019;59(8):3370-3388. "
                    "doi:10.1021/acs.jcim.9b00237"
                ),
                plugin_ref="prediction.chemprop_checkpoint@0.1.0",
                gate_plugin="prediction.numeric_evidence_gate@0.1.0",
                requires=("chemprop", "torch"),
                defaults={
                    "schema_version": 1,
                    "bundle_dir": "",
                    "expected_bundle_sha256": "",
                    "batch_size": 256,
                    # Deliberately false. See the trust field below.
                    "allow_unsafe_model_deserialization": False,
                },
                gate_defaults={
                    "schema_version": 1,
                    "semantics_label": "Project model; review its training domain.",
                },
                thresholds=(
                    ThresholdField(
                        name="bundle_dir",
                        label="Checkpoint bundle directory",
                        kind="text",
                        help=(
                            "Absolute path to the directory containing "
                            "molcascade_chemprop.yaml and the saved checkpoint."
                        ),
                    ),
                    ThresholdField(
                        name="expected_bundle_sha256",
                        label="Bundle digest",
                        kind="text",
                        help=(
                            "Run `molcascade model-bundle <dir>` and paste bundle_sha256. "
                            "It pins the run to exactly these model bytes."
                        ),
                    ),
                    ThresholdField(
                        name="allow_unsafe_model_deserialization",
                        label="I accept that loading this checkpoint runs code",
                        kind="boolean",
                        help=(
                            "A Chemprop checkpoint is a Torch checkpoint, so the digest "
                            "above proves provenance but not safety. The stage refuses "
                            "to start until this is ticked."
                        ),
                    ),
                    ThresholdField(
                        name="endpoint_id",
                        label="Endpoint",
                        kind="text",
                        target="gate",
                        help="The endpoint_id declared in the bundle manifest.",
                    ),
                    *_window(
                        lower=-1e9,
                        upper=1e9,
                        step=0.1,
                        help_text=(
                            "Inclusive window on the predicted value. A classification "
                            "head emits a probability in 0-1; a regression head emits "
                            "the unit recorded in the manifest."
                        ),
                    ),
                    _BATCH,
                ),
                notes=(
                    "For an endpoint you have your own measurements for. Train with "
                    "Chemprop's own CLI, then place the saved checkpoint and a "
                    "molcascade_chemprop.yaml manifest in one directory and run "
                    "'molcascade model-bundle <dir>' for the digest.\n\n"
                    "This is the complement to the ONNX bundle rather than a "
                    "competitor: a message-passing network learns its own "
                    "featurization from the molecular graph, which an ONNX bundle -- "
                    "fed a fixed-width vector MolCascade computes -- cannot express. "
                    "The price is that loading it executes code, so it needs the trust "
                    "flag that the ONNX path does not.\n\n"
                    "A multitask checkpoint contributes one endpoint here. Declare "
                    "n_tasks and task_index in the manifest so the column that decides "
                    "is chosen deliberately and a reshaped checkpoint fails closed "
                    "instead of scoring on a different endpoint."
                ),
            ),
            BackendOption(
                id="openadmet",
                label="OpenADMET models",
                engine="OpenADMET",
                summary=(
                    "Open ADMET model collection with per-prediction uncertainty when "
                    "run as an ensemble."
                ),
                license_spdx="MIT",
                citation=(
                    "OpenADMET. https://openadmet.org — an open data and model consortium; "
                    "cite the specific released model actually used, not the project."
                ),
                requires=("openadmet_models",),
                notes="Reviewed research option; no MolCascade adapter is registered yet.",
            ),
        ),
    ),
    CriterionSpec(
        id="synthesizability",
        label="Synthesizability score",
        question="How hard does a fast proxy think this molecule is to make?",
        stage="synthesis",
        evidence="synthesis_score",
        summary=(
            "A score is a fast proxy for how hard something looks to make. It is not a "
            "route and it is not a step count. Two scores that disagree are more "
            "informative than one score you trust."
        ),
        options=(
            BackendOption(
                id="rdkit_sa_score",
                label="SA Score (Ertl and Schuffenhauer)",
                engine="RDKit",
                summary=(
                    "The standard 1-10 synthetic accessibility score; higher means "
                    "harder. Still the strongest cheap default."
                ),
                plugin_ref="synthesis.rdkit_sa_score@0.1.0",
                gate_plugin="synthesis.numeric_evidence_gate@0.1.0",
                license_spdx="BSD-3-Clause",
                citation=(
                    "Ertl P, Schuffenhauer A. Estimation of synthetic accessibility score "
                    "of drug-like molecules based on molecular complexity and fragment "
                    "contributions. J Cheminform. 2009;1:8. doi:10.1186/1758-2946-1-8"
                ),
                # Measured through the adapter at about 248 molecules/s. Cheap
                # enough to sit above route search, which is the entire reason a
                # fragment-contribution proxy is still worth having.
                throughput_per_second=248,
                recommended=True,
                gate_defaults={
                    "schema_version": 1,
                    "expected_direction": "HIGHER_HARDER",
                    "maximum": 6.0,
                },
                thresholds=(
                    ThresholdField(
                        name="maximum",
                        label="Maximum SA score",
                        target="gate",
                        minimum=1,
                        maximum=10,
                        step=0.5,
                        nullable=True,
                        help="Higher scores are harder to make. 6 is a common triage line.",
                    ),
                    ThresholdField(
                        name="minimum",
                        label="Minimum SA score",
                        target="gate",
                        minimum=1,
                        maximum=10,
                        step=0.5,
                        nullable=True,
                        help="Rarely used; keeps out trivially simple fragments.",
                    ),
                ),
            ),
            BackendOption(
                id="scscore",
                label="SCScore (Coley, reaction-trained)",
                engine="MolCascade + numpy",
                summary=(
                    "A 1-5 complexity score learned from 12 million Reaxys reactions, so "
                    "it reflects how molecules are actually built rather than how rare "
                    "their fragments are. Runs locally in numpy with no extra Python "
                    "package and no GPU."
                ),
                plugin_ref="synthesis.scscore@0.1.0",
                gate_plugin="synthesis.numeric_evidence_gate@0.1.0",
                license_spdx="MIT",
                citation=(
                    "Coley CW, Rogers L, Green WH, Jensen KF. SCScore: synthetic "
                    "complexity learned from a reaction corpus. J Chem Inf Model. "
                    "2018;58(2):252-261. doi:10.1021/acs.jcim.7b00622"
                ),
                requires_assets=("scscore",),
                defaults={
                    "schema_version": 1,
                    "weights_path": (
                        "asset:scscore/models/full_reaxys_model_1024bool/"
                        "model.ckpt-10654.as_numpy.json.gz"
                    ),
                    "expected_weights_sha256": "",
                    "fingerprint": "bits",
                },
                gate_defaults={
                    "schema_version": 1,
                    "expected_direction": "HIGHER_HARDER",
                    "maximum": 3.5,
                },
                notes=(
                    "Run 'molcascade assets fetch scscore' once. The weights are "
                    "verified against a pinned digest before every run and the stage "
                    "refuses to start if they do not match; MolCascade never downloads "
                    "anything during a screen.\n\n"
                    "The file is a gzipped JSON array of matrices, evaluated in numpy. "
                    "Nothing in it is executed and no TensorFlow install is needed."
                ),
                thresholds=(
                    ThresholdField(
                        name="weights_path",
                        label="Weight file",
                        kind="text",
                        help=(
                            "Leave as the 'asset:' reference to use the fetched weights. "
                            "Replace with an absolute path only to point at your own copy."
                        ),
                    ),
                    ThresholdField(
                        name="expected_weights_sha256",
                        label="Weight file digest",
                        kind="text",
                        help=(
                            "Only needed for a file given by absolute path; an 'asset:' "
                            "reference already carries its own pinned digest. Paste the "
                            "output of `sha256sum`."
                        ),
                    ),
                    ThresholdField(
                        name="fingerprint",
                        label="Model variant",
                        kind="choice",
                        choices=(
                            ("bits", "Binary fingerprint (*_1024bool, *_2048bool)"),
                            ("counts", "Folded counts (*_1024uint8)"),
                        ),
                        help=(
                            "Must match the weight file. The width is read from the "
                            "weights, but bits and counts cannot be told apart."
                        ),
                    ),
                    ThresholdField(
                        name="maximum",
                        label="Maximum SC score",
                        target="gate",
                        minimum=1,
                        maximum=5,
                        step=0.1,
                        nullable=True,
                        help=(
                            "Higher means more complex. Around 3.5 separates routine "
                            "medicinal chemistry from multi-step targets."
                        ),
                    ),
                    ThresholdField(
                        name="minimum",
                        label="Minimum SC score",
                        target="gate",
                        minimum=1,
                        maximum=5,
                        step=0.1,
                        nullable=True,
                        help="Rarely used; keeps out trivially simple fragments.",
                    ),
                ),
            ),
            BackendOption(
                id="rascore",
                label="RAscore retrosynthetic accessibility",
                engine="RAscore",
                summary="Classifier trained on AiZynthFinder solvability of ChEMBL molecules.",
                license_spdx="MIT",
                citation=(
                    "Thakkar A, Chadimová V, Bjerrum EJ, Engkvist O, Reymond JL. "
                    "Retrosynthetic accessibility score (RAscore). Chem Sci. "
                    "2021;12:3339-3349. doi:10.1039/D0SC05401A"
                ),
                requires=("RAscore",),
                notes="Reviewed research option; no MolCascade adapter is registered yet.",
            ),
            BackendOption(
                id="syba",
                label="SYBA Bayesian classifier",
                engine="SYBA",
                summary="Fragment-based Bayesian easy/hard classifier.",
                license_spdx="MIT",
                citation=(
                    "Voršilák M, Kolář M, Čmelo I, Svozil D. SYBA: Bayesian estimation of "
                    "synthetic accessibility of organic compounds. J Cheminform. "
                    "2020;12:35. doi:10.1186/s13321-020-00439-2"
                ),
                requires=("syba",),
                notes="Reviewed research option; no MolCascade adapter is registered yet.",
            ),
        ),
    ),
    CriterionSpec(
        id="retrosynthesis_routes",
        label="Retrosynthesis route search",
        question="Can a route be found to purchasable stock, and in how many steps?",
        stage="synthesis",
        evidence="synthesis_score",
        summary=(
            "The only component that produces genuine step counts. Route search is "
            "orders of magnitude slower than a score and belongs late in the funnel."
        ),
        options=(
            BackendOption(
                id="aizynthfinder",
                label="AiZynthFinder 4",
                engine="AiZynthFinder",
                summary="Monte-Carlo tree search over template policies against a stock file.",
                license_spdx="MIT",
                citation=(
                    "Genheden S, Thakkar A, Chadimová V, Reymond JL, Engkvist O, Bjerrum "
                    "E. AiZynthFinder: a fast, robust and flexible open-source software "
                    "for retrosynthetic planning. J Cheminform. 2020;12:70. "
                    "doi:10.1186/s13321-020-00472-1"
                ),
                plugin_ref="synthesis.aizynthfinder@0.1.0",
                gate_plugin="synthesis.numeric_evidence_gate@0.1.0",
                # Deliberately empty. Every other option lists the packages it
                # imports, and this one imports nothing: aizynthfinder must *not*
                # be installed in this environment, so probing for it here would
                # report the healthy case as broken and the broken case as fine.
                # What has to exist is an executable somewhere else, and only the
                # stage's own settings know where.
                requires=(),
                defaults={
                    "schema_version": 1,
                    "executable": "",
                    "config_path": "",
                    "max_molecules": 1000,
                    "timeout_per_molecule_seconds": 600.0,
                    "unsolved_score": 99.0,
                },
                gate_defaults={
                    "schema_version": 1,
                    "expected_direction": "HIGHER_HARDER",
                    "maximum": 6.0,
                },
                notes=(
                    "Runs in its own environment, which you create once and "
                    "MolCascade never touches. Measured, not assumed: resolving "
                    "aizynthfinder 4.4.1 into this environment would downgrade RDKit "
                    "from 2026.03 to 2023.09 and NumPy from 2.x to 1.26, which "
                    "changes every descriptor, every policy digest and every ADMET "
                    "checkpoint in the cascade -- and pulls in Jupyter, dask, "
                    "metaflow and boto3 besides. So the adapter talks to aizynthcli "
                    "across a process boundary instead.\n\n"
                    "Set it up once:\n"
                    "  conda create -n aizynth \"python>=3.10,<3.13\"\n"
                    "  conda run -n aizynth python -m pip install aizynthfinder\n"
                    "  conda run -n aizynth download_public_data <folder>\n"
                    "Then paste the path that 'conda run -n aizynth which aizynthcli' "
                    "prints, and the path to the config.yml that download_public_data "
                    "wrote. MolCascade fetches neither the package nor the policy and "
                    "stock files; the run stops before the first molecule if either "
                    "path is wrong.\n\n"
                    "Cost argues for placing it last: tree search is seconds to "
                    "minutes per molecule, so it belongs on the few hundred or "
                    "thousand that survive everything else, never on the library. "
                    "The stage refuses more than 'max_molecules' inputs for that "
                    "reason.\n\n"
                    "The score is the number of reactions in the top-scored route. A "
                    "target with no route to your stock cannot be left blank -- the "
                    "column is not nullable -- so it is recorded as the unsolved "
                    "sentinel, far outside any real step count, and flagged "
                    "ROUTE_NOT_FOUND so the threshold below rejects it rather than "
                    "ranking it."
                ),
                thresholds=(
                    ThresholdField(
                        name="executable",
                        label="aizynthcli path",
                        kind="text",
                        environment_variable="MOLCASCADE_AIZYNTHFINDER_EXECUTABLE",
                        help=(
                            "Absolute path to aizynthcli inside the environment you "
                            "made for it. Not a bare command name: that environment is "
                            "deliberately not on this one's PATH."
                        ),
                    ),
                    ThresholdField(
                        name="config_path",
                        label="AiZynthFinder config file",
                        kind="text",
                        environment_variable="MOLCASCADE_AIZYNTHFINDER_CONFIG_PATH",
                        help=(
                            "Absolute path to the YAML naming your expansion policy "
                            "and stock. Its bytes are hashed into the method identity, "
                            "so changing the policy changes the recorded method."
                        ),
                    ),
                    ThresholdField(
                        name="max_molecules",
                        label="Most molecules to search",
                        kind="number",
                        minimum=1,
                        maximum=100_000,
                        step=100,
                        help=(
                            "The stage stops before the first search if the tier hands "
                            "it more than this. Raise it only once you have timed one "
                            "molecule and multiplied."
                        ),
                    ),
                    ThresholdField(
                        name="timeout_per_molecule_seconds",
                        label="Time budget per molecule",
                        kind="number",
                        unit="s",
                        minimum=1,
                        maximum=86_400,
                        step=30,
                        help=(
                            "Multiplied by the molecules each worker sees to cap the "
                            "whole call. A deadlock guard on top of the search budget "
                            "in your AiZynthFinder config, not a replacement for it."
                        ),
                    ),
                    ThresholdField(
                        name="unsolved_score",
                        label="Score for an unsolved target",
                        kind="number",
                        minimum=0,
                        maximum=1_000_000,
                        step=1,
                        help=(
                            "Recorded when no route to stock is found. Keep it above "
                            "the step limit below so unsolved molecules are rejected."
                        ),
                    ),
                    ThresholdField(
                        name="maximum",
                        label="Most steps allowed",
                        kind="number",
                        target="gate",
                        minimum=1,
                        maximum=20,
                        step=1,
                        nullable=True,
                        help=(
                            "Molecules whose shortest found route is longer than this "
                            "are dropped, as are those with no route at all."
                        ),
                    ),
                ),
            ),
            BackendOption(
                id="syntheseus",
                label="Syntheseus",
                engine="Syntheseus",
                summary="Unified search interface over several single-step retro models.",
                license_spdx="MIT",
                citation=(
                    "Maziarz K, Tripp A, Liu G, et al. Re-evaluating retrosynthesis "
                    "algorithms with Syntheseus. arXiv:2310.19796 (2023). "
                    "https://github.com/microsoft/syntheseus"
                ),
                requires=("syntheseus",),
                notes="Reviewed research option; no MolCascade adapter is registered yet.",
            ),
        ),
    ),
    CriterionSpec(
        id="ligand_conformers",
        label="3D conformer generation",
        question="What geometry do the docking engines start from?",
        stage="ligand_prep",
        evidence="ligand_conformer",
        filters=False,
        summary=(
            "Embeds one 3D structure per molecule and keeps it as evidence. It "
            "rejects nothing; molecules RDKit cannot embed are recorded as such "
            "and are simply absent from the engines' input below."
        ),
        options=(
            BackendOption(
                id="rdkit_etkdgv3",
                label="RDKit ETKDGv3",
                engine="RDKit",
                summary=(
                    "Distance geometry with experimental torsion preferences, "
                    "optionally relaxed with MMFF94s."
                ),
                plugin_ref="docking.rdkit_conformers@0.1.0",
                license_spdx="BSD-3-Clause",
                citation=(
                    "Wang S, Witek J, Landrum GA, Riniker S. Improving conformer "
                    "generation for small rings and macrocycles based on distance "
                    "geometry and experimental torsional-angle preferences. J Chem "
                    "Inf Model. 2020;60(4):2044-2058. doi:10.1021/acs.jcim.0c00025"
                ),
                requires=("rdkit",),
                recommended=True,
                # Measured on this machine over twenty drug-like molecules:
                # 10.5/s embedding alone, 5.3/s with MMFF94s on top. The
                # minimisation is worth roughly half the tier's cost, which is
                # why it is a switch rather than an assumption.
                throughput_per_second=5,
                defaults={
                    "schema_version": 1,
                    "conformers_per_molecule": 1,
                    "embed_attempts": 3,
                    "minimize": True,
                    "max_minimize_iterations": 500,
                    "prune_rms_threshold": 0.5,
                },
                notes=(
                    "Put this above the docking tier and nowhere else. It exists so "
                    "that a consensus between two engines is a consensus about one "
                    "molecule: each engine reads the same conformer table, joined on "
                    "parent_id, instead of embedding its own.\n\n"
                    "The seed is part of the recorded method and each molecule's own "
                    "seed is derived from its identity, not from its row, so the same "
                    "library embeds to the same coordinates however the work was "
                    "split across workers or resumed after an interruption.\n\n"
                    "Vina-family search re-generates torsions itself, so one conformer "
                    "is the honest default; more of them costs linearly and buys little "
                    "unless a pose predictor downstream is treating the input geometry "
                    "as the answer."
                ),
                thresholds=(
                    ThresholdField(
                        name="conformers_per_molecule",
                        label="Conformers per molecule",
                        kind="integer",
                        minimum=1,
                        maximum=64,
                        step=1,
                        help=(
                            "Kept per molecule. Above one, the engines below still dock "
                            "the conformer named in their own settings; the rest are "
                            "evidence."
                        ),
                    ),
                    ThresholdField(
                        name="minimize",
                        label="Relax with MMFF94s",
                        kind="boolean",
                        help=(
                            "Roughly halves throughput -- 10/s becomes 5/s on one core, "
                            "measured. Leave it on unless the tier is the bottleneck."
                        ),
                    ),
                    ThresholdField(
                        name="embed_attempts",
                        label="Embedding attempts",
                        kind="integer",
                        minimum=1,
                        maximum=20,
                        step=1,
                        help=(
                            "ETKDG occasionally needs several tries for macrocycles and "
                            "heavily bridged systems before it finds a valid geometry."
                        ),
                    ),
                    ThresholdField(
                        name="prune_rms_threshold",
                        label="Prune below RMSD",
                        kind="number",
                        unit="Å",
                        minimum=0.0,
                        maximum=5.0,
                        step=0.1,
                        help=(
                            "Discards a conformer closer than this to one already kept. "
                            "Ignored when only one conformer is requested."
                        ),
                    ),
                    ThresholdField(
                        name="seed",
                        label="Random seed",
                        kind="integer",
                        minimum=0,
                        maximum=2_147_483_647,
                        step=1,
                        help=(
                            "Changing it changes every coordinate, so it is part of the "
                            "recorded method rather than a run-time convenience."
                        ),
                    ),
                    _BATCH,
                ),
            ),
        ),
    ),
    CriterionSpec(
        id="docking_score",
        label="Docking score",
        question="Does this molecule fit the binding site, and how well?",
        stage="docking",
        evidence="docking_score",
        summary=(
            "Docks against a receptor named at run time with '--receptor', not in "
            "this file: the cascade is the screening policy and is meant to be "
            "reused, while the protein is what changes between campaigns. Needs a "
            "3D conformer tier above it."
        ),
        options=(
            BackendOption(
                id="unidock",
                label="Uni-Dock",
                engine="Uni-Dock",
                summary=(
                    "AutoDock Vina's search as a CUDA kernel: the same scoring "
                    "functions and the same box, thousands of ligands resident on the "
                    "card at once."
                ),
                plugin_ref="docking.unidock@0.2.0",
                gate_plugin="docking.numeric_evidence_gate@0.1.0",
                license_spdx="Apache-2.0",
                citation=(
                    "Yu Y, Cai C, Wang J, Bo Z, Zhu Z, Zheng H. Uni-Dock: GPU-accelerated "
                    "docking enables ultralarge virtual screening. J Chem Theory Comput. "
                    "2023;19(11):3336-3345. doi:10.1021/acs.jctc.2c01145"
                ),
                # meeko is the only piece that installs into this environment; the
                # engine itself is an executable path, which is why it is not
                # probed as an import here.
                requires=("meeko",),
                recommended=True,
                # Upstream reports ~0.1 s per ligand on one card at
                # 'search_mode balance'. Per GPU lane, not per core.
                throughput_per_second=10,
                defaults={
                    "schema_version": 1,
                    "executable": "",
                    "scoring": "vina",
                    "search_mode": "balance",
                    "num_modes": 1,
                    "conformer_index": 0,
                    "keep_poses": True,
                    "max_molecules": 200_000,
                },
                gate_defaults={
                    "schema_version": 1,
                    "engine_id": "unidock",
                    "expected_score_kind": "VINA_KCAL_MOL",
                    "expected_direction": "LOWER_STRONGER",
                    "maximum": -9.5,
                },
                notes=(
                    "Linux and an NVIDIA card only: it has no CPU path and takes no "
                    "device flag, so the card it uses is whichever one MolCascade "
                    "leaves visible to the shard. Set 'executable' to the absolute "
                    "path of the 'unidock' binary; MolCascade does not build or "
                    "install it.\n\n"
                    "There are no weights to fetch. Vina's scoring functions are "
                    "empirical physics, so nothing about this engine is downloadable "
                    "and nothing about it is a model version.\n\n"
                    "You supply a receptor PDB and MolCascade derives the PDBQT the "
                    "engine reads, once, with meeko, before the first molecule. That "
                    "is the fragile step -- protonation and unusual residues are where "
                    "it fails -- and the run stops there rather than three tiers in. "
                    "Pass '--receptor-pdbqt' with your own preparation when it cannot "
                    "cope.\n\n"
                    "A Vina score is a ranking, not an affinity. The default of -9.5 "
                    "kcal/mol is not the conventional -7: this tier sits below seven "
                    "cheaper ones, so it is handed molecules that already look like "
                    "drugs, and -7 then keeps most of what it is given -- measured at "
                    "88% on a real library, which is a tier that costs GPU hours and "
                    "decides almost nothing. Neither number is a measurement, and the "
                    "published correlation with experiment is weak enough that this "
                    "belongs at the bottom of a funnel rather than at the top."
                ),
                thresholds=(
                    ThresholdField(
                        name="executable",
                        label="unidock path",
                        kind="text",
                        environment_variable="MOLCASCADE_UNIDOCK_EXECUTABLE",
                        help=(
                            "Absolute path to the binary. Not a bare command name: this "
                            "is a tool you installed, and the run records which one."
                        ),
                    ),
                    ThresholdField(
                        name="scoring",
                        label="Scoring function",
                        kind="choice",
                        choices=(
                            ("vina", "Vina (empirical, kcal/mol)"),
                            ("vinardo", "Vinardo (reparameterised Vina)"),
                            ("ad4", "AutoDock4 (force-field)"),
                        ),
                        help=(
                            "Three different scales, not three settings of one. Change "
                            "this and change the gate's expected scale to match, or the "
                            "run stops rather than comparing kcal/mol against something "
                            "else."
                        ),
                    ),
                    ThresholdField(
                        name="search_mode",
                        label="Search effort",
                        kind="choice",
                        choices=(
                            ("fast", "Fast"),
                            ("balance", "Balanced"),
                            ("detail", "Detailed"),
                        ),
                        help=(
                            "Uni-Dock's own presets for exhaustiveness and step count. "
                            "'detail' is several times the cost of 'balance'."
                        ),
                    ),
                    ThresholdField(
                        name="num_modes",
                        label="Poses kept per ligand",
                        kind="integer",
                        minimum=1,
                        maximum=20,
                        step=1,
                        help=(
                            "One row of evidence per pose. The threshold below reads the "
                            "best of them; the rest are there to be looked at."
                        ),
                    ),
                    ThresholdField(
                        name="keep_poses",
                        label="Store the docked pose",
                        kind="boolean",
                        help=(
                            "On by default, because 'molcascade trace' writes the kept "
                            "poses as SDF and cannot recover them later. Turn it off for "
                            "a scores-only screen: 45,000 poses is a few hundred "
                            "megabytes of molblock."
                        ),
                    ),
                    ThresholdField(
                        name="max_molecules",
                        label="Most molecules to dock",
                        kind="integer",
                        minimum=1,
                        maximum=10_000_000,
                        step=1_000,
                        help=(
                            "The stage stops before the first ligand if the tier hands it "
                            "more than this. It is the guard against pointing docking at "
                            "a whole library by accident."
                        ),
                    ),
                    ThresholdField(
                        name="expected_score_kind",
                        label="Score scale the gate expects",
                        kind="choice",
                        target="gate",
                        choices=(
                            ("VINA_KCAL_MOL", "Vina (kcal/mol)"),
                            ("VINARDO_KCAL_MOL", "Vinardo (kcal/mol)"),
                            ("AD4_KCAL_MOL", "AutoDock4 (kcal/mol)"),
                        ),
                        help=(
                            "Must match the scoring function above. The gate refuses to "
                            "run on evidence from a different scale instead of quietly "
                            "applying the wrong window to it."
                        ),
                    ),
                    *_window(
                        unit="kcal/mol",
                        minimum_label="Strongest score allowed",
                        maximum_label="Weakest score allowed",
                        lower=-30.0,
                        upper=10.0,
                        step=0.1,
                        help_text=(
                            "Inclusive, and lower is stronger. The upper bound is the "
                            "filter; a lower bound is worth setting only to catch scores "
                            "too good to be real."
                        ),
                    ),
                ),
            ),
            BackendOption(
                id="karmadock",
                label="KarmaDock",
                engine="KarmaDock",
                summary=(
                    "Predicts a pose in one forward pass instead of searching for one, "
                    "which is why it is about fifty times faster than a Vina-family "
                    "engine."
                ),
                plugin_ref="docking.karmadock@0.3.0",
                gate_plugin="docking.numeric_evidence_gate@0.1.0",
                license_spdx="Apache-2.0",
                citation=(
                    "Zhang X, Zhang O, Shen C, et al. Efficient and accurate large "
                    "library ligand docking with KarmaDock. Nat Comput Sci. "
                    "2023;3:789-804. doi:10.1038/s43588-023-00511-5"
                ),
                # Nothing to import: it runs under its own interpreter, and
                # probing this environment for it would report the healthy case
                # as broken.
                requires=(),
                # Upstream reports ~0.02 s per ligand on one card.
                throughput_per_second=50,
                defaults={
                    "schema_version": 1,
                    "executable": "",
                    "repo_path": "",
                    "engine_batch_size": 64,
                    "keep_poses": True,
                    "max_molecules": 200_000,
                },
                gate_defaults={
                    "schema_version": 1,
                    "engine_id": "karmadock",
                    "expected_score_kind": "KARMADOCK_MDN",
                    "expected_direction": "HIGHER_STRONGER",
                    "minimum": 45.0,
                },
                notes=(
                    "Runs in an environment of its own, which you create once and "
                    "MolCascade never touches. Measured, not assumed: KarmaDock pins "
                    "rdkit 2022.09.1 against this project's rdkit 2024.9 or newer, and "
                    "resolving it into this environment would change every descriptor "
                    "and every identity in the cascade above it. So 'executable' is the "
                    "python inside its environment, and 'repo_path' is the checkout it "
                    "runs from.\n\n"
                    "Its checkpoint is not something MolCascade can fetch for you: "
                    "utils/virtual_screening.py loads the weights from a fixed path "
                    "inside its own checkout with no flag to redirect it, so a copy "
                    "downloaded anywhere else would never be read. Download them into "
                    "the checkout as KarmaDock documents (Zenodo record 7789066) and "
                    "set 'weights_path' to the file it loads -- that is what puts the "
                    "model version into the recorded method rather than leaving two "
                    "checkpoints indistinguishable.\n\n"
                    "It takes no box. The site comes from the reference ligand's atoms, "
                    "which is why '--reference-ligand' is the pocket definition this "
                    "engine can actually use, and why a cascade holding it and Uni-Dock "
                    "together needs the ligand rather than six numbers alone.\n\n"
                    "The score is a unitless ranking from a mixture-density network, "
                    "higher being stronger. It is not kcal/mol and does not convert to "
                    "one; the default of 45 is a starting point to calibrate against "
                    "your own known actives, not a published cut. It is paired with "
                    "Uni-Dock's -9.5 so that the two engines reject comparable shares "
                    "of a tier: at 50 this one did nearly all of the filtering, which "
                    "makes a two-engine tier an expensive way to ask one engine."
                ),
                thresholds=(
                    ThresholdField(
                        name="executable",
                        label="KarmaDock python path",
                        kind="text",
                        environment_variable="MOLCASCADE_KARMADOCK_EXECUTABLE",
                        help=(
                            "Absolute path to the python inside KarmaDock's own "
                            "environment. Deliberately not this one's."
                        ),
                    ),
                    ThresholdField(
                        name="repo_path",
                        label="KarmaDock checkout",
                        kind="text",
                        environment_variable="MOLCASCADE_KARMADOCK_REPO_PATH",
                        help=(
                            "Absolute path to the checkout. Its utils/virtual_screening.py "
                            "is the entry point and its trained_models/ is where the "
                            "weights have to sit; the script has no flag for either."
                        ),
                    ),
                    ThresholdField(
                        name="weights_path",
                        label="Checkpoint file",
                        kind="text",
                        nullable=True,
                        help=(
                            "The checkpoint inside the checkout that KarmaDock will load. "
                            "Recorded for identity only -- naming it is what distinguishes "
                            "two model versions in the run's provenance."
                        ),
                    ),
                    ThresholdField(
                        name="pocket_pdb_path",
                        label="Pocket PDB override",
                        kind="text",
                        nullable=True,
                        help=(
                            "Optional. Set this when the geometric selection picks up the "
                            "wrong chain or misses a residue that matters; leave it empty "
                            "and KarmaDock derives its own from the reference ligand."
                        ),
                    ),
                    ThresholdField(
                        name="engine_batch_size",
                        label="Ligands per forward pass",
                        kind="integer",
                        minimum=1,
                        maximum=4_096,
                        step=16,
                        help=(
                            "GPU memory, not search effort. Lower it if the card runs out; "
                            "raising it past what fits buys nothing."
                        ),
                    ),
                    ThresholdField(
                        name="keep_poses",
                        label="Store the repaired pose",
                        kind="boolean",
                        help=(
                            "On by default, as everywhere in this tier: 'molcascade "
                            "trace' exports these as SDF, and a repaired pose that was "
                            "discarded cannot be rebuilt from the score."
                        ),
                    ),
                    ThresholdField(
                        name="max_molecules",
                        label="Most molecules to dock",
                        kind="integer",
                        minimum=1,
                        maximum=10_000_000,
                        step=1_000,
                        help=(
                            "The stage stops before the first ligand if the tier hands it "
                            "more than this."
                        ),
                    ),
                    *_window(
                        minimum_label="Lowest score allowed",
                        maximum_label="Highest score allowed",
                        lower=0.0,
                        upper=1_000.0,
                        step=1.0,
                        help_text=(
                            "Inclusive, and higher is stronger. Unitless: calibrate it on "
                            "molecules you already know the answer for before trusting it "
                            "to reject anything."
                        ),
                    ),
                ),
            ),
        ),
    ),
    CriterionSpec(
        id="docking_cnn_rescore",
        label="CNN pose rescoring",
        question="Does a convolutional network agree that this pose is a binder?",
        stage="docking_rescore",
        evidence="docking_score",
        summary=(
            "A second, independent opinion on the poses a cheaper engine already "
            "kept. Seconds per ligand, so its position in the funnel is the whole "
            "question of whether it is affordable."
        ),
        options=(
            BackendOption(
                id="gnina",
                label="GNINA 1.3",
                engine="GNINA",
                summary=(
                    "Vina-family search followed by a 3D convolutional network scoring "
                    "the resulting poses."
                ),
                plugin_ref="docking.gnina@0.3.0",
                gate_plugin="docking.numeric_evidence_gate@0.1.0",
                license_spdx="GPL-2.0-or-later",
                citation=(
                    "McNutt AT, Li Y, Meli R, Aggarwal R, Koes DR. GNINA 1.3: the next "
                    "increment in molecular docking with deep learning. J Cheminform. "
                    "2025;17:28. doi:10.1186/s13321-025-00973-x"
                ),
                requires=(),
                requires_license_optin=True,
                # Deliberately not recommended and off in every shipped cascade:
                # it is the one option here whose default position is "nowhere".
                recommended=False,
                # Upstream reports 2-10 s per ligand on one card with
                # '--cnn_scoring rescore'. Fractional per second, and that is the
                # point: at 0.3/s a million molecules is over a hundred GPU-days.
                throughput_per_second=0.3,
                defaults={
                    "schema_version": 1,
                    "executable": "",
                    "cnn_scoring": "rescore",
                    "exhaustiveness": 8,
                    "rank_by": "cnn_affinity",
                    "num_modes": 1,
                    "conformer_index": 0,
                    "keep_poses": True,
                    "allow_cpu": False,
                    # Two orders of magnitude below the other two engines, for the
                    # same reason the throughput figure is: this is a shortlist
                    # tool, and the cap is where that gets enforced rather than
                    # merely recommended.
                    "max_molecules": 20_000,
                },
                gate_defaults={
                    "schema_version": 1,
                    "engine_id": "gnina",
                    "expected_score_kind": "CNN_AFFINITY",
                    "expected_direction": "HIGHER_STRONGER",
                    "minimum": 6.0,
                },
                notes=(
                    "Blocked until you say otherwise. GNINA is GPL-2.0 -- only because "
                    "it links Open Babel, but the terms are the terms -- so a run has "
                    "to permit copyleft backends with '--allow-copyleft' before it will "
                    "start, and the run records that the permission was given so a "
                    "methods section can say so. MolCascade cannot make that decision "
                    "for an organisation, so it defaults to the reversible answer.\n\n"
                    "Set 'executable' to the absolute path of the 'gnina' binary. "
                    "Linux only. Its CNN weights are compiled into the binary, so there "
                    "is nothing to download and nothing to pin -- which also means the "
                    "binary's version *is* the model version.\n\n"
                    "Position is everything here. At two to ten seconds a ligand, "
                    "45,000 molecules is one to five GPU-days and a million is out of "
                    "the question; a few thousand survivors of a cheaper docking tier "
                    "is under an hour. That is why this ships as its own tier below the "
                    "parallel one rather than as a third engine inside it, and why the "
                    "molecule cap is set two orders of magnitude lower than the "
                    "others'.\n\n"
                    "CNNaffinity is a predicted log-affinity, so 6 is roughly one "
                    "micromolar. Unlike the search, the network is not bit-stable "
                    "across GPU models: two cards can disagree in the last decimal, so "
                    "treat a threshold as a threshold and not as a reproducible "
                    "boundary."
                ),
                thresholds=(
                    ThresholdField(
                        name="executable",
                        label="gnina path",
                        kind="text",
                        environment_variable="MOLCASCADE_GNINA_EXECUTABLE",
                        help=(
                            "Absolute path to the binary. MolCascade does not build or "
                            "install it."
                        ),
                    ),
                    ThresholdField(
                        name="cnn_scoring",
                        label="How much CNN",
                        kind="choice",
                        choices=(
                            ("rescore", "Rescore poses (published setting)"),
                            ("refinement", "Refine against the network"),
                            ("metrorescore", "Metropolis rescore"),
                            ("metrorefine", "Metropolis refine"),
                            ("all", "All of them"),
                            ("none", "None -- empirical scoring only"),
                        ),
                        help=(
                            "'rescore' searches empirically and then scores the poses "
                            "with the network, which is what the published accuracy "
                            "figures are for. 'refinement' optimises against the network "
                            "too and costs several times more. 'none' turns GNINA into a "
                            "slower Uni-Dock."
                        ),
                    ),
                    ThresholdField(
                        name="rank_by",
                        label="Which number is the score",
                        kind="choice",
                        choices=(
                            ("cnn_affinity", "CNNaffinity (predicted log-affinity)"),
                            ("cnn_score", "CNNscore (pose quality, 0-1)"),
                            ("affinity", "Vina affinity (kcal/mol)"),
                        ),
                        help=(
                            "The other two are still computed and one is kept alongside. "
                            "Choosing a CNN number with the network off is refused rather "
                            "than silently producing an empty shortlist."
                        ),
                    ),
                    ThresholdField(
                        name="exhaustiveness",
                        label="Search effort",
                        kind="integer",
                        minimum=1,
                        maximum=512,
                        step=1,
                        help=(
                            "Monte-Carlo effort before any CNN work. Vina's own default "
                            "is 8."
                        ),
                    ),
                    ThresholdField(
                        name="num_modes",
                        label="Poses kept per ligand",
                        kind="integer",
                        minimum=1,
                        maximum=20,
                        step=1,
                        help="One row of evidence per pose; the gate reads the best.",
                    ),
                    ThresholdField(
                        name="keep_poses",
                        label="Store the docked pose",
                        kind="boolean",
                        help=(
                            "On by default; 'molcascade trace' exports the kept poses "
                            "as SDF. Turn it off for a scores-only screen."
                        ),
                    ),
                    ThresholdField(
                        name="allow_cpu",
                        label="Allow the CPU path",
                        kind="boolean",
                        help=(
                            "Deliberately awkward. GNINA's CPU path works, and a "
                            "shortlist that takes an hour on a card takes weeks on it."
                        ),
                    ),
                    ThresholdField(
                        name="max_molecules",
                        label="Most molecules to rescore",
                        kind="integer",
                        minimum=1,
                        maximum=10_000_000,
                        step=1_000,
                        help=(
                            "The stage stops before the first ligand if the tier hands it "
                            "more than this. Raise it only after timing one molecule and "
                            "multiplying."
                        ),
                    ),
                    ThresholdField(
                        name="expected_score_kind",
                        label="Score scale the gate expects",
                        kind="choice",
                        target="gate",
                        choices=(
                            ("CNN_AFFINITY", "CNNaffinity (predicted log-affinity)"),
                            ("CNN_SCORE", "CNNscore (pose quality, 0-1)"),
                            ("VINA_KCAL_MOL", "Vina affinity (kcal/mol)"),
                        ),
                        help=(
                            "Must match what is ranked above. -8.5 is a strong Vina score "
                            "and an impossible CNN one; 0.9 is the reverse, and a window "
                            "typed against the wrong scale would quietly accept or reject "
                            "everything."
                        ),
                    ),
                    ThresholdField(
                        name="expected_direction",
                        label="Which way is stronger",
                        kind="choice",
                        target="gate",
                        choices=(
                            ("HIGHER_STRONGER", "Higher is stronger (both CNN scales)"),
                            ("LOWER_STRONGER", "Lower is stronger (Vina affinity)"),
                        ),
                        help=(
                            "Exposed only here, because GNINA is the one engine whose "
                            "ranking can change direction: its two CNN scales go up and "
                            "its Vina affinity goes down. Get it wrong and the gate stops "
                            "the run rather than inverting the shortlist."
                        ),
                    ),
                    *_window(
                        minimum_label="Weakest score allowed",
                        maximum_label="Strongest score allowed",
                        lower=-30.0,
                        upper=30.0,
                        step=0.1,
                        help_text=(
                            "Inclusive. Higher is stronger for both CNN scales and lower "
                            "is stronger for Vina affinity, so which end filters depends "
                            "on the scale chosen above."
                        ),
                    ),
                ),
            ),
        ),
    ),
    CriterionSpec(
        id="reference_similarity",
        label="Similarity to reference leads",
        question="How close is this molecule to the project's known actives?",
        stage="similarity",
        evidence="applicability",
        summary=(
            "Maximum Tanimoto similarity to a reference set. Use a minimum to stay "
            "near known chemistry, or a maximum to demand novelty."
        ),
        options=(
            BackendOption(
                id="rdkit_reference_similarity",
                label="RDKit exact reference similarity",
                engine="RDKit",
                summary=(
                    "Exact maximum Tanimoto against an inline reference list, with "
                    "explicit comparison budgets."
                ),
                plugin_ref="applicability.rdkit_reference_similarity@0.1.0",
                license_spdx="BSD-3-Clause",
                citation=(
                    "Rogers D, Hahn M. Extended-connectivity fingerprints. J Chem Inf "
                    "Model. 2010;50(5):742-754. doi:10.1021/ci100050t"
                ),
                # The one option whose cost is not a single number: this compares
                # every candidate against every reference, so throughput falls
                # roughly linearly with the reference set. Measured through the
                # adapter: ~700 molecules/s against 8 references, ~610 against
                # 200, ~380 against 2,000 and ~60 against 20,000. The figure here
                # is the small-reference-set case that this backend is for; past
                # about a thousand references the indexed FPSim2 option below
                # overtakes it and keeps going.
                throughput_per_second=610,
                recommended=True,
                defaults={"schema_version": 1, "objective": "annotate"},
                thresholds=(
                    ThresholdField(
                        name="reference_path",
                        label="Reference lead file",
                        kind="text",
                        nullable=True,
                        required=True,
                        help="Local CSV/TSV/SMI file holding the project's known actives.",
                    ),
                    ThresholdField(
                        name="objective",
                        label="Objective",
                        kind="choice",
                        choices=(
                            ("annotate", "Annotate only — record similarity, filter nothing"),
                            ("analogue", "Analogue — keep molecules close to the leads"),
                            ("novel", "Novelty — keep molecules far from the leads"),
                        ),
                        help=(
                            "Similarity and novelty are opposing objectives. Pick the one "
                            "this project actually wants."
                        ),
                    ),
                    ThresholdField(
                        name="analogue_min_similarity",
                        label="Minimum similarity (analogue objective)",
                        minimum=0,
                        maximum=1,
                        step=0.05,
                        nullable=True,
                        help="Raise to stay near known actives.",
                    ),
                    ThresholdField(
                        name="novelty_max_similarity",
                        label="Maximum similarity (novelty objective)",
                        minimum=0,
                        maximum=1,
                        step=0.05,
                        nullable=True,
                        help="Lower to demand novelty instead of resemblance.",
                    ),
                ),
            ),
            BackendOption(
                id="fpsim2",
                label="FPSim2 indexed similarity",
                engine="FPSim2",
                summary=(
                    "BitBound-pruned popcount search built for reference sets far "
                    "larger than an exact scan can afford."
                ),
                plugin_ref="applicability.fpsim2_reference_similarity@0.1.0",
                license_spdx="MIT",
                citation=(
                    "Félix E. FPSim2: simple package for fast molecular similarity "
                    "searches. EMBL-EBI. https://github.com/chembl/FPSim2"
                ),
                requires=("FPSim2",),
                # Measured against the exact backend on 300 candidates: at 200
                # references it is slower (611 vs 785 q/s), at 2,000 it draws
                # ahead (519 vs 376), at 20,000 it is 3.6x (221 vs 61).  The
                # rate quoted is the 20,000-reference figure, since that is the
                # size at which anyone would choose this option.
                throughput_per_second=220,
                notes=(
                    "Only worth choosing above roughly a thousand references — below "
                    "that, building the index costs more than the comparisons it "
                    "saves, and the exact backend is faster. Coefficients come back "
                    "in single precision, so a molecule sitting exactly on a "
                    "threshold can land on the other side of it from the exact "
                    "backend."
                ),
                defaults={"schema_version": 1, "objective": "annotate"},
                thresholds=(
                    ThresholdField(
                        name="reference_path",
                        label="Reference lead file",
                        kind="text",
                        nullable=True,
                        required=True,
                        help="Local CSV/TSV/SMI file holding the reference set.",
                    ),
                    ThresholdField(
                        name="objective",
                        label="Objective",
                        kind="choice",
                        choices=(
                            ("annotate", "Annotate only — record similarity, filter nothing"),
                            ("analogue", "Analogue — keep molecules close to the leads"),
                            ("novel", "Novelty — keep molecules far from the leads"),
                        ),
                        help=(
                            "Similarity and novelty are opposing objectives. Pick the one "
                            "this project actually wants."
                        ),
                    ),
                    ThresholdField(
                        name="analogue_min_similarity",
                        label="Minimum similarity (analogue objective)",
                        minimum=0,
                        maximum=1,
                        step=0.05,
                        nullable=True,
                        help="Raise to stay near known actives.",
                    ),
                    ThresholdField(
                        name="novelty_max_similarity",
                        label="Maximum similarity (novelty objective)",
                        minimum=0,
                        maximum=1,
                        step=0.05,
                        nullable=True,
                        help="Lower to demand novelty instead of resemblance.",
                    ),
                    ThresholdField(
                        name="n_workers",
                        label="Query threads",
                        kind="integer",
                        minimum=1,
                        maximum=256,
                        step=1,
                        help=(
                            "Threads FPSim2 uses inside one query. Leave at 1 unless this "
                            "stage is the only thing running."
                        ),
                    ),
                ),
            ),
        ),
    ),
    CriterionSpec(
        id="custom_model",
        label="Custom model prediction",
        question="What does my own trained model say about this molecule?",
        stage="custom",
        evidence="prediction",
        summary=(
            "Point at a model bundle you exported yourself. MolCascade computes the "
            "features, runs the graph locally, and thresholds one named endpoint."
        ),
        options=(
            BackendOption(
                id="custom_onnx_bundle",
                label="MolCascade model bundle (ONNX)",
                engine="ONNX Runtime",
                summary=(
                    "A directory holding molcascade_model.yaml, one exported graph and "
                    "its digest. The featurizer is MolCascade's own, so no third-party "
                    "Python is executed."
                ),
                plugin_ref="prediction.custom_model@0.1.0",
                gate_plugin="prediction.numeric_evidence_gate@0.1.0",
                license_spdx="MolCascade",
                citation=(
                    "ONNX Runtime. https://onnxruntime.ai — cite the model you exported "
                    "into the bundle; this line names only the runtime that executed it."
                ),
                requires=("onnxruntime",),
                recommended=True,
                defaults={"schema_version": 1, "bundle_dir": "", "expected_bundle_sha256": ""},
                gate_defaults={
                    "schema_version": 1,
                    "semantics_label": "Project model; review its training domain.",
                },
                thresholds=(
                    ThresholdField(
                        name="bundle_dir",
                        label="Model bundle directory",
                        kind="text",
                        help=(
                            "Absolute path to the directory containing "
                            "molcascade_model.yaml and the exported graph."
                        ),
                    ),
                    ThresholdField(
                        name="expected_bundle_sha256",
                        label="Bundle digest",
                        kind="text",
                        help=(
                            "Run `molcascade model-bundle <dir>` and paste bundle_sha256. "
                            "It pins the run to exactly these model bytes."
                        ),
                    ),
                    ThresholdField(
                        name="endpoint_id",
                        label="Endpoint",
                        kind="text",
                        target="gate",
                        help="The endpoint_id declared in the bundle manifest.",
                    ),
                    *_window(
                        lower=-1e9,
                        upper=1e9,
                        step=0.1,
                        help_text="Inclusive window on the predicted value.",
                    ),
                ),
            ),
        ),
    ),
    CriterionSpec(
        id="binding_affinity",
        label="Predicted binding affinity",
        question="What affinity does a structure-aware model predict for my target?",
        stage="custom",
        evidence="prediction",
        filters=True,
        summary=(
            "Needs a target structure in addition to the ligand. Structure-aware "
            "affinity models rank; they do not produce calibrated Ki."
        ),
        options=(
            BackendOption(
                id="boltz2_affinity",
                label="Boltz-2 affinity head",
                engine="Boltz",
                summary=(
                    "Open-weight co-folding model with an affinity head; useful for "
                    "ranking, not for absolute Ki."
                ),
                license_spdx="MIT",
                citation=(
                    "Passaro S, Corso G, Wohlwend J, et al. Boltz-2: towards accurate and "
                    "efficient binding affinity prediction. bioRxiv (2025). "
                    "https://github.com/jwohlwend/boltz"
                ),
                requires=("boltz", "torch"),
                notes=(
                    "Reviewed research option. Requires a GPU and a target structure; "
                    "no MolCascade adapter is registered yet."
                ),
            ),
            BackendOption(
                id="custom_affinity_bundle",
                label="Your own affinity model",
                engine="ONNX Runtime",
                summary=(
                    "Train on your own Ki/IC50 table and export a bundle; then use the "
                    "custom-model criterion above."
                ),
                notes="Use the custom model bundle criterion; it emits the same evidence.",
                # The only entry in the catalogue whose reference is not ours to
                # give: the model is the user's, so the paper is theirs too. Said
                # explicitly rather than left blank, because a blank reference and
                # a tool that was never used look identical in a methods section.
                citation=(
                    "Cite your own model and its training set. This option is a signpost "
                    "to the custom model bundle criterion, not a method of its own; "
                    "the bundle's own metadata is what a reader needs."
                ),
            ),
        ),
    ),
)


CRITERIA_BY_ID: dict[str, CriterionSpec] = {spec.id: spec for spec in CRITERIA}


def criteria_for_stage(stage_id: str) -> tuple[CriterionSpec, ...]:
    return tuple(spec for spec in CRITERIA if spec.stage == stage_id)


def find_option(criterion_id: str, option_id: str) -> BackendOption | None:
    spec = CRITERIA_BY_ID.get(criterion_id)
    if spec is None:
        return None
    return next((option for option in spec.options if option.id == option_id), None)


def option_for_plugin(plugin_ref: str) -> tuple[CriterionSpec, BackendOption] | None:
    """Find which criterion a registered plugin implements."""

    for spec in CRITERIA:
        for option in spec.options:
            if option.plugin_ref == plugin_ref:
                return spec, option
    return None


def catalogue_json(*, available_plugins: Iterable[str] | None = None) -> dict[str, Any]:
    """Serialize the catalogue for the browser, marking unavailable adapters.

    An option keeps ``executable`` only when its adapter is registered in the
    running installation, so a builder generated on a machine without an
    optional backend cannot export a configuration that machine cannot run.
    """

    registered = None if available_plugins is None else set(available_plugins)
    stages = [group.as_json() for group in STAGE_GROUPS]
    criteria: list[dict[str, Any]] = []
    for spec in CRITERIA:
        payload = spec.as_json()
        for option, option_payload in zip(spec.options, payload["options"], strict=True):
            registered_here = (
                option.executable
                if registered is None
                else bool(option.plugin_ref and option.plugin_ref in registered)
            )
            option_payload["executable"] = registered_here
            if option.executable and not registered_here:
                option_payload["notes"] = (
                    option.notes or "Adapter is not registered in this installation."
                )
        criteria.append(payload)
    return {"stages": stages, "criteria": criteria}


__all__ = [
    "CRITERIA",
    "CRITERIA_BY_ID",
    "STAGE_GROUPS",
    "BackendOption",
    "CriterionSpec",
    "StageGroup",
    "ThresholdField",
    "catalogue_json",
    "criteria_for_stage",
    "find_option",
    "option_for_plugin",
]
