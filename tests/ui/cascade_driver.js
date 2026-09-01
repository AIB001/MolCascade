/* Drive the offline cascade builder through a scripted session.
 *
 * Usage: node cascade_driver.js <stub.js> <cascade.js> <ids.json> <payload.json>
 *
 * Prints one JSON object describing what the builder produced at each step.
 * Every interaction goes through the same listeners a real click or drop would
 * reach, so the assertions in the Python test are about the shipped behaviour
 * rather than about a parallel reimplementation of it.
 */
"use strict";

const fs = require("fs");
const path = require("path");

const [stubPath, cascadePath, idsPath, payloadPath] = process.argv.slice(2);
const { makeDocument } = require(path.resolve(stubPath));
const ids = JSON.parse(fs.readFileSync(idsPath, "utf8"));
const payloadText = fs.readFileSync(payloadPath, "utf8");

const built = makeDocument(ids);
const downloads = [];

global.document = built.document;
global.CSS = { escape: (value) => String(value).replace(/["\\]/g, "\\$&") };
global.Blob = function Blob(parts) {
  this.parts = parts;
  downloads.push(parts.join(""));
};
global.URL = { createObjectURL: () => "blob:test", revokeObjectURL: () => {} };
global.FileReader = function FileReader() {};
/* Every POST the page attempts, and what the stub answered.  The reply is
 * controlled from the environment so one driver can play both the run where the
 * server takes the file and the run where it refuses -- the second being the
 * one that has to end in a browser download rather than a lost cascade. */
const posts = [];
const fetchFailure = process.env.DRIVER_FETCH_FAILURE || "";
global.window = {
  confirm: () => true,
  setTimeout: (fn) => 0,
  fetch: (url, init) => {
    posts.push({ url: url, method: init.method, headers: init.headers, body: init.body });
    if (fetchFailure === "network") return Promise.reject(new Error("connection refused"));
    if (fetchFailure === "refused") {
      return Promise.resolve({
        ok: false,
        json: () => Promise.resolve({ ok: false, error: "wrong or missing builder token" }),
      });
    }
    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ ok: true, path: payloadEndpointPath() }),
    });
  },
};

function payloadEndpointPath() {
  const endpoint = JSON.parse(payloadText).save_endpoint;
  return endpoint ? endpoint.path : null;
}

document.getElementById("builder-payload").value = payloadText;

require(path.resolve(cascadePath));

/* ------------------------------------------------------------- inspection */

const funnel = document.getElementById("funnel");
const tray = document.getElementById("tray");
const inspector = document.getElementById("inspector");
const machine = document.getElementById("machine");
const review = document.getElementById("review-body");

function tierCards() {
  return funnel.all((node) => node.tagName === "ARTICLE" && node.classList.contains("tier"));
}

function blocks() {
  return funnel.all((node) => node.classList.contains("block"));
}

function shape() {
  return tierCards().map((card) => ({
    id: card.dataset.tier,
    criteria: card
      .all((node) => node.classList.contains("block"))
      .map((block) => block.dataset.criterion),
  }));
}

/* The canvas in document order, keeping only what claims to be a real stage.
 * This is the list the faithfulness test compares against `lower_cascade`. */
function canvasStages() {
  return funnel.all((node) => Boolean(node.dataset.stage)).map((node) => node.dataset.stage);
}

function openReview() {
  document.getElementById("action-review").click();
}

function exported() {
  openReview();
  return JSON.parse(review.textContent);
}

function reviewPlan() {
  openReview();
  return document
    .getElementById("review-plan")
    .all((node) => node.classList.contains("plan-stage"))
    .map((node) => node.textContent);
}

function focusKey(key) {
  const found = funnel.querySelector('[data-focus-key="' + key + '"]');
  if (!found) throw new Error("no control with focus key " + key);
  return found;
}

function status() {
  const bar = document.getElementById("status-bar");
  return { className: bar.className, text: bar.textContent };
}

/* A drag is three events in a real browser and three events here.  The builder
 * keeps the payload in a module variable rather than in `dataTransfer`, because
 * `dragover` -- the handler that decides whether a drop is legal -- is not
 * allowed to read the transfer object; the stub follows the same contract. */
function transfer() {
  return { setData: () => {}, effectAllowed: "", dropEffect: "" };
}

function dragTo(source, target) {
  source.dispatch("dragstart", { dataTransfer: transfer() });
  const over = target.dispatch("dragover", { dataTransfer: transfer() });
  target.dispatch("drop", { dataTransfer: transfer() });
  source.dispatch("dragend");
  return over;
}

function socketOf(tierId) {
  const found = funnel.all(
    (node) => node.dataset.drop === "socket" && node.dataset.tier === tierId
  );
  if (!found.length) throw new Error("no socket for tier " + tierId);
  return found[0];
}

function slotOf(tierId, index) {
  const found = funnel.all(
    (node) =>
      node.dataset.drop === "slot" &&
      node.dataset.tier === tierId &&
      node.dataset.index === String(index)
  );
  if (!found.length) throw new Error("no slot " + index + " in tier " + tierId);
  return found[0];
}

const steps = {};

// Editing a threshold has to reach the gate settings, not the producer's.
function inspectorFields() {
  const found = new Map();
  for (const wrapper of inspector.all((node) => node.classList.contains("field"))) {
    const label = wrapper.childNodes[0];
    const control = wrapper.childNodes[1];
    if (!label || !control) continue;
    if (["INPUT", "SELECT", "TEXTAREA"].indexOf(control.tagName) < 0) continue;
    found.set(label.textContent, control);
  }
  return found;
}

/* ------------------------------------------------------------------ steps */

steps.boot = {
  shape: shape(),
  tierModes: exported().tiers.map((tier) => tier.mode),
  downloadDisabled: document.getElementById("action-download").disabled,
  inspectorText: inspector.textContent,
  machineText: machine.textContent,
  trayBricks: tray.all((node) => node.classList.contains("brick")).length,
  canvasStages: canvasStages(),
};

/* A pin is the answer to a question only some arrangements ask.
 *
 * Two docking engines in one tier both emit `docking_score/v1`, and each stage's
 * table holds only its own engine's rows, so a block below that reads the score
 * has to name the one it means -- `lower_cascade` refuses to guess.  The shipped
 * cascade answers it.  What has to stay true is that the answer tracks the
 * funnel: switch the named engine off and the question stops being asked, so the
 * pin must go rather than be exported pointing at a stage that is not there.
 */
function pinOf(criterionId) {
  for (const tier of exported().tiers) {
    for (const criterion of tier.criteria) {
      if (criterion.id === criterionId) return criterion.evidence_from || null;
    }
  }
  return null;
}

const pinnedBoot = (() => {
  for (const tier of exported().tiers) {
    for (const criterion of tier.criteria) {
      const contracts = Object.keys(criterion.evidence_from || {});
      if (!contracts.length) continue;
      return {
        criterion: criterion.id,
        contract: contracts[0],
        stage: criterion.evidence_from[contracts[0]],
      };
    }
  }
  return null;
})();
if (!pinnedBoot) throw new Error("the starter cascade pins no evidence source");

// What the inspector offers for it, so the operator can change the choice
// without editing the exported file.
blocks().filter((node) => node.dataset.criterion === pinnedBoot.criterion)[0].click();
const pinField = Array.from(inspectorFields()).filter(
  (entry) => entry[0] === pinnedBoot.contract
)[0];
focusKey("ct:" + pinnedBoot.stage).click();
const pinWithoutIt = pinOf(pinnedBoot.criterion);
const statusWithoutIt = status();
const downloadWithoutIt = document.getElementById("action-download").disabled;
focusKey("ct:" + pinnedBoot.stage).click();

/* And the case that had none of this: a block the operator adds themselves.
 * `newCriterion` builds it from the catalogue entry, which cannot know what
 * sits above it, so it arrives with no pin at all -- and used to be exported
 * that way, into a cascade lowering refused for an ambiguity the page had
 * already drawn.  Removed and re-added through the same two controls an
 * operator would use, so the answer has to be reached rather than remembered.
 */
const pinnedTier = shape().filter((tier) =>
  tier.criteria.some((id) => id === pinnedBoot.criterion)
)[0];
const pinnedSpec = (() => {
  for (const tier of exported().tiers) {
    for (const criterion of tier.criteria) {
      if (criterion.id === pinnedBoot.criterion) return criterion.criterion;
    }
  }
  return null;
})();
const pinnedLabel = JSON.parse(payloadText).criteria.filter(
  (spec) => spec.id === pinnedSpec
)[0].label;
focusKey("cx:" + pinnedBoot.criterion).click();
focusKey("add:" + pinnedTier.id).click();
const readded = document
  .getElementById("picker-list")
  .all((node) => node.classList.contains("picker-item"))
  .filter((item) => item.textContent.indexOf(pinnedLabel) === 0);
if (readded.length !== 1) throw new Error("the picker no longer offers " + pinnedSpec);
readded[0].click();
const reborn = shape().filter((tier) => tier.id === pinnedTier.id)[0];

steps.evidencePin = {
  readded: {
    spec: pinnedSpec,
    criterion: reborn.criteria[reborn.criteria.length - 1],
    pin: pinOf(reborn.criteria[reborn.criteria.length - 1]),
    downloadDisabled: document.getElementById("action-download").disabled,
  },
  boot: pinnedBoot,
  offered: pinField ? pinField[1].all((node) => node.tagName === "OPTION").map((o) => o.value) : null,
  selected: pinField ? pinField[1].value : null,
  withoutIt: pinWithoutIt,
  statusWithoutIt: statusWithoutIt,
  downloadDisabledWithoutIt: downloadWithoutIt,
  restored: pinOf(pinnedBoot.criterion),
};

// The complaint the whole redesign answers: moving a module between tiers.
// Here it is one click on the block's own up arrow.
const secondTier = steps.boot.shape[1];
focusKey("cu:" + secondTier.criteria[0]).click();
steps.moveAcrossTier = { shape: shape(), status: status() };

// Assembling by hand: drag a brick out of the tray into the last tier.  This is
// the gesture the redesign is named for, so it is exercised, not assumed.
const lastTierId = shape()[shape().length - 1].id;
const brick = tray.all((node) => node.classList.contains("brick") && !node.disabled)[0];
if (!brick) throw new Error("the tray offers no runnable block");
const brickSpec = brick.dataset.spec;
dragTo(brick, socketOf(lastTierId));
steps.dragFromTray = {
  spec: brickSpec,
  shape: shape(),
  status: status(),
  landed: exported()
    .tiers.find((tier) => tier.id === lastTierId)
    .criteria.map((criterion) => criterion.criterion),
};

// And drag the block it created into the tier above, at a chosen position.
const shapeNow = shape();
const donor = shapeNow[shapeNow.length - 1];
const receiver = shapeNow[shapeNow.length - 2];
const travellingId = donor.criteria[donor.criteria.length - 1];
const travelling = blocks().filter((node) => node.dataset.criterion === travellingId)[0];
dragTo(travelling, slotOf(receiver.id, 0));
steps.dragBetweenTiers = {
  moved: travellingId,
  from: donor.id,
  to: receiver.id,
  shape: shape(),
  status: status(),
};

// Select the first tier, then switch it to a consensus vote from the inspector.
tierCards()[0].childNodes[0].dispatch("click");
const atLeast = inspector.all(
  (node) =>
    node.tagName === "INPUT" && node.getAttribute("type") === "radio" && node.value === "at_least"
);
if (atLeast.length !== 1) throw new Error("expected one at_least radio, saw " + atLeast.length);
atLeast[0].dispatch("change");
steps.consensus = {
  tier: exported().tiers[0],
  inspectorText: inspector.textContent,
};

// Add a criterion through the picker dialog.
const targetTier = shape()[0].id;
focusKey("add:" + targetTier).click();
const picker = document.getElementById("picker");
const pickerList = document.getElementById("picker-list");
steps.picker = {
  open: picker.open,
  offered: pickerList
    .all((node) => node.classList.contains("picker-item"))
    .map((item) => ({ text: item.textContent, disabled: item.disabled })),
};
const wanted = pickerList
  .all((node) => node.classList.contains("picker-item"))
  .filter((item) => item.textContent.indexOf("Synthesizability score") === 0);
if (wanted.length !== 1) throw new Error("expected one synthesizability entry");
wanted[0].click();
steps.added = { shape: shape(), status: status() };

const fields = inspectorFields();
steps.thresholdLabels = Array.from(fields.keys());
const maximum = fields.get("Maximum SA score");
if (!maximum) throw new Error("no maximum control: " + steps.thresholdLabels.join(" | "));
// Deliberately not a value any shipped default uses, so the Python side can
// tell an edit that landed on this criterion from one that landed on its twin.
maximum.value = "3.25";
maximum.dispatch("change");

const finalConfig = exported();
steps.final = {
  config: finalConfig,
  downloadDisabled: document.getElementById("action-download").disabled,
  // What the user sees, in the order they see it, and what the review dialog
  // promises will run.  The Python side checks both against the real lowering.
  canvasStages: canvasStages(),
  reviewPlan: reviewPlan(),
};

// An unsatisfiable consensus must block export rather than be silently repaired.
const firstTierId = shape()[0].id;
const rows = blocks().filter((row) => row.dataset.tier === firstTierId);
focusKey("ct:" + rows[0].dataset.criterion).click();
focusKey("ct:" + rows[1].dataset.criterion).click();
steps.unsatisfiable = {
  status: status(),
  downloadDisabled: document.getElementById("action-download").disabled,
  minimumPasses: exported().tiers[0].minimum_passes,
  // Disabled blocks are still drawn, but they must stop claiming a stage.
  canvasStages: canvasStages(),
  disabledCriteria: [rows[0].dataset.criterion, rows[1].dataset.criterion],
};
focusKey("ct:" + rows[0].dataset.criterion).click();
focusKey("ct:" + rows[1].dataset.criterion).click();

const restored = JSON.stringify(exported(), null, 2) + "\n";
document.getElementById("action-download").click();
steps.download = { count: downloads.length, matches: downloads[0] === restored };

/* A docking block cannot be finished in this file -- the receptor belongs to
 * the campaign, not to the screening policy -- so the tier has to say what the
 * command line still owes it while the cascade is being written, rather than
 * an hour later when the run refuses to start. */
function needsOf(tierId) {
  const card = tierCards().filter((node) => node.dataset.tier === tierId)[0];
  if (!card) throw new Error("no tier card for " + tierId);
  return card.all((node) => node.classList.contains("tier-needs")).map((n) => n.textContent);
}

/* Docking ships enabled in the starter cascade now, and the conformer tier that
 * feeds it ships with it.  Everything below is the story of an operator *adding*
 * docking -- the tier declaring what the command line still owes it, the
 * evidence producer appearing above it, both withdrawn again when the block is
 * turned off -- and none of that can be observed on a cascade that already has
 * docking in it.  So the shipped pair is removed first, through the same delete
 * control an operator would use, and what was removed is recorded: a starter
 * that stops shipping docking must fail here rather than quietly reduce this
 * whole section to a re-run of the defaults.
 *
 * Consumers first, then the feeder.  Deleting a producer while a block above
 * still reads it is refused -- that refusal is a guarantee of its own, checked
 * further down -- so the reverse order would leave the session wedged in an
 * error state.  The post-docking metrics are consumers too: they read the
 * docking score, so they go with it or the page stays in the error the export
 * check further down would otherwise be reporting about them. */
const catalogueOptions = {};
for (const spec of JSON.parse(payloadText).criteria) {
  for (const option of spec.options) {
    if (option.plugin_ref) catalogueOptions[option.plugin_ref] = option;
  }
}
const shippedDocking = exported().tiers.filter((tier) =>
  tier.criteria.some((criterion) => criterion.backend.indexOf("docking.") === 0)
);
const shippedNeeds = [];
for (const tier of shippedDocking) {
  for (const criterion of tier.criteria) {
    const option = catalogueOptions[criterion.backend];
    for (const contract of option ? option.needs_evidence : []) {
      if (shippedNeeds.indexOf(contract) < 0) shippedNeeds.push(contract);
    }
  }
}
const shippedFeeders = exported().tiers.filter(
  (tier) =>
    !shippedDocking.some((docked) => docked.id === tier.id) &&
    tier.criteria.some((criterion) => {
      const option = catalogueOptions[criterion.backend];
      if (!option) return false;
      return option.produces_evidence.some((contract) => shippedNeeds.indexOf(contract) >= 0);
    })
);
/* Tiers left reading a contract no surviving tier above them produces, found by
 * walking the funnel in order rather than by name: the metrics below docking
 * today, whatever is added below them tomorrow. */
function orphanedWithout(removed) {
  const orphaned = [];
  for (;;) {
    const gone = removed.concat(orphaned);
    const available = [];
    const found = [];
    for (const tier of exported().tiers) {
      if (gone.indexOf(tier.id) >= 0) continue;
      const needs = [];
      for (const criterion of tier.criteria) {
        const option = catalogueOptions[criterion.backend];
        for (const contract of option ? option.needs_evidence : []) needs.push(contract);
      }
      if (needs.some((contract) => available.indexOf(contract) < 0)) found.push(tier.id);
      for (const criterion of tier.criteria) {
        const option = catalogueOptions[criterion.backend];
        for (const contract of option ? option.produces_evidence : []) available.push(contract);
      }
    }
    if (!found.length) return orphaned;
    for (const id of found) orphaned.push(id);
  }
}

const dockingNeighbourhood = shippedDocking.concat(shippedFeeders).map((tier) => tier.id);
const shippedConsumers = orphanedWithout(dockingNeighbourhood);
const removedShipped = shippedConsumers.concat(dockingNeighbourhood);
for (const tierId of removedShipped) {
  focusKey("tx:" + tierId).click();
}
steps.shippedDocking = {
  removed: removedShipped,
  // Recorded by the role each tier played rather than as one count, so that a
  // tier added below docking tomorrow shows up as a consumer here instead of
  // as an off-by-one in a number nobody can read.
  docking: shippedDocking.map((tier) => tier.id),
  feeders: shippedFeeders.map((tier) => tier.id),
  consumers: shippedConsumers,
  remaining: shape().map((tier) => tier.id),
  status: status(),
  downloadDisabled: document.getElementById("action-download").disabled,
};

const dockTierId = shape()[shape().length - 1].id;
const needsBefore = needsOf(dockTierId);
const shapeBeforeDocking = shape();
focusKey("add:" + dockTierId).click();
const dockingEntry = document
  .getElementById("picker-list")
  .all((node) => node.classList.contains("picker-item"))
  .filter((item) => item.textContent.indexOf("Docking score") === 0);
if (dockingEntry.length !== 1) throw new Error("expected one docking entry");
dockingEntry[0].click();
const dockingCriteria = shape()[shape().length - 1].criteria;
const dockingId = dockingCriteria[dockingCriteria.length - 1];
const needsAfter = needsOf(dockTierId);

/* Some engines read evidence they do not compute: Uni-Dock and GNINA dock a 3D
 * structure and generate none.  Nothing on the page tells an operator which of
 * the offered engines is like that, so choosing one used to export a file that
 * `lower_cascade` refused outright.  Adding the dependency is the fix, and this
 * records where it landed -- it must be a *serial* tier directly above the block
 * that reads it, because a pure evidence producer has no threshold and a
 * parallel tier demands a decision from everything in it. */
const shapeAfterDocking = shape();
const insertedIds = shapeAfterDocking
  .map((tier) => tier.id)
  .filter((id) => !shapeBeforeDocking.some((was) => was.id === id));
const exportedAfterDocking = exported();
function exportedTier(tierId) {
  return exportedAfterDocking.tiers.filter((tier) => tier.id === tierId)[0] || null;
}
steps.evidence = {
  addStatus: status(),
  downloadDisabled: document.getElementById("action-download").disabled,
  tiersBefore: shapeBeforeDocking.map((tier) => tier.id),
  tiersAfter: shapeAfterDocking.map((tier) => tier.id),
  inserted: insertedIds,
  insertedTiers: insertedIds.map(exportedTier),
  dockTier: exportedTier(dockTierId),
};

/* Where the engine is installed is the one question this browser cannot answer,
 * and it must not pretend to.  It used to: a blank engine path disabled Download
 * and told the operator to fill it in or export a variable "on the machine that
 * will run this cascade" -- advice that cannot be taken from the machine drawing
 * the page, which is usually a laptop with no docking engine on it at all.  So
 * the export goes ahead with the field blank, the page says who will answer it,
 * and `preflight_engine_paths` refuses on the host that can actually look.  The
 * Python side takes the file downloaded here and pushes it through exactly that
 * sequence. */
const dockingBlock = blocks().filter((node) => node.dataset.criterion === dockingId)[0];
if (!dockingBlock) throw new Error("no block for " + dockingId);
dockingBlock.click();
const engineFields = inspectorFields();
// Located by the variable it advertises rather than by its label, because which
// engine the picker offers first is not what is under test here.  The variable
// moved from the placeholder into the help text when the field stopped being
// required, so this reads the whole field rather than one attribute of it.
let engineLabel = null;
let engineControl = null;
let engineHelp = null;
for (const [label, control] of engineFields) {
  const text = control.parentNode.textContent || "";
  if (text.indexOf("MOLCASCADE_") < 0 || text.indexOf("EXECUTABLE") < 0) continue;
  engineLabel = label;
  engineControl = control;
  engineHelp = text;
  break;
}
steps.installationPaths = {
  blankStatus: status(),
  blankDownloadDisabled: document.getElementById("action-download").disabled,
  labels: Array.from(engineFields.keys()),
  label: engineLabel,
  placeholder: engineControl ? engineControl.getAttribute("placeholder") : null,
  help: engineHelp,
  variable: engineHelp ? (/MOLCASCADE_[A-Z0-9_]+/.exec(engineHelp) || [null])[0] : null,
};
if (!engineControl) {
  throw new Error("no engine path control: " + steps.installationPaths.labels.join(" | "));
}
// The claim in one line: a cascade that names no installation still leaves the
// browser.  What the Python side then does with it is the other half.
document.getElementById("action-download").click();
steps.installationPaths.blankDownload = downloads[downloads.length - 1];
steps.installationPaths.downloadsSoFar = downloads.length;
// Filling it in is still allowed -- an engine installed somewhere unusual is
// exactly what the field is for -- and has to reach the exported settings.
engineControl.value = "/opt/envs/karmadock/bin/python";
engineControl.dispatch("change");
steps.installationPaths.filledStatus = status();
steps.installationPaths.filledDownloadDisabled =
  document.getElementById("action-download").disabled;
steps.installationPaths.filledSettings = exported()
  .tiers.reduce((all, tier) => all.concat(tier.criteria), [])
  .filter((criterion) => criterion.id === dockingId)[0].settings;

// Disabling the block withdraws the demand with it: a tier that compiles into
// no stage asks the command line for nothing.
focusKey("ct:" + dockingId).click();
steps.runTimeNeeds = {
  offeredDisabled: dockingEntry[0].disabled,
  before: needsBefore,
  after: needsAfter,
  whenTurnedOff: needsOf(dockTierId),
  elsewhere: shape()
    .filter((tier) => tier.id !== dockTierId)
    .map((tier) => needsOf(tier.id)),
};

/* The inserted producer is an ordinary block afterwards -- movable, editable,
 * deletable -- so deleting it has to be caught rather than quietly allowed.  It
 * is the same state a cascade loaded from disk can arrive in, which is the case
 * the automatic insertion cannot cover. */
focusKey("ct:" + dockingId).click();
const producerTierId = steps.evidence.inserted[0];
if (producerTierId) {
  focusKey("tx:" + producerTierId).click();
}
steps.evidenceWithdrawn = {
  removed: producerTierId || null,
  status: status(),
  downloadDisabled: document.getElementById("action-download").disabled,
  tiers: shape().map((tier) => tier.id),
};

/* Where the exported file goes.
 *
 * With no endpoint in the payload -- the document written to disk -- the button
 * makes a Blob and nothing is posted anywhere.  With one, the same button posts
 * instead, and the interesting part is the failure: a server that refuses, or a
 * command the operator has already stopped, must still leave the operator
 * holding the file.  Both runs go through the shipped listener, so the token
 * header, the body and the fallback are the real ones. */
/* The steps above deliberately leave the canvas unexportable, and where a file
 * goes is only a question about a cascade that is ready to run.  Restoring the
 * starter through the shipped button is the shortest way back to one. */
function recordExport() {
  document.getElementById("action-reset").click();
  steps.export = {
    endpoint: JSON.parse(payloadText).save_endpoint,
    downloadsBefore: downloads.length,
    postsBefore: posts.length,
  };
  document.getElementById("action-download").click();
  steps.export.posted = posts.slice(steps.export.postsBefore);
  steps.export.downloadedAfterPost = downloads.length - steps.export.downloadsBefore;
  steps.export.status = status();
}

/* Two ticks, for two different reasons.  The steps above click Download as
 * well, and with an endpoint in the payload those clicks post too -- their
 * promises settle after this script's synchronous phase, so the counters are
 * only trustworthy once a turn of the event loop has drained them.  The second
 * tick is then the export's own: its status line and any fallback download
 * exist a microtask after the click returns, not before. */
setImmediate(() => {
  recordExport();
  setImmediate(() => {
    steps.export.settledStatus = status();
    steps.export.settledDownloads = downloads.length - steps.export.downloadsBefore;
    steps.export.settledFile = downloads[downloads.length - 1] || null;
    process.stdout.write(JSON.stringify(steps));
  });
});
