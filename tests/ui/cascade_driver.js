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
global.window = {
  confirm: () => true,
  setTimeout: (fn) => 0,
};

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

process.stdout.write(JSON.stringify(steps));
