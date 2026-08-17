"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { setLoadingState } = require("../ui-state.js");

const html = fs.readFileSync(path.join(__dirname, "..", "index.html"), "utf8");

function createElements() {
  const classes = new Set();
  const overlay = { classList: { add: (name) => classes.add(name), contains: (name) => classes.has(name) } };
  const labelAttrs = new Map();
  const label = {
    textContent: "",
    setAttribute: (name, value) => labelAttrs.set(name, value),
    getAttribute: (name) => labelAttrs.get(name),
  };
  const barAttrs = new Map();
  const bar = {
    style: {},
    parentElement: { setAttribute: (name, value) => barAttrs.set(name, value) },
    getAttribute: (name) => barAttrs.get(name),
  };
  return { overlay, label, bar };
}

test("analyzer textarea exposes its supporting guidance", () => {
  assert.match(html, /<textarea[^>]*aria-label="Passage to review"[^>]*aria-describedby="inputHint"/);
  assert.match(html, /<p class="input-hint" id="inputHint">For a steadier estimate/);
  assert.match(html, /id="analyzeBtn" aria-label="Scan this passage"/);
});

test("built-in samples and model metadata match the local classifier", () => {
  assert.match(html, /id="sampleMixed"/);
  assert.match(html, /mixed: `Large language models have revolutionized/);
  assert.match(html, /mixed: `[^`]*I've been trying to fix this bug/);
  assert.match(html, /<div class="mstat-v">Calibrated browser model<\/div>/);
});

test("idle and report states use user-facing guidance", () => {
  assert.match(html, /class="empty-preview"/);
  assert.match(html, /WHAT YOU'LL SEE/);
  assert.match(html, /SCAN COMPLETE/);
  assert.match(html, /Sentence-level evidence/);
  assert.match(html, /Patterns behind the estimate/);
  assert.doesNotMatch(html, /<div class="mstat-v">LinearSVC<\/div>/);
  assert.doesNotMatch(html, /Find the signal|Your text, explained|What influenced the score/);
  assert.match(html, /Local analysis/);
  assert.match(html, /Evidence you can inspect/);
  assert.match(html, /Where is my passage processed/);
});

test("loading overlay exposes live status and progress semantics", () => {
  assert.match(html, /id="loadLabel" aria-live="polite"/);
  assert.match(html, /role="progressbar" aria-label="Preparing local writing model"/);
  assert.match(html, /id="loadSkip" type="button"/);
});

test("loading state updates status, progress, and accessible value", () => {
  const elements = createElements();
  setLoadingState(elements, "downloading");
  assert.equal(elements.label.textContent, "Loading local scan data…");
  assert.equal(elements.bar.style.width, "35%");
  assert.equal(elements.bar.getAttribute("aria-valuenow"), "35");
  assert.equal(elements.label.getAttribute("aria-live"), "polite");
});

test("ready state completes and dismisses the overlay", () => {
  const elements = createElements();
  setLoadingState(elements, "ready");
  assert.equal(elements.label.textContent, "Your scanner is ready.");
  assert.equal(elements.bar.style.width, "100%");
  assert.equal(elements.overlay.classList.contains("done"), true);
});

test("unavailable state keeps local mode understandable", () => {
  const elements = createElements();
  setLoadingState(elements, "unavailable");
  assert.match(elements.label.textContent, /quick scan is ready/);
  assert.equal(elements.bar.style.background, "var(--amber, #fbbf24)");
  assert.equal(elements.overlay.classList.contains("done"), true);
});

test("skipped state dismisses the overlay without pretending the model is ready", () => {
  const elements = createElements();
  setLoadingState(elements, "skipped");
  assert.equal(elements.overlay.classList.contains("done"), true);
  assert.equal(elements.label.textContent, "");
});
