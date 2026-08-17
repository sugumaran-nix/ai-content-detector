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
  assert.match(html, /<textarea[^>]*aria-label="Text to analyze"[^>]*aria-describedby="inputHint"/);
  assert.match(html, /<p class="input-hint" id="inputHint">For a steadier result/);
  assert.match(html, /id="analyzeBtn" aria-label="Run private scan"/);
});

test("built-in samples and model metadata match the local classifier", () => {
  assert.match(html, /id="sampleMixed"/);
  assert.match(html, /mixed: `Large language models have revolutionized/);
  assert.match(html, /mixed: `[^`]*I've been trying to fix this bug/);
  assert.match(html, /<div class="mstat-v">Calibrated local model<\/div>/);
});

test("idle and report states use user-facing guidance", () => {
  assert.match(html, /class="empty-preview"/);
  assert.match(html, /REPORT PREVIEW/);
  assert.match(html, /ANALYSIS COMPLETE/);
  assert.match(html, /Sentence evidence/);
  assert.doesNotMatch(html, /<div class="mstat-v">LinearSVC<\/div>/);
  assert.doesNotMatch(html, /Private by default|No API|No uploads/);
  assert.match(html, /Runs in your browser/);
  assert.match(html, /Explainable signals/);
});

test("loading overlay exposes live status and progress semantics", () => {
  assert.match(html, /id="loadLabel" aria-live="polite"/);
  assert.match(html, /role="progressbar" aria-label="Loading local analysis engine"/);
  assert.match(html, /id="loadSkip" type="button"/);
});

test("loading state updates status, progress, and accessible value", () => {
  const elements = createElements();
  setLoadingState(elements, "downloading");
  assert.equal(elements.label.textContent, "Downloading model weights…");
  assert.equal(elements.bar.style.width, "35%");
  assert.equal(elements.bar.getAttribute("aria-valuenow"), "35");
  assert.equal(elements.label.getAttribute("aria-live"), "polite");
});

test("ready state completes and dismisses the overlay", () => {
  const elements = createElements();
  setLoadingState(elements, "ready");
  assert.equal(elements.label.textContent, "Ready to scan.");
  assert.equal(elements.bar.style.width, "100%");
  assert.equal(elements.overlay.classList.contains("done"), true);
});

test("unavailable state keeps local mode understandable", () => {
  const elements = createElements();
  setLoadingState(elements, "unavailable");
  assert.match(elements.label.textContent, /limited scan mode is ready/);
  assert.equal(elements.bar.style.background, "var(--amber, #fbbf24)");
  assert.equal(elements.overlay.classList.contains("done"), true);
});

test("skipped state dismisses the overlay without pretending the model is ready", () => {
  const elements = createElements();
  setLoadingState(elements, "skipped");
  assert.equal(elements.overlay.classList.contains("done"), true);
  assert.equal(elements.label.textContent, "");
});
