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
});

test("loading overlay exposes live status and progress semantics", () => {
  assert.match(html, /id="loadLabel" aria-live="polite"/);
  assert.match(html, /role="progressbar" aria-label="Loading language model"/);
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
  assert.equal(elements.label.textContent, "Ready.");
  assert.equal(elements.bar.style.width, "100%");
  assert.equal(elements.overlay.classList.contains("done"), true);
});

test("unavailable state keeps local mode understandable", () => {
  const elements = createElements();
  setLoadingState(elements, "unavailable");
  assert.match(elements.label.textContent, /local signal mode is ready/);
  assert.equal(elements.bar.style.background, "var(--amber, #fbbf24)");
  assert.equal(elements.overlay.classList.contains("done"), true);
});

test("skipped state dismisses the overlay without pretending the model is ready", () => {
  const elements = createElements();
  setLoadingState(elements, "skipped");
  assert.equal(elements.overlay.classList.contains("done"), true);
  assert.equal(elements.label.textContent, "");
});
