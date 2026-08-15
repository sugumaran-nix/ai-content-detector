"use strict";

function setLoadingState(elements, state) {
  const { overlay, label, bar } = elements;
  if (!overlay || !label || !bar) throw new Error("Loading state elements are required");

  const states = {
    loading: { label: "Loading language model…", width: "15%", color: "var(--accent)" },
    downloading: { label: "Downloading model weights…", width: "35%", color: "var(--accent)" },
    ready: { label: "Ready.", width: "100%", color: "var(--accent)" },
    unavailable: { label: "Classifier unavailable — local signal mode is ready.", width: "100%", color: "var(--amber, #fbbf24)" },
  };

  if (state === "skipped") {
    overlay.classList.add("done");
    return;
  }

  const next = states[state];
  if (!next) throw new Error(`Unknown loading state: ${state}`);
  label.textContent = next.label;
  label.setAttribute("aria-live", "polite");
  bar.style.width = next.width;
  bar.style.background = next.color;
  bar.parentElement?.setAttribute("aria-valuenow", next.width.replace("%", ""));

  if (state === "ready" || state === "unavailable") overlay.classList.add("done");
}

if (typeof module !== "undefined" && module.exports) module.exports = { setLoadingState };
if (typeof window !== "undefined") window.AppUI = { setLoadingState };
