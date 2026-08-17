# AI Text Detector

A privacy-first AI-generated-text detector with an explainable browser experience and an optional FastAPI inference service. The static frontend runs locally in the browser using exported classifier weights and a compact reference language model; the backend provides the same calibrated feature pipeline for API consumers.

> **Important:** AI-text detection is probabilistic. Results are signals for review, not proof of authorship. Short, paraphrased, translated, domain-specific, or heavily edited passages can be difficult to classify reliably.

**[Live demo](https://ai-content-detector-nine.vercel.app)** · **[GitHub repository](https://github.com/sugumaran-nix/ai-content-detector)**

---

## Features

The browser interface provides guided text entry, quick-start examples, plain-text file loading, drag-and-drop support, live word and character feedback, keyboard shortcuts, dark/light themes, accessible loading and result states, sentence-level highlighting, an 11-signal diagnostic grid, copy/export actions, and session-only recent-analysis history. Text entered into the static frontend is not sent to a server.

The optional backend exposes full analysis, a lightweight verdict endpoint, batch analysis, model metadata, health/readiness reporting, exact-origin CORS, request IDs, response timing, request-size protection, and an in-process sliding-window rate limiter. Model artifacts are loaded and validated before being cached for inference.

---

## Model and scoring contract

The project’s public verdict bands are consistent across the browser and backend:

| Verdict | Calibrated AI probability |
|---|---:|
| Likely human | `≤ 0.30` |
| Mixed / uncertain | `> 0.30` and `< 0.70` |
| Likely AI-generated | `≥ 0.70` |

The classifier uses eleven statistical features: perplexity, burstiness, type-token ratio, average sentence length, sentence-length variance, average word length, stopword ratio, punctuation density, repetition score, part-of-speech entropy, and readability. The calibration step shrinks extreme scores when feature values or vocabulary appear out of distribution.

The model metadata includes training-distribution accuracy and AUC. Those metrics are useful for comparing model builds, but they are not a guarantee of real-world accuracy on new writing styles.

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla HTML/CSS/JavaScript with no build step |
| Browser scoring | Exported calibrated classifier weights plus reference LM JSON |
| Backend API | FastAPI, Pydantic, scikit-learn, NLTK, NumPy |
| Model artifacts | Trusted local pickle artifacts and browser-safe JSON exports |
| Deployment | Vercel/static hosting for the frontend; Render or Docker for the backend |

---

## Project structure

```
ai-content-detector/
├── index.html                  # Full frontend — single file, no build step
├── render.yaml                 # Render deployment configuration
├── Dockerfile                  # Root/Hugging Face Spaces image
├── test/ui.test.js             # Node.js frontend regression tests
└── backend/
    ├── app.py                  # FastAPI service and API contracts
    ├── inference.py            # Model loading, calibration, and scoring
    ├── features.py             # Canonical 11-feature extraction pipeline
    ├── train.py                # Classifier training and artifact export
    ├── reference_lm.py         # Reference language-model construction
    ├── export_lm_json.py       # Backend LM to browser JSON export
    ├── export_weights.py       # Backend classifier to browser JSON export
    ├── requirements.txt        # Runtime and test dependencies
    ├── tests/test_app.py       # FastAPI and inference regression tests
    └── model/                  # Trusted local model artifacts
```

---

## How it works

1. The page loads `model.json` and `lm.json` from the same origin and keeps them in browser memory.
2. The browser extracts the eleven statistical signals locally from the passage.
3. The calibrated classifier produces a probability, confidence score, three-tier verdict, and sentence-level annotations.
4. Recent analyses remain in the current tab’s session storage and can be restored without leaving the browser.

No browser API calls are required for the static frontend, and no entered text is uploaded by that frontend.

## Local development and validation

Serve the static frontend through a local HTTP server so the model assets can load:

```bash
npx --yes serve .
```

Run the full validation suite from the repository root:

```bash
python3 -m compileall -q backend
node --check ui-state.js
npm test
PYTHONPATH=backend pytest -q backend/tests
```

To run the backend locally:

```bash
sudo pip3 install -r backend/requirements.txt
cd backend
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

See [docs/API.md](docs/API.md) for the service contract.

## Deployment

The static frontend can be deployed to Vercel or another static host without a build step. The repository also includes Render and Docker configurations for the backend. Both images prefetch NLTK data, build the reference language model and classifier, and run the API as a non-root user. The default rate limiter is process-local, so multi-worker or multi-replica deployments should add a shared edge limiter.

---

## 📄 License

MIT
