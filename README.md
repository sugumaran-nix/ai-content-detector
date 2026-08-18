# AI Content Detector

> Paste any passage — get an instant verdict. Fully client-side inference, no text ever leaves your browser.

![HTML](https://img.shields.io/badge/HTML-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=111827)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Vercel](https://img.shields.io/badge/Deployed_on_Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)

**[🚀 Live Demo](https://ai-content-detector-nine.vercel.app)** · **[API Docs](docs/API.md)** · **[Benchmark](docs/BENCHMARK.md)**

> **Note:** AI-text detection is probabilistic. Results are signals for review, not proof of authorship. Short, paraphrased, translated, or heavily edited passages can be difficult to classify reliably.

---

## ✨ Features

- **Privacy-first** — static frontend scores text entirely in-browser; no text is uploaded to any server
- **11-signal diagnostic grid** — perplexity, burstiness, type-token ratio, POS entropy, readability, and more
- **Sentence-level highlighting** — see exactly which sentences push the score toward AI or human
- **Three-tier verdict** — calibrated probability bands (Likely Human / Mixed / Likely AI)
- **No build step** — single `index.html` file, served anywhere
- **Optional FastAPI backend** — same calibrated pipeline exposed as a REST API for integrations
- **Drag-and-drop file loading** — plain-text files supported directly in the browser
- **Session history** — recent analyses persist in tab session storage, never sent anywhere

---

## 📊 Verdict Bands & Benchmark

Consistent across browser and backend:

| Verdict | Calibrated AI Probability |
|---|---|
| Likely Human | ≤ 0.30 |
| Mixed / Uncertain | > 0.30 and < 0.70 |
| Likely AI-generated | ≥ 0.70 |

Local benchmark (30 iterations per sample, 3 warmups):

| Sample | Predicted | AI Probability | Median latency |
|---|---|---|---|
| AI text | Likely AI | 0.8962 | 42ms |
| Human text | Likely Human | 0.0719 | 52ms |
| Mixed text | Mixed | 0.6398 | 91ms |

**Accuracy: 100% on benchmark suite · p95 latency: 92ms**

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla HTML / CSS / JavaScript — zero build step |
| Browser scoring | Exported LinearSVC weights + reference LM (`model.json`, `lm.json`) |
| Backend API | FastAPI, scikit-learn, NLTK, NumPy |
| Model | LinearSVC with 11 statistical features, calibrated probability output |
| Deployment | Vercel (frontend) · Render / Docker (backend) |

---

## 📁 Project Structure

```
ai-content-detector/
├── index.html                  # Full frontend — single file, no build step
├── model.json                  # Exported LinearSVC weights (browser-safe)
├── lm.json                     # Reference language model (browser-safe)
├── ui-state.js                 # Frontend state module
├── render.yaml                 # Render deployment config
├── Dockerfile                  # Root / Hugging Face Spaces image
├── test/
│   └── ui.test.js              # Frontend regression tests
├── benchmarks/
│   ├── benchmark.py            # Inference latency + accuracy benchmark
│   └── local_results.json      # Latest benchmark results
├── docs/
│   ├── API.md                  # Full API contract
│   ├── BENCHMARK.md            # Benchmark methodology
│   ├── INTEGRATION.md          # Integration guide
│   └── SECURITY_AUDIT.md       # Security audit notes
└── backend/
    ├── app.py                  # FastAPI service
    ├── inference.py            # Model loading, calibration, scoring
    ├── features.py             # 11-feature extraction pipeline
    ├── train.py                # Classifier training + artifact export
    ├── reference_lm.py         # Reference language model construction
    ├── export_lm_json.py       # LM → browser JSON export
    ├── export_weights.py       # Classifier → browser JSON export
    ├── requirements.txt
    ├── tests/test_app.py       # FastAPI + inference regression tests
    └── model/                  # Trained model artifacts
```

---

## 🧠 How It Works

### Browser Inference (no server needed)

```
Text input
    ↓
11 statistical features extracted locally
  • Perplexity against reference LM
  • Burstiness (sentence-length variance pattern)
  • Type-token ratio, avg sentence/word length
  • Stopword ratio, punctuation density
  • Repetition score, POS entropy, readability
    ↓
Exported LinearSVC weights applied in-browser
    ↓
Calibrated probability → Verdict + sentence-level highlights
```

Everything runs in-browser from `model.json` and `lm.json`. No API call is made.

### Optional Backend (API consumers)

The FastAPI backend runs the same feature pipeline server-side with additional endpoints for batch analysis, model metadata, and health/readiness checks.

---

## 🚀 Getting Started

### Run the frontend

```bash
npx --yes serve .
```

Open [http://localhost:3000](http://localhost:3000) — no install, no build.

### Run the backend locally

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

See [docs/API.md](docs/API.md) for the full API contract.

### Run the full test suite

```bash
python3 -m compileall -q backend
node --check ui-state.js
npm test
PYTHONPATH=backend pytest -q backend/tests
```

---

## 🐳 Docker

```bash
docker build -t ai-content-detector .
docker run -p 8000:8000 ai-content-detector
```

---

## 🔌 REST API

### `POST /analyze`

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "The concept of emergence refers to properties..."}'
```

**Response:**

```json
{
  "verdict": "likely_ai",
  "ai_probability": 0.87,
  "confidence": "high",
  "features": {
    "perplexity": 11240.5,
    "burstiness": 4821.3,
    "type_token_ratio": 0.74
  },
  "sentence_scores": [...]
}
```

See [docs/API.md](docs/API.md) for all endpoints including `/analyze/batch`, `/model/info`, and `/health`.

---

## 🚨 Known Limitations

- Short passages (< 150 words) produce less reliable scores — the feature pipeline needs enough signal to work with
- Heavily paraphrased, translated, or domain-specific text can fool the classifier
- The rate limiter is process-local — multi-worker or multi-replica deployments should add a shared edge limiter
- Model metrics reflect training-distribution accuracy, not a guarantee on unseen writing styles

---

## 📄 License

MIT
