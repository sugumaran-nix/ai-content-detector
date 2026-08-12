# AI-Generated Text Detector

> Paste any text — get a verdict in seconds. 11 statistical signals, sentence-level highlighting, and a full diagnostic readout to explain exactly why the model thinks it's AI or human.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Backend-Render-46E3B7?style=for-the-badge&logo=render&logoColor=black)
![Vercel](https://img.shields.io/badge/Frontend-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)


<!--
**[🚀 Live Demo](https://ai-content-detector-nine.vercel.app)** · **[API Docs](https://ai-content-detector-hodw.onrender.com/docs)**

---

-->

<!-- Add a demo GIF here once recorded -->
<!-- ![Demo](./demo.gif) -->

> 📸 **Demo GIF coming soon** — record a paste → analysis flow and drop it here as `demo.gif`

---

## ✨ Features

- **11 interpretable signals** — perplexity, burstiness, type-token ratio, sentence length variance, POS entropy, readability (Flesch), stopword ratio, punctuation density, repetition score, avg word/sentence length
- **Sentence-level breakdown** — each sentence scored and highlighted individually; fragments under 4 words return `null` rather than a misleading score
- **Three-tier verdict** — `likely_ai` (≥60%), `mixed` (40–60%), `likely_human` (≤40%) with a confidence score
- **Batch API** — analyze up to 10 texts in a single request via `POST /batch`
- **Fast path** — `POST /analyze/lite` returns verdict + probability only, skipping the full feature breakdown
- **Rate limiting** — 60 req/min per IP via in-process sliding window, no Redis needed
- **Auto model build** — if `classifier.pkl` is missing on startup, the server trains it automatically from the Brown corpus
- **Dark / Light theme** — full CSS custom property theming, respects system preference

---

## 🤖 Model

| Property | Detail |
|---|---|
| Algorithm | LinearSVC (via `CalibratedClassifierCV`) — selected by 5-fold CV over LR, RF, LinearSVC |
| Training data | 450 AI samples + 450 human samples (NLTK Brown corpus) |
| Features | 11 statistical features (see below) |
| Split | 80/20 stratified train/test |
| Evaluation | Test accuracy + AUC reported at `/model-info` |

**The 11 features:**

| # | Feature | AI signal |
|---|---|---|
| 1 | Perplexity | AI text is more predictable vs. reference English bigram LM |
| 2 | Burstiness | Humans vary sentence complexity more than LLMs |
| 3 | Type-token ratio | Lexical diversity — AI tends toward repetition |
| 4 | Avg sentence length | AI sentences cluster around a mean |
| 5 | Sentence length variance | Low variance → AI |
| 6 | Avg word length | AI prefers slightly longer, formal words |
| 7 | Stopword ratio | Functional word distribution differs by source |
| 8 | Punctuation density | AI uses punctuation more uniformly |
| 9 | Repetition score | Fraction of repeated word bigrams |
| 10 | POS entropy | Grammatical tag diversity |
| 11 | Readability (Flesch) | AI text clusters in the 30–60 range |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla HTML/CSS/JS — no framework, no build step |
| Backend | FastAPI 0.110+, Python 3.11 |
| ML | scikit-learn, NLTK, NumPy |
| Reference LM | Custom bigram language model (Brown corpus) |
| Container | Docker |
| Deployment | Vercel (frontend) + Render (backend) |

---

## 📁 Project Structure

```
ai-content-detector/
├── index.html                  # Full frontend — single file, no build step
├── render.yaml                 # Render deploy config (build + start commands)
├── Dockerfile                  # Root-level Docker (for local full-stack)
└── backend/
    ├── app.py                  # FastAPI app — all endpoints, middleware, rate limiting
    ├── inference.py            # Document + sentence scoring pipeline
    ├── features.py             # 11 feature extractors
    ├── reference_lm.py         # Bigram LM — build, save, load, perplexity
    ├── train.py                # Dataset build + model selection + training
    ├── requirements.txt
    ├── Dockerfile              # Backend-only container
    ├── model/
    │   ├── classifier.pkl      # Trained model bundle (scaler + LinearSVC + metadata)
    │   └── reference_lm.pkl   # Serialized bigram LM
    └── data/
        ├── ai_samples.py       # AI text generation for training data
        ├── build_dataset.py    # Dataset assembly script
        └── data.csv            # Raw training data
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Or Docker

### Run locally

```bash
git clone https://github.com/sugumaran-nix/ai-content-detector.git
cd ai-content-detector/backend
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

On first startup, if `model/classifier.pkl` is missing, the server auto-builds it (~1–2 min).

Open `index.html` in your browser — the backend URL is pre-set to the live Render instance. To use your local backend instead, update line 1508 in `index.html`:

```js
let API_URL = "http://localhost:8000";
```

### Run with Docker

```bash
cd backend
docker build -t ai-content-detector .
docker run -p 8000:8000 ai-content-detector
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness + model status |
| `GET` | `/model-info` | Classifier metadata, accuracy, AUC, feature names |
| `POST` | `/analyze` | Full analysis — verdict, features, sentence breakdown |
| `POST` | `/analyze/lite` | Verdict + probability only (fast path) |
| `POST` | `/batch` | Analyze up to 10 texts in one request |

**Example:**

```bash
curl -X POST https://ai-content-detector-hodw.onrender.com/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here — at least 8 words."}'
```

**Response:**
```json
{
  "label": "likely_ai",
  "ai_probability": 0.82,
  "confidence": 0.64,
  "features": { "perplexity": 312.4, "burstiness": 18.2, ... },
  "sentences": [
    { "text": "Your text here.", "ai_probability": 0.79 }
  ],
  "model_name": "LinearSVC",
  "n_words": 120,
  "n_sentences": 6
}
```

**Rate limit:** 60 POST requests / 60 seconds per IP. Returns `429` with `Retry-After` header when exceeded.

---

## 🌐 Deployment

### Backend — Render

`render.yaml` handles everything — NLTK downloads, LM build, model training, and server start. Just connect the repo on [render.com](https://render.com).

Set this env var on Render:

| Key | Value |
|---|---|
| `ALLOWED_ORIGINS` | `https://ai-content-detector-nine.vercel.app` |

### Frontend — Vercel

Drop the repo on [vercel.com](https://vercel.com) — no build config needed, `index.html` is served as a static file directly.

---

## 📄 License

MIT
