# AI-Generated Text Detector

Paste any text — get a verdict in seconds. DistilBERT fine-tuned on HC3, running entirely in your browser. Sentence-level highlighting and 11 supplementary statistical signals explain exactly why.

![DistilBERT](https://img.shields.io/badge/DistilBERT-ONNX-FF6B6B?style=for-the-badge&logo=huggingface&logoColor=white)
![Transformers.js](https://img.shields.io/badge/Transformers.js-2.17-FFD21E?style=for-the-badge&logo=javascript&logoColor=black)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![Vercel](https://img.shields.io/badge/Frontend-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Model-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

**[🚀 Live Demo](https://ai-content-detector-nine.vercel.app)** · **[Model on HuggingFace](https://huggingface.co/Sugum4r4n/ai-content-detector-bert)**

---

## ✨ Features

- **DistilBERT inference in the browser** — ONNX quantized model runs via Transformers.js, no server involved
- **94.1% accuracy, 0.991 AUC** — fine-tuned on HC3 (real ChatGPT vs real human answers from Reddit/StackExchange)
- **11 supplementary statistical signals** — perplexity, burstiness, type-token ratio, sentence length variance, POS entropy, readability (Flesch), stopword ratio, punctuation density, repetition score, avg word/sentence length
- **Sentence-level breakdown** — every sentence scored and highlighted individually by BERT
- **Three-tier verdict** — likely_ai (≥60%), mixed (40–60%), likely_human (≤40%) with confidence score
- **Zero data collection** — model downloads once and caches in your browser, text never touches a server
- **Dark / Light theme** — full CSS custom property theming, respects system preference

---

## 🤖 Model

| Property | Detail |
|---|---|
| Base model | distilbert-base-uncased |
| Fine-tuned on | ai-text-detection-pile (5,000 human + 5,000 AI samples) |
| Training accuracy | 94.15% |
| ROC AUC | 0.9913 |
| Export format | ONNX INT8 quantized via Optimum |
| Inference | Browser-side via @xenova/transformers@2.17.2 |
| Hosted | Sugum4r4n/ai-content-detector-bert on HuggingFace Hub |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla HTML/CSS/JS — no framework, no build step |
| ML inference | Transformers.js + ONNX Runtime Web |
| Model training | PyTorch, HuggingFace Transformers, Optimum |
| Training data | ai-text-detection-pile |
| Model hosting | HuggingFace Hub |
| Deployment | Vercel (static) |

---

## 📁 Project Structure

```
ai-content-detector/
├── index.html                  # Full frontend — single file, no build step
├── render.yaml                 # Legacy backend config (unused)
├── .gitignore
└── backend/                    # Training pipeline (not deployed)
    ├── train.py                # Fine-tune DistilBERT on HC3
    ├── reference_lm.py         # Bigram LM for perplexity signal
    ├── export_lm_json.py       # Export reference_lm.pkl → lm.json
    ├── export_weights.py       # Export classifier.pkl → model.json
    ├── features.py             # 11 statistical feature extractors
    ├── inference.py            # Scoring pipeline
    ├── app.py                  # FastAPI app (legacy, unused)
    ├── requirements.txt
    └── model/
        ├── classifier.pkl      # Legacy LinearSVC (unused)
        └── reference_lm.pkl   # Bigram LM for perplexity display
```

---

## 🚀 How it works

1. Page loads — Transformers.js downloads the quantized DistilBERT ONNX model from HuggingFace (~65 MB, cached after first load)
2. You paste text — 11 statistical signals are extracted locally in JS
3. BERT runs — full text and each sentence scored by DistilBERT entirely in your browser
4. Verdict rendered — three-tier label, confidence bar, sentence highlighting, signal grid

No API calls. No backend. No tracking.

---

## 🌐 Deployment

Drop the repo on vercel.com — no build config needed. index.html is served as a static file. The ONNX model is fetched at runtime from HuggingFace. No environment variables needed.

---

## 📄 License

MIT
