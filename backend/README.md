---
title: AI Content Detector
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# AI-Generated Text Detector — API

FastAPI backend for the AI-Generated Text Detector.

## Endpoints
- `GET /health`
- `GET /model-info`
- `POST /analyze` — `{ "text": "..." }`