---
title: AI Content Detector
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
---

# AI-Generated Text Detector — API

FastAPI backend for the AI-Generated Text Detector. See the main project
README (in the GitHub repo root, one level up) for full methodology,
architecture, and limitations.

## Endpoints
- `GET /health`
- `GET /model-info`
- `POST /analyze` — `{ "text": "..." }`
