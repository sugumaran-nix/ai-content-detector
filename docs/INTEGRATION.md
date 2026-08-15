# AI Content Detector Backend Integration Guide

This guide explains how to run, integrate, deploy, and operate the updated FastAPI backend. The API is stateless at the application layer: callers submit text, receive a result, and correlate requests with `X-Request-ID`. The current service has no authentication, user accounts, or server-side persistence.

## Architecture

The backend runs as a FastAPI application under Uvicorn. At startup it ensures the reference language model and classifier bundle exist, loads the classifier into process memory, and exposes readiness through `/health`. Text is transformed into eleven statistical features before the trained classifier produces an AI probability. The `/analyze` endpoint adds sentence-level analysis; `/analyze/lite` returns only the verdict fields; `/batch` repeats document-level inference for multiple inputs.

| Component | Responsibility |
|---|---|
| `backend/app.py` | HTTP API, validation, CORS, rate limiting, request metadata, health checks, and response models. |
| `backend/inference.py` | Classifier bundle loading, feature extraction orchestration, document and sentence scoring. |
| `backend/features.py` | NLTK tokenization, statistical feature engineering, and readability calculations. |
| `backend/model/` | Serialized classifier and reference language model artifacts. These files must be treated as trusted build outputs. |
| `render.yaml` | Render deployment configuration. |
| `Dockerfile` | Container build for Hugging Face Spaces or compatible Docker platforms. |

## Local setup

Use Python 3.11 or newer. Create an isolated environment and install the backend requirements:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

Run the service from the `backend` directory so imports resolve correctly:

```bash
cd backend
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Verify readiness and inspect the generated documentation:

```bash
curl -i http://127.0.0.1:8000/health
open http://127.0.0.1:8000/docs
```

The first startup can build or load model artifacts. In container deployments, the Dockerfile performs the build during image creation so runtime startup does not need to download corpora or train the classifier.

## Frontend integration

The existing static frontend can call the backend from a browser when its deployed origin appears in `ALLOWED_ORIGINS`. Keep the API base URL in deployment configuration rather than embedding environment-specific URLs throughout the code.

```js
const API_BASE = window.__API_BASE__ || "http://localhost:8000";

async function analyzeText(text) {
  const response = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Request-ID": crypto.randomUUID(),
    },
    body: JSON.stringify({ text }),
  });

  const requestId = response.headers.get("X-Request-ID");
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.detail || `API request failed (${response.status})`);
    error.requestId = requestId;
    error.status = response.status;
    throw error;
  }
  return { ...payload, requestId };
}
```

A production frontend should display a friendly error message, log the request ID for support diagnostics, and avoid rendering raw server error strings into HTML. The backend already removes internal exception details from `500` and `503` responses.

## CORS configuration

Configure exact origins as a comma-separated environment variable:

```bash
ALLOWED_ORIGINS=https://ai-content-detector-nine.vercel.app,http://localhost:5173
```

The service now defaults to exact local development origins. Do not use `ALLOWED_ORIGINS=*` for a public production deployment; production startup rejects the wildcard and requires explicit origins. If a reverse proxy terminates TLS, ensure it forwards the browser’s `Origin` header and preserves the API response headers.

## Environment variables

| Variable | Default | Purpose |
|---|---:|---|
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000,http://localhost:4173,http://127.0.0.1:4173` | Exact origins allowed by CORS. Production must set explicit origins. |
| `MAX_CHARS` | `8000` | Maximum characters per text input. |
| `MIN_WORDS` | `8` | Minimum whitespace-separated words for document analysis. |
| `RATE_LIMIT` | `60` | POST requests allowed per client IP per window. |
| `RATE_WINDOW` | `60` | Rate-limit window in seconds. |
| `MAX_BATCH` | `10` | Maximum number of texts in a batch. |
| `MAX_RATE_KEYS` | `10000` | Maximum in-memory client keys retained by the rate limiter. |
| `NLTK_DATA` | platform-dependent | Optional NLTK corpus directory. |
| `PORT` | platform-provided | Runtime port used by Render-style platforms. |

The in-process rate limiter is appropriate for a single worker or small free-tier deployment. Its in-memory key store is bounded by `MAX_RATE_KEYS`. For multiple workers or multiple replicas, place a shared rate limiter at the gateway or use a shared store; otherwise each process maintains an independent counter.

## Deployment with Docker

Build and run the repository’s root container:

```bash
docker build -t ai-content-detector .
docker run --rm \
  -p 7860:7860 \
  -e ALLOWED_ORIGINS=https://your-frontend.example \
  -e RATE_LIMIT=60 \
  ai-content-detector
```

The image downloads the required NLTK data and builds model artifacts during the image build. Do not use untrusted pickle files or replace the model artifacts at runtime. Python pickle deserialization can execute arbitrary code if the input is attacker-controlled; the repository’s serialized files must therefore come only from the trusted build pipeline.

## Deployment with Render

The checked-in `render.yaml` defines a Python web service rooted at `backend`. It installs dependencies, downloads NLTK data into the configured directory, builds the reference language model and classifier, starts Uvicorn with one worker, and uses `/health` as its health-check path.

Before deployment, review these settings:

```yaml
ALLOWED_ORIGINS: https://your-frontend.example
RATE_LIMIT: "60"
RATE_WINDOW: "60"
MAX_BATCH: "10"
```

Set secrets and environment-specific values in the hosting provider’s secret/configuration UI rather than committing them to Git. The current backend does not need a secret key for its public unauthenticated API; if authentication is added later, store signing keys only in the provider’s secret manager.

## Monitoring and operations

Use `/health` for liveness/readiness checks. Record `X-Request-ID` and `X-Process-Time` in edge or application logs. A sustained increase in `X-Process-Time`, `429` responses, or `503` responses should trigger investigation. Do not log submitted text by default because it may contain private documents.

The generated OpenAPI schema is available at `/openapi.json`. Keep `docs/API.md` synchronized with model and endpoint changes, and regenerate client SDKs from the OpenAPI schema when the response models change. FastAPI documents the generated schema and interactive interfaces automatically.[1]

## Troubleshooting

| Symptom | Likely cause | Resolution |
|---|---|---|
| `503 Model service is not ready.` | Model artifacts are absent or could not be loaded. | Review startup logs, verify `backend/model/`, and rebuild the image or run the training/build steps. |
| Browser CORS error | Frontend origin is not in `ALLOWED_ORIGINS`. | Add the exact scheme, host, and port; restart the service. |
| `422 Text too short` | Input contains fewer than `MIN_WORDS` words. | Send a longer passage or lower the environment value deliberately. |
| `429 Rate limit exceeded` | Client exceeded the POST window. | Respect `Retry-After`, reduce concurrency, or adjust limits at the deployment boundary. |
| Slow first request | NLTK/model initialization or cold start. | Prebuild artifacts and corpora in the container; keep one warm instance where practical. |
| Different limits across replicas | In-process limiter is local to each worker. | Enforce a shared limit at the reverse proxy or gateway. |

## References

[1]: https://fastapi.tiangolo.com/tutorial/metadata/ "FastAPI metadata and generated documentation"
