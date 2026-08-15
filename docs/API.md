# AI Content Detector API

The backend is a FastAPI service that estimates whether a text passage is likely human-written, AI-generated, or mixed. It exposes interactive OpenAPI documentation at `/docs`, a ReDoc view at `/redoc`, and the machine-readable schema at `/openapi.json`. FastAPI generates these interfaces from the declared request and response models.[1]

## Base URL and protocol

Use the deployed HTTPS origin as the base URL. For local development, the default command is:

```bash
cd backend
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

The API accepts and returns JSON. Clients should send `Content-Type: application/json` for POST requests. The service does not require authentication in the current deployment; place it behind an authenticated gateway before exposing it to private or high-value workloads.

## Common request and response behavior

Every response includes the following operational headers:

| Header | Meaning |
|---|---|
| `X-Request-ID` | Correlation identifier. A valid client value containing only letters, digits, `.`, `_`, `:`, or `-` and no more than 64 characters is preserved; otherwise the server generates an eight-character identifier. |
| `X-Process-Time` | Server-side processing time, formatted as milliseconds. |
| `X-Content-Type-Options` | Always `nosniff`. |
| `X-Frame-Options` | Always `DENY`. |
| `Referrer-Policy` | Always `no-referrer`. |
| `Permissions-Policy` | Disables camera, microphone, and geolocation access. |

The service enforces a configurable sliding-window rate limit on POST requests. Defaults are **60 requests per 60 seconds per client IP**, with at most `MAX_RATE_KEYS` client keys retained in memory. A rejected request returns `429 Too Many Requests` with a `Retry-After` header.

The default maximum input is **8,000 characters** and the default minimum is **8 whitespace-separated words**. Configure these values with `MAX_CHARS` and `MIN_WORDS`. Batch requests default to a maximum of **10 items**, controlled by `MAX_BATCH`.

## `GET /health`

Returns model readiness information for deployment health checks.

### Successful response: `200 OK`

```json
{
  "status": "ok",
  "model": "LinearSVC",
  "lm_loaded": true,
  "version": "2.0.0"
}
```

### Unavailable response: `503 Service Unavailable`

```json
{
  "status": "unavailable",
  "detail": "Model service is not ready."
}
```

The endpoint intentionally does not expose internal exception messages or filesystem paths.

## `GET /model-info`

Returns classifier metadata and the feature order used by the trained model.

```json
{
  "model_name": "LinearSVC",
  "test_accuracy": 0.994,
  "test_auc": 1.0,
  "feature_names": [
    "perplexity",
    "burstiness",
    "type_token_ratio",
    "avg_sentence_length",
    "sentence_length_variance",
    "avg_word_length",
    "stopword_ratio",
    "punctuation_density",
    "repetition_score",
    "pos_entropy",
    "readability"
  ],
  "n_classes": 2,
  "labels": ["likely_human", "mixed", "likely_ai"],
  "thresholds": {
    "likely_ai": 0.7,
    "likely_human": 0.3
  }
}
```

Treat the reported test metrics as model metadata, not as a guarantee of real-world accuracy.

## `POST /analyze`

Performs full document analysis, including the overall verdict, probability, confidence, raw feature values, and sentence-level scores.

### Request body

```json
{
  "text": "Paste a document with at least eight words here. Longer passages generally provide steadier statistical signals."
}
```

The `text` field is trimmed before validation. Blank strings, non-string values, inputs over `MAX_CHARS`, or inputs below `MIN_WORDS` are rejected.

### Successful response: `200 OK`

```json
{
  "label": "likely_human",
  "ai_probability": 0.18,
  "confidence": 0.64,
  "features": {
    "perplexity": 183.42,
    "burstiness": 44.11,
    "type_token_ratio": 0.82,
    "avg_sentence_length": 16.5,
    "sentence_length_variance": 5.2,
    "avg_word_length": 4.7,
    "stopword_ratio": 0.41,
    "punctuation_density": 0.02,
    "repetition_score": 0.03,
    "pos_entropy": 2.84,
    "readability": 61.8
  },
  "sentences": [
    {
      "text": "Paste a document with at least eight words here.",
      "ai_probability": 0.18
    }
  ],
  "model_name": "LinearSVC",
  "n_words": 20,
  "n_sentences": 2
}
```

Sentence probabilities are `null` when a sentence is too short to score reliably or when sentence-level extraction fails. The document-level result remains available.

## `POST /analyze/lite`

Returns only the verdict fields needed by lightweight embeds or list views.

### Request

```bash
curl -sS "$BASE_URL/analyze/lite" \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: web-demo-001' \
  -d '{"text":"This is a sufficiently long passage for the detector to evaluate its statistical characteristics."}'
```

### Response

```json
{
  "label": "mixed",
  "ai_probability": 0.51,
  "confidence": 0.02,
  "model_name": "LinearSVC"
}
```

## `POST /batch`

Analyzes up to `MAX_BATCH` texts in one request. Each item is returned in its original order. A short item produces an item-level error rather than failing the entire batch.

### Request

```json
{
  "texts": [
    "This first passage contains enough words to be evaluated by the API.",
    "This second passage is also long enough for a result."
  ]
}
```

### Response

```json
[
  {
    "index": 0,
    "label": "likely_human",
    "ai_probability": 0.22,
    "confidence": 0.56,
    "error": null
  },
  {
    "index": 1,
    "label": "error",
    "ai_probability": 0.0,
    "confidence": 0.0,
    "error": "Too short (3 words, need 8+)."
  }
]
```

Batch validation rejects blank, non-string, oversized, or more-than-maximum entries with `422 Unprocessable Entity`.

## Error contract

| Status | Meaning | Typical body |
|---|---|---|
| `200` | Request completed successfully. | Endpoint-specific response. |
| `422` | JSON shape or input validation failed. | FastAPI validation details or a minimum-word message. |
| `429` | Rate limit exceeded. | `{"detail":"Rate limit exceeded ..."}` plus `Retry-After`. |
| `500` | Unexpected inference failure. | `{"detail":"Inference failed."}` without internal exception details. |
| `503` | Model service is not ready. | `{"detail":"Model service is not ready."}`. |

## CORS

Set `ALLOWED_ORIGINS` to a comma-separated list of exact browser origins, for example:

```bash
ALLOWED_ORIGINS=https://ai-content-detector-nine.vercel.app,http://localhost:5173
```

The repository’s deployment configuration uses the production frontend origin. Avoid `*` in production. FastAPI’s CORS middleware requires explicit origin configuration when a browser client needs controlled cross-origin access.[2]

## Client examples

### JavaScript

```js
const response = await fetch(`${API_BASE}/analyze/lite`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-Request-ID": crypto.randomUUID(),
  },
  body: JSON.stringify({ text }),
});

if (!response.ok) {
  const error = await response.json().catch(() => ({}));
  throw new Error(error.detail || `Request failed with ${response.status}`);
}

const result = await response.json();
```

### Python

```python
import requests

response = requests.post(
    f"{api_base}/analyze",
    json={"text": text},
    headers={"X-Request-ID": "python-client-001"},
    timeout=30,
)
response.raise_for_status()
result = response.json()
```

## References

[1]: https://fastapi.tiangolo.com/tutorial/metadata/ "FastAPI metadata and generated documentation"

[2]: https://fastapi.tiangolo.com/tutorial/cors/ "FastAPI CORS configuration"
