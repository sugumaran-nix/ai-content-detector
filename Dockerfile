# ── Hugging Face Spaces / Docker deployment ──────────────────────────────────
# Dockerfile lives at repo root. HF Spaces Docker SDK picks it up automatically.
# Build context is the repo root; all paths below are relative to it.
FROM python:3.11-slim

WORKDIR /app

# System deps: gcc needed for some wheel builds
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Install Python deps from backend/requirements.txt
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire backend source
COPY backend/ .

# Pre-download NLTK corpora at build time → no runtime network dependency
ENV NLTK_DATA=/app/nltk_data
RUN python -c "import nltk; nltk.data.path.insert(0, '/app/nltk_data'); \
    [nltk.download(p, quiet=True) for p in \
    ['brown','punkt','punkt_tab','stopwords','averaged_perceptron_tagger','averaged_perceptron_tagger_eng']]"

# Build reference LM + train classifier — image ships ready-to-serve
RUN python reference_lm.py && python train.py
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

# HF Spaces Docker SDK default port; Render uses $PORT (overridden at runtime)
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
