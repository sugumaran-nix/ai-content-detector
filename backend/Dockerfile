FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal — this is a pure-Python/CPU stack (no torch/transformers).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-fetch NLTK corpora at build time so the container has no runtime
# network dependency, then build the dataset + train the model so the
# image ships ready-to-serve.
RUN python -c "import nltk; [nltk.download(p, quiet=True) for p in ['brown','punkt','punkt_tab','stopwords','averaged_perceptron_tagger','averaged_perceptron_tagger_eng']]" \
    && python data/build_dataset.py \
    && python -c "from reference_lm import build_reference_lm, save_reference_lm; save_reference_lm(build_reference_lm())" \
    && python train.py

# Hugging Face Spaces (Docker SDK) expects the app on port 7860 by default.
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
