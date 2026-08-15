# Inference Benchmark Report

## Scope and method

The benchmark compares the current local classifier path with the previous frontend path that dynamically imported Transformers.js and downloaded a roughly 40 MB remote model on first use. The local benchmark uses the repository’s three built-in samples—AI-written, human-written, and half-AI/half-human mixed text—with three warm-up calls and thirty timed inference calls per sample. Accuracy is measured against those declared sample labels, not against a production evaluation corpus.

The benchmark runner is reproducible with:

```bash
python3 benchmarks/benchmark.py
```

It writes the machine-readable result to `benchmarks/local_results.json`.

## Results

| Path | Cold-start behavior | Warm inference | Sample accuracy | Notes |
|---|---:|---:|---:|---|
| Current local classifier | Local `model.json` and `lm.json` assets load without a remote transformer dependency. | Median **52.8 ms** across samples; worst-sample p95 **99.6 ms**. | **100% (3/3)** | Backend thresholds now use a 0.70 AI / 0.30 human boundary, leaving the middle band as mixed. |
| Previous remote-transformer setup | The old page was opened from the pre-local-model commit and remained on “Loading language model…” during the browser observation window, more than six seconds after navigation. | Not measurable because the remote pipeline never reached a ready state in the benchmark environment. | Not measurable | The comparison is reported as unavailable rather than inventing a remote score. The old path depended on an external CDN/model download and could not complete reproducibly here. |

The local sample-level results were:

| Sample | Expected | Predicted | AI probability | Median | p95 |
|---|---|---|---:|---:|---:|
| AI-written | `likely_ai` | `likely_ai` | 0.8962 | 43.1 ms | 44.3 ms |
| Human-written | `likely_human` | `likely_human` | 0.0719 | 52.8 ms | 56.2 ms |
| Half-AI/half-human | `mixed` | `mixed` | 0.6398 | 92.4 ms | 99.6 ms |

## Interpretation

The local path is the only path that completed end-to-end in the benchmark environment. It avoids a large remote transformer download and produces deterministic, explainable feature-based results from repository-shipped assets. The measured latency is warm-process inference latency; it excludes the first process startup and model artifact loading time. The three-sample accuracy is a smoke-test signal, not a claim of real-world detector accuracy. The application’s own documentation correctly warns that training-split metrics do not generalize to all writing styles, short text, paraphrases, translations, or non-native English.

The backend threshold adjustment was necessary for contract consistency: the mixed sample produced a calibrated probability of 0.6398, which is no longer treated as `likely_ai` because the public API now reserves probabilities from 0.30 through 0.70 for the `mixed` band. The browser’s verified mixed sample renders at 50% in the local UI.

## Security verification attached to this benchmark

The same verification run includes backend tests, frontend tests, Python and JavaScript syntax checks, `pip-audit`, Bandit, and `git diff --check`. The remaining pickle and deterministic-randomness findings are documented as trusted build-artifact or non-cryptographic data-generation exceptions rather than runtime user-input vulnerabilities. The production CORS guard now rejects wildcard or empty origins, and batch inference errors return a stable generic message instead of internal exception text.
