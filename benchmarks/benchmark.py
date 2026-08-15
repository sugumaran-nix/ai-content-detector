from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from inference import predict_document

SAMPLES = {
    "ai": {
        "label": "likely_ai",
        "text": "Large language models have revolutionized the field of natural language processing in recent years. These sophisticated systems leverage transformer architectures to generate coherent and contextually relevant text across a wide variety of domains. The implications for various industries are profound and far-reaching. Organizations are increasingly adopting these technologies to enhance productivity and streamline their operations. Furthermore, the potential applications continue to expand as ongoing research progresses toward more capable and efficient systems. It is worth noting that while these models demonstrate impressive capabilities, they also present significant challenges related to alignment, interpretability, and responsible deployment.",
    },
    "human": {
        "label": "likely_human",
        "text": "I've been trying to fix this bug for three days and honestly I'm losing my mind. The weird part is it only happens on Tuesdays? Or at least that's what the logs say. My coworker thinks I'm joking but I pulled the timestamps — yeah, Tuesdays. Something to do with the weekly batch job that runs Monday night I think. Anyway I added extra logging and I think I'm getting close. Will update tomorrow if I figure it out. Also the coffee machine is broken again which is not helping my mental state at all.",
    },
    "mixed": {
        "label": "mixed",
        "text": "Large language models have revolutionized the field of natural language processing in recent years. These sophisticated systems leverage transformer architectures to generate coherent and contextually relevant text across a wide variety of domains. The implications for various industries are profound and far-reaching. Organizations are increasingly adopting these technologies to enhance productivity and streamline their operations. Furthermore, the potential applications continue to expand as ongoing research progresses toward more capable and efficient systems. It is worth noting that while these models demonstrate impressive capabilities, they also present significant challenges related to alignment, interpretability, and responsible deployment. I've been trying to fix this bug for three days and honestly I'm losing my mind. The weird part is it only happens on Tuesdays? Or at least that's what the logs say. My coworker thinks I'm joking but I pulled the timestamps — yeah, Tuesdays. Something to do with the weekly batch job that runs Monday night I think. Anyway I added extra logging and I think I'm getting close. Will update tomorrow if I figure it out. Also the coffee machine is broken again which is not helping my mental state at all.",
    },
}

WARMUPS = 3
ITERATIONS = 30
results = []
for name, sample in SAMPLES.items():
    for _ in range(WARMUPS):
        predict_document(sample["text"])
    timings = []
    predictions = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        result = predict_document(sample["text"])
        timings.append((time.perf_counter() - start) * 1000)
        predictions.append(result)
    result = predictions[-1]
    results.append({
        "sample": name,
        "expected_label": sample["label"],
        "predicted_label": result["label"],
        "ai_probability": result["ai_probability"],
        "correct": result["label"] == sample["label"],
        "median_ms": round(statistics.median(timings), 3),
        "p95_ms": round(sorted(timings)[int(ITERATIONS * 0.95) - 1], 3),
        "min_ms": round(min(timings), 3),
        "max_ms": round(max(timings), 3),
    })

summary = {
    "warmups": WARMUPS,
    "iterations_per_sample": ITERATIONS,
    "accuracy": round(sum(r["correct"] for r in results) / len(results), 4),
    "median_ms": round(statistics.median(r["median_ms"] for r in results), 3),
    "p95_ms": round(max(r["p95_ms"] for r in results), 3),
    "results": results,
}
output = ROOT / "benchmarks" / "local_results.json"
output.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
