"""Backend API regression tests."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as app_module
import inference


LONG_TEXT = "This is a sufficiently long sample text for analysis. " * 3
REPORT_TEXT = "Cygni system, possibly an extrasolar planet. The planet is distinctive for the interaction of its strong gravity with the centrifugal force due to its fast rotation, giving it a gradient in the perceived force of gravity from 3 g on the equator to 665 g on the planet's poles (diagram pictured). It is inhabited by native lifeforms, including an intelligent centipede-like species, the Mesklinities. Mesklin is considered an example of hard science fiction worldbuilding, an exotic milieu that accords with known facts and laws of physics. The planet is dissimilar to Earth, its inhabitants are commonly regarded as humanlike in behaviour."


def test_analyze_request_strips_and_rejects_blank_text():
    assert app_module.AnalyzeRequest(text="  hello world  ").text == "hello world"
    with pytest.raises(ValidationError, match="cannot be blank"):
        app_module.AnalyzeRequest(text="   ")


def test_batch_request_rejects_blank_non_string_and_oversized_items():
    assert app_module.BatchRequest(texts=[" first ", "second"]).texts == ["first", "second"]
    with pytest.raises(ValidationError, match="index 1"):
        app_module.BatchRequest(texts=["valid", "   "])
    with pytest.raises(ValidationError, match="must be a string"):
        app_module.BatchRequest(texts=["valid", 42])
    with pytest.raises(ValidationError, match="character limit"):
        app_module.BatchRequest(texts=["x" * (app_module.MAX_CHARS + 1)])


def test_analyze_endpoint_returns_request_metadata_and_prediction(monkeypatch):
    monkeypatch.setattr(app_module, "_ensure_models", lambda: None)
    monkeypatch.setattr(app_module.inference, "get_bundle", lambda: {"model_name": "test-model"})
    monkeypatch.setattr(
        app_module.inference,
        "predict_document",
        lambda text: {
            "label": "likely_human",
            "ai_probability": 0.12,
            "confidence": 0.76,
            "features": {"readability": 62.4},
            "sentences": [{"text": text, "ai_probability": 0.12}],
            "model_name": "test-model",
            "n_words": len(text.split()),
            "n_sentences": 1,
        },
    )

    with TestClient(app_module.app) as client:
        response = client.post("/analyze", headers={"X-Request-ID": "test-request"}, json={"text": LONG_TEXT})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request"
    assert "X-Process-Time" in response.headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Permissions-Policy"] == "camera=(), microphone=(), geolocation=()"
    assert response.json()["label"] == "likely_human"


def test_invalid_request_id_is_replaced(monkeypatch):
    monkeypatch.setattr(app_module, "_ensure_models", lambda: None)
    monkeypatch.setattr(app_module.inference, "get_bundle", lambda: {"model_name": "test-model"})
    monkeypatch.setattr(app_module.inference, "predict_document", lambda text: {
        "label": "mixed", "ai_probability": 0.5, "confidence": 0.0,
        "features": {}, "sentences": [], "model_name": "test-model",
        "n_words": len(text.split()), "n_sentences": 0,
    })

    with TestClient(app_module.app) as client:
        response = client.post("/analyze", headers={"X-Request-ID": "<script>"}, json={"text": LONG_TEXT})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] != "<script>"
    assert len(response.headers["X-Request-ID"]) == 8


def test_inference_errors_do_not_leak_exception_details(monkeypatch):
    monkeypatch.setattr(app_module, "_ensure_models", lambda: None)
    monkeypatch.setattr(app_module.inference, "get_bundle", lambda: {"model_name": "test-model"})
    monkeypatch.setattr(app_module.inference, "predict_document", lambda text: (_ for _ in ()).throw(RuntimeError("secret backend path")))

    with TestClient(app_module.app) as client:
        response = client.post("/analyze", json={"text": LONG_TEXT})

    assert response.status_code == 500
    assert response.json()["detail"] == "Inference failed."
    assert "secret backend path" not in response.text


def test_analyze_endpoint_rejects_blank_and_too_short_text(monkeypatch):
    monkeypatch.setattr(app_module, "_ensure_models", lambda: None)
    monkeypatch.setattr(app_module.inference, "get_bundle", lambda: {"model_name": "test-model"})

    with TestClient(app_module.app) as client:
        blank = client.post("/analyze", json={"text": "   "})
        short = client.post("/analyze", json={"text": "one two"})

    assert blank.status_code == 422
    assert short.status_code == 422


def test_reported_factual_passage_is_not_overconfidently_human():
    result = inference.predict_document(REPORT_TEXT)
    assert result["label"] == "mixed"
    assert 0.30 < result["ai_probability"] < 0.70
    assert result["confidence"] < 0.10


def test_probability_thresholds_match_public_verdict_bands():
    assert inference.label_for_probability(0.70) == "likely_ai"
    assert inference.label_for_probability(0.30) == "likely_human"
    assert inference.label_for_probability(0.50) == "mixed"


def test_batch_errors_do_not_leak_exception_details(monkeypatch):
    monkeypatch.setattr(app_module, "_ensure_models", lambda: None)
    monkeypatch.setattr(app_module.inference, "get_bundle", lambda: {"model_name": "test-model"})
    monkeypatch.setattr(app_module.inference, "predict_document", lambda text: (_ for _ in ()).throw(RuntimeError("private stack path")))

    with TestClient(app_module.app) as client:
        response = client.post("/batch", json={"texts": [LONG_TEXT]})

    assert response.status_code == 200
    assert response.json()[0]["error"] == "Inference failed for this item."
    assert "private stack path" not in response.text


def test_health_reports_unavailable_without_leaking_internal_error(monkeypatch):
    def fail_bundle():
        raise FileNotFoundError("internal model path")

    monkeypatch.setattr(app_module.inference, "get_bundle", fail_bundle)
    response = app_module.health()

    assert response.status_code == 503
    assert response.body
    assert b"Model service is not ready." in response.body
    assert b"internal model path" not in response.body
