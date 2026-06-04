import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import main
from fastapi.testclient import TestClient

def test_health():
    client = TestClient(main.app)
    response = client.get("/")
    assert response.status_code == 200

def test_query_returns_structured_response(monkeypatch):
    monkeypatch.setattr(main, "_indexes_loaded", lambda: True)
    monkeypatch.setattr(main, "load_indexes", lambda: None)
    monkeypatch.setattr(main, "hybrid_retrieve", lambda question, top_k: [
        {"chunk": "Paris is in France.", "source": "doc1.txt", "rrf_score": 1.0, "ce_score": 0.9}
    ])
    monkeypatch.setattr(main, "run_rag_agent", lambda question, results, history: {
        "answer": "Paris is in France.",
        "validation": "PASS",
        "validation_score": 97,
        "retries_used": 0,
        "confidence": 97,
        "status": "success",
        "fail_reason": "",
        "best_validation_score": 97,
    })

    with TestClient(main.app) as client:
        response = client.post("/query", json={"question": "Where is Paris?", "session_id": "t1"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Paris is in France."
    assert body["validation"] == "PASS"
    assert body["validation_score"] == 97
    assert body["confidence"] == 97
    assert body["status"] == "success"

def test_query_stream_returns_sse(monkeypatch):
    monkeypatch.setattr(main, "_indexes_loaded", lambda: True)
    monkeypatch.setattr(main, "load_indexes", lambda: None)
    monkeypatch.setattr(main, "hybrid_retrieve", lambda question, top_k: [
        {"chunk": "Paris is in France.", "source": "doc1.txt", "rrf_score": 1.0, "ce_score": 0.9}
    ])
    monkeypatch.setattr(main, "run_rag_agent", lambda question, results, history: {
        "answer": "Paris is in France.",
        "validation": "PASS",
        "validation_score": 97,
        "retries_used": 0,
        "confidence": 97,
        "status": "success",
        "fail_reason": "",
        "best_validation_score": 97,
    })

    with TestClient(main.app) as client:
        with client.stream("POST", "/query/stream", json={"question": "Where is Paris?", "session_id": "t1"}) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: start" in body
    assert "event: chunk" in body
    assert "event: final" in body
