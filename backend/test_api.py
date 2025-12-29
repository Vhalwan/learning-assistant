import json
from fastapi.testclient import TestClient
import pytest

from backend.api import app

client = TestClient(app)


def test_query_endpoint_monkeypatch(monkeypatch):
    """
    Monkeypatch rag_answer_from_embeddings to avoid external LLM calls and test the API response shape.
    """
    def fake_rag(question, embeddings_path, top_k=3, use_faiss=False, faiss_index_path=None,
                 use_safe=None, use_query_expansion=False, return_meta=False, llm_call=None, embed_call=None):
        # return the rich tuple (answer, retrieved_chunks, prompt, provenance)
        answer = f"Fake answer to: {question}"
        retrieved = [
            {"score": 0.9, "id": "fake1", "text": "Fake chunk 1", "pos": 0, "vec": None}
        ]
        prompt = "FAKE PROMPT"
        provenance = {"sentences": [], "by_chunk": {}}
        return answer, retrieved, prompt, provenance

    monkeypatch.setattr("backend.api.rag_answer_from_embeddings", fake_rag)

    payload = {
        "question": "What is X?",
        "embeddings_path": "data/processed/fake.json",
        "top_k": 1,
        "use_faiss": False
    }
    resp = client.post("/query", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body and isinstance(body["answer"], str)
    assert "retrieved" in body and isinstance(body["retrieved"], list)
    assert "latency_s" in body and body["latency_s"] >= 0


def test_build_append_generate_status_endpoints(monkeypatch):
    # Patch rebuild_index_if_needed
    monkeypatch.setattr("backend.api.rebuild_index_if_needed", lambda embeddings_path, index_path, force=False: True)
    # Patch get_index_status
    monkeypatch.setattr("backend.api.get_index_status", lambda idx: {"exists": True, "n_vectors": 2, "dim": 8, "created_at": "now", "index_mtime": 0})
    # Build index
    resp = client.post("/build_index", json={"embeddings_path": "data/processed/fake.json"})
    assert resp.status_code == 200
    j = resp.json()
    assert j["built"] is True
    assert "status" in j and isinstance(j["status"], dict)

    # Patch append_to_index
    monkeypatch.setattr("backend.api.append_to_index", lambda embeddings_path, index_path: 3)
    resp2 = client.post("/append_index", json={"embeddings_path": "data/processed/fake.json"})
    assert resp2.status_code == 200
    j2 = resp2.json()
    assert j2["appended"] == 3

    # Patch generate_quiz_from_context
    fake_quiz = {"stem": "s", "questions": []}
    monkeypatch.setattr("backend.api.generate_quiz_from_context", lambda stem, context_text, n=5: fake_quiz)
    resp3 = client.post("/generate_quiz", json={"stem": "s", "context_text": "ctx", "n": 2})
    assert resp3.status_code == 200
    j3 = resp3.json()
    assert j3["quiz"] == fake_quiz

    # Status endpoint
    resp4 = client.get("/status", params={"index_path": "data/processed/fake.index"})
    assert resp4.status_code == 200
    j4 = resp4.json()
    assert "status" in j4 and j4["status"]["exists"] is True
