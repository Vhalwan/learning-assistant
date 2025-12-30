# backend/tests/test_api_auth.py
import os
from fastapi.testclient import TestClient
import backend.api as api_mod


def test_api_token_required_and_allows_valid_token(monkeypatch):
    # Ensure API_TOKEN is set for this test (simulate production guard)
    monkeypatch.setenv("API_TOKEN", "test-token-123")

    # Patch get_index_status (imported into backend.api) so /status won't fail on file IO
    monkeypatch.setattr(api_mod, "get_index_status", lambda idx: {"exists": False, "n_vectors": 0, "dim": 0})

    client = TestClient(api_mod.app)

    # 1) Missing Authorization header -> 401
    r = client.get("/status", params={"embeddings_path": "data/processed/nonexistent.json"})
    assert r.status_code == 401, f"expected 401 when token missing, got {r.status_code} payload: {r.text}"

    # 2) Malformed Authorization header -> 401
    r2 = client.get("/status", params={"embeddings_path": "data/processed/nonexistent.json"}, headers={"Authorization": "BadHeader token"})
    assert r2.status_code == 401

    # 3) Correct header -> proceed (should return 200 because we mocked get_index_status)
    r3 = client.get(
        "/status",
        params={"embeddings_path": "data/processed/nonexistent.json"},
        headers={"Authorization": "Bearer test-token-123"},
    )
    assert r3.status_code == 200
    data = r3.json()
    assert "status" in data or "faiss_available" in data
