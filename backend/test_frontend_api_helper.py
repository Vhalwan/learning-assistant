# frontend/tests/test_frontend_api_helper.py
import pytest

# We will import the helper from the Streamlit app module
# (it defines `call_query_api`), and monkeypatch the requests.post used inside it.
def test_call_query_api_requests_monkeypatched(monkeypatch):
    fake_resp = {
        "answer": "42",
        "retrieved": [{"id": "c1", "text": "chunk text"}],
        "prompt": "PROMPT",
        "provenance": {"sentences": [{"sentence": "Ans", "chunk_id": "c1", "score": 0.9}]},
        "latency_s": 0.123,
    }

    class DummyResp:
        def __init__(self, data):
            self._data = data
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return self._data

    def fake_post(url, json=None, headers=None, timeout=None):
        # very light validation to ensure payload shape
        assert "question" in (json or {})
        return DummyResp(fake_resp)

    # import here so tests do not import before monkeypatch setup (but we monkeypatch the attribute on the module)
    import frontend.app as appmod

    # Replace the requests.post used inside frontend.app
    monkeypatch.setattr(appmod.requests, "post", fake_post)

    out = appmod.call_query_api(
        question="What is 6 * 7?",
        embeddings_path="data/processed/x.json",
        top_k=3,
        use_faiss=False,
        faiss_index_path=None,
        api_base="http://dummy",
        token="",
    )

    assert out["answer"] == "42"
    assert isinstance(out.get("retrieved"), list)
    assert out["latency_s"] == pytest.approx(0.123, rel=1e-6)
