import json
import os
import numpy as np
import pytest
import tempfile

from backend.vectorstore.faiss_store import (
    FAISS_AVAILABLE,
    build_faiss_index,
    append_to_index,
    load_faiss_index,
    get_index_status,
    search_faiss,
)

pytestmark = pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not available in this environment")

def write_json(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

def test_faiss_append_roundtrip(tmp_path):
    """
    Build index from file1 (1 vector), append file2 (1 vector),
    then assert metadata updated and searches return expected top result.
    """
    # prepare embeddings files
    emb1 = [
        {"id": "a", "text": "alpha", "embedding": [1.0, 0.0]}
    ]
    emb2 = [
        {"id": "b", "text": "beta", "embedding": [0.0, 1.0]}
    ]
    emb1_path = tmp_path / "emb1.json"
    emb2_path = tmp_path / "emb2.json"
    write_json(str(emb1_path), emb1)
    write_json(str(emb2_path), emb2)

    index_path = tmp_path / "test.index"

    # build index from first file
    build_faiss_index(str(emb1_path), str(index_path))

    status1 = get_index_status(str(index_path))
    assert status1["exists"] is True
    assert status1["n_vectors"] == 1

    # append second file
    appended = append_to_index(str(emb2_path), str(index_path))
    assert isinstance(appended, int)
    assert appended == 1

    status2 = get_index_status(str(index_path))
    assert status2["exists"] is True
    assert status2["n_vectors"] == 2

    # load index and metas and check ids
    index, metas = load_faiss_index(str(index_path))
    ids = [m.get("id") for m in metas]
    assert "a" in ids
    assert "b" in ids

    # query for vector similar to emb2 -> expect top id 'b'
    qvec = np.array([0.0, 1.0], dtype=np.float32)
    results = search_faiss(index, metas, qvec, k=1)
    assert len(results) >= 1
    top_score, top_meta, top_pos = results[0]
    assert top_meta.get("id") == "b" or top_meta.get("text") == "beta"
    # score should be close to 1.0 for identical normalized vectors
    assert top_score > 0.9
