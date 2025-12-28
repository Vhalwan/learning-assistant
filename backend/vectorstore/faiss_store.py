# backend/vectorstore/faiss_store.py
import json
import os
import numpy as np
from datetime import datetime

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

META_EXT = ".meta.json"


def _ensure_faiss_available():
    if not FAISS_AVAILABLE or faiss is None:
        raise RuntimeError(
            "FAISS is not available in this environment. "
            "Install with: pip install faiss-cpu"
        )

def build_faiss_index(embeddings_json_path: str, index_path: str) -> None:
    """
    Build a FAISS index (IndexFlatIP) and save it to index_path.
    Also writes metadata JSON with same basename + .meta.json that maps positions -> id/text
    and includes n_vectors, dim, created_at (UTC isoformat).
    """
    _ensure_faiss_available()

    with open(embeddings_json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    if len(rows) == 0:
        raise ValueError("No rows found in embeddings file.")

    vecs = np.array([r["embedding"] for r in rows], dtype=np.float32)
    vecs = np.ascontiguousarray(vecs)

    # Normalize L2 so IndexFlatIP about equivalent cosine similarity
    faiss.normalize_L2(vecs)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    faiss.write_index(index, index_path)

    metas = [{"id": r.get("id"), "text": r.get("text")} for r in rows]
    meta_obj = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "n_vectors": int(vecs.shape[0]),
        "dim": int(dim),
        "items": metas,
    }
    with open(index_path + META_EXT, "w", encoding="utf-8") as f:
        json.dump(meta_obj, f, ensure_ascii=False, indent=2)


def load_faiss_index(index_path: str):
    """
    Loads a FAISS index and its metadata.
    Returns (index, metas) where metas is list of dicts with keys "id" and "text".
    """
    _ensure_faiss_available()
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not os.path.exists(index_path + META_EXT):
        raise FileNotFoundError(f"FAISS metadata not found: {index_path + META_EXT}")

    index = faiss.read_index(index_path)
    with open(index_path + META_EXT, "r", encoding="utf-8") as f:
        meta_obj = json.load(f)
    metas = meta_obj.get("items", [])
    return index, metas


def search_faiss(index, metas, query_vec: np.ndarray, k: int = 3):
    """
    Search FAISS index with a single query vector.
    - query_vec: 1-D array-like (will be normalized internally)
    Returns list of (score, meta_dict, position_index)
    """
    _ensure_faiss_available()
    q = np.array(query_vec, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, k)
    scores = scores[0].tolist()
    idxs = idxs[0].tolist()

    results = []
    for score, idx in zip(scores, idxs):
        if idx < 0:
            continue
        meta = metas[idx]
        results.append((float(score), meta, int(idx)))
    return results


def get_index_status(index_path: str):
    """
    Returns status dict:
      { exists: bool, n_vectors: int|None, dim: int|None, created_at: str|None, index_mtime: float|None }
    """
    if not os.path.exists(index_path) or not os.path.exists(index_path + META_EXT):
        return {"exists": False, "n_vectors": None, "dim": None, "created_at": None, "index_mtime": None}

    try:
        with open(index_path + META_EXT, "r", encoding="utf-8") as f:
            meta_obj = json.load(f)
        n_vectors = meta_obj.get("n_vectors")
        dim = meta_obj.get("dim")
        created_at = meta_obj.get("created_at")
    except Exception:
        n_vectors = None
        dim = None
        created_at = None

    index_mtime = os.path.getmtime(index_path)
    return {"exists": True, "n_vectors": n_vectors, "dim": dim, "created_at": created_at, "index_mtime": index_mtime}


def rebuild_index_if_needed(embeddings_json_path: str, index_path: str, force: bool = False) -> bool:
    """
    Rebuilds the FAISS index if:
      - force=True OR
      - index missing OR
      - embeddings file mtime > index file mtime

    Returns True if rebuild happened, False if no-op.
    """
    if force:
        build_faiss_index(embeddings_json_path, index_path)
        return True

    # if either file missing then rebuild
    if not os.path.exists(index_path) or not os.path.exists(index_path + META_EXT):
        build_faiss_index(embeddings_json_path, index_path)
        return True

    emb_mtime = os.path.getmtime(embeddings_json_path)
    idx_mtime = os.path.getmtime(index_path)
    if emb_mtime > idx_mtime:
        build_faiss_index(embeddings_json_path, index_path)
        return True

    return False
