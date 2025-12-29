# backend/vectorstore/summarize_file.py
import json
import os
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

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
    _ensure_faiss_available()
    with open(embeddings_json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if len(rows) == 0:
        raise ValueError("No rows found in embeddings file.")
    vecs = np.array([r["embedding"] for r in rows], dtype=np.float32)
    vecs = np.ascontiguousarray(vecs)
    # Normalize L2 so IndexFlatIP approximates cosine similarity
    faiss.normalize_L2(vecs)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    # write index and metadata atomically where possible
    tmp_index = index_path + ".tmp"
    try:
        faiss.write_index(index, tmp_index)
        os.replace(tmp_index, index_path)
    except Exception as e:
        logger.exception("Failed to write FAISS index: %s", e)
        raise
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
    _ensure_faiss_available()
    q = np.array(query_vec, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(q)
    try:
        scores, idxs = index.search(q, k)
    except Exception as e:
        logger.exception("FAISS search error: %s", e)
        raise
    scores = scores[0].tolist()
    idxs = idxs[0].tolist()
    results = []
    for score, idx in zip(scores, idxs):
        if idx < 0:
            continue
        meta = metas[idx] if idx < len(metas) else {"id": None, "text": None}
        results.append((float(score), meta, int(idx)))
    return results

def get_index_status(index_path: str):
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
    if force:
        build_faiss_index(embeddings_json_path, index_path)
        return True
    if not os.path.exists(index_path) or not os.path.exists(index_path + META_EXT):
        build_faiss_index(embeddings_json_path, index_path)
        return True
    emb_mtime = os.path.getmtime(embeddings_json_path)
    idx_mtime = os.path.getmtime(index_path)
    if emb_mtime > idx_mtime:
        build_faiss_index(embeddings_json_path, index_path)
        return True
    return False

def append_to_index(embeddings_json_path: str, index_path: str) -> int:
    """
    Append vectors from embeddings_json_path to existing FAISS index at index_path.

    Returns:
        int: number of vectors appended.

    Behavior:
      - If FAISS is not available -> raises RuntimeError
      - If index_path or its meta is missing -> builds a new index from embeddings_json_path and returns n_vectors (same as build)
      - Ensures dim compatibility between existing index metadata and new embeddings
      - Normalizes new vectors before adding (so IndexFlatIP acts as cosine)
      - Updates <index_path>.meta.json by appending new items and updating n_vectors
    """
    _ensure_faiss_available()

    if not os.path.exists(embeddings_json_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_json_path}")

    # load new rows
    with open(embeddings_json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not rows:
        logger.info("No rows to append from %s", embeddings_json_path)
        return 0

    new_vecs = np.array([r["embedding"] for r in rows], dtype=np.float32)
    new_vecs = np.ascontiguousarray(new_vecs)
    faiss.normalize_L2(new_vecs)
    new_count = int(new_vecs.shape[0])
    new_dim = int(new_vecs.shape[1])

    # If index or meta missing, build fresh index instead of append
    if not os.path.exists(index_path) or not os.path.exists(index_path + META_EXT):
        logger.info("Index or metadata missing; building fresh index at %s", index_path)
        build_faiss_index(embeddings_json_path, index_path)
        return new_count

    # load existing meta to check dim and n_vectors
    try:
        with open(index_path + META_EXT, "r", encoding="utf-8") as f:
            meta_obj = json.load(f)
    except Exception as e:
        logger.exception("Failed to read FAISS meta file: %s", e)
        raise

    existing_dim = meta_obj.get("dim")
    existing_n = meta_obj.get("n_vectors", 0)
    if existing_dim is not None and int(existing_dim) != new_dim:
        raise ValueError(f"Dimension mismatch: existing index dim={existing_dim}, new embeddings dim={new_dim}")

    # load index
    try:
        index = faiss.read_index(index_path)
    except Exception as e:
        logger.exception("Failed to read FAISS index at %s: %s", index_path, e)
        raise

    # add vectors
    try:
        index.add(new_vecs)
    except Exception as e:
        logger.exception("Failed to append vectors to FAISS index: %s", e)
        raise

    # atomic write index
    tmp_index = index_path + ".tmp"
    try:
        faiss.write_index(index, tmp_index)
        os.replace(tmp_index, index_path)
    except Exception as e:
        logger.exception("Failed to write FAISS index after append: %s", e)
        raise

    # append metadata items
    new_metas = [{"id": r.get("id"), "text": r.get("text")} for r in rows]
    meta_obj.setdefault("items", [])
    meta_obj["items"].extend(new_metas)
    meta_obj["n_vectors"] = int(existing_n) + new_count
    # record update timestamp
    meta_obj["updated_at"] = datetime.utcnow().isoformat() + "Z"
    # ensure dim present
    meta_obj["dim"] = int(new_dim)

    # write meta file atomically
    tmp_meta = index_path + META_EXT + ".tmp"
    try:
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta_obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp_meta, index_path + META_EXT)
    except Exception as e:
        logger.exception("Failed to write FAISS meta file after append: %s", e)
        raise

    logger.info("Appended %d vectors to index %s (new total: %d)", new_count, index_path, meta_obj["n_vectors"])
    return new_count
