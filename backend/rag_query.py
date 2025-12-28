# backend/rag_query.py
"""
RAG wrapper supporting:
 - deterministic (safe) query embeddings
 - provider-backed query embeddings (when USE_SAFE_EMBEDDINGS=0)
 - FAISS search with graceful fallback to NumPy linear search

Public API:
  rag_answer_from_embeddings(question, embeddings_path, top_k=3,
                             use_faiss=False, faiss_index_path=None, use_safe=None)
Returns:
  (answer_str, list_of_retrieved_texts)
"""

import os
import json
import hashlib
import logging
from typing import Tuple, List, Optional

import numpy as np
from dotenv import load_dotenv

from backend.summarize_file import summarize_with_gemini
from backend.create_embeddings import load_embeddings  # returns ids, texts, vecs
from backend.embeddings_provider import get_embedding_provider, deterministic_vector

# faiss helpers
try:
    from backend.vectorstore.faiss_store import load_faiss_index, search_faiss
    _faiss_available = True
except Exception:
    _faiss_available = False

load_dotenv()
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_EMBED_FILE = "data/processed/lecture1_embeddings.json"


def _embed_query(question: str, use_safe: Optional[bool], dim: Optional[int]) -> np.ndarray:
    """
    Get a query embedding vector as numpy array (float32), normalized.
    - use_safe True -> deterministic_vector
    - use_safe False -> provider.embed([question])
    - use_safe None -> read env var USE_SAFE_EMBEDDINGS
    """
    if use_safe is None:
        use_safe = os.getenv("USE_SAFE_EMBEDDINGS", "1").lower() in ("1", "true", "yes")

    if use_safe:
        # deterministic_vector returns a python list
        v = deterministic_vector(question, dim=dim or int(os.getenv("EMBED_DIM", "1536")))
        arr = np.array(v, dtype=np.float32)
        arr = arr / (np.linalg.norm(arr) + 1e-12)
        return arr

    provider = get_embedding_provider()
    vecs = provider.embed([question], dim=dim or int(os.getenv("EMBED_DIM", "1536")), batch_size=1)
    if not vecs or len(vecs) != 1:
        raise RuntimeError("Embedding provider failed to return a vector for the query")
    arr = np.array(vecs[0], dtype=np.float32)
    arr = arr / (np.linalg.norm(arr) + 1e-12)
    return arr


def _top_k_similar_numpy(query_vec: np.ndarray, all_vecs: np.ndarray, k: int = 3) -> Tuple[List[int], List[float]]:
    """Return top-k indices and sims using numpy cosine similarity (assumes vectors may not be normalized)."""
    if all_vecs.size == 0:
        return [], []
    M = np.array(all_vecs, dtype=np.float32)
    M_norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    sims = M_norm @ q_norm
    idx = np.argsort(sims)[::-1][:k]
    return idx.tolist(), sims[idx].tolist()


def rag_answer_from_embeddings(
    question: str,
    embeddings_path: str,
    top_k: int = 3,
    use_faiss: bool = False,
    faiss_index_path: Optional[str] = None,
    use_safe: Optional[bool] = None
) -> Tuple[str, List[str]]:
    """
    Programmatic RAG wrapper.

    Parameters
    - question: the user question
    - embeddings_path: path to embeddings JSON (if searching via NumPy) OR used to derive default index path
    - top_k: number of retrieved chunks
    - use_faiss: try FAISS search first (if True)
    - faiss_index_path: explicit .index path to load; if None and use_faiss True, will try embeddings_path with ".index" suffix
    - use_safe: override safe-mode embedding for the query; if None, reads env var USE_SAFE_EMBEDDINGS.

    Returns (answer_string, retrieved_texts)
    """
    if not question or not question.strip():
        raise ValueError("Question must be non-empty")

    # Prepare query vector
    dim = None
    try:
        if os.path.exists(embeddings_path):
            with open(embeddings_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            if rows and isinstance(rows, list) and "embedding" in rows[0]:
                dim = len(rows[0]["embedding"])
    except Exception:
        logger.debug("Could not infer dimension from embeddings file; letting provider decide")

    qvec = _embed_query(question, use_safe=use_safe, dim=dim)

    # FAISS search attempt
    retrieved_texts: List[str] = []
    used_faiss = False
    if use_faiss:
        if not _faiss_available:
            logger.warning("FAISS requested but not available; falling back to numpy search.")
        else:
            # determine index path
            idx_path = faiss_index_path
            if not idx_path:
                # derive from the embeddings_path
                base = os.path.splitext(embeddings_path)[0]
                idx_path = base + ".index"
            try:
                index, metas = load_faiss_index(idx_path)
                results = search_faiss(index, metas, qvec, k=top_k)
                # results: list of (score, meta, idx)
                if results:
                    retrieved_texts = [r[1].get("text", "") for r in results]
                    used_faiss = True
                else:
                    logger.info("FAISS returned no results; falling back to numpy search")
            except Exception as e:
                logger.warning("Failed to load/search FAISS index (%s). Falling back to numpy. Error: %s", idx_path, e)

    # If not using FAISS or FAISS failed then fallback to numpy linear search
    if not used_faiss:
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        ids, texts, vecs = load_embeddings(embeddings_path)
        if len(ids) == 0:
            raise RuntimeError("Embeddings file contained no rows")
        idxs, sims = _top_k_similar_numpy(qvec, vecs, k=top_k)
        retrieved_texts = [texts[i] for i in idxs]

    # Build prompt and call LLM
    context_parts = []
    # Keep same ordering as retrieval
    for i, txt in enumerate(retrieved_texts):
        context_parts.append(f"[chunk:{i}]\n{txt}")

    prompt_context = "\n\n".join(context_parts)
    prompt = (
        "Answer the question using ONLY the following context. "
        "If the answer cannot be determined, say 'Not enough information.'\n\n"
        f"Context:\n{prompt_context}\n\nQuestion:\n{question}\n\nAnswer:"
    )

    answer = summarize_with_gemini(prompt)
    return answer, retrieved_texts


if __name__ == "__main__":
    # simple CLI test
    path = os.getenv("EMBED_FILE", DEFAULT_EMBED_FILE)
    if not os.path.exists(path):
        print("Embeddings file not found; create embeddings first.")
        raise SystemExit(1)

    q = input("Question: ").strip() or "What is the main idea?"
    ans, texts = rag_answer_from_embeddings(q, path, top_k=3, use_faiss=False)
    print("\n--- Retrieved ---")
    for t in texts:
        print(t[:300], "...\n---")
    print("\n--- Answer ---\n")
    print(ans)
