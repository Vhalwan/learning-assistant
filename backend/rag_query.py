"""
RAG wrapper supporting:
 - deterministic (safe) query embeddings
 - provider-backed query embeddings (when USE_SAFE_EMBEDDINGS=0)
 - FAISS search with graceful fallback to NumPy linear search
 - lightweight query expansion (toggleable)
 - small in-process LRU caching keyed to question + embeddings mtime
 - optional metadata return: retrieved chunks (with score/id/pos/vec),
   prompt_used, and provenance mapping sentences->chunk ids

This file exposes two high-level operations to support the UI design:
  1) rag_answer_from_embeddings(...)  -- Q&A mode (fast, uses top_k chunks only)
  2) rag_generate_summary_from_embeddings(...) -- Summary mode (document-level)

Notes:
 - The Q&A path will **not** produce a document-level summary.
 - The Summary path retrieves all chunks (or top_k if provided) and produces a summary only.
"""
from __future__ import annotations
import os
import json
import hashlib
import logging
import re
from collections import OrderedDict
from typing import Tuple, List, Optional, Callable, Any, Dict

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

# LRU cache
class _LRUCache:
    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self._od = OrderedDict()

    def get(self, key):
        v = self._od.get(key)
        if v is None:
            return None
        self._od.move_to_end(key)
        return v

    def set(self, key, value):
        self._od[key] = value
        self._od.move_to_end(key)
        if len(self._od) > self.maxsize:
            self._od.popitem(last=False)

    def clear(self):
        self._od.clear()

_RAG_CACHE = _LRUCache(maxsize=256)

# helpers
def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _embed_query(question: str, use_safe: Optional[bool], dim: Optional[int],
                 embed_call: Optional[Callable[[List[str], Optional[int]], List[List[float]]]] = None) -> np.ndarray:
    """
    Return normalized numpy vector (float32).
    embed_call signature: embed_call(texts: List[str], dim: Optional[int]) -> list[list[float]]
    If embed_call is None, use deterministic_vector for safe mode or provider otherwise.
    """
    if use_safe is None:
        use_safe = os.getenv("USE_SAFE_EMBEDDINGS", "1").lower() in ("1", "true", "yes")

    if embed_call is not None:
        vec = embed_call([question], dim)
        if not vec or len(vec) != 1:
            raise RuntimeError("embed_call failed to return a single vector")
        arr = np.array(vec[0], dtype=np.float32)
    else:
        if use_safe:
            arr = np.array(deterministic_vector(question, dim=dim or int(os.getenv("EMBED_DIM", "1536"))), dtype=np.float32)
        else:
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
    # normalize rows
    M_norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    sims = M_norm @ q_norm
    idx = np.argsort(sims)[::-1][:k]
    return idx.tolist(), sims[idx].tolist()

def _simple_query_expand(question: str, use_safe: bool, llm_call: Optional[Callable[[str], str]] = None) -> List[str]:
    """Return up to 2 short expansion keywords/phrases."""
    try:
        if not use_safe and llm_call:
            prompt = (
                "Provide 1-2 short alternative keywords or short phrases (comma-separated) "
                f"to expand this search query: \"{question}\"\n\nReturn only the keywords separated by commas."
            )
            resp = llm_call(prompt)
            if isinstance(resp, dict):
                resp = str(resp)
            if not isinstance(resp, str):
                return []
            parts = [p.strip() for p in re.split(r'[,\n;]+', resp) if p.strip()]
            return parts[:2]
    except Exception as e:
        logger.debug("Query expansion LLM failed: %s", e)

    # safe deterministic fallback last 2 words)
    words = re.findall(r"\w+", question)
    if not words:
        return []
    return words[-2:][-2:]

def _embed_texts_for_provenance(texts: List[str], use_safe: bool, dim: Optional[int],
                                 embed_call: Optional[Callable[[List[str], Optional[int]], List[List[float]]]] = None) -> List[np.ndarray]:
    """Return list of normalized np vectors; uses embed_call if provided else provider/deterministic."""
    if embed_call is not None:
        vecs = embed_call(texts, dim)
        out = []
        for v in vecs:
            arr = np.array(v, dtype=np.float32)
            arr = arr / (np.linalg.norm(arr) + 1e-12)
            out.append(arr)
        return out

    if use_safe:
        out = []
        for t in texts:
            v = np.array(deterministic_vector(t, dim=dim or int(os.getenv("EMBED_DIM", "1536"))), dtype=np.float32)
            out.append(v / (np.linalg.norm(v) + 1e-12))
        return out

    provider = get_embedding_provider()
    vecs = provider.embed(texts, dim=dim or int(os.getenv("EMBED_DIM", "1536")), batch_size=1)
    arr = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    arr = arr / norms
    return [arr[i] for i in range(arr.shape[0])]

# Utility to strip a trailing "Key Concepts" block from model answers in QA mode.
# We preserve the main answer and optionally capture the key concepts text (not used in UI by default).
_KEY_CONCEPTS_PATTERNS = [
    r"\n+\d+\s*Key Concepts\s*:\s*",   # "5 Key Concepts:"
    r"\n+Key Concepts\s*:\s*",
    r"\n+Key concepts\s*:\s*",
    r"\n+Key concepts / highlights\s*:\s*",
    r"\n+Key concepts / highlights\s*-",  # fallback
]

def _separate_key_concepts_block(answer: str) -> Tuple[str, Optional[str]]:
    """
    If the answer contains a trailing 'Key Concepts' (or similar) block, split and return (main_answer, key_block).
    If not found, returns (answer, None).
    """
    if not isinstance(answer, str):
        return str(answer), None
    # search for any of the patterns (case-insensitive by lowercasing search)
    # simpler approach: find first occurrence of a line that starts with "Key Concepts" (case-insensitive)
    m = re.search(r"\n+(?:\d+\s*)?Key\s+Concepts?\b", answer, flags=re.IGNORECASE)
    if not m:
        # also try "Key concepts / highlights" variations
        m2 = re.search(r"\n+Key\s+concepts\b", answer, flags=re.IGNORECASE)
        if m2:
            idx = m2.start()
            return answer[:idx].strip(), answer[idx:].strip()
        return answer.strip(), None
    idx = m.start()
    main = answer[:idx].strip()
    rest = answer[idx:].strip()
    return main, rest

# ----------------------
# Q&A (fast, targeted)
# ----------------------
def rag_answer_from_embeddings(
    question: str,
    embeddings_path: str,
    top_k: int = 3,
    use_faiss: bool = False,
    faiss_index_path: Optional[str] = None,
    use_safe: Optional[bool] = None,
    use_query_expansion: bool = False,
    return_meta: bool = False,
    llm_call: Optional[Callable[[str], str]] = None,
    embed_call: Optional[Callable[[List[str], Optional[int]], List[List[float]]]] = None,
) -> Tuple[Any, Any]:
    """
    Q&A mode -- returns concise answer using ONLY the top_k retrieved chunks.
    If return_meta is False (default): returns (answer_str, [texts...])
    If return_meta is True: returns (answer_str, retrieved_chunks, prompt_used, provenance)
    """
    if not question or not question.strip():
        raise ValueError("Question must be non-empty")

    # prepare embeddings file mtime for cache key
    emb_mtime = 0.0
    if os.path.exists(embeddings_path):
        emb_mtime = os.path.getmtime(embeddings_path)

    # cache key
    qhash = _sha256_hex(question)[:16]
    cache_key = ("rag_qa", qhash, embeddings_path, int(emb_mtime), top_k, bool(use_faiss), bool(use_query_expansion), bool(use_safe))
    cached = _RAG_CACHE.get(cache_key)
    if cached is not None:
        logger.debug("RAG QA cache hit")
        cached_qvec, retrieved_chunks = cached
    else:
        # possibly expand query
        if use_query_expansion:
            expansions = _simple_query_expand(question, use_safe=(use_safe if use_safe is not None else (os.getenv("USE_SAFE_EMBEDDINGS","1").lower() in ("1","true","yes"))), llm_call=llm_call)
            if expansions:
                expanded_query = question + " " + " ".join(expansions)
            else:
                expanded_query = question
        else:
            expanded_query = question

        # infer dim from embeddings file if possible
        dim = None
        try:
            if os.path.exists(embeddings_path):
                with open(embeddings_path, "r", encoding="utf-8") as f:
                    rows = json.load(f)
                if rows and isinstance(rows, list) and "embedding" in rows[0]:
                    dim = len(rows[0]["embedding"])
        except Exception:
            logger.debug("Could not infer dimension from embeddings file; letting provider decide")

        qvec = _embed_query(expanded_query, use_safe=use_safe, dim=dim, embed_call=embed_call)

        # Attempt FAISS
        retrieved_chunks: List[Dict[str, Any]] = []
        used_faiss = False
        if use_faiss and _faiss_available:
            try:
                idx_path = faiss_index_path or (os.path.splitext(embeddings_path)[0] + ".index")
                index, metas = load_faiss_index(idx_path)
                results = search_faiss(index, metas, qvec, k=top_k)
                if results:
                    for score, meta, pos in results:
                        retrieved_chunks.append({
                            "score": float(score),
                            "id": meta.get("id") or pos,
                            "text": meta.get("text") or "",
                            "pos": int(pos),
                            "vec": None
                        })
                    used_faiss = True
                else:
                    logger.info("FAISS returned no results; will fallback to numpy")
            except Exception as e:
                logger.warning("FAISS search failed; falling back to numpy. Error: %s", e)

        # fallback to numpy linear search or if FAISS not requested/available
        if not used_faiss:
            if not os.path.exists(embeddings_path):
                raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
            ids, texts, vecs = load_embeddings(embeddings_path)
            if len(ids) == 0:
                raise RuntimeError("Embeddings file contained no rows")
            idxs, sims = _top_k_similar_numpy(qvec, vecs, k=top_k)
            retrieved_chunks = []
            for i, sim in zip(idxs, sims):
                retrieved_chunks.append({
                    "score": float(sim),
                    "id": ids[i] if i < len(ids) else i,
                    "text": texts[i] if i < len(texts) else "",
                    "pos": int(i),
                    "vec": vecs[i].tolist() if hasattr(vecs[i], "tolist") else list(vecs[i])
                })
        else:
            # we used FAISS: populate vecs using embeddings file
            try:
                ids, texts, vecs = load_embeddings(embeddings_path)
                for ch in retrieved_chunks:
                    pos = int(ch.get("pos"))
                    if 0 <= pos < len(vecs):
                        ch["vec"] = np.array(vecs[pos], dtype=np.float32).tolist()
                    else:
                        ch["vec"] = None
            except Exception:
                logger.debug("Could not load embeddings file to populate vectors for provenance")

        # cache the qvec + retrieval
        try:
            _RAG_CACHE.set(cache_key, (qvec, retrieved_chunks))
        except Exception:
            pass

    if cached is not None:
        qvec, retrieved_chunks = cached

    # Build prompt
    context_parts = []
    for ch in retrieved_chunks:
        context_parts.append(f"[chunk:pos={ch.get('pos')} id={ch.get('id')} score={ch.get('score'):.4f}]\n{ch.get('text')}")
    prompt_context = "\n\n".join(context_parts)
    prompt = (
        "Answer the question using ONLY the following context. "
        "Cite each sentence in your answer by referring to the chunk id in square brackets, e.g. [chunk:id]. "
        "If the answer cannot be determined, say 'Not enough information.'\n\n"
        f"Context:\n{prompt_context}\n\nQuestion:\n{question}\n\nAnswer:"
    )

    # LLM call
    if llm_call is None:
        answer = summarize_with_gemini(prompt)
    else:
        answer = llm_call(prompt)
        if isinstance(answer, dict):
            answer = answer.get("answer") or json.dumps(answer)

    if not isinstance(answer, str):
        answer = str(answer)

    # Strip trailing "Key Concepts" block if present (so the Q&A UI stays concise)
    try:
        answer_main, key_block = _separate_key_concepts_block(answer)
        # For Q&A mode we deliberately return the concise answer (without appended key concepts block).
        # If return_meta is True we still produce provenance; key_block isn't returned separately by the API,
        # but could be attached to provenance if desired.
        answer = answer_main
        if key_block:
            # attach to provenance for debugging/inspection if return_meta requested
            # provenance will be populated later; temporarily stash it in a local var
            extra_key_block = key_block
        else:
            extra_key_block = None
    except Exception:
        extra_key_block = None

    # Provenance mapping
    provenance = {"sentences": [], "by_chunk": {}}
    try:
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()][:8]
        if sents:
            dim = None
            try:
                if os.path.exists(embeddings_path):
                    with open(embeddings_path, "r", encoding="utf-8") as f:
                        rows = json.load(f)
                    if rows and isinstance(rows, list) and "embedding" in rows[0]:
                        dim = len(rows[0]["embedding"])
            except Exception:
                pass
            sent_vecs = _embed_texts_for_provenance(sents, use_safe=(use_safe if use_safe is not None else (os.getenv("USE_SAFE_EMBEDDINGS","1").lower() in ("1","true","yes"))), dim=dim, embed_call=embed_call)
            for si, s in enumerate(sents):
                best_score = -1.0
                best_chunk_id = None
                best_pos = None
                for ch in retrieved_chunks:
                    if ch.get("vec") is None:
                        continue
                    cvec = np.array(ch.get("vec"), dtype=np.float32)
                    sc = float(np.dot(sent_vecs[si], cvec) / (np.linalg.norm(cvec) + 1e-12))
                    if sc > best_score:
                        best_score = sc
                        best_chunk_id = ch.get("id")
                        best_pos = ch.get("pos")
                provenance["sentences"].append({"sentence": s, "chunk_id": best_chunk_id, "pos": best_pos, "score": float(best_score)})
                provenance["by_chunk"].setdefault(best_chunk_id, []).append(si)
    except Exception as e:
        logger.debug("Provenance mapping failed: %s", e)

    # Attach any extracted key_block for inspection
    if extra_key_block:
        provenance["extracted_key_block"] = extra_key_block

    # Prepare return values
    if not return_meta:
        texts = [ch.get("text", "") for ch in retrieved_chunks]
        return answer, texts

    # else return rich tuple
    return answer, retrieved_chunks, prompt, provenance

# ----------------------
# Summary (document-level)
# ----------------------
def rag_generate_summary_from_embeddings(
    embeddings_path: str,
    summary_type: str = "brief",
    top_k: Optional[int] = None,
    use_safe: Optional[bool] = None,
    llm_call: Optional[Callable[[str], str]] = None,
    embed_call: Optional[Callable[[List[str], Optional[int]], List[List[float]]]] = None,
    return_meta: bool = False,
) -> Tuple[Any, Any]:
    """
    Document-level summary mode.

    - embeddings_path: path to embeddings JSON produced by create_embeddings_for_text
    - summary_type: "brief" or "detailed" (affects prompt instruction)
    - top_k: if None -> use ALL chunks; otherwise use first top_k chunks
    - return_meta: if True returns (summary, used_chunks, prompt_used) else (summary, [texts...])

    This function concatenates the requested chunks (in document order) and asks the LLM to summarize.
    """
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    ids, texts, vecs = load_embeddings(embeddings_path)
    if len(ids) == 0:
        raise RuntimeError("Embeddings file contained no rows")

    # choose chunks: default = all (in order). If top_k is provided, choose first top_k
    if top_k is None or top_k <= 0 or top_k >= len(texts):
        chosen_idxs = list(range(len(texts)))
    else:
        chosen_idxs = list(range(min(top_k, len(texts))))

    # compose context (keeping chunk separators and ids for traceability)
    context_parts = []
    used_chunks = []
    for i in chosen_idxs:
        used_chunks.append({"id": ids[i], "pos": i, "text": texts[i]})
        context_parts.append(f"[chunk:pos={i} id={ids[i]}]\n{texts[i]}")

    prompt_context = "\n\n".join(context_parts)

    # build summary prompt
    if summary_type == "detailed":
        instruct = "Write a DETAILED, structured summary of the following lecture. Include section headings, key concepts, and bullet-pointed takeaways. Keep it faithful to the text and do not hallucinate facts."
    else:
        instruct = "Write a BRIEF summary (3-6 sentences) capturing the main ideas and key takeaways of the following lecture. Be concise and factual."

    prompt = f"{instruct}\n\nContext:\n{prompt_context}\n\nSummary:"

    # LLM call
    if llm_call is None:
        summary = summarize_with_gemini(prompt)
    else:
        summary = llm_call(prompt)
        if isinstance(summary, dict):
            summary = summary.get("summary") or json.dumps(summary)

    if not isinstance(summary, str):
        summary = str(summary)

    # Extract variable-length key concepts using the LLM
    key_concepts: List[str] = []
    try:
        kc_prompt = (
            "Extract short key concepts / phrases from the summary below. "
            "Return them as a comma-separated list. Aim for a concise set (typically 3-8 items), but return however many are appropriate.\n\nSummary:\n" + summary
        )
        if llm_call is None:
            kc_resp = summarize_with_gemini(kc_prompt)
        else:
            kc_resp = llm_call(kc_prompt)
        if isinstance(kc_resp, dict):
            kc_resp = str(kc_resp)
        if isinstance(kc_resp, str):
            parts = [p.strip() for p in re.split(r'[,\n;]+', kc_resp) if p.strip()]
            key_concepts = parts[:16]  # allow up to a sensible cap
    except Exception:
        logger.debug("Key concepts extraction failed; continuing without them")

    out = {"summary": summary.strip(), "key_concepts": key_concepts}

    if not return_meta:
        texts_out = [c["text"] for c in used_chunks]
        return out, texts_out

    return out, used_chunks, prompt
