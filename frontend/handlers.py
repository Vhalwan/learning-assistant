# frontend/handlers.py
import os
from typing import Optional, List, Dict, Any, Tuple
import requests

# backend imports (same as the original app.py used)
from backend.llm_client import get_llm_call
from backend.summarize_file import summarize_with_gemini
from backend.create_embeddings import create_embeddings_for_text, load_embeddings, EMBED_DIM
from backend.embeddings_provider import deterministic_vector, get_embedding_provider
from backend.rag_query import (
    rag_answer_from_embeddings, rag_generate_summary_from_embeddings, rag_chat_answer,
)
from backend.generate_quiz import generate_quiz_from_context, generate_mcq_from_context
from backend.study_srs import SRSManager, INTERVALS
from backend.quiz_storage import save_quiz_items, load_quiz_item_by_id, load_all_quiz_items

# faiss builder (optional)
try:
    from backend.vectorstore.faiss_store import build_faiss_index
    _faiss_builder_available = True
except Exception:
    _faiss_builder_available = False

# local helper api wrappers
from frontend.ui_helpers import (
    call_query_api, call_summarize_api, call_chat_api
)


def init_llm():
    """Return callable llm if available (same behavior as original app.py)"""
    try:
        llm = get_llm_call()
        return llm
    except Exception:
        return None


def create_embeddings_if_needed(text: str, embeddings_path: str, dim: int = EMBED_DIM, recreate: bool = False) -> int:
    """Create embeddings for given text, returning number of rows created."""
    if not recreate and os.path.exists(embeddings_path):
        # nothing to do
        _, _, vecs = load_embeddings(embeddings_path)
        return len(vecs) if hasattr(vecs, "__len__") else 0
    rows = create_embeddings_for_text(text, embeddings_path, dim=dim)
    return len(rows)


def load_embeddings_wrapper(embeddings_path: str):
    """Safe wrapper around load_embeddings"""
    try:
        ids, texts, vecs = load_embeddings(str(embeddings_path))
        return ids, texts, vecs
    except Exception:
        return [], [], []


def perform_query(
    question: str,
    embeddings_path: str,
    top_k: int,
    use_faiss: bool,
    faiss_index_path: Optional[str] = None,
    use_api_mode: bool = False,
    api_base: str = None,
    token: str = "",
    llm_call = None,
) -> Dict[str, Any]:
    """Run the retrieval + LLM answer. Returns a dict with keys: answer, retrieved, prompt, provenance, latency."""
    api_base = api_base or os.getenv("API_BASE", "http://localhost:8000")
    if use_api_mode:
        resp = call_query_api(
            question=question,
            embeddings_path=embeddings_path,
            top_k=top_k,
            use_faiss=use_faiss,
            faiss_index_path=faiss_index_path,
            api_base=api_base,
            token=token,
        )
        return {
            "answer": resp.get("answer"),
            "retrieved": resp.get("retrieved", []),
            "prompt": resp.get("prompt"),
            "provenance": resp.get("provenance"),
            "latency": resp.get("latency_s", None),
        }
    else:
        # local mode: call rag_answer_from_embeddings
        ans, retrieved, prompt, provenance = rag_answer_from_embeddings(
            question,
            embeddings_path,
            top_k=top_k,
            use_faiss=use_faiss,
            faiss_index_path=faiss_index_path,
            use_safe=(True if os.environ.get("USE_SAFE_EMBEDDINGS", "1") in ("1", "true", "yes") else False),
            use_query_expansion=False,
            return_meta=True,
            llm_call=llm_call,
        )
        return {
            "answer": ans,
            "retrieved": retrieved or [],
            "prompt": prompt,
            "provenance": provenance,
            "latency": None,
        }


def perform_summary(
    embeddings_path: str,
    summary_type: str = "brief",
    top_k: Optional[int] = None,
    use_api_mode: bool = False,
    api_base: str = None,
    token: str = "",
    llm_call = None,
) -> Dict[str, Any]:
    api_base = api_base or os.getenv("API_BASE", "http://localhost:8000")
    if use_api_mode:
        resp = call_summarize_api(
            embeddings_path=embeddings_path,
            summary_type=summary_type,
            top_k=top_k,
            api_base=api_base,
            token=token,
        )
        out = resp.get("out", {}) or {}
        return {
            "summary": out.get("summary", ""),
            "key_concepts": out.get("key_concepts", []) or [],
            "used_chunks": resp.get("used_chunks", []) or [],
        }
    else:
        out, used_chunks = rag_generate_summary_from_embeddings(
            embeddings_path,
            summary_type=summary_type,
            top_k=top_k,
            use_safe=(True if os.environ.get("USE_SAFE_EMBEDDINGS", "1") in ("1", "true", "yes") else False),
            return_meta=False,
            llm_call=llm_call,
        )
        return {
            "summary": out.get("summary", ""),
            "key_concepts": out.get("key_concepts", []) or [],
            "used_chunks": used_chunks or [],
        }


def perform_chat(
    question: str,
    embeddings_path: str,
    history: Optional[List[Dict[str, str]]],
    top_k: int,
    use_faiss: bool,
    faiss_index_path: Optional[str] = None,
    use_api_mode: bool = False,
    api_base: str = None,
    token: str = "",
    llm_call = None,
) -> Dict[str, Any]:
    api_base = api_base or os.getenv("API_BASE", "http://localhost:8000")
    if use_api_mode:
        resp = call_chat_api(
            question=question,
            embeddings_path=embeddings_path,
            history=history,
            top_k=top_k,
            use_faiss=use_faiss,
            faiss_index_path=faiss_index_path,
            api_base=api_base,
            token=token,
        )
        return {
            "answer": resp.get("answer"),
            "history": resp.get("history"),
            "retrieved": resp.get("retrieved", []),
            "prompt": resp.get("prompt"),
            "provenance": resp.get("provenance"),
        }
    else:
        qa_out = rag_chat_answer(
            question,
            embeddings_path,
            history=history,
            top_k=top_k,
            use_faiss=use_faiss,
            faiss_index_path=faiss_index_path,
            use_safe=(True if os.environ.get("USE_SAFE_EMBEDDINGS", "1") in ("1", "true", "yes") else False),
            use_query_expansion=False,
            return_meta=True,
            llm_call=llm_call,
        )
        if isinstance(qa_out, (tuple, list)):
            # keep compatibility with the original code which sometimes returns tuple
            ans = qa_out[0]
            updated_history = qa_out[1] if len(qa_out) >= 2 else None
            retrieved = qa_out[2] if len(qa_out) >= 3 else []
            prompt_used = qa_out[3] if len(qa_out) >= 4 else None
            provenance = qa_out[4] if len(qa_out) >= 5 else None
            return {
                "answer": ans,
                "history": updated_history,
                "retrieved": retrieved or [],
                "prompt": prompt_used,
                "provenance": provenance,
            }
        else:
            return {"answer": str(qa_out), "history": None, "retrieved": [], "prompt": None, "provenance": None}


def generate_quiz(
    stem: str,
    context_text: str,
    n: int,
    use_api_mode: bool = False,
    api_base: str = None,
    token: str = "",
    llm_call = None,
) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    """Generate MCQ quiz items. Returns (quiz_items, latency)"""
    api_base = api_base or os.getenv("API_BASE", "http://localhost:8000")
    if use_api_mode:
        url = f"{api_base.rstrip('/')}/generate_quiz_live"
        payload = {"stem": stem, "context_text": context_text, "n": int(n), "type": "mcq"}
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        out = resp.json()
        quiz_items = out.get("quiz", []) or []
        latency = out.get("latency_s")
        # ensure ids are prefixed with stem like original app
        for itm in quiz_items:
            if "id" in itm and not str(itm["id"]).startswith(f"{stem}_"):
                itm["id"] = f"{stem}_{itm['id']}"
        return quiz_items, latency
    else:
        quiz_items = generate_mcq_from_context(context_text, n=int(n), llm_call=llm_call)
        # prefix IDs to match expected pattern and ensure uniqueness
        for idx, itm in enumerate(quiz_items, start=1):
            if "id" in itm:
                if not str(itm["id"]).startswith(f"{stem}_"):
                    itm["id"] = f"{stem}_{itm['id']}"
            else:
                itm["id"] = f"{stem}_q{idx}"
        return quiz_items, None


def build_index(embeddings_path: str, index_path: str):
    """Build FAISS index using backend helper."""
    if not _faiss_builder_available:
        raise RuntimeError("FAISS builder not available (faiss-cpu not installed).")
    return build_faiss_index(str(embeddings_path), str(index_path))


def save_quiz_to_disk(stem: str, quiz_items: List[Dict[str, Any]]):
    """Save quiz items to disk via backend helper and propagate exceptions upward."""
    save_quiz_items(stem, quiz_items)


def load_all_quiz_items_wrapper():
    return load_all_quiz_items()


def load_quiz_item_by_id_wrapper(qid: str):
    return load_quiz_item_by_id(qid)
