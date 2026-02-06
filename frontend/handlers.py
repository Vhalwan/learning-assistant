# frontend/handlers.py
import os
import re
from typing import Optional, List, Dict, Any, Tuple
import requests
from backend.confusion_analysis import analyze_confusion
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
from backend.confusion_store import record_quiz_result as _record_quiz_result, get_top_confusions as _get_top_confusions
from backend.confusion_analysis import analyze_confusion, _shorten
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

_PREFIX_PATTERNS = [
    r"^\s*(?:q(?:uestion)?\s*\d*[:\-\.)]?|problem\s*\d*[:\-\.)]?|quiz\s*\d*[:\-\.)]?)\s*",
    r"^\s*(?:which of the following|choose the correct answer|select the correct answer|select one|pick one)\s*[:\-]?\s*",
    r"^\s*(?:true or false|t\/f)\s*[:\-]?\s*",
]


def extract_concept_from_mcq_stem(question_text: str, max_len: int = 80) -> str:
    """Extract a deterministic short concept label from an MCQ-style question."""
    text = " ".join((question_text or "").strip().split())
    if not text:
        return "Unknown concept"

    cleaned = text
    for pat in _PREFIX_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)

    # Remove obvious option-like suffixes from first option marker onward.
    option_markers = [
        r"\s+[A-Da-d][\)\.:]\s+",
        r"\s+\([A-Da-d]\)\s+",
        r"\s+1[\)\.:]\s+",
        r"\s+\*\s+",
    ]
    cut = len(cleaned)
    for marker in option_markers:
        m = re.search(marker, cleaned)
        if m:
            cut = min(cut, m.start())
    cleaned = cleaned[:cut].strip(" -:;,.?!")
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Try to keep only the leading noun-phrase-ish part before helper clauses.
    splitters = [" is ", " are ", " was ", " were ", " can ", " does ", " do ", " means ", " refers to "]
    lowered = f" {cleaned.lower()} "
    split_at = None
    for token in splitters:
        idx = lowered.find(token)
        if idx > 0:
            split_at = idx
            break
    if split_at:
        cleaned = cleaned[:split_at].strip(" -:;,.?!")

    # Convert question forms to a concise label.
    cleaned = re.sub(r"^(what|which|why|how|when|where|who)\s+(is|are|was|were)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(define|describe|explain|identify)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" -:;,.?!")

    if 3 <= len(cleaned) <= max_len:
        return cleaned

    # Fallback: bounded cleaned snippet from original question.
    fallback = re.sub(r"\s+", " ", text).strip(" -:;,.?!")
    if len(fallback) > max_len:
        fallback = fallback[:max_len].rsplit(" ", 1)[0].rstrip(" -:;,.?!") + "..."
    return fallback or "Unknown concept"


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



def record_quiz_result(qid: str, question: str, is_correct: bool, stem: str = "") -> None:
    """
    Frontend-callable wrapper to persist a quiz result.
    Non-blocking: logs but doesn't raise in the UI path.
    """
    try:
        concept = extract_concept_from_mcq_stem(question or "")
        _record_quiz_result(
            qid=qid,
            question=question or "",
            is_correct=bool(is_correct),
            stem=stem or "",
            concept=concept,
        )
    except Exception as e:
        # Keep UI stable; log to console
        print(f"[confusion_store] failed to record quiz result: {e}")


def load_persisted_confusions(limit: int = 10):
    try:
        return _get_top_confusions(limit)
    except Exception as e:
        print(f"[confusion_store] failed to load confusions: {e}")
        return []


def perform_confusion_analysis(
    history: Optional[List[Dict[str, str]]] = None,
    quiz_submissions: Optional[List[Dict[str, Any]]] = None,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    top_n: int = 5,
    stem: str = "",
    llm_call = None,
) -> List[Dict[str, Any]]:
    """
    Return a prioritized list of *persisted* quiz confusions only (no chat-derived placeholders).
    Rules enforced here:
      - Only include persisted entries coming from record_quiz_result (backend/confusion_store).
      - Exclude any persisted entry with wrong_count <= 0.
      - Map persisted entries into the UI-friendly schema:
          { concept, status, reason, evidence, signal_strength }
      - Sort by wrong_count desc and return up to top_n items (default 5).
    This ensures the Confused UI shows *only* real quiz mistakes and never shows the "clear/0" placeholders.
    """
    try:
        # Prefer persisted store as the single source of truth for "real confusion"
        persisted = load_persisted_confusions(limit=top_n * 2) or []
    except Exception as e:
        print(f"[confusion_analysis] failed to load persisted confusions: {e}")
        persisted = []

    # Filter to only real confusion items from quizzes (wrong_count > 0)
    real = []
    for p in persisted:
        try:
            wrong_count = int(p.get("wrong_count", 0))
        except Exception:
            wrong_count = 0
        if wrong_count <= 0:
            # do not include zero-count placeholders
            continue

        qid = p.get("qid") or ""
        entry_stem = (p.get("stem") or "").strip()
        if stem:
            matches = False
            if entry_stem and entry_stem == stem:
                matches = True
            elif qid and qid.startswith(f"{stem}_"):
                matches = True
            if not matches:
                continue
        qtext = (p.get("question") or "").strip()
        concept = (p.get("concept") or "").strip() or extract_concept_from_mcq_stem(qtext or qid or "")
        concept = _shorten(concept or qid or "Unknown concept", 200)

        status = "confused" if wrong_count > 1 else "shaky"
        reason = f"Persisted incorrect answers (count={wrong_count})"

        evidence = [{"type": "quiz", "qid": qid, "question": qtext, "meta": p}]

        real.append({
            "concept": concept,
            "original_question": qtext,
            "status": status,
            "reason": reason,
            "evidence": evidence,
            "signal_strength": wrong_count,
        })

    # Sort by signal_strength (wrong_count) desc
    try:
        real_sorted = sorted(real, key=lambda r: -int(r.get("signal_strength", 0)))
    except Exception:
        real_sorted = real

    # Enforce UI cap: at most top_n (recommended 3-5; callers may pass 5)
    return real_sorted[:int(top_n or 5)]



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
