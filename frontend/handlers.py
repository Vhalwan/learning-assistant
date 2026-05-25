# frontend/handlers.py
import os
import hashlib
import math
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
from backend.concept_policy import build_concept_card_id, slugify_concept_id
from backend.concept_storage import load_concepts, save_concepts
try:
    from backend.concept_storage import load_concepts_with_meta as _load_concepts_with_meta
except ImportError:
    _load_concepts_with_meta = None
from backend.study_srs import SRSManager, INTERVALS
from backend.quiz_storage import save_quiz_items, load_quiz_item_by_id, load_all_quiz_items
from backend.confusion_store import (
    record_quiz_result as _record_quiz_result,
    get_top_confusions as _get_top_confusions,
    delete_confusion_entries as _delete_confusion_entries,
    _coerce_mcq_history,
)
from backend.confusion_analysis import analyze_confusion
# faiss builder (optional)
try:
    from backend.vectorstore.faiss_store import build_faiss_index
    _faiss_builder_available = True
except Exception:
    _faiss_builder_available = False

# local helper api wrappers
from frontend.runtime_ui_helpers import (
    call_query_api, call_summarize_api, call_chat_api
)

try:
    from backend.generate_quiz import generate_concepts_from_context as _generate_concepts_from_context
except ImportError:
    _generate_concepts_from_context = None

def init_llm(api_key: Optional[str] = None):
    """Return callable llm if available. Optional api_key overrides env GEMINI_API_KEY."""
    try:
        return get_llm_call(api_key=api_key)
    except Exception:
        return None


def _stable_quiz_item_id(stem: str, item: Dict[str, Any], idx: int) -> str:
    question = " ".join(str(item.get("question") or "").split()).strip().lower()
    if not question:
        return f"{stem}_q{idx}"
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]
    return f"{stem}_q_{digest}"


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


def ensure_concepts_for_lecture(
    stem: str,
    text: str,
    llm_call=None,
    doc_id: str = "",
    max_concepts: int = 8,
    force: bool = False,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Ensure the lecture has a persisted concept map.

    Returns (concepts, created_or_replaced). Concepts are lecture-level topics
    that MCQs and weak-topic progress share across retrieval chunks.
    """
    clean_stem = (stem or "").strip()
    if not clean_stem:
        return [], False

    effective_doc_id = (doc_id or hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]).strip()
    if not force and _load_concepts_with_meta is not None:
        meta = _load_concepts_with_meta(clean_stem)
        if meta and isinstance(meta, dict):
            concepts = meta.get("concepts") or []
            meta_doc_id = (meta.get("doc_id") or "").strip()
            if concepts and meta_doc_id == effective_doc_id:
                return concepts, False

    if _generate_concepts_from_context is None or llm_call is None:
        return [], False

    concepts = _generate_concepts_from_context(
        text or "",
        max_concepts=max(5, min(int(max_concepts or 8), 10)),
        llm_call=llm_call,
    )
    if concepts:
        save_concepts(clean_stem, concepts, doc_id=effective_doc_id)
        return concepts, True
    return [], False


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



def record_quiz_result(
    qid: str,
    question: str,
    is_correct: bool,
    stem: str = "",
    question_item: Optional[Dict[str, Any]] = None,
    chosen_answer: str = "",
    doc_id: str = "",
) -> None:
    """
    Frontend-callable wrapper to persist a quiz result.
    Non-blocking: logs but doesn't raise in the UI path.
    """
    try:
        concept_label = ""
        concept_id = ""
        if isinstance(question_item, dict):
            concept_label = (question_item.get("concept_label") or "").strip()
            concept_id = (question_item.get("concept_id") or "").strip()
        print(f"[debug] qid={qid} concept_id={concept_id} concept_label={concept_label}")
        _record_quiz_result(
            qid=qid,
            question=question or "",
            is_correct=bool(is_correct),
            stem=stem or "",
            concept=concept_label,
            concept_label=concept_label,
            concept_id=concept_id,
            question_item=question_item or {},
            chosen_answer=chosen_answer or "",
            doc_id=doc_id or "",
        )
    except Exception as e:
        # Keep UI stable; log to console
        print(f"[confusion_store] failed to record quiz result: {e}")


def load_persisted_confusions(
    limit: Optional[int] = 10,
    stem: str = "",
    doc_id: str = "",
    only_wrong: bool = False,
):
    try:
        return _get_top_confusions(limit=limit, stem=stem, doc_id=doc_id, only_wrong=only_wrong)
    except Exception as e:
        print(f"[confusion_store] failed to load confusions: {e}")
        return []


def delete_confusion_entries(keys: List[str]) -> int:
    try:
        return int(_delete_confusion_entries(keys or []))
    except Exception as e:
        print(f"[confusion_store] failed to delete confusions: {e}")
        return 0


def _concept_group_key(row: Dict[str, Any], stem: str = "", doc_id: str = "") -> str:
    card_id = (row.get("concept_bucket_key") or row.get("card_id") or row.get("store_key") or "").strip()
    if card_id.startswith("concept:"):
        return card_id
    concept_id = (row.get("concept_id") or "").strip()
    concept_label = (row.get("concept_label") or "").strip()
    concept_key = build_concept_card_id(
        concept_id,
        stem=stem or row.get("stem", ""),
        doc_id=doc_id or row.get("doc_id", ""),
        concept_label=concept_label,
    )
    if concept_key:
        return concept_key
    return card_id or (row.get("chunk_id") or row.get("source_chunk_id") or "").strip()


def _merge_persisted_by_concept(rows: List[Dict[str, Any]], stem: str = "", doc_id: str = "") -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for raw in rows or []:
        if not isinstance(raw, dict):
            continue
        key = _concept_group_key(raw, stem=stem, doc_id=doc_id)
        if not key:
            continue
        incoming = dict(raw)
        if key not in merged:
            incoming["card_id"] = key
            incoming["store_key"] = key
            incoming["concept_bucket_key"] = key
            merged[key] = incoming
            continue

        base = merged[key]
        base["total_attempts"] = int(base.get("total_attempts", 0) or 0) + int(incoming.get("total_attempts", 0) or 0)
        base["wrong_attempts"] = int(base.get("wrong_attempts", 0) or 0) + int(incoming.get("wrong_attempts", 0) or 0)
        base["correct_attempts"] = int(base.get("correct_attempts", 0) or 0) + int(incoming.get("correct_attempts", 0) or 0)

        base["mcq_history"] = _coerce_mcq_history(
            list(base.get("mcq_history") or []) + list(incoming.get("mcq_history") or []),
            card_question=(
                (base.get("question") or base.get("last_question") or "")
                or (incoming.get("question") or incoming.get("last_question") or "")
            ),
            last_qid=(
                (base.get("quiz_question_id") or base.get("last_mcq_id") or "")
                or (incoming.get("quiz_question_id") or incoming.get("last_mcq_id") or "")
            ),
            default_choices=base.get("choices") if isinstance(base.get("choices"), dict) else incoming.get("choices"),
            default_answer=(base.get("answer") or base.get("correct_answer") or incoming.get("answer") or ""),
        )

        linked = []
        for chunk_id in (
            list(base.get("linked_chunk_ids") or [])
            + list(incoming.get("linked_chunk_ids") or [])
            + [base.get("chunk_id"), incoming.get("chunk_id"), incoming.get("source_chunk_id")]
        ):
            clean = str(chunk_id or "").strip()
            if clean and clean not in linked:
                linked.append(clean)
        base["linked_chunk_ids"] = linked
        if not (base.get("chunk_id") or "").strip() and linked:
            base["chunk_id"] = linked[0]

        for field in ("concept_id", "concept_label", "title", "concept", "source_chunk_preview", "source_chunk"):
            if not (base.get(field) or "").strip() and (incoming.get(field) or "").strip():
                base[field] = incoming.get(field)

        incoming_seen = str(incoming.get("last_seen") or incoming.get("last_updated") or "")
        base_seen = str(base.get("last_seen") or base.get("last_updated") or "")
        if incoming_seen >= base_seen:
            for field in (
                "last_seen", "last_updated", "last_mcq_id", "quiz_question_id",
                "question", "last_question", "choices", "answer", "correct_answer",
                "explanation", "last_chosen_answer", "last_is_correct",
            ):
                if field in incoming:
                    base[field] = incoming.get(field)

    for row in merged.values():
        wrong = int(row.get("wrong_attempts", row.get("wrong_count", 0)) or 0)
        total = int(row.get("total_attempts", wrong + int(row.get("correct_attempts", 0) or 0)) or 0)
        total = max(total, wrong)
        error_rate = (wrong / total) if total else 0.0
        score = (wrong + 1.0) / (total + 2.0)
        row["error_rate"] = error_rate
        row["score"] = score
        row["final_score"] = score * math.log(1 + total) if total > 0 else 0.0
    return list(merged.values())


def perform_confusion_analysis(
    history: Optional[List[Dict[str, str]]] = None,
    quiz_submissions: Optional[List[Dict[str, Any]]] = None,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    top_n: int = 5,
    stem: str = "",
    doc_id: str = "",
    llm_call = None,
) -> List[Dict[str, Any]]:
    """
    Return persisted concept-based confusion cards ranked by their normalized score.
    """
    try:
        persisted = load_persisted_confusions(
            limit=None,
            stem=stem,
            doc_id=doc_id,
            only_wrong=True,
        ) or []
    except Exception as e:
        print(f"[confusion_analysis] failed to load persisted confusions: {e}")
        persisted = []

    # Debugging: surface counts to help diagnose missing entries during UI tests
    try:
        print(f"[confusion_analysis] loaded_persisted={len(persisted)} top_n={top_n} stem='{stem}' doc_id='{doc_id}'")
    except Exception:
        pass

    persisted = _merge_persisted_by_concept(persisted, stem=stem, doc_id=doc_id)

    real: List[Dict[str, Any]] = []
    for p in persisted:
        wrong_attempts = int(p.get("wrong_attempts", p.get("wrong_count", 0)) or 0)
        total_attempts = int(
            p.get(
                "total_attempts",
                wrong_attempts + int(p.get("correct_attempts", p.get("correct_count", 0)) or 0),
            )
            or 0
        )
        total_attempts = max(total_attempts, wrong_attempts)
        if wrong_attempts <= 0:
            continue

        error_rate = float(
            p.get("error_rate", (wrong_attempts / total_attempts) if total_attempts else 0.0)
        )
        score = float(p.get("score", (wrong_attempts + 1.0) / (total_attempts + 2.0)))
        final_score = float(
            p.get("final_score", score * math.log(1 + total_attempts) if total_attempts > 0 else 0.0)
        )

        chunk_id = (p.get("chunk_id") or p.get("source_chunk_id") or "").strip()
        concept_id = (p.get("concept_id") or "").strip()
        concept_label = (p.get("concept_label") or "").strip()
        source_chunk_preview = (p.get("source_chunk_preview") or p.get("source_chunk") or "").strip()
        display_title = (
            (p.get("title") or "").strip()
            or concept_label
            or source_chunk_preview
            or chunk_id
            or "Unlabeled concept"
        )
        store_key = (p.get("store_key") or p.get("card_id") or "").strip()
        last_seen = (p.get("last_seen") or p.get("last_updated") or p.get("last_wrong") or p.get("last_correct") or "").strip()
        last_question = (p.get("question") or p.get("last_question") or "").strip()
        last_qid = (p.get("quiz_question_id") or p.get("last_mcq_id") or "").strip()

        evidence: List[Dict[str, Any]] = []
        history_ids = []
        history_question_map: Dict[str, str] = {}
        history_choices_map: Dict[str, dict] = {}
        for entry in (p.get("mcq_history") or []):
            if isinstance(entry, dict):
                qid = str(entry.get("qid") or "").strip()
                if qid:
                    history_ids.append(qid)
                    if entry.get("question"):
                        history_question_map[qid] = str(entry["question"]).strip()
                    if isinstance(entry.get("choices"), dict):
                        history_choices_map[qid] = entry["choices"]
            elif str(entry or "").strip():
                history_ids.append(str(entry).strip())
        seen_qids = set()
        for mcq_id in reversed(history_ids):
            if mcq_id in seen_qids:
                continue
            seen_qids.add(mcq_id)
            quiz_item = load_quiz_item_by_id(mcq_id) or {}
            evidence_question = (
                (quiz_item.get("question") or "").strip()
                or history_question_map.get(mcq_id, "")
                or (last_question if mcq_id == last_qid else "")
            )
            evidence_choices = (
                quiz_item.get("choices")
                if isinstance(quiz_item.get("choices"), dict)
                else history_choices_map.get(mcq_id)
                or (p.get("choices") if mcq_id == last_qid and isinstance(p.get("choices"), dict) else {})
            )
            evidence_answer = (
                (quiz_item.get("answer") or "").strip().upper()
                or ((p.get("answer") or "").strip().upper() if mcq_id == last_qid else "")
            )
            evidence_explanation = (
                (quiz_item.get("explanation") or quiz_item.get("brief_explanation") or "").strip()
                or ((p.get("explanation") or "").strip() if mcq_id == last_qid else "")
            )
            detailed = quiz_item.get("detailed_explanation") if isinstance(quiz_item.get("detailed_explanation"), dict) else {}
            why_wrong = detailed.get("why_wrong") if isinstance(detailed.get("why_wrong"), dict) else {}
            evidence_meta = {
                "store_key": store_key,
                "qid": mcq_id,
                "question": evidence_question,
                "choices": evidence_choices,
                "answer": evidence_answer,
                "explanation": evidence_explanation,
                "question_type": (quiz_item.get("question_type") or "").strip(),
                "why_wrong": why_wrong,
                "last_chosen_answer": (
                    (p.get("last_chosen_answer") or "").strip().upper()
                    if mcq_id == last_qid
                    else ""
                ),
                "wrong_count": wrong_attempts,
            }
            evidence.append({
                "type": "quiz",
                "qid": mcq_id,
                "question": evidence_question,
                "meta": evidence_meta,
            })
            if len(evidence) >= 5:
                break

        if not evidence:
            fallback_quiz = load_quiz_item_by_id(last_qid) if last_qid else {}
            fallback_detailed = (
                fallback_quiz.get("detailed_explanation")
                if isinstance(fallback_quiz.get("detailed_explanation"), dict)
                else {}
            )
            fallback_why_wrong = (
                fallback_detailed.get("why_wrong")
                if isinstance(fallback_detailed.get("why_wrong"), dict)
                else {}
            )
            evidence.append({
                "type": "quiz",
                "qid": last_qid,
                "question": last_question,
                "meta": {
                    "store_key": store_key,
                    "qid": last_qid,
                    "question": last_question,
                    "choices": (p.get("choices") or {}) if isinstance(p.get("choices"), dict) else {},
                    "answer": (p.get("answer") or "").strip().upper(),
                    "explanation": (p.get("explanation") or "").strip(),
                    "question_type": (fallback_quiz.get("question_type") or "").strip(),
                    "why_wrong": fallback_why_wrong,
                    "last_chosen_answer": (p.get("last_chosen_answer") or "").strip().upper(),
                    "wrong_count": wrong_attempts,
                },
            })

        real.append({
            "item_type": "mcq",
            "origin": "quiz_mcq",
            "is_mcq": True,
            "card_id": (p.get("card_id") or store_key),
            "store_key": store_key,
            "chunk_id": chunk_id,
            "linked_chunk_ids": p.get("linked_chunk_ids") or ([chunk_id] if chunk_id else []),
            "quiz_question_id": last_qid,
            "original_question": last_question,
            "choices": (p.get("choices") or {}) if isinstance(p.get("choices"), dict) else {},
            "answer": (p.get("answer") or "").strip().upper(),
            "explanation": (p.get("explanation") or "").strip(),
            "last_is_correct": bool(p.get("last_is_correct", False)),
            "concept_id": concept_id,
            "concept_label": concept_label,
            "source_chunk_id": chunk_id,
            "source_chunk_preview": source_chunk_preview,
            "concept_bucket_key": (p.get("card_id") or store_key),
            "concept": display_title,
            "title": display_title,
            "concept_unlabeled": False,
            "last_seen": last_seen,
            "error_count": wrong_attempts,
            "wrong_attempts": wrong_attempts,
            "total_attempts": total_attempts,
            "error_rate": error_rate,
            "score": score,
            "final_score": final_score,
            "status": "confused" if wrong_attempts > 1 else "shaky",
            "reason": f"Error rate {error_rate:.0%} across {total_attempts} attempt(s).",
            "evidence": evidence,
            "signal_strength": wrong_attempts,
            "mcq_history": history_ids,
        })

    try:
        real_sorted = sorted(
            real,
            key=lambda r: (
                float(r.get("final_score", 0.0)),
                int(r.get("wrong_attempts", 0)),
                int(r.get("total_attempts", 0)),
                str(r.get("last_seen") or ""),
            ),
            reverse=True,
        )
    except Exception:
        real_sorted = list(real)

    try:
        print(f"[confusion_analysis] cards={len(real)} returning={len(real_sorted[:int(top_n or 5)])}")
    except Exception:
        pass

    return real_sorted[:int(top_n or 5)]



def generate_quiz(
    stem: str,
    context_text: str,
    n: int,
    use_api_mode: bool = False,
    api_base: str = None,
    token: str = "",
    llm_call = None,
    session_chunk_counts: Optional[Dict[str, int]] = None,
    exclude_questions: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    """Generate MCQ quiz items. Returns (quiz_items, latency)"""
    def _ensure_concept_fields(items: List[Dict[str, Any]]):
        for itm in items:
            if not isinstance(itm, dict):
                continue
            concept_label = (itm.get("concept_label") or itm.get("concept") or "").strip()
            concept_id = (itm.get("concept_id") or "").strip()
            if concept_label and not (itm.get("concept_label") or "").strip():
                itm["concept_label"] = concept_label
            if concept_label and not concept_id:
                itm["concept_id"] = slugify_concept_id(concept_label, fallback="concept")

    api_base = api_base or os.getenv("API_BASE", "http://localhost:8000")
    doc_id = hashlib.sha1((context_text or "").encode("utf-8")).hexdigest()[:12]
    concepts: List[Dict[str, Any]] = []
    if _load_concepts_with_meta:
        meta = _load_concepts_with_meta(stem)
        if meta and isinstance(meta, dict):
            concepts = meta.get("concepts") or []
            meta_doc_id = (meta.get("doc_id") or "").strip()
            if not meta_doc_id or meta_doc_id != doc_id:
                concepts = []

    exclude_payload: Optional[List[str]] = None
    if exclude_questions:
        exclude_payload = [str(q).strip() for q in exclude_questions if str(q).strip()][:20]

    if use_api_mode:
        url = f"{api_base.rstrip('/')}/generate_quiz_live"
        payload = {
            "stem": stem,
            "context_text": context_text,
            "n": int(n),
            "type": "mcq",
            "session_chunk_counts": session_chunk_counts,
        }
        if concepts:
            payload["concepts"] = concepts
        if exclude_payload:
            payload["exclude_questions"] = exclude_payload
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        timeout_s = min(180, max(60, 35 + 12 * int(n)))
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
        resp.raise_for_status()
        out = resp.json()
        quiz_items = out.get("quiz", []) or []
        latency = out.get("latency_s")
        # Use stable IDs based on normalized question text to prevent duplicates in SRS.
        for idx, itm in enumerate(quiz_items, start=1):
            itm["id"] = _stable_quiz_item_id(stem, itm, idx)
        _ensure_concept_fields(quiz_items)
        return quiz_items, latency
    else:
        if not concepts:
            if _generate_concepts_from_context is None:
                concepts = []
            else:
                concepts = _generate_concepts_from_context(context_text, max_concepts=8, llm_call=llm_call)
            try:
                save_concepts(stem, concepts, doc_id=doc_id)
            except Exception:
                pass
        exclude_norms = None
        if exclude_payload:
            try:
                from backend.generate_quiz import _normalize_question_key
                exclude_norms = {_normalize_question_key(q) for q in exclude_payload}
            except Exception:
                import re as _re
                exclude_norms = {_re.sub(r"\s+", " ", q.strip().lower()) for q in exclude_payload}

        try:
            quiz_items = generate_mcq_from_context(
                context_text,
                n=int(n),
                llm_call=llm_call,
                concepts=concepts,
                session_chunk_counts=session_chunk_counts,
                exclude_question_norms=exclude_norms,
            )
        except TypeError:
            try:
                quiz_items = generate_mcq_from_context(
                    context_text,
                    n=int(n),
                    llm_call=llm_call,
                    concepts=concepts,
                    session_chunk_counts=session_chunk_counts,
                )
            except TypeError:
                quiz_items = generate_mcq_from_context(context_text, n=int(n), llm_call=llm_call)
        # Use stable IDs based on normalized question text to prevent duplicates in SRS.
        for idx, itm in enumerate(quiz_items, start=1):
            itm["id"] = _stable_quiz_item_id(stem, itm, idx)
        _ensure_concept_fields(quiz_items)
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
