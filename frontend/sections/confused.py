# frontend/sections/confused.py
"""
Confused? Quick prioritized list section extracted from frontend/app.py

Expose a single function:
    render(st, stem, embeddings_path, index_path, use_faiss_search, llm)

This module mirrors the UI and behavior originally in app.py without changes.
"""

import os
import re
from pathlib import Path
from typing import Any

import streamlit as st

# handlers and backend helpers (same as in app.py)
from frontend.handlers import perform_confusion_analysis, perform_query
from backend.study_srs import SRSManager


API_DEFAULT = os.getenv("API_BASE", "http://localhost:8000")

def _shorten(text: str, limit: int = 140) -> str:
    if not text:
        return ""
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rsplit(" ", 1)[0] + "..."


def render(st: Any, stem: str, embeddings_path: Path, index_path: Path, use_faiss_search: bool, llm):
    """
    Render the 'Confused? Quick prioritized list' UI for the given document (stem).
    Keep behavior identical to previous inline implementation.

    Args:
        st: streamlit module (passed from app.py)
        stem: document stem (string) used to build session keys
        embeddings_path: Path to embeddings file (Path or str)
        index_path: Path to faiss index file (Path or str)
        use_faiss_search: whether FAISS search is enabled (bool)
        llm: local LLM object (can be None)
    """
    # session/state keys that were used inline in app.py
    hist_key = f"chat_history_{stem}"
    quiz_state_key = f"quiz_items_{stem}"

    st.markdown('<a id="confused-quick-prioritized-list"></a>', unsafe_allow_html=True)
    st.markdown("## Confused? Quick prioritized list")
    st.write("These are concepts you repeatedly missed in quizzes — review them before moving on.")

    # gather session-local quiz context (not authoritative)
    history = st.session_state.get(hist_key, []) or []
    quiz_items_session = st.session_state.get(quiz_state_key, []) or []

    quiz_submissions = []
    for q in quiz_items_session:
        qid = q.get("id")
        submit_key = f"{quiz_state_key}_sub_{qid}"
        sub = st.session_state.get(submit_key)
        if sub:
            quiz_submissions.append({
                "id": qid,
                "question": q.get("question", "") or "",
                "is_correct": bool(sub.get("is_correct", False))
            })

    # fetch persisted top confusions (handlers returns only real quiz-based confusions)
    try:
        top_limit = 5  # cap to 3-5 as requested (handlers also enforces)
        results = perform_confusion_analysis(
            history=history,
            quiz_submissions=quiz_submissions,
            retrieved_chunks=[],
            top_n=top_limit,
            stem=stem,
            llm_call=llm,
        ) or []
    except Exception as e:
        st.error("Failed to compute prioritized confusions.")
        st.exception(e)
        results = []

    # require positive signal_strength (wrong_count) — safety filter
    real_confusions = [r for r in results if int(r.get("signal_strength", 0)) > 0]

    # When empty, show one calm line only (no cards/icons)
    if not real_confusions:
        st.write("No repeated quiz mistakes yet 👍")
    else:
        preview_limit = min(len(real_confusions), 5)
        for idx, item in enumerate(real_confusions[:preview_limit], start=1):
            with st.container():
                st.markdown('<div class="la-card"></div>', unsafe_allow_html=True)
                concept = item.get("concept", "(no concept)")
                strength = int(item.get("signal_strength", 0))
                st.markdown(f"**{idx}. {concept}** — missed {strength} time{'s' if strength != 1 else ''}.")
                short_concept = _shorten(concept, 120) or concept
                st.markdown(f"### 🤔 Confused Card {idx}: {short_concept}")
                st.markdown(f"**Question / concept**")
                st.write(concept)
                st.markdown(f"**Progress / Review stats**")
                st.write(f"📊 Missed {strength} time{'s' if strength != 1 else ''}")
                if item.get("reason"):
                    st.caption(item.get("reason"))
                st.info("You struggled with this concept. Review the explanation or add it to SRS.")
                evidence = item.get("evidence", []) or []
                if evidence:
                    with st.expander("Show evidence"):
                        for e in evidence:
                            if e.get("type") in ("quiz", "persisted"):
                                meta = e.get("meta", {}) or {}
                                qid = meta.get("qid") or e.get("qid") or "<no-id>"
                                qtext = (meta.get("question") or e.get("question") or "")[:400]
                                st.write(f"- Quiz: id={qid} — {qtext}")
                            else:
                                st.write(f"- {str(e)[:400]}")

                with st.container():
                    st.markdown('<div class="la-action-bar"></div>', unsafe_allow_html=True)
                    action_cols = st.columns(2)
                    with action_cols[0]:
                        # Explain simply (kept — student-first)
                        explain_btn_key = f"conf_explain_{stem}_{idx}"
                        if st.button("Explain simply", key=explain_btn_key):
                            try:
                                candidate_index_path = str(index_path) if index_path and Path(index_path).exists() else None
                                raw_concept = item.get("concept", "")
                                explain_q = f"Explain this simply and give 1 short example: {raw_concept}"
                                resp = perform_query(
                                    question=explain_q,
                                    embeddings_path=str(embeddings_path),
                                    top_k=3,
                                    use_faiss=bool(use_faiss_search),
                                    faiss_index_path=candidate_index_path,
                                    use_api_mode=st.session_state.get("use_api_mode", False),
                                    api_base=os.getenv("API_BASE", API_DEFAULT),
                                    token=st.session_state.get("api_token", "") or "",
                                    llm_call=llm,
                                )
                                st.markdown("**Explanation**")
                                st.write(resp.get("answer", "(no answer returned)"))
                                if resp.get("retrieved"):
                                    with st.expander("Show retrieved chunks used for this explanation"):
                                        for rc in resp.get("retrieved", []):
                                            st.write(rc.get("text", "")[:1000] + ("..." if len(rc.get("text", "")) > 1000 else ""))
                            except Exception as e:
                                st.error("Failed to generate explanation.")
                                st.exception(e)

                    with action_cols[1]:
                        # ➕ Add to SRS (idempotent)
                        add_srs_key = f"conf_add_srs_{stem}_{idx}"
                        if st.button("➕ Add to SRS", key=add_srs_key):
                            try:
                                mgr = SRSManager()
                                # try to extract canonical quiz id from evidence metadata
                                card_id = None
                                question_text = ""
                                for e in evidence:
                                    meta = e.get("meta", {}) or {}
                                    if meta.get("qid"):
                                        card_id = meta.get("qid")
                                        question_text = meta.get("question", "") or question_text
                                        break
                                    if e.get("qid"):
                                        card_id = e.get("qid")
                                        question_text = e.get("question", "") or question_text
                                        break

                                # fallback deterministic id derived from concept+stem
                                if not card_id:
                                    # make a safe short token from the concept
                                    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", concept).strip("_")[:40] or "conf"
                                    card_id = f"{stem}_{safe}"

                                    question_text = question_text or concept

                                mgr.ensure_card(card_id, meta={"question": question_text or "", "stem": stem})
                                if "srs_quiz_items_cache" not in st.session_state:
                                    st.session_state["srs_quiz_items_cache"] = {}
                                st.session_state["srs_quiz_items_cache"][card_id] = {
                                    "id": card_id,
                                    "question": question_text or "",
                                    "choices": {},
                                    "answer": "",
                                    "explanation": "",
                                }
                                st.success(f"Added to SRS: {card_id}")
                            except Exception as e:
                                st.error("Failed to add to SRS.")
                                st.exception(e)

    st.markdown("---")
