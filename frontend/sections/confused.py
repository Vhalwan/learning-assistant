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

def _deterministic_confusion_id(stem: str, concept_text: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", concept_text).strip("_")[:40] or "conf"
    return f"{stem}_{safe}"


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
    st.markdown("## 🤝 Confused? Coaching list")
    st.write("These concepts are worth reviewing next.")

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
    merged = {}
    for r in real_confusions:
        key = re.sub(r"[^a-z0-9 ]+", "", (r.get("concept", "") or "").lower())
        key = " ".join(key.split())
        if not key:
            key = r.get("concept", "")
        if key not in merged:
            merged[key] = r
        else:
            merged[key]["signal_strength"] = int(merged[key].get("signal_strength", 0)) + int(r.get("signal_strength", 0))
            merged[key]["evidence"] = (merged[key].get("evidence", []) or []) + (r.get("evidence", []) or [])
    real_confusions = list(merged.values())

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
                st.markdown(f"**{idx}. {concept}** — this concept is worth reviewing.")
                st.markdown(f"### 🤔 Confused Card {idx}: {concept}")
                st.markdown("**Concept**")
                st.write(concept)
                st.markdown(f"**Progress / Review stats**")
                st.write(f"📊 Review signal: {strength}")
                if item.get("reason"):
                    st.caption(item.get("reason"))
                st.info("You can quickly reinforce this with a simple explanation or move it into SRS.")
                evidence = item.get("evidence", []) or []
                evidence_quiz_rows = []

                # Build a deduplicated list of candidate original MCQs from evidence
                mcq_candidates = []
                seen_mcq = set()
                for e in evidence:
                    meta = e.get("meta", {}) or {}
                    qid = meta.get("qid") or e.get("qid")
                    question = meta.get("question") or e.get("question") or ""
                    stable_id = str(qid).strip() if qid else ""
                    dedupe_key = (stable_id, str(question).strip())
                    if dedupe_key in seen_mcq:
                        continue
                    seen_mcq.add(dedupe_key)
                    if not stable_id and not question:
                        continue
                    preview = _shorten(question or "(question text unavailable)", 100)
                    if stable_id:
                        label = f"{preview} [id={stable_id}]"
                    else:
                        label = f"{preview} [id=<no-id>]"
                    mcq_candidates.append({"id": stable_id or None, "question": question, "label": label})

                fallback_label = "No original MCQ available"
                if not mcq_candidates:
                    mcq_candidates = [{"id": None, "question": "", "label": fallback_label}]

                selected_mcq_key = f"conf_selected_mcq_{stem}_{idx}"
                selected_mcq_label = st.selectbox(
                    "Original question(s) [used for Add to SRS only]",
                    options=[c["label"] for c in mcq_candidates],
                    key=selected_mcq_key,
                    help="Explain Simply and Ask Follow-up always use the concept. Pick a question here only when adding to SRS.",
                )
                selected_mcq = next(
                    (c for c in mcq_candidates if c["label"] == selected_mcq_label),
                    mcq_candidates[0],
                )

                if evidence:
                    with st.expander("Show evidence"):
                        for e in evidence:
                            if e.get("type") in ("quiz", "persisted"):
                                meta = e.get("meta", {}) or {}
                                qid = meta.get("qid") or e.get("qid") or "<no-id>"
                                qtext = (meta.get("question") or e.get("question") or "")[:400]
                                if qid and qid != "<no-id>" and qtext:
                                    evidence_quiz_rows.append({"qid": str(qid), "question": qtext})
                                ctext = meta.get("concept") or ""
                                if ctext:
                                    st.write(f"- Quiz: id={qid} — concept: {ctext}")
                                if qtext:
                                    st.write(f"  - original question: {qtext}")
                            else:
                                st.write(f"- {str(e)[:400]}")

                # Centralized action input resolution for this card
                extracted_concept = item.get("concept", "") or concept
                deterministic_fallback_id = _deterministic_confusion_id(stem, extracted_concept)

                selected_qid = selected_mcq.get("id")
                selected_question = selected_mcq.get("question") or ""

                with st.container():
                    st.markdown('<div class="la-action-bar"></div>', unsafe_allow_html=True)
                    action_cols = st.columns(3)
                    with action_cols[0]:
                        # Explain simply (kept — student-first)
                        explain_btn_key = f"conf_explain_{stem}_{idx}"
                        if st.button("Explain simply", key=explain_btn_key):
                            try:
                                candidate_index_path = str(index_path) if index_path and Path(index_path).exists() else None
                                compact_evidence = []
                                for row in evidence_quiz_rows[:2]:
                                    compact_evidence.append(f"- [{row['qid']}] {_shorten(row['question'], 120)}")
                                explain_q = (
                                    "You are tutoring a learner on a confusion point. "
                                    "Explain simply and provide one short concrete example.\n\n"
                                    f"Confused concept: {extracted_concept}\n"
                                    "Compact evidence snippets:\n"
                                    f"{chr(10).join(compact_evidence) if compact_evidence else '- None available'}"
                                )
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
                        follow_key = f"conf_follow_{stem}_{idx}"
                        if st.button("Follow up", key=follow_key):
                            follow_prompt = (
                                "I'm reviewing this concept and I'm confused. Please teach it clearly.\n\n"
                                f"Concept: {extracted_concept}\n"
                                "Use the quiz misses only as evidence. "
                                "Start with a plain-language explanation, then give one intuitive example, "
                                "and finally ask me one short check question to test my understanding."
                            )
                            st.session_state[f"chat_input_{stem}"] = follow_prompt
                            st.success("Loaded a follow-up prompt into Chat input. Open Chat to review/edit it, then press Send.")
                    with action_cols[2]:
                        # ➕ Add to SRS (idempotent)
                        add_srs_key = f"conf_add_srs_{stem}_{idx}"
                        if st.button("➕ Add to SRS", key=add_srs_key):
                            try:
                                mgr = SRSManager()
                                # Use selected original MCQ (if available) as primary source
                                card_id = selected_qid or deterministic_fallback_id
                                question_text = selected_question or extracted_concept

                                # fallback deterministic id derived from concept+stem
                                if not card_id:
                                    # make a safe short token from the concept
                                    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", concept).strip("_")[:40] or "conf"
                                    card_id = f"{stem}_{safe}"

                                    question_text = question_text or concept

                                mgr.ensure_card(card_id, meta={"question": question_text or "", "stem": stem, "source_reason": "Added from Confused review"})
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
