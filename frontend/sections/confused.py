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
from frontend.handlers import perform_confusion_analysis, perform_query, delete_confusion_entries
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
    doc_id_key = f"doc_id_{stem}"
    doc_id = st.session_state.get(doc_id_key, "")
    followup_notice_key = f"conf_followup_notice_{stem}"

    st.markdown('<a id="confused-quick-prioritized-list"></a>', unsafe_allow_html=True)
    st.markdown("## Confused? Coaching list")
    st.write("These are the concepts most worth reviewing next.")
    st.caption("Start with the top card, get a simple explanation, or queue a follow-up prompt for Chat.")

    followup_notice = st.session_state.get(followup_notice_key)
    notice_rendered = False

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
            doc_id=doc_id,
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
        concept_id = (r.get("concept_id") or "").strip()
        concept_label = (r.get("concept_label") or r.get("concept") or "").strip()
        if r.get("concept_unlabeled"):
            concept_label = ""
        if concept_id:
            key = f"id:{concept_id}"
        else:
            label_key = re.sub(r"[^a-z0-9 ]+", "", concept_label.lower())
            label_key = " ".join(label_key.split())
            if label_key:
                key = f"label:{label_key}"
            else:
                evidence = r.get("evidence", []) or []
                first = evidence[0] if evidence else {}
                meta = first.get("meta", {}) or {}
                qid = meta.get("qid") or first.get("qid") or ""
                key = f"qid:{qid}" if qid else f"idx:{len(merged)}"
        if key not in merged:
            merged[key] = r
        else:
            merged[key]["signal_strength"] = int(merged[key].get("signal_strength", 0)) + int(r.get("signal_strength", 0))
            merged[key]["evidence"] = (merged[key].get("evidence", []) or []) + (r.get("evidence", []) or [])
            if not merged[key].get("concept_label") and concept_label:
                merged[key]["concept_label"] = concept_label
            if not merged[key].get("concept_id") and concept_id:
                merged[key]["concept_id"] = concept_id
    real_confusions = list(merged.values())
    real_confusions.sort(key=lambda r: -int(r.get("signal_strength", 0)))

    # When empty, show one calm line only (no cards/icons)
    if not real_confusions:
        st.success("No repeated quiz mistakes yet. This section will fill in after the same concept is missed more than once.")
    else:
        preview_limit = min(len(real_confusions), 5)
        for idx, item in enumerate(real_confusions[:preview_limit], start=1):
            with st.container():
                st.markdown('<div class="la-card"></div>', unsafe_allow_html=True)
                concept = item.get("concept_label") or item.get("concept") or ""
                if not concept:
                    first_evidence = (item.get("evidence") or [{}])[0]
                    meta = first_evidence.get("meta", {}) or {}
                    qid = meta.get("qid") or first_evidence.get("qid") or ""
                    if qid:
                        concept = f"Unlabeled concept ({qid})"
                    else:
                        concept = "Unlabeled concept"
                strength = int(item.get("signal_strength", 0))
                header_cols = st.columns([3.2, 1.2])
                with header_cols[0]:
                    st.markdown(f"### {idx}. {concept}")
                    st.write("This concept showed up as a repeated weak spot in your recent quiz answers.")
                    if item.get("reason"):
                        st.caption(item.get("reason"))
                with header_cols[1]:
                    st.metric("Signals", strength)
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
                    "Source question for Add to SRS",
                    options=[c["label"] for c in mcq_candidates],
                    key=selected_mcq_key,
                    help="Explain simply and Ask follow-up in Chat use the concept directly. Pick a question here only for Add to SRS.",
                )
                st.caption("The dropdown only affects Add to SRS. Explain simply and Chat follow-up always use the concept itself.")
                selected_mcq = next(
                    (c for c in mcq_candidates if c["label"] == selected_mcq_label),
                    mcq_candidates[0],
                )

                if evidence:
                    with st.expander("Why this concept was flagged"):
                        for e in evidence:
                            if e.get("type") in ("quiz", "persisted"):
                                meta = e.get("meta", {}) or {}
                                qid = meta.get("qid") or e.get("qid") or "<no-id>"
                                qtext = (meta.get("question") or e.get("question") or "")[:400]
                                if qid and qid != "<no-id>" and qtext:
                                    evidence_quiz_rows.append({"qid": str(qid), "question": qtext})
                                ctext = meta.get("concept_label") or meta.get("concept") or ""
                                if ctext:
                                    st.write(f"- Quiz: id={qid} — concept: {ctext}")
                                if qtext:
                                    st.write(f"  - original question: {qtext}")
                            else:
                                st.write(f"- {str(e)[:400]}")

                delete_keys = []
                for e in evidence:
                    meta = e.get("meta", {}) or {}
                    store_key = (meta.get("store_key") or "").strip()
                    if not store_key:
                        qid = meta.get("qid") or e.get("qid")
                        if qid:
                            store_key = str(qid).strip()
                        else:
                            qtext = meta.get("question") or e.get("question") or ""
                            if qtext:
                                store_key = qtext[:120]
                    if store_key:
                        delete_keys.append(store_key)
                delete_keys = sorted(set(delete_keys))

                # Centralized action input resolution for this card
                extracted_concept = item.get("concept_label") or item.get("concept") or concept
                deterministic_fallback_id = _deterministic_confusion_id(stem, extracted_concept)

                if (
                    not notice_rendered
                    and isinstance(followup_notice, dict)
                    and followup_notice.get("card_id") == deterministic_fallback_id
                ):
                    st.info(followup_notice.get("message", "Follow-up prompt queued for chat."))
                    notice_rendered = True

                selected_qid = selected_mcq.get("id")
                selected_question = selected_mcq.get("question") or ""

                with st.container():
                    st.markdown('<div class="la-action-bar"></div>', unsafe_allow_html=True)
                    st.markdown("**Next step**")
                    primary_action_cols = st.columns(2)
                    secondary_action_cols = st.columns(2)
                    with primary_action_cols[0]:
                        # Explain simply (kept — student-first)
                        explain_btn_key = f"conf_explain_{stem}_{idx}"
                        if st.button("Explain simply", key=explain_btn_key, type="primary", use_container_width=True):
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

                    with primary_action_cols[1]:
                        follow_key = f"conf_follow_{stem}_{idx}"
                        if st.button("Ask follow-up in Chat", key=follow_key, use_container_width=True):
                            follow_prompt = (
                                f"Concept: {extracted_concept}\n"
                                "The user struggled with this concept in recent quizzes. Explain it clearly, give simple examples, and highlight key points that might cause confusion."
                            )
                            st.session_state[f"chat_pending_input_{stem}"] = follow_prompt
                            st.session_state[followup_notice_key] = {
                                "card_id": deterministic_fallback_id,
                                "message": f"Follow-up prompt queued for '{extracted_concept}'. Open the Chat tab to review or send it.",
                            }
                            try:
                                st.rerun()
                            except Exception:
                                try:
                                    st.experimental_rerun()
                                except Exception:
                                    pass
                    with secondary_action_cols[0]:
                        # ➕ Add to SRS (idempotent)
                        add_srs_key = f"conf_add_srs_{stem}_{idx}"
                        if st.button("Add to SRS", key=add_srs_key, use_container_width=True):
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
                    with secondary_action_cols[1]:
                        delete_key = f"conf_delete_{stem}_{idx}"
                        if st.button("Delete this card", key=delete_key, use_container_width=True):
                            if not delete_keys:
                                st.warning("No persisted entries found to delete.")
                            else:
                                removed = delete_confusion_entries(delete_keys)
                                st.success(f"Deleted {removed} confusion item(s).")
                                try:
                                    st.rerun()
                                except Exception:
                                    try:
                                        st.experimental_rerun()
                                    except Exception:
                                        pass
        if notice_rendered:
            st.session_state.pop(followup_notice_key, None)
    st.markdown("---")
