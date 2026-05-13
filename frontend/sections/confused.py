# frontend/sections/confused.py
"""
Confused? Quick prioritized list section extracted from frontend/app.py

Expose a single function:
    render(st, stem, embeddings_path, index_path, use_faiss_search, llm)

This module mirrors the UI and behavior originally in app.py without changes.
"""

import os
import re
from datetime import datetime
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


def _build_mcq_from_item(item: dict, selected_mcq: dict) -> dict:
    selected_q = (selected_mcq.get("question") or "").strip() if isinstance(selected_mcq, dict) else ""
    base_q = (item.get("original_question") or item.get("question") or "").strip()
    question = selected_q or base_q
    choices = item.get("choices", {}) if isinstance(item.get("choices"), dict) else {}
    answer = (item.get("answer") or "").strip().upper()
    explanation = (item.get("explanation") or "").strip()
    qid = (selected_mcq.get("id") or item.get("quiz_question_id") or "").strip() if isinstance(selected_mcq, dict) else ""
    return {
        "id": qid or None,
        "question": question,
        "choices": choices,
        "answer": answer,
        "explanation": explanation,
    }


def _choice_with_text(letter: str, choices: dict) -> str:
    clean_letter = str(letter or "").strip().upper()
    if not clean_letter:
        return ""
    option_text = ""
    if isinstance(choices, dict):
        option_text = " ".join(str(choices.get(clean_letter, "") or "").split())
    return f"{clean_letter}. {option_text}" if option_text else clean_letter


def _format_flagged_reason(evidence_entry: dict, is_most_recent: bool = False) -> str:
    """Return a short one-line reason for a single wrong-answer evidence row.

    Keeps bullets compact: trimmed question text plus the correct answer letter.
    The "you chose" hint is only added for the most recent wrong attempt because
    the persisted card stores only the latest chosen answer, so attaching it to
    older evidence rows would be misleading.
    """
    meta = evidence_entry.get("meta", {}) or {}
    raw_question = str(meta.get("question") or evidence_entry.get("question") or "")
    question = _shorten(raw_question, limit=100)

    chosen = str(meta.get("last_chosen_answer") or "").strip().upper()
    correct = str(meta.get("answer") or "").strip().upper()

    tail_parts = []
    if is_most_recent and chosen and correct and chosen != correct:
        tail_parts.append(f"chose {chosen}")
    if correct:
        tail_parts.append(f"correct: {correct}")

    if question and tail_parts:
        return f"{question} — " + ", ".join(tail_parts)
    if question:
        return question
    if tail_parts:
        return "Wrong attempt — " + ", ".join(tail_parts)
    return "Wrong quiz attempt for this concept."


def _format_timestamp(value: str) -> str:
    raw = " ".join(str(value or "").split()).strip()
    if not raw:
        return ""
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return raw
    return dt.strftime("%Y-%m-%d %H:%M")


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
    quiz_generation_key = f"{quiz_state_key}_generation"
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
    generation_token = int(st.session_state.get(quiz_generation_key, 0))
    for q in quiz_items_session:
        qid = q.get("id")
        submit_key = f"{quiz_state_key}_sub_{generation_token}_{qid}"
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
    real_confusions.sort(
        key=lambda r: (
            float(r.get("final_score", 0.0)),
            int(r.get("wrong_attempts", 0)),
            int(r.get("total_attempts", 0)),
            str(r.get("last_seen") or ""),
        ),
        reverse=True,
    )

    # When empty, show one calm line only (no cards/icons)
    if not real_confusions:
        st.success("No wrong quiz answers are recorded for this lecture yet. When you miss a question, the matching concept will appear here.")
    else:
        preview_limit = min(len(real_confusions), 5)
        for idx, item in enumerate(real_confusions[:preview_limit], start=1):
            with st.container():
                st.markdown('<div class="la-card"></div>', unsafe_allow_html=True)
                concept = item.get("title") or item.get("concept") or item.get("concept_label") or "Unlabeled concept"
                strength = int(item.get("signal_strength", 0))
                total_attempts = int(item.get("total_attempts", 0) or 0)
                error_rate = float(item.get("error_rate", 0.0) or 0.0)
                last_seen = _format_timestamp(item.get("last_seen", ""))
                header_cols = st.columns([3.0, 1.0, 1.0])
                with header_cols[0]:
                    st.markdown(f"### {idx}. {concept}")
                    if strength > 1:
                        st.write("You have missed multiple quiz questions tied to this concept.")
                    else:
                        st.write("You missed a quiz question tied to this concept.")
                    if item.get("reason"):
                        st.caption(item.get("reason"))
                    if last_seen:
                        st.caption(f"Last seen: {last_seen}")
                with header_cols[1]:
                    st.metric("Error rate", f"{error_rate:.0%}")
                with header_cols[2]:
                    st.metric("Attempts", total_attempts)
                evidence = item.get("evidence", []) or []
                evidence_quiz_rows = []
                item_type = (item.get("item_type") or ("mcq" if item.get("is_mcq") else "concept")).strip().lower()
                is_mcq_item = item_type == "mcq"

                # Build a deduplicated list of WRONG-only candidate MCQs from evidence.
                # `evidence` is sourced from the card's mcq_history, which the
                # confusion store only appends to on incorrect attempts, so every
                # entry here represents a question the user actually got wrong.
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
                    label = question or "(question text unavailable)"
                    mcq_candidates.append({"id": stable_id or None, "question": question, "label": label})

                # Only fall back to the card's latest question if it really was a
                # wrong attempt. The persisted `last_question`/`original_question`
                # field is updated on every recorded answer (correct or wrong), so
                # we must gate this behind `last_is_correct == False` to avoid
                # leaking a correctly-answered question into the SRS dropdown.
                if is_mcq_item and not bool(item.get("last_is_correct", False)):
                    mcq_payload = _build_mcq_from_item(item, {"id": item.get("quiz_question_id"), "question": item.get("original_question")})
                    if mcq_payload.get("question"):
                        fallback_id = mcq_payload.get("id") or ""
                        fallback_label = mcq_payload.get("question")
                        candidate = {"id": fallback_id or None, "question": mcq_payload.get("question"), "label": fallback_label}
                        if (candidate["id"], candidate["question"]) not in seen_mcq:
                            mcq_candidates.insert(0, candidate)

                fallback_label = "No wrong-answered question available"
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
                if selected_mcq.get("question"):
                    st.caption(selected_mcq.get("question"))
                st.caption("Type: MCQ confusion item" if is_mcq_item else "Type: Concept confusion item")

                if evidence:
                    with st.expander("Why this weak area was flagged"):
                        st.caption("Last 5 wrong quiz attempts for this concept.")
                        seen_q = set()
                        reason_rows = []
                        first_quiz_seen = False
                        for e in evidence:
                            if e.get("type") in ("quiz", "persisted"):
                                meta = e.get("meta", {}) or {}
                                qtext = " ".join(str(meta.get("question") or e.get("question") or "").split())
                                qid = str(meta.get("qid") or e.get("qid") or "").strip()
                                if qtext:
                                    dedupe_key = (qid, qtext.strip().lower())
                                    if dedupe_key not in seen_q:
                                        seen_q.add(dedupe_key)
                                        reason_rows.append(
                                            _format_flagged_reason(e, is_most_recent=not first_quiz_seen)
                                        )
                                        first_quiz_seen = True
                                if qid and qtext:
                                    evidence_quiz_rows.append({"qid": qid, "question": qtext})
                            else:
                                extra = _shorten(" ".join(str(e).split()), limit=120)
                                if extra:
                                    reason_rows.append(extra)
                        if reason_rows:
                            for row in reason_rows[:5]:
                                st.write(f"- {row}")
                        else:
                            st.write("- Repeated incorrect quiz answers for this concept.")

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
                deterministic_fallback_id = (
                    item.get("store_key")
                    or item.get("card_id")
                    or item.get("chunk_id")
                    or _deterministic_confusion_id(stem, extracted_concept)
                )

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
                            st.session_state[f"open_chat_tab_{stem}"] = True
                            st.session_state[f"chat_focus_input_{stem}"] = True
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

                                topic_heading = (
                                    (item.get("title") or item.get("concept") or item.get("concept_label") or concept or "")
                                    .strip()
                                )
                                if is_mcq_item:
                                    mcq_payload = _build_mcq_from_item(item, selected_mcq)
                                    card_id = card_id or mcq_payload.get("id") or deterministic_fallback_id
                                    question_text = mcq_payload.get("question") or question_text
                                    card_meta = {
                                        "item_type": "mcq",
                                        "origin": "confused_mcq",
                                        "quiz_question_id": mcq_payload.get("id") or "",
                                        "question": question_text or "",
                                        "choices": mcq_payload.get("choices") or {},
                                        "answer": mcq_payload.get("answer") or "",
                                        "explanation": mcq_payload.get("explanation") or "",
                                        "stem": stem,
                                        "source_reason": "Added from Confused review",
                                        "concept_label": (item.get("concept_label") or extracted_concept or topic_heading),
                                        "concept_id": (item.get("concept_id") or "").strip(),
                                        "concept": topic_heading or extracted_concept,
                                        "title": topic_heading or extracted_concept,
                                    }
                                else:
                                    card_meta = {
                                        "item_type": "concept",
                                        "origin": "confused_concept",
                                        "question": question_text or "",
                                        "stem": stem,
                                        "source_reason": "Added from Confused review",
                                        "concept_label": (item.get("concept_label") or extracted_concept or topic_heading),
                                        "concept_id": (item.get("concept_id") or "").strip(),
                                        "concept": topic_heading or extracted_concept,
                                        "title": topic_heading or extracted_concept,
                                    }
                                mgr.ensure_card(card_id, meta=card_meta)
                                if "srs_quiz_items_cache" not in st.session_state:
                                    st.session_state["srs_quiz_items_cache"] = {}
                                if is_mcq_item:
                                    st.session_state["srs_quiz_items_cache"][card_id] = {
                                        "id": card_id,
                                        "question": card_meta.get("question", ""),
                                        "choices": card_meta.get("choices", {}) or {},
                                        "answer": card_meta.get("answer", ""),
                                        "explanation": card_meta.get("explanation", ""),
                                        "item_type": "mcq",
                                    }
                                else:
                                    st.session_state["srs_quiz_items_cache"][card_id] = {
                                        "id": card_id,
                                        "question": card_meta.get("question", ""),
                                        "choices": {},
                                        "answer": "",
                                        "explanation": "",
                                        "item_type": "concept",
                                    }
                                st.success(f"Added to SRS: {card_id}")
                                try:
                                    st.rerun()
                                except Exception:
                                    try:
                                        st.experimental_rerun()
                                    except Exception:
                                        pass
                            except Exception as e:
                                st.error("Failed to add to SRS.")
                                st.exception(e)
                    with secondary_action_cols[1]:
                        delete_key = f"conf_delete_{stem}_{idx}"
                        if st.button("Reset", key=delete_key, use_container_width=True):
                            if not delete_keys:
                                st.warning("No persisted entries found to delete.")
                            else:
                                removed = delete_confusion_entries(delete_keys)
                                st.success(f"Reset {removed} confusion card(s).")
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
