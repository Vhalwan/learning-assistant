"""
Quiz UI extraction from frontend/app.py — UX-optimized presentation only.

Expose:
    def render(st, stem, text, llm, hist_key):
        ...

Behavior & state keys are unchanged — only presentation/layout has been improved.
"""

import uuid
import hashlib
import requests
from pathlib import Path
from typing import Any, List, Dict
import os
import streamlit as st

# Handlers / helpers (same as app.py)
from frontend.handlers import (
    generate_quiz,
    save_quiz_to_disk,
    record_quiz_result,
)

# Backend SRS manager (same as app.py)
from backend.study_srs import SRSManager


def _mark_selection_made(sel_key: str):
    """
    Streamlit on_change helper used to mark a selection as made.
    Kept simple and identical to the inline helper in app.py.
    """
    st.session_state[f"{sel_key}_made"] = True


def render(st: Any, stem: str, text: str, llm, hist_key: str):
    """
    Render the Quiz generation + Quiz item UI.

    Args:
        st: the streamlit module (passed from app.py)
        stem: document stem string
        text: extracted text from uploaded PDF (used as context for quiz generation)
        llm: local LLM object (can be None)
        hist_key: chat history session_state key (passed through to keep same scope naming)
    """
    API_DEFAULT = os.getenv("API_BASE", "http://localhost:8000") if "os" in globals() else "http://localhost:8000"

    st.markdown("---")
    st.markdown('<a id="study-quiz-mcq-v1"></a>', unsafe_allow_html=True)
    st.subheader("📝 Study / Quiz (MCQ v1)")
    st.markdown(
        "Generate short multiple-choice quizzes from your lecture. "
        "Read each card, select an answer, then **Check answer**. "
        "Use **Add to SRS** to save questions you'd like to review later."
    )

    # session_state key for storing generated quiz for this document
    quiz_state_key = f"quiz_items_{stem}"
    doc_id_key = f"doc_id_{stem}"
    if text is not None:
        doc_id = hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]
        st.session_state[doc_id_key] = doc_id
    else:
        doc_id = st.session_state.get(doc_id_key, "")

    # Top controls inline: number input + generate button
    cols_top = st.columns([2, 1])
    with cols_top[0]:
        n_q = st.number_input("Number of quiz items", min_value=1, max_value=20, value=5, help="How many MCQs to generate from this lecture.")
    with cols_top[1]:
        gen_key = f"gen_quiz_{stem}"
        # keep the same button key as original
        if st.button("Generate quiz", key=gen_key, type="primary", use_container_width=True):
            if not text:
                st.warning("No document text extracted.")
            else:
                with st.spinner("Generating quiz..."):
                    try:
                        if st.session_state.get("use_api_mode", False):
                            quiz_items, latency = generate_quiz(
                                stem=stem,
                                context_text=text,
                                n=int(n_q),
                                use_api_mode=True,
                                api_base=os.getenv("API_BASE", API_DEFAULT),
                                token=st.session_state.get("api_token", "") or "",
                                llm_call=None,
                            )
                            if latency:
                                st.success(f"Quiz generated: {len(quiz_items)} items. (latency: {latency:.3f}s)")
                            else:
                                st.success(f"Quiz generated: {len(quiz_items)} items.")
                        else:
                            quiz_items, _ = generate_quiz(
                                stem=stem,
                                context_text=text,
                                n=int(n_q),
                                use_api_mode=False,
                                llm_call=llm,
                            )
                            st.success(f"Quiz generated: {len(quiz_items)} items (local).")

                        st.session_state[quiz_state_key] = quiz_items

                        try:
                            save_quiz_to_disk(stem, quiz_items)
                            if "srs_disk_quiz_items_cache" in st.session_state:
                                del st.session_state["srs_disk_quiz_items_cache"]
                        except Exception as e:
                            st.warning(f"Quiz saved to session but failed to save to disk: {e}")

                    except requests.HTTPError as he:
                        try:
                            detail = he.response.json().get("detail", str(he))
                        except Exception:
                            detail = str(he)
                        st.error(f"API quiz generation failed: {detail}")
                        st.exception(he)
                    except Exception as e:
                        st.error("Quiz generation failed.")
                        st.exception(e)

    # load quiz items from session_state
    quiz_items = st.session_state.get(quiz_state_key, [])

    def _mark_selection_made_local(sel_key: str):
        # wrap to use same name as previous callback; this will call the module-level helper
        _mark_selection_made(sel_key)

    st.markdown("")  # small spacer

    # Display MCQs with choices A-D and stable widget keys
    if not quiz_items:
        st.info("No quiz items generated yet. Click 'Generate quiz' to create items.")
        return

    # Render each question as a clean "card" with clear spacing & buttons
    seen_qids: Dict[str, int] = {}
    for idx, q in enumerate(quiz_items, start=1):
        with st.container():
            st.markdown('<div class="la-card"></div>', unsafe_allow_html=True)
            qid = q.get("id", str(uuid.uuid4()))
            seen_qids[qid] = seen_qids.get(qid, 0) + 1
            key_suffix = f"{qid}_{seen_qids[qid]}" if seen_qids[qid] > 1 else qid
            q_text = q.get("question", "")
            choices = q.get("choices", {}) or {}
            answer_letter = q.get("answer", None)
            explanation_text = q.get("explanation", "") or ""

            selection_key = f"{quiz_state_key}_sel_{key_suffix}"
            submit_key = f"{quiz_state_key}_sub_{key_suffix}"
            srs_key = f"{quiz_state_key}_srs_{key_suffix}"
            exp_key = f"{quiz_state_key}_expander_{key_suffix}"
            if exp_key not in st.session_state:
                st.session_state[exp_key] = False

            # Determine submission state early (unchanged logic)
            already_submitted = bool(st.session_state.get(submit_key))

            # Show a compact header with index, id (short), and status
            short_id = qid.split("-")[0] if "-" in qid else qid
            if seen_qids[qid] > 1:
                short_id = f"{short_id}-{seen_qids[qid]}"
            status_icon = ""
            if already_submitted:
                submitted = st.session_state.get(submit_key, {})
                if submitted.get("is_correct"):
                    status_icon = " ✅"
                else:
                    status_icon = " ❌"

            # Title preview (was expander title)
            title_preview = q_text[:120].replace("\n", " ")
            header_title = f"Q{idx} ({short_id}...){status_icon}: {title_preview}"
            st.markdown(f"### {header_title}")

            # Display full question with better typography
            st.markdown(f"**Question {idx}:**")
            # Use a monospace block for very long single-line questions to preserve readability,
            # otherwise normal markdown which will wrap gracefully.
            if "\n" in q_text or len(q_text) > 300:
                # long text — show a scrollable text area (read-only) to allow comfortable scanning
                st.text_area(label="", value=q_text, height=140, key=f"{quiz_state_key}_qtext_{key_suffix}", disabled=True)
            else:
                st.write(q_text)

            st.markdown("---")

            # Build dropdown options (identical strings to original behavior)
            placeholder = "Select an answer"
            dropdown_options = []
            for label in ["A", "B", "C", "D"]:
                opt_text = choices.get(label, "")
                dropdown_options.append(f"{label}. {opt_text}")
            dropdown_with_placeholder = [placeholder] + dropdown_options

            # Show the selectable control in a left column and a visual list of choices on the right
            c1, c2 = st.columns([1, 1])
            with c1:
                # Keep the same selectbox behavior (preserves stored values used elsewhere)
                try:
                    pre_index = 0
                    if selection_key in st.session_state:
                        cur = st.session_state.get(selection_key)
                        if cur in dropdown_with_placeholder:
                            pre_index = dropdown_with_placeholder.index(cur)
                        else:
                            pre_index = 0
                    st.selectbox(
                        "Choose an answer",
                        dropdown_with_placeholder,
                        index=pre_index,
                        key=selection_key,
                        disabled=already_submitted,
                        on_change=_mark_selection_made_local,
                        args=(selection_key,),
                    )
                except Exception:
                    st.selectbox(
                        "Choose an answer",
                        dropdown_with_placeholder,
                        key=selection_key,
                        disabled=already_submitted,
                        on_change=_mark_selection_made_local,
                        args=(selection_key,),
                    )

            # Visual list of choices to make scanning easier (read-only)
            with c2:
                st.markdown("**Choices:**")
                # present each choice on its own line with bold label
                for label in ["A", "B", "C", "D"]:
                    text = choices.get(label, "")
                    if text:
                        st.markdown(f"- **{label}.** {text}")
                    else:
                        st.markdown(f"- **{label}.** _(no option)_")

            # Derive the chosen letter maintaining original placeholder logic
            chosen_letter = None
            if already_submitted:
                submitted = st.session_state.get(submit_key, {})
                chosen_letter = submitted.get("chosen")
            else:
                sel_display = st.session_state.get(selection_key)
                if sel_display and sel_display != placeholder:
                    chosen_letter = sel_display.split(".", 1)[0].strip()

            # Action buttons grouped in a single bar
            with st.container():
                st.markdown('<div class="la-action-bar"></div>', unsafe_allow_html=True)
                st.markdown("**Actions**")
                btn_col_left, btn_col_mid, btn_col_right = st.columns([1, 1, 1])
                with btn_col_left:
                    check_key = f"{quiz_state_key}_check_{key_suffix}"
                    check_disabled = (chosen_letter is None) or already_submitted
                    if st.button("Check answer", key=check_key, disabled=check_disabled, use_container_width=True):
                        is_correct = (chosen_letter == answer_letter) if (chosen_letter and answer_letter) else False
                        st.session_state[submit_key] = {
                            "chosen": chosen_letter,
                            "is_correct": is_correct,
                        }
                        # --- persist the quiz result to confusion store so it survives restart ---
                        try:
                            # use question id and text if available
                            record_quiz_result(
                                qid=qid,
                                question=q_text,
                                is_correct=is_correct,
                                stem=stem,
                                question_item=q,
                                doc_id=doc_id,
                            )
                        except Exception as e:
                            # don't crash the UI; just log
                            print(f"[ui] failed to persist quiz result: {e}")

                        # mark state that would previously open the expander
                        st.session_state[exp_key] = True
                        # Force a rerun so the Confused section reloads persisted entries immediately
                        try:
                            st.experimental_rerun()
                        except Exception:
                            # If rerun fails (e.g., during testing), continue gracefully
                            pass

                with btn_col_mid:
                    # Toggle showing explanation manually (keeps same expander key so state persists)
                    show_expl_key = f"{quiz_state_key}_showex_{key_suffix}"
                    if show_expl_key not in st.session_state:
                        st.session_state[show_expl_key] = False
                    if st.button("Show / hide explanation", key=f"{quiz_state_key}_toggleexp_{key_suffix}", use_container_width=True):
                        st.session_state[show_expl_key] = not st.session_state[show_expl_key]
                        st.session_state[exp_key] = True  # keep the question visible when toggling

                with btn_col_right:
                    # Start SRS (keeps same key and behavior)
                    if st.button("Add to SRS", key=srs_key, use_container_width=True):
                        try:
                            mgr = SRSManager()
                            mgr.ensure_card(qid, meta={"question": q_text or "", "stem": stem, "source_reason": "Added from quiz review"})
                            if "srs_quiz_items_cache" not in st.session_state:
                                st.session_state["srs_quiz_items_cache"] = {}
                            st.session_state["srs_quiz_items_cache"][qid] = {
                                "id": qid,
                                "question": q_text or "",
                                "choices": choices or {},
                                "answer": answer_letter,
                                "explanation": explanation_text or "",
                            }
                            st.session_state[f"{srs_key}_done"] = True
                            st.info(f"Registered card {qid} in SRS.")
                        except Exception as e:
                            st.error("Failed to register SRS card.")
                            st.exception(e)

            st.markdown("")  # slight spacing

            # After submission: show feedback in a dedicated nicely formatted panel
            submitted = st.session_state.get(submit_key)
            if submitted:
                chosen = submitted.get("chosen")
                is_correct = submitted.get("is_correct", False)

                if is_correct:
                    # Success box with explanation (if provided)
                    if explanation_text:
                        st.success("✅ Correct — well done!")
                        # explanation area shown inline
                        st.markdown("**Explanation**")
                        if len(explanation_text) > 300:
                            st.text_area("", value=explanation_text, height=160, disabled=True)
                        else:
                            st.write(explanation_text)
                    else:
                        st.success("✅ Correct")
                else:
                    # Incorrect path: show what they chose and the correct answer
                    correct_display = "(not provided)"
                    if answer_letter and choices.get(answer_letter):
                        correct_display = f"{answer_letter}. {choices.get(answer_letter)}"
                    user_choice_text = ""
                    if chosen:
                        user_choice_text = choices.get(chosen, "")
                        st.error(f"❌ Incorrect — you chose **{chosen}**. {user_choice_text}\n\n**Correct:** {correct_display}")
                    else:
                        st.error(f"❌ Incorrect.\n\n**Correct:** {correct_display}")

                    # show explanation if available, inline
                    if explanation_text:
                        st.markdown("**Explanation**")
                        if len(explanation_text) > 300:
                            st.text_area("", value=explanation_text, height=160, disabled=True)
                        else:
                            st.write(explanation_text)

            # Optionally show a small divider between questions
            st.markdown("---")
    total_answered = 0
    total_wrong = 0
    for q in quiz_items:
        qid = q.get("id", "")
        submit_key = f"{quiz_state_key}_sub_{qid}"
        submitted = st.session_state.get(submit_key)
        if submitted:
            total_answered += 1
            if not submitted.get("is_correct", False):
                total_wrong += 1

    if total_answered:
        st.markdown("### 🔁 Next step recommendations")
        st.info(f"You answered {total_answered} question(s). Missed: {total_wrong}.")
        if total_wrong > 0:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Add missed concepts to SRS", key=f"quiz_push_missed_{stem}", use_container_width=True):
                    try:
                        mgr = SRSManager()
                        added = 0
                        for q in quiz_items:
                            qid = q.get("id", "")
                            submit_key = f"{quiz_state_key}_sub_{qid}"
                            sub = st.session_state.get(submit_key)
                            if sub and not sub.get("is_correct", False):
                                mgr.ensure_card(qid, meta={"question": q.get("question", ""), "stem": stem, "source_reason": "Added because you missed this in quiz"})
                                added += 1
                        st.success(f"Added {added} missed concept(s) to SRS.")
                    except Exception as e:
                        st.error(f"Could not add missed concepts: {e}")
            with c2:
                if st.button("Review top confused concepts", key=f"quiz_focus_confused_{stem}", use_container_width=True):
                    st.session_state[f"focus_confused_{stem}"] = True
                    st.success("Jump to Confused section below.")

    # End loop over quiz items — UI improved but all keys + behavior preserved.
