# frontend/sections/quiz.py
"""
Quiz UI extraction from frontend/app.py

Expose:
    def render(st, stem, text, llm, hist_key):
        ...

No behavioral changes; preserves session_state keys and widget keys.
"""

import uuid
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
    st.subheader("Study / Quiz (MCQ v1)")

    # session_state key for storing generated quiz for this document
    quiz_state_key = f"quiz_items_{stem}"
    n_q = st.number_input("Number of quiz items", min_value=1, max_value=20, value=5)
    gen_key = f"gen_quiz_{stem}"
    if st.button("Generate Quiz from lecture", key=gen_key):
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

    # Display MCQs with choices A-D and stable widget keys
    if not quiz_items:
        st.info("No quiz items generated yet. Click 'Generate Quiz from lecture' to create items.")
    else:
        for q in quiz_items:
            qid = q.get("id", str(uuid.uuid4()))
            q_text = q.get("question", "")
            choices = q.get("choices", {}) or {}
            answer_letter = q.get("answer", None)
            explanation_text = q.get("explanation", "") or ""

            selection_key = f"{quiz_state_key}_sel_{qid}"
            submit_key = f"{quiz_state_key}_sub_{qid}"
            srs_key = f"{quiz_state_key}_srs_{qid}"
            exp_key = f"{quiz_state_key}_expander_{qid}"
            if exp_key not in st.session_state:
                st.session_state[exp_key] = False

            with st.expander(f"Q ({qid}): {q_text[:140]}", expanded=st.session_state[exp_key]):
                st.write(q_text)

                dropdown_options = []
                for label in ["A", "B", "C", "D"]:
                    opt_text = choices.get(label, "")
                    dropdown_options.append(f"{label}. {opt_text}")

                placeholder = "Select an answer"
                dropdown_with_placeholder = [placeholder] + dropdown_options

                already_submitted = bool(st.session_state.get(submit_key))

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

                chosen_letter = None
                if already_submitted:
                    submitted = st.session_state.get(submit_key, {})
                    chosen_letter = submitted.get("chosen")
                else:
                    sel_display = st.session_state.get(selection_key)
                    if sel_display and sel_display != placeholder:
                        chosen_letter = sel_display.split(".", 1)[0].strip()

                check_key = f"{quiz_state_key}_check_{qid}"
                check_disabled = (chosen_letter is None) or already_submitted

                if st.button("Check answer", key=check_key, disabled=check_disabled):
                    is_correct = (chosen_letter == answer_letter) if (chosen_letter and answer_letter) else False
                    st.session_state[submit_key] = {
                        "chosen": chosen_letter,
                        "is_correct": is_correct,
                    }
                    # --- persist the quiz result to confusion store so it survives restart ---
                    try:
                        # use question id and text if available
                        record_quiz_result(qid=qid, question=q_text, is_correct=is_correct)
                    except Exception as e:
                        # don't crash the UI; just log
                        print(f"[ui] failed to persist quiz result: {e}")

                    st.session_state[exp_key] = True

                submitted = st.session_state.get(submit_key)
                if submitted:
                    chosen = submitted.get("chosen")
                    is_correct = submitted.get("is_correct", False)

                    if is_correct:
                        if explanation_text:
                            st.success(f"✅ Correct\n\n{explanation_text}")
                        else:
                            st.success("✅ Correct")
                    else:
                        correct_display = "(not provided)"
                        if answer_letter and choices.get(answer_letter):
                            correct_display = f"{answer_letter}. {choices.get(answer_letter)}"
                        user_choice_text = ""
                        if chosen:
                            user_choice_text = choices.get(chosen, "")
                            st.error(f"❌ Incorrect — you chose {chosen}. {user_choice_text}\n\nCorrect: {correct_display}")
                        else:
                            st.error(f"❌ Incorrect.\n\nCorrect: {correct_display}")

                        if explanation_text:
                            st.write("Explanation:")
                            st.write(explanation_text)

                if st.button(f"Start SRS for {qid}", key=srs_key):
                    try:
                        mgr = SRSManager()
                        mgr.ensure_card(qid)
                        st.session_state[f"{srs_key}_done"] = True
                        st.info(f"Registered card {qid} in SRS.")
                    except Exception as e:
                        st.error("Failed to register SRS card.")
                        st.exception(e)
