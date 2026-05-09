"""
Quiz UI — UX-optimized presentation.

Changes vs previous version:
  - After answer submission the flat explanation block is replaced with:
      1. A short 1-2 line brief_explanation shown inline (no label prefix).
      2. An st.expander("Show reasoning") that contains:
           • Why the correct answer is correct  (why_correct)
           • Why each wrong option is wrong     (why_wrong, skips correct letter)
           • Source from lecture                (source_chunk, styled as a blockquote)
  - Falls back gracefully when new fields are absent (old quiz items on disk).
  - All session_state keys, widget keys, and submission logic are unchanged.
"""

import uuid
import hashlib
import time
import re
import requests
from pathlib import Path
from typing import Any, List, Dict
import os
import streamlit as st

from frontend.handlers import (
    generate_quiz,
    save_quiz_to_disk,
    record_quiz_result,
)
from backend.study_srs import SRSManager


def _normalize_quiz_question_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _quiz_session_chunk_counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for q in items or []:
        cid = (q.get("chunk_id") or q.get("source_chunk_id") or "").strip().lower()
        if not cid:
            continue
        counts[cid] = counts.get(cid, 0) + 1
    return counts


def _mark_selection_made(sel_key: str):
    st.session_state[f"{sel_key}_made"] = True


def _fallback_brief(q_text: str, choices: Dict[str, str], answer_letter: str) -> str:
    ans = (answer_letter or "").strip().upper()
    if ans and isinstance(choices, dict) and choices.get(ans):
        return (
            f"The best-supported answer is **{ans}. {choices.get(ans)}** based on the question context."
        )
    if q_text:
        return "Review the question and key concept, then retry."
    return "No explanation available for this item."


def _render_detailed_explanation(
    detailed: Dict,
    choices: Dict[str, str],
    answer_letter: str,
) -> None:
    """
    Render the contents of the 'Show reasoning' expander.
    Handles missing / empty fields gracefully.
    """
    ans = (answer_letter or "").strip().upper()

    # ── Why the correct answer is correct ──────────────────────────────
    why_correct = (detailed.get("why_correct") or "").strip()
    if why_correct:
        correct_label = f"{ans}. {choices.get(ans, '')}" if ans and choices.get(ans) else ans
        st.markdown(f"**✅ Why {correct_label} is correct**")
        st.markdown(why_correct)
    else:
        # No detailed reasoning from the LLM — don't repeat the brief.
        st.markdown("*Detailed reasoning was not generated for this question.*")

    st.markdown("")

    # ── Why each distractor is wrong ───────────────────────────────────
    why_wrong: Dict = detailed.get("why_wrong") or {}
    distractor_lines = []
    for letter in ["A", "B", "C", "D"]:
        if letter == ans:
            continue   # skip correct letter
        reason = (why_wrong.get(letter) or "").strip()
        choice_text = (choices.get(letter) or "").strip()
        if reason:
            distractor_lines.append(f"**{letter}. {choice_text}** — {reason}")
        elif choice_text:
            distractor_lines.append(f"**{letter}. {choice_text}** — *(no explanation provided)*")

    if distractor_lines:
        st.markdown("**❌ Why the other options are wrong**")
        for line in distractor_lines:
            st.markdown(f"- {line}")

    # ── Source chunk ───────────────────────────────────────────────────
    source_chunk = (detailed.get("source_chunk") or "").strip()
    if source_chunk:
        st.markdown("")
        st.markdown("**📖 From the lecture**")
        # Render as a blockquote-style info box
        st.info(f"*\"{source_chunk}\"*")


def render(st: Any, stem: str, text: str, llm, hist_key: str):
    """
    Render the Quiz generation + Quiz item UI.
    """
    API_DEFAULT = os.getenv("API_BASE", "http://localhost:8000") if "os" in globals() else "http://localhost:8000"

    st.markdown("---")
    st.markdown('<a id="study-quiz"></a>', unsafe_allow_html=True)
    st.subheader("📝 Study / Quiz")
    st.markdown(
        "Generate short multiple-choice quizzes from your lecture. "
        "Read each card, select an answer, then **Check answer**. "
        "Use **Add to SRS** to save questions you'd like to review later."
    )

    quiz_state_key = f"quiz_items_{stem}"
    quiz_generation_key = f"{quiz_state_key}_generation"
    wave_breaks_key = f"{quiz_state_key}_wave_breaks"
    if quiz_generation_key not in st.session_state:
        st.session_state[quiz_generation_key] = 0
    doc_id_key = f"doc_id_{stem}"
    if text is not None:
        doc_id = hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]
        st.session_state[doc_id_key] = doc_id
    else:
        doc_id = st.session_state.get(doc_id_key, "")

    cols_top = st.columns([2, 1])
    with cols_top[0]:
        n_q = st.number_input("Number of quiz items", min_value=1, max_value=20, value=5,
                               help="How many MCQs to generate from this lecture.")
    with cols_top[1]:
        gen_key = f"gen_quiz_{stem}"
        if st.button("Generate quiz", key=gen_key, type="primary", use_container_width=True):
            if not text:
                st.warning("No document text extracted.")
            else:
                request_sig = hashlib.sha1(
                    f"{stem}|{int(n_q)}|{(text or '')[:1000]}|{len(text or '')}".encode("utf-8")
                ).hexdigest()
                req_key = f"{quiz_state_key}_last_request"
                last_req = st.session_state.get(req_key, {})
                now_ts = time.time()
                if (
                    isinstance(last_req, dict)
                    and last_req.get("sig") == request_sig
                    and (now_ts - float(last_req.get("ts", 0.0))) < 8.0
                    and st.session_state.get(quiz_state_key)
                ):
                    st.info("Using the most recent generated quiz (skipped duplicate request).")
                else:
                    with st.spinner("Generating quiz..."):
                        try:
                            if st.session_state.get("use_api_mode", False):
                                quiz_items, latency = generate_quiz(
                                    stem=stem, context_text=text, n=int(n_q),
                                    use_api_mode=True,
                                    api_base=os.getenv("API_BASE", API_DEFAULT),
                                    token=st.session_state.get("api_token", "") or "",
                                    llm_call=None,
                                    session_chunk_counts=None,
                                )
                                msg = f"Quiz generated: {len(quiz_items)} items."
                                if latency:
                                    msg += f" (latency: {latency:.3f}s)"
                                st.success(msg)
                            else:
                                quiz_items, _ = generate_quiz(
                                    stem=stem, context_text=text, n=int(n_q),
                                    use_api_mode=False, llm_call=llm,
                                    session_chunk_counts=None,
                                )
                                st.success(f"Quiz generated: {len(quiz_items)} items (local).")

                            st.session_state[quiz_state_key] = quiz_items
                            st.session_state[wave_breaks_key] = []
                            st.session_state[quiz_generation_key] = (
                                int(st.session_state.get(quiz_generation_key, 0)) + 1
                            )
                            st.session_state[req_key] = {"sig": request_sig, "ts": now_ts}

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

    quiz_items = st.session_state.get(quiz_state_key, [])
    generation_token = int(st.session_state.get(quiz_generation_key, 0))

    def _mark_selection_made_local(sel_key: str):
        _mark_selection_made(sel_key)

    st.markdown("")

    if not quiz_items:
        st.info("No quiz items generated yet. Click 'Generate quiz' to create items.")
        return

    seen_qids: Dict[str, int] = {}
    quiz_render_items: List[Dict[str, Any]] = []

    wave_breaks_raw = st.session_state.get(wave_breaks_key, [])
    wave_breaks = set(wave_breaks_raw) if isinstance(wave_breaks_raw, list) else set()

    for idx, q in enumerate(quiz_items, start=1):
        if idx in wave_breaks:
            st.markdown("---")
            st.markdown("### Additional questions")

        with st.container():
            st.markdown('<div class="la-card"></div>', unsafe_allow_html=True)
            qid = q.get("id", str(uuid.uuid4()))
            seen_qids[qid] = seen_qids.get(qid, 0) + 1
            key_suffix = f"{qid}_{seen_qids[qid]}" if seen_qids[qid] > 1 else qid
            q_text = q.get("question", "")
            choices = q.get("choices", {}) or {}
            answer_letter = q.get("answer", None)

            # ── explanation fields (new schema with old-item fallback) ──
            brief_explanation = (q.get("brief_explanation") or q.get("explanation") or "").strip()
            detailed_explanation: Dict = q.get("detailed_explanation") or {}

            selection_key = f"{quiz_state_key}_sel_{generation_token}_{key_suffix}"
            submit_key   = f"{quiz_state_key}_sub_{generation_token}_{key_suffix}"
            srs_key      = f"{quiz_state_key}_srs_{generation_token}_{key_suffix}"
            exp_key      = f"{quiz_state_key}_expander_{generation_token}_{key_suffix}"
            if exp_key not in st.session_state:
                st.session_state[exp_key] = False

            quiz_render_items.append({
                "question": q,
                "qid": qid,
                "key_suffix": key_suffix,
                "submit_key": submit_key,
            })

            already_submitted = bool(st.session_state.get(submit_key))

            status_icon = ""
            if already_submitted:
                submitted = st.session_state.get(submit_key, {})
                status_icon = " ✅" if submitted.get("is_correct") else " ❌"

            st.markdown(f"### Q{idx}{status_icon}")
            if q_text:
                st.markdown(q_text)
            st.markdown("---")

            placeholder = "Select an answer"
            dropdown_options = [f"{lbl}. {choices.get(lbl, '')}" for lbl in ["A", "B", "C", "D"]]
            dropdown_with_placeholder = [placeholder] + dropdown_options

            c1, c2 = st.columns([1, 1])
            with c1:
                try:
                    pre_index = 0
                    if selection_key in st.session_state:
                        cur = st.session_state.get(selection_key)
                        if cur in dropdown_with_placeholder:
                            pre_index = dropdown_with_placeholder.index(cur)
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

            with c2:
                st.markdown("**Choices:**")
                correct_letter_upper = (answer_letter or "").strip().upper()
                chosen_letter_post = ""
                if already_submitted:
                    chosen_letter_post = (
                        st.session_state.get(submit_key, {}).get("chosen", "") or ""
                    ).strip().upper()
                for label in ["A", "B", "C", "D"]:
                    opt = choices.get(label, "")
                    # After submission, mark the correct answer green and
                    # the chosen-but-wrong answer red so users can scan the
                    # outcome at a glance without rereading the alert above.
                    marker = ""
                    if already_submitted:
                        if label == correct_letter_upper:
                            marker = "✅ "
                        elif label == chosen_letter_post:
                            marker = "❌ "
                    if opt:
                        st.markdown(f"- {marker}**{label}.** {opt}")
                    else:
                        st.markdown(f"- {marker}**{label}.** _(no option)_")

            # Derive chosen letter
            chosen_letter = None
            if already_submitted:
                chosen_letter = st.session_state.get(submit_key, {}).get("chosen")
            else:
                sel_display = st.session_state.get(selection_key)
                if sel_display and sel_display != placeholder:
                    chosen_letter = sel_display.split(".", 1)[0].strip()

            # Action buttons
            with st.container():
                st.markdown('<div class="la-action-bar"></div>', unsafe_allow_html=True)
                st.markdown("**Actions**")
                btn_col_left, btn_col_right = st.columns([1, 1])
                with btn_col_left:
                    check_key = f"{quiz_state_key}_check_{key_suffix}"
                    check_disabled = (chosen_letter is None) or already_submitted
                    if st.button("Check answer", key=check_key,
                                 disabled=check_disabled, use_container_width=True):
                        is_correct = (
                            (chosen_letter == answer_letter)
                            if (chosen_letter and answer_letter) else False
                        )
                        st.session_state[submit_key] = {
                            "chosen": chosen_letter,
                            "is_correct": is_correct,
                        }
                        try:
                            record_quiz_result(
                                qid=qid,
                                question=q_text,
                                is_correct=is_correct,
                                stem=stem,
                                question_item=q,
                                chosen_answer=chosen_letter or "",
                                doc_id=doc_id,
                            )
                        except Exception as e:
                            print(f"[ui] failed to persist quiz result: {e}")

                        st.session_state[exp_key] = True
                        try:
                            st.experimental_rerun()
                        except Exception:
                            pass

                with btn_col_right:
                    if st.button("Add to SRS", key=srs_key, use_container_width=True):
                        try:
                            mgr = SRSManager()
                            mgr.ensure_card(
                                qid,
                                meta={
                                    "question": q_text or "",
                                    "choices": choices or {},
                                    "answer": answer_letter,
                                    "brief_explanation": brief_explanation,
                                    "detailed_explanation": detailed_explanation,
                                    "stem": stem,
                                    "item_type": "mcq",
                                    "origin": "quiz_mcq",
                                    "quiz_question_id": qid,
                                    "source_reason": "Added from quiz review",
                                },
                            )
                            if "srs_quiz_items_cache" not in st.session_state:
                                st.session_state["srs_quiz_items_cache"] = {}
                            st.session_state["srs_quiz_items_cache"][qid] = {
                                "id": qid,
                                "question": q_text or "",
                                "choices": choices or {},
                                "answer": answer_letter,
                                "brief_explanation": brief_explanation,
                                "detailed_explanation": detailed_explanation,
                                "item_type": "mcq",
                                "origin": "quiz_mcq",
                            }
                            st.session_state[f"{srs_key}_done"] = True
                            st.info("Added to SRS.")
                            try:
                                st.rerun()
                            except Exception:
                                try:
                                    st.experimental_rerun()
                                except Exception:
                                    pass
                        except Exception as e:
                            st.error("Failed to register SRS card.")
                            st.exception(e)

            st.markdown("")

            # ── Post-submission feedback (NEW 2-layer layout) ────────────
            submitted = st.session_state.get(submit_key)
            if submitted:
                chosen = submitted.get("chosen")
                is_correct = submitted.get("is_correct", False)

                if is_correct:
                    st.success("✅ Correct — well done!")
                else:
                    correct_display = "(not provided)"
                    if answer_letter and choices.get(answer_letter):
                        correct_display = f"{answer_letter}. {choices.get(answer_letter)}"
                    if chosen:
                        chosen_text = choices.get(chosen, "")
                        st.error(
                            f"❌ Incorrect — you chose **{chosen}**. {chosen_text}\n\n"
                            f"**Correct:** {correct_display}"
                        )
                    else:
                        st.error(f"❌ Incorrect.\n\n**Correct:** {correct_display}")

                # ── Layer 1: brief explanation (inline, no label) ─────────
                effective_brief = brief_explanation or _fallback_brief(
                    q_text, choices, answer_letter
                )
                if effective_brief:
                    st.markdown(effective_brief)

                # ── Layer 2: expandable detailed reasoning ────────────────
                # Only show the expander when the LLM produced genuinely distinct
                # detailed content — never open it just to repeat the brief.
                has_detail = bool(
                    (detailed_explanation.get("why_correct") or "").strip()
                    or any(
                        (detailed_explanation.get("why_wrong") or {}).get(l, "").strip()
                        for l in ["A", "B", "C", "D"]
                        if l != (answer_letter or "").upper()
                    )
                    or (detailed_explanation.get("source_chunk") or "").strip()
                )

                if has_detail:
                    with st.expander("🔽 Show reasoning"):
                        _render_detailed_explanation(
                            detailed_explanation,
                            choices,
                            answer_letter or "",
                        )

            st.markdown("---")

    # ── Aggregate progress across the current batch ───────────────────────
    total_questions = len(quiz_items)
    total_answered = 0
    total_correct = 0
    total_wrong = 0
    wrong_topic_counts: Dict[str, int] = {}
    for render_item in quiz_render_items:
        submitted = st.session_state.get(render_item["submit_key"])
        if not submitted:
            continue
        total_answered += 1
        if submitted.get("is_correct", False):
            total_correct += 1
            continue
        total_wrong += 1
        q = render_item["question"]
        topic = (
            (q.get("concept_label") or "").strip()
            or (q.get("source_chunk_preview") or "").strip()
            or (q.get("source_chunk") or "").strip()
        )
        if topic:
            # Keep the topic display short for the summary card.
            if len(topic) > 120:
                topic = topic[:120].rstrip() + "…"
            wrong_topic_counts[topic] = wrong_topic_counts.get(topic, 0) + 1

    is_quiz_complete = total_questions > 0 and total_answered == total_questions

    # ── Completion panel: score, accuracy, encouragement, continue ────────
    if is_quiz_complete:
        accuracy_pct = (total_correct / total_questions) * 100.0 if total_questions else 0.0
        if accuracy_pct >= 90:
            encouragement = "🎯 Excellent work — you've got a strong grip on this lecture."
        elif accuracy_pct >= 70:
            encouragement = "✨ You're improving — a couple more rounds will lock this in."
        elif accuracy_pct >= 50:
            encouragement = "💡 Keep practicing this lecture — you're getting there."
        else:
            encouragement = "📚 Keep practicing — try reviewing the lecture and giving it another go."

        st.markdown("### ✅ Quiz complete")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Score", f"{total_correct} / {total_questions}")
        with m2:
            st.metric("Accuracy", f"{accuracy_pct:.0f}%")
        st.markdown(encouragement)

        if wrong_topic_counts:
            top_topic, _ = max(
                wrong_topic_counts.items(),
                key=lambda kv: kv[1],
            )
            st.markdown(f"**🧠 You struggled most with:** {top_topic}")

        # Continue Quiz: append N more questions without resetting prior state.
        cont_key = f"{quiz_state_key}_continue_{generation_token}"
        cont_n = max(1, int(n_q))
        if st.button(
            f"🔄 Continue with {cont_n} more question(s)",
            key=cont_key,
            type="primary",
            use_container_width=True,
        ):
            if not text:
                st.warning("No document text — cannot continue.")
            else:
                with st.spinner("Generating more questions..."):
                    try:
                        chunk_counts = _quiz_session_chunk_counts(quiz_items)
                        if st.session_state.get("use_api_mode", False):
                            more_items, _ = generate_quiz(
                                stem=stem,
                                context_text=text,
                                n=cont_n,
                                use_api_mode=True,
                                api_base=os.getenv("API_BASE", API_DEFAULT),
                                token=st.session_state.get("api_token", "") or "",
                                llm_call=None,
                                session_chunk_counts=chunk_counts,
                            )
                        else:
                            more_items, _ = generate_quiz(
                                stem=stem,
                                context_text=text,
                                n=cont_n,
                                use_api_mode=False,
                                llm_call=llm,
                                session_chunk_counts=chunk_counts,
                            )

                        existing_ids = {
                            (item.get("id") or "")
                            for item in quiz_items
                            if item.get("id")
                        }
                        existing_qnorm = {
                            _normalize_quiz_question_text(item.get("question") or "")
                            for item in quiz_items
                        }
                        appended = []
                        for item in more_items or []:
                            tid = (item.get("id") or "").strip()
                            qnorm = _normalize_quiz_question_text(item.get("question") or "")
                            if tid and tid in existing_ids:
                                continue
                            if qnorm and qnorm in existing_qnorm:
                                continue
                            appended.append(item)
                            if tid:
                                existing_ids.add(tid)
                            if qnorm:
                                existing_qnorm.add(qnorm)

                        if appended:
                            first_new_idx = len(quiz_items) + 1
                            waves = st.session_state.get(wave_breaks_key, [])
                            if not isinstance(waves, list):
                                waves = []
                            waves.append(first_new_idx)
                            st.session_state[wave_breaks_key] = waves

                            combined = list(quiz_items) + appended
                            st.session_state[quiz_state_key] = combined
                            try:
                                save_quiz_to_disk(stem, combined)
                                if "srs_disk_quiz_items_cache" in st.session_state:
                                    del st.session_state["srs_disk_quiz_items_cache"]
                            except Exception as e:
                                st.warning(
                                    f"Continued in session but failed to save to disk: {e}"
                                )
                            st.success(f"Added {len(appended)} new question(s).")
                            try:
                                st.rerun()
                            except Exception:
                                try:
                                    st.experimental_rerun()
                                except Exception:
                                    pass
                        else:
                            st.info(
                                "No new unique questions were generated — try clicking again "
                                "or change the number of items."
                            )
                    except requests.HTTPError as he:
                        try:
                            detail = he.response.json().get("detail", str(he))
                        except Exception:
                            detail = str(he)
                        st.error(f"Continue failed: {detail}")
                    except Exception as e:
                        st.error("Continue failed.")
                        st.exception(e)

        st.markdown("")

    # ── Next-step recommendations (preserved logic for missed items) ──────
    if total_answered and total_wrong > 0:
        st.markdown("### 🔁 Next step recommendations")
        st.markdown(f"You answered {total_answered} question(s). Missed: {total_wrong}.")
        if st.button("Add missed concepts to SRS",
                     key=f"quiz_push_missed_{stem}", use_container_width=True):
            try:
                mgr = SRSManager()
                added = 0
                seen_card_ids = set()
                if "srs_quiz_items_cache" not in st.session_state:
                    st.session_state["srs_quiz_items_cache"] = {}
                for render_item in quiz_render_items:
                    sub = st.session_state.get(render_item["submit_key"])
                    if not sub or sub.get("is_correct", False):
                        continue
                    q = render_item["question"]
                    base_qid = render_item["qid"] or ""
                    key_suffix = render_item["key_suffix"]
                    card_id = (
                        f"{base_qid}_{generation_token}_{key_suffix}"
                        if base_qid
                        else f"missed_{generation_token}_{key_suffix}"
                    )
                    if card_id in seen_card_ids:
                        continue
                    seen_card_ids.add(card_id)
                    mgr.ensure_card(
                        card_id,
                        meta={
                            "question": q.get("question", ""),
                            "choices": q.get("choices", {}) or {},
                            "answer": q.get("answer", None),
                            "brief_explanation": q.get("brief_explanation", ""),
                            "detailed_explanation": q.get("detailed_explanation", {}),
                            "stem": stem,
                            "item_type": "mcq",
                            "origin": "quiz_mcq",
                            "quiz_question_id": base_qid or card_id,
                            "source_reason": "Added because you missed this in quiz session",
                        },
                    )
                    st.session_state["srs_quiz_items_cache"][card_id] = {
                        "id": card_id,
                        "question": q.get("question", ""),
                        "choices": q.get("choices", {}) or {},
                        "answer": q.get("answer", None),
                        "brief_explanation": q.get("brief_explanation", ""),
                        "detailed_explanation": q.get("detailed_explanation", {}),
                        "item_type": "mcq",
                        "origin": "quiz_mcq",
                    }
                    added += 1
                st.success(f"Added {added} missed concept(s) to SRS.")
                st.markdown("Jump to Confused section below.")
            except Exception as e:
                st.error(f"Could not add missed concepts: {e}")

    elif total_answered and total_wrong == 0 and not is_quiz_complete:
        # Partial progress with no misses yet — keep the lightweight nudge.
        # Once the user finishes the batch, the completion panel above takes
        # over with a richer score / continue affordance.
        st.markdown("### 🔁 Next step recommendations")
        st.markdown(f"You answered {total_answered} question(s). Missed: 0.")
        st.markdown("Great progress so far — keep going.")