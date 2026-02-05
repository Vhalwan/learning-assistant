"""
Spaced Repetition (SRS) UI extracted from frontend/app.py — UX-optimized presentation only.

Expose:
    def render(st):
        ...

All SRS logic, state keys and persistence are unchanged — only presentation/layout
and affordances are improved for readability and UX.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

# Backend SRS manager and intervals (same as used by app.py)
from backend.study_srs import SRSManager, INTERVALS

# Handler wrappers used to load quiz items from disk if missing
from frontend.handlers import load_all_quiz_items_wrapper, load_quiz_item_by_id_wrapper

def _shorten(text: str, limit: int = 80) -> str:
    if not text:
        return ""
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rsplit(" ", 1)[0] + "..."


def _card_stem(card_id: str, meta: Dict | None) -> str:
    if meta and meta.get("stem"):
        return str(meta.get("stem"))
    if card_id and "_" in card_id:
        return card_id.split("_", 1)[0]
    return ""


def _infer_stem_from_session(session: Dict) -> str | None:
    """
    Try to infer the current `stem` used in app.py from session keys.
    Strategy:
      0. use session["current_stem"] if present
      1. find stems that appear in both quiz_items_{stem} and chat_history_{stem}
      2. else use the first quiz_items_{stem}
      3. else use the first chat_history_{stem}
      4. else return None
    """
    if session.get("current_stem"):
        return session.get("current_stem")
    keys = list(session.keys())
    quiz_keys = [k for k in keys if k.startswith("quiz_items_")]
    chat_keys = [k for k in keys if k.startswith("chat_history_")]

    quiz_stems = [k[len("quiz_items_"):] for k in quiz_keys]
    chat_stems = [k[len("chat_history_"):] for k in chat_keys]

    common = [s for s in quiz_stems if s in chat_stems]
    if common:
        return common[0]
    if quiz_stems:
        return quiz_stems[0]
    if chat_stems:
        return chat_stems[0]
    return None


def _pretty_date_delta(iso_ts: str) -> str:
    """Return a human friendly days-until string for an ISO datetime or empty string."""
    if not iso_ts:
        return "N/A"
    try:
        next_due_dt = datetime.fromisoformat(iso_ts)
        days_until = (next_due_dt - datetime.utcnow()).days
        if days_until <= 0:
            return "Due now"
        if days_until == 1:
            return "Due in 1 day"
        return f"Due in {days_until} days"
    except Exception:
        # fallback to short iso prefix
        return iso_ts[:10] if iso_ts else "N/A"

def _card_doc_name(card_id: str) -> str:
    if "_" not in card_id:
        return "Unknown"
    return card_id.rsplit("_", 1)[0]


def render(st: Any):
    """
    Render the SRS review UI. Attempts to infer the current `stem` from st.session_state.
    Preserves all session_state keys and SRSManager behavior exactly.
    """
    # Infer stem (same behavior as original)
    stem = _infer_stem_from_session(st.session_state)
    quiz_state_key = f"quiz_items_{stem}" if stem else None

    st.markdown("---")
    st.markdown('<a id="spaced-repetition-review"></a>', unsafe_allow_html=True)
    st.subheader("📚 Spaced Repetition Review")

    with st.expander("ℹ️ What is Spaced Repetition?", expanded=False):
        st.markdown(
            """
            **Spaced Repetition** helps you remember information long-term by reviewing it at increasing intervals.

            **How it works (quick):**
            - Correct → schedule further out (1 → 3 → 7 → 14 → 30 days)
            - Incorrect → reset to 1 day to strengthen recall

            **To get started:**
            1. Generate a quiz from your PDF above
            2. Click **Start SRS** on questions you want to review later
            3. Come back here to review cards when they're due
            """
        )

    try:
        srs_mgr = SRSManager()
        all_cards = list(srs_mgr._data.keys())
        due_cards = srs_mgr.get_due_cards()

        # Top-level guidance if no cards exist (unchanged logic)
        if not all_cards:
            st.info(
                "📝 **No cards registered yet.**\n\n"
                "To start using spaced repetition:\n"
                "1. Generate a quiz from your PDF above\n"
                "2. Click **'Start SRS'** on any question you want to review later\n"
                "3. Come back here to review when cards are due!"
            )
            return

        # If there is an active quiz in the current session, show tips about unregistered questions
        current_quiz = st.session_state.get(quiz_state_key, []) if quiz_state_key else []
        if current_quiz:
            registered_ids = set(all_cards)
            unregistered = [q for q in current_quiz if q.get("id") not in registered_ids]
            if unregistered:
                st.info(
                    f"💡 **Tip:** You have {len(unregistered)} quiz question(s) generated above. "
                    "Click **'Start SRS'** on any question to add it to your spaced repetition review!"
                )
        show_all_lectures = st.checkbox(
            "📚 Show cards from all lectures",
            value=False,
            help="When disabled, SRS shows only cards tied to the currently uploaded lecture.",
            key="srs_show_all_lectures",
        )

        if stem and not show_all_lectures:
            filtered_cards = [cid for cid in all_cards if _card_stem(cid, srs_mgr.get_card_meta(cid)) == stem]
            filtered_due_cards = [cid for cid in due_cards if _card_stem(cid, srs_mgr.get_card_meta(cid)) == stem]
        else:
            filtered_cards = all_cards
            filtered_due_cards = due_cards

        if stem and not show_all_lectures and not filtered_cards:
            st.info("No SRS cards found for the current lecture yet. Start SRS on a quiz question to add one.")
        # --- Metrics row (clear visual hierarchy) ---
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            st.metric("Total cards", len(filtered_cards))
        with col_stats2:
            st.metric(
                "Due now",
                len(filtered_due_cards),
                delta=None if len(filtered_due_cards) == 0 else f"{len(filtered_due_cards)} to review",
            )
        with col_stats3:
            reviewed_count = sum(
                1 for cid in filtered_cards if srs_mgr.get_card_meta(cid).get("review_count", 0) > 0
            )
            st.metric("Reviewed", reviewed_count)

        # If there are due cards, present them grouped by originating document for scanability
        if filtered_due_cards:
            st.success(f"🎯 You have {len(filtered_due_cards)} card(s) due for review")
            st.markdown("---")

            # Build cache of quiz items (session cache key matches original app behavior)
            cache_key = "srs_quiz_items_cache"
            if cache_key not in st.session_state:
                st.session_state[cache_key] = {}

            all_quiz_items = st.session_state[cache_key].copy()

            # Merge any quiz questions generated in this session
            current_quiz = st.session_state.get(quiz_state_key, []) if quiz_state_key else []
            for q in current_quiz:
                all_quiz_items[q.get("id")] = q

            # Fill missing cards from disk cache if possible (same logic as original)
            missing_card_ids = [cid for cid in filtered_due_cards if cid not in all_quiz_items]
            if missing_card_ids:
                disk_cache_key = "srs_disk_quiz_items_cache"
                if disk_cache_key not in st.session_state:
                    disk_quiz_items = load_all_quiz_items_wrapper()
                    filtered = {}
                    for card_id, item in disk_quiz_items.items():
                        question = item.get("question", "")
                        if question and "placeholder" not in question.lower() and len(question) > 20:
                            filtered[card_id] = item
                    st.session_state[disk_cache_key] = filtered
                else:
                    filtered = st.session_state[disk_cache_key]

                for card_id in missing_card_ids:
                    if card_id in filtered:
                        all_quiz_items[card_id] = filtered[card_id]
                        st.session_state[cache_key][card_id] = filtered[card_id]
                    else:
                        quiz_item = load_quiz_item_by_id_wrapper(card_id)
                        if quiz_item:
                            all_quiz_items[card_id] = quiz_item
                            st.session_state[cache_key][card_id] = quiz_item

            reviewed_this_session = st.session_state.get("srs_reviewed_this_session", set())

            # Group due cards by document name to make long lists easier to scan
            grouped: Dict[str, List[str]] = {}
            for cid in filtered_due_cards:
                doc_name = _card_doc_name(cid)
                grouped.setdefault(doc_name, []).append(cid)

            # Render groups inline (previously expanders)
            for doc_name, card_ids in grouped.items():
                label = doc_name if doc_name else "Current lecture"
                st.markdown(f"### 📄 {label} — {len(card_ids)} due")
                for idx_offset, card_id in enumerate(card_ids, 1):
                    # preserve original index semantics by computing a stable index
                    # find global index in due_cards to match original numbering if needed
                    try:
                        global_idx = filtered_due_cards.index(card_id) + 1
                    except Exception:
                        global_idx = idx_offset

                    if card_id in reviewed_this_session:
                        continue

                    card_meta = srs_mgr.get_card_meta(card_id)
                    quiz_item = all_quiz_items.get(card_id)
                    doc_name_local = _card_doc_name(card_id)
                    stored_question = (card_meta or {}).get("question", "")
                    if not quiz_item:
                        with st.container():
                            st.markdown('<div class="la-card"></div>', unsafe_allow_html=True)
                            # Placeholder / missing question case — preserve original behavior
                            card_title = _shorten(stored_question or card_id, 80) or card_id
                            st.markdown(f"### 📄 Card {global_idx}: {card_title}")
                            is_placeholder = "placeholder" in card_id.lower()
                            if stored_question:
                                st.markdown("**Question**")
                                if "\n" in stored_question or len(stored_question) > 300:
                                    st.text_area(label="", value=stored_question, height=120, key=f"srs_qtext_{card_id}", disabled=True)
                                else:
                                    st.write(stored_question)
                            else:
                                if doc_name_local != "Unknown":
                                    st.warning(f"📄 **Card from: {doc_name_local}**")
                                    st.info(
                                        f"💡 **Tip:** Upload the PDF '{doc_name_local}.pdf' and generate a quiz to see the full question."
                                    )
                                else:
                                    st.info(
                                        "💡 **Tip:** This card was registered from a previous session. "
                                        "Upload the same PDF and generate a quiz to see the full question."
                                    )
                            st.caption("🗑️ Remove card deletes only this SRS card, not the original quiz question.")
                            if st.button(f"🗑️ Remove card", key=f"remove_{card_id}"):
                                try:
                                    srs_mgr.delete_card(card_id)
                                    st.success("Card removed!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to remove card: {e}")

                            st.markdown("**💡 Review this question. Try to recall the answer before checking.**")

                            # Simple Yes / No columns
                            col_correct, col_incorrect = st.columns(2)
                            with col_correct:
                                if st.button(f"✅ Yes, I remember this", key=f"srs_correct_{card_id}", type="primary", use_container_width=True):
                                    srs_mgr.mark_review(card_id, correct=True)
                                    reviewed_this_session.add(card_id)
                                    st.session_state["srs_reviewed_this_session"] = reviewed_this_session
                                    st.success("🎉 Excellent! This card will be scheduled for review in a longer interval.")
                                    st.balloons()
                                    st.rerun()

                            with col_incorrect:
                                if st.button(f"❌ No, I need to review", key=f"srs_incorrect_{card_id}", use_container_width=True):
                                    srs_mgr.mark_review(card_id, correct=False)
                                    reviewed_this_session.add(card_id)
                                    st.session_state["srs_reviewed_this_session"] = reviewed_this_session
                                    st.info("📚 No problem! This card will come up again soon to help strengthen your memory.")
                                    st.rerun()

                            if card_meta:
                                review_count = card_meta.get("review_count", 0)
                                interval_idx = card_meta.get("interval_index", 0)
                                next_due = card_meta.get("next_due", "")
                                pretty = _pretty_date_delta(next_due)
                                st.markdown("**Progress / Review stats**")
                                st.caption(
                                    f"📊 Reviewed {review_count} time(s) | Interval: {INTERVALS[interval_idx]} day(s) | {pretty}"
                                )

                            st.markdown("---")
                    else:
                        with st.container():
                            st.markdown('<div class="la-card"></div>', unsafe_allow_html=True)
                            # Full question rendering (keeps original behavior and keys)
                            q_text = quiz_item.get("question", "") or stored_question
                            card_title = _shorten(q_text or card_id, 80) or card_id
                            st.markdown(f"### 📄 Card {global_idx}: {card_title}")

                            choices = quiz_item.get("choices", {}) or {}
                            answer_letter = quiz_item.get("answer", None)
                            explanation = quiz_item.get("explanation", "")

                            # Question area with graceful handling for long text
                            st.markdown("**Question**")
                            if q_text:
                                if "\n" in q_text or len(q_text) > 300:
                                    st.text_area(label="", value=q_text, height=120, key=f"srs_qtext_{card_id}", disabled=True)
                                else:
                                    st.info("Upload the PDF and regenerate the quiz to see the full question.")
                            else:
                                st.write(q_text)

                            # Show choices in a compact list for scanning
                            if choices:
                                st.markdown("**Answer Choices**")
                                for label in ["A", "B", "C", "D"]:
                                    if label in choices:
                                        st.write(f"**{label}.** {choices[label]}")

                            show_answer_key = f"srs_show_{card_id}"
                            if show_answer_key not in st.session_state:
                                st.session_state[show_answer_key] = False

                            st.markdown("---")

                            # Primary CTA for revealing answer — large and clear
                            if not st.session_state[show_answer_key]:
                                st.info("💡 Review this question. Try to recall the answer before checking.")
                                st.caption("🗑️ Remove card deletes only this SRS card, not the original quiz question.")
                                if st.button("🗑️ Remove card", key=f"remove_{card_id}"):
                                    try:
                                        srs_mgr.delete_card(card_id)
                                        st.success("Card removed!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to remove card: {e}")
                                if st.button("🔍 Show Answer", key=f"show_{card_id}", type="primary", use_container_width=True):
                                    st.session_state[show_answer_key] = True
                                    st.rerun()
                            else:
                                # Show the correct answer clearly
                                st.markdown("**✅ Correct Answer**")
                                if answer_letter and choices.get(answer_letter):
                                    st.success(f"**{answer_letter}.** {choices[answer_letter]}")
                                elif answer_letter:
                                    st.success(f"{answer_letter}")
                                else:
                                    st.info("Answer choices aren't available for this card.")
                                # Show explanation if present
                                if explanation:
                                    st.markdown("**Explanation**")
                                    # long explanations put in text area for scrollability
                                    if len(explanation) > 300:
                                        st.info("")  # small visual gap
                                        st.text_area("", value=explanation, height=140, disabled=True)
                                    else:
                                        st.info(explanation)

                                st.markdown("---")
                                st.markdown("**Did you get it correct?** ✅ / ❌")

                                # Yes / No buttons with consistent messaging
                                col_correct, col_incorrect = st.columns(2)

                                with col_correct:
                                    if st.button(f"✅ Yes, I got it correct!", key=f"srs_correct_{card_id}", type="primary", use_container_width=True):
                                        srs_mgr.mark_review(card_id, correct=True)
                                        reviewed_this_session.add(card_id)
                                        st.session_state["srs_reviewed_this_session"] = reviewed_this_session
                                        st.session_state[show_answer_key] = False
                                        st.success("🎉 Excellent! This card will be scheduled for review in a longer interval.")
                                        st.balloons()
                                        st.rerun()

                                with col_incorrect:
                                    if st.button(f"❌ No, I got it wrong", key=f"srs_incorrect_{card_id}", use_container_width=True):
                                        srs_mgr.mark_review(card_id, correct=False)
                                        reviewed_this_session.add(card_id)
                                        st.session_state["srs_reviewed_this_session"] = reviewed_this_session
                                        st.session_state[show_answer_key] = False
                                        st.info("📚 That's okay! This card will come up again soon to help you learn it better.")
                                        st.rerun()

                                # Show progress info (next due)
                                if card_meta:
                                    review_count = card_meta.get("review_count", 0)
                                    interval_idx = card_meta.get("interval_index", 0)
                                    next_due = card_meta.get("next_due", "")
                                    pretty = _pretty_date_delta(next_due)
                                    st.markdown("**Progress / Review stats**")
                                    st.caption(
                                        f"📊 Reviewed {review_count} time(s) | Interval: {INTERVALS[interval_idx]} day(s) | {pretty}"
                                    )

                                st.markdown("---")
            # End grouped due cards
        else:
            st.info("✅ **No cards due for review right now!** Great job staying on top of your studies. 🎉")

        st.markdown("---")

        # Show all cards toggle (preserve original behavior)
        if st.checkbox("📋 Show all my SRS cards", help="View all cards you've registered, including those not due yet"):
            display_cards = all_cards
            if stem:
                display_cards = [cid for cid in all_cards if _card_doc_name(cid) == stem]

            if not display_cards:
                if stem:
                    st.info(f"No cards registered yet for **{stem}**.")
                else:
                    st.info("No cards registered yet.")
            else:
                cache_key = "srs_quiz_items_cache"
                cached_items = st.session_state.get(cache_key, {})

                disk_cache_key = "srs_disk_quiz_items_cache"
                if disk_cache_key not in st.session_state:
                    disk_quiz_items = load_all_quiz_items_wrapper()
                    filtered = {cid: item for cid, item in disk_quiz_items.items()
                               if item.get("question", "") and "placeholder" not in item.get("question", "").lower()}
                    st.session_state[disk_cache_key] = filtered
                else:
                    filtered = st.session_state[disk_cache_key]

                display_items = {**cached_items, **filtered}

                st.markdown("### All Your Study Cards")
                if stem and not show_all_lectures:
                    visible_cards = filtered_cards
                else:
                    visible_cards = all_cards
                # Render a compact list with expandable details to avoid long scrolls (now inline)
                for card_id in display_cards:
                    meta = srs_mgr.get_card_meta(card_id)
                    quiz_item = display_items.get(card_id)

                    if not quiz_item:
                        quiz_item = load_quiz_item_by_id_wrapper(card_id)
                    stored_question = meta.get("question", "") if meta else ""
                    if quiz_item:
                        question_preview = quiz_item.get("question", "")[:100] + "..." if len(quiz_item.get("question", "")) > 100 else quiz_item.get("question", "")
                    else:
                        doc_name = _card_doc_name(card_id)
                        preview_source = stored_question or f"Card from {doc_name} (question not available)"
                        question_preview = _shorten(preview_source, 120)

                    is_due = card_id in filtered_due_cards
                    status_icon = "🎯" if is_due else "✅"
                    status_text = "Due now" if is_due else "Not due"

                    next_due = meta.get("next_due", "") if meta else ""
                    review_count = meta.get("review_count", 0) if meta else 0
                    interval_idx = meta.get("interval_index", 0) if meta else 0

                    # Render card header inline
                    if quiz_item:
                        question_text = quiz_item.get("question", "")
                    else:
                        question_text = stored_question or card_id + " (question missing)"
                    card_title = _shorten(question_text, 80)
                    st.markdown(f"#### {status_icon} 📄 {card_title} — {status_text}")
                    st.write(question_preview)
                    pretty = _pretty_date_delta(next_due)
                    st.caption(f"📊 Reviewed {review_count} time(s) | Interval: {INTERVALS[interval_idx]} day(s) | {pretty}")
                    # Optionally show full text if available
                    full_question = quiz_item.get("question", "") if quiz_item else stored_question
                    if full_question:
                        if len(full_question) > 250:
                            st.text_area("", value=full_question, height=120, disabled=True)
                        else:
                            st.write(full_question)
                    st.caption("🗑️ Remove card deletes only this SRS card, not the original quiz question.")
                    if st.button("🗑️ Remove card", key=f"remove_all_{card_id}"):
                        try:
                            srs_mgr.delete_card(card_id)
                            st.success("Card removed!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to remove card: {e}")

                    st.markdown("")  # spacing

    except Exception as e:
        st.error("Error loading SRS data.")
        st.exception(e)
