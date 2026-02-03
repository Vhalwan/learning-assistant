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


def _infer_stem_from_session(session: Dict) -> str | None:
    """
    Try to infer the current `stem` used in app.py from session keys.
    Strategy:
      1. find stems that appear in both quiz_items_{stem} and chat_history_{stem}
      2. else use the first quiz_items_{stem}
      3. else use the first chat_history_{stem}
      4. else return None
    """
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

        # --- Metrics row (clear visual hierarchy) ---
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            st.metric("Total cards", len(all_cards))
        with col_stats2:
            st.metric("Due now", len(due_cards), delta=None if len(due_cards) == 0 else f"{len(due_cards)} to review")
        with col_stats3:
            reviewed_count = sum(1 for cid in all_cards if srs_mgr.get_card_meta(cid).get("review_count", 0) > 0)
            st.metric("Reviewed", reviewed_count)

        # If there are due cards, present them grouped by originating document for scanability
        if due_cards:
            st.success(f"🎯 You have {len(due_cards)} card(s) due for review")
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
            missing_card_ids = [cid for cid in due_cards if cid not in all_quiz_items]
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
            for cid in due_cards:
                doc_name = cid.rsplit("_", 1)[0] if "_" in cid else "Unknown"
                grouped.setdefault(doc_name, []).append(cid)

            # Render groups inline (previously expanders)
            for doc_name, card_ids in grouped.items():
                st.markdown(f"### 📄 {doc_name} — {len(card_ids)} due")
                for idx_offset, card_id in enumerate(card_ids, 1):
                    # preserve original index semantics by computing a stable index
                    # find global index in due_cards to match original numbering if needed
                    try:
                        global_idx = due_cards.index(card_id) + 1
                    except Exception:
                        global_idx = idx_offset

                    if card_id in reviewed_this_session:
                        continue

                    card_meta = srs_mgr.get_card_meta(card_id)
                    quiz_item = all_quiz_items.get(card_id)
                    doc_name_local = card_id.rsplit("_", 1)[0] if "_" in card_id else "Unknown"

                    if not quiz_item:
                        with st.container():
                            st.markdown('<div class="la-card"></div>', unsafe_allow_html=True)
                            # Placeholder / missing question case — preserve original behavior
                            st.markdown(f"### Card {global_idx}: {card_id}")
                            is_placeholder = "placeholder" in card_id.lower() or (card_meta and card_meta.get("review_count", 0) == 0)
                            if doc_name_local != "Unknown":
                                st.warning(f"📄 **Card from: {doc_name_local}**")
                                if is_placeholder:
                                    st.info(
                                        "💡 This appears to be a placeholder card from an old session. "
                                        "You can delete it and register new questions from the quiz section above."
                                    )
                                else:
                                    st.info(
                                        f"💡 **Tip:** Upload the PDF '{doc_name_local}.pdf' and generate a quiz to see the full question."
                                    )
                            else:
                                st.info(
                                    "💡 **Tip:** This card was registered from a previous session. "
                                    "Upload the same PDF and generate a quiz to see the full question."
                                )
                            st.markdown("**Do you remember this topic?**")
                            st.caption("Think about what you learned. Can you recall the key concepts?")
                            if is_placeholder:
                                if st.button(f"🗑️ Remove this placeholder card", key=f"remove_{card_id}"):
                                    try:
                                        del srs_mgr._data[card_id]
                                        srs_mgr._save()
                                        st.success("Card removed!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to remove card: {e}")

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
                                st.caption(f"📊 Progress: Reviewed {review_count} time(s) | Current interval: {INTERVALS[interval_idx]} days")

                            st.markdown("---")
                    else:
                        with st.container():
                            st.markdown('<div class="la-card"></div>', unsafe_allow_html=True)
                            # Full question rendering (keeps original behavior and keys)
                            st.markdown(f"### Card {global_idx}: Review Question")

                            q_text = quiz_item.get("question", "")
                            choices = quiz_item.get("choices", {}) or {}
                            answer_letter = quiz_item.get("answer", None)
                            explanation = quiz_item.get("explanation", "")

                            # Question area with graceful handling for long text
                            st.markdown("**📝 Question:**")
                            if "\n" in q_text or len(q_text) > 300:
                                st.text_area(label="", value=q_text, height=120, key=f"srs_qtext_{card_id}", disabled=True)
                            else:
                                st.write(q_text)

                            # Show choices in a compact list for scanning
                            st.markdown("**🔤 Answer Choices:**")
                            for label in ["A", "B", "C", "D"]:
                                if label in choices:
                                    st.write(f"**{label}.** {choices[label]}")

                            show_answer_key = f"srs_show_{card_id}"
                            if show_answer_key not in st.session_state:
                                st.session_state[show_answer_key] = False

                            st.markdown("---")

                            # Primary CTA for revealing answer — large and clear
                            if not st.session_state[show_answer_key]:
                                st.info("💭 Think about your answer, then reveal it.")
                                if st.button("🔍 Show Answer", key=f"show_{card_id}", type="primary", use_container_width=True):
                                    st.session_state[show_answer_key] = True
                                    st.rerun()
                            else:
                                # Show the correct answer clearly
                                st.markdown("**✅ Correct Answer:**")
                                if answer_letter and choices.get(answer_letter):
                                    st.success(f"**{answer_letter}.** {choices[answer_letter]}")

                                # Show explanation if present
                                if explanation:
                                    st.markdown("**📖 Explanation:**")
                                    # long explanations put in text area for scrollability
                                    if len(explanation) > 300:
                                        st.info("")  # small visual gap
                                        st.text_area("", value=explanation, height=140, disabled=True)
                                    else:
                                        st.info(explanation)

                                st.markdown("---")
                                st.markdown("**🎯 Did you get it right?**")

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
                                    st.caption(f"📊 Progress: Reviewed {review_count} time(s) | Interval: {INTERVALS[interval_idx]} days | {pretty}")

                                st.markdown("---")
            # End grouped due cards
        else:
            st.info("✅ **No cards due for review right now!** Great job staying on top of your studies. 🎉")

        st.markdown("---")

        # Show all cards toggle (preserve original behavior)
        if st.checkbox("📋 Show all my SRS cards", help="View all cards you've registered, including those not due yet"):
            if not all_cards:
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
                # Render a compact list with expandable details to avoid long scrolls (now inline)
                for card_id in all_cards:
                    meta = srs_mgr.get_card_meta(card_id)
                    quiz_item = display_items.get(card_id)

                    if not quiz_item:
                        quiz_item = load_quiz_item_by_id_wrapper(card_id)

                    if quiz_item:
                        question_preview = quiz_item.get("question", "")[:100] + "..." if len(quiz_item.get("question", "")) > 100 else quiz_item.get("question", "")
                    else:
                        doc_name = card_id.rsplit("_", 1)[0] if "_" in card_id else "Unknown"
                        question_preview = f"Card from {doc_name} (question not available)"

                    is_due = card_id in due_cards
                    status_icon = "🎯" if is_due else "✅"
                    status_text = "Due now" if is_due else "Not due"

                    next_due = meta.get("next_due", "") if meta else ""
                    review_count = meta.get("review_count", 0) if meta else 0
                    interval_idx = meta.get("interval_index", 0) if meta else 0

                    # Render card header inline
                    st.markdown(f"#### {status_icon} {card_id} — {status_text}")
                    st.write(question_preview)
                    pretty = _pretty_date_delta(next_due)
                    st.caption(f"📊 Reviewed {review_count} time(s) | Interval: {INTERVALS[interval_idx]} days | {pretty}")
                    # Optionally show full text if available
                    if quiz_item and quiz_item.get("question", ""):
                        if len(quiz_item.get("question", "")) > 250:
                            st.text_area("", value=quiz_item.get("question", ""), height=120, disabled=True)
                        else:
                            st.write(quiz_item.get("question", ""))

                    st.markdown("")  # spacing

    except Exception as e:
        st.error("Error loading SRS data.")
        st.exception(e)
