"""
SRS review and browse UI.

Expose:
    def render_srs_section(st, stem=None):
        Renders a tabbed section: Review (SRS) | Browse cards

    # Legacy single-mode exports (still usable standalone)
    def render(st, stem=None): ...
    def render_browse(st, stem=None): ...
"""

from datetime import datetime
from typing import Any, Dict, List

import streamlit as st

from backend.study_srs import INTERVALS, SRSManager
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
        return card_id.rsplit("_", 1)[0]
    return ""


def _infer_stem_from_session(session: Dict) -> str | None:
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
        return iso_ts[:10] if iso_ts else "N/A"


def _format_interval(card_meta: Dict | None) -> str:
    meta = card_meta or {}
    interval_days = meta.get("interval_days")
    if interval_days is not None:
        try:
            days = float(interval_days)
            if days < 1:
                minutes = max(1, int(round(days * 1440)))
                return f"{minutes} min"
            day_text = int(days) if float(days).is_integer() else round(days, 2)
            return f"{day_text} day(s)"
        except Exception:
            pass

    interval_idx = int(meta.get("interval_index", 0) or 0)
    interval_idx = max(0, min(interval_idx, len(INTERVALS) - 1))
    return f"{INTERVALS[interval_idx]} day(s)"


def _is_mcq_card(quiz_item: Dict | None, card_meta: Dict | None) -> bool:
    quiz_item = quiz_item or {}
    card_meta = card_meta or {}
    explicit_type = (quiz_item.get("item_type") or card_meta.get("item_type") or "").strip().lower()
    if explicit_type:
        return explicit_type == "mcq"
    choices = quiz_item.get("choices") if isinstance(quiz_item.get("choices"), dict) else card_meta.get("choices")
    return isinstance(choices, dict) and any(str(choices.get(k, "")).strip() for k in ["A", "B", "C", "D"])


def _hydrate_card_quiz_item(card_id: str, quiz_item: Dict | None, card_meta: Dict | None) -> Dict | None:
    quiz_item = quiz_item if isinstance(quiz_item, dict) else {}
    card_meta = card_meta if isinstance(card_meta, dict) else {}

    quiz_choices = quiz_item.get("choices") if isinstance(quiz_item.get("choices"), dict) else {}
    meta_choices = card_meta.get("choices") if isinstance(card_meta.get("choices"), dict) else {}
    merged = {
        "id": quiz_item.get("id") or card_meta.get("quiz_question_id") or card_id,
        "question": quiz_item.get("question", "") or card_meta.get("question", ""),
        "choices": quiz_choices or meta_choices,
        "answer": quiz_item.get("answer", "") or card_meta.get("answer", ""),
        "explanation": quiz_item.get("explanation", "") or card_meta.get("explanation", ""),
        "item_type": quiz_item.get("item_type", "") or card_meta.get("item_type", ""),
        "origin": quiz_item.get("origin", "") or card_meta.get("origin", ""),
    }

    if not any(
        [
            str(merged.get("question", "")).strip(),
            merged.get("choices"),
            str(merged.get("answer", "")).strip(),
            str(merged.get("explanation", "")).strip(),
        ]
    ):
        return None
    return merged


def _get_lecture_card_ids(srs_mgr: SRSManager, stem: str | None) -> List[str]:
    if not stem:
        return []
    return [cid for cid in srs_mgr._data.keys() if _card_stem(cid, srs_mgr.get_card_meta(cid)) == stem]


def _load_quiz_items_for_cards(st_module: Any, quiz_state_key: str | None, card_ids: List[str]) -> Dict[str, Dict]:
    cache_key = "srs_quiz_items_cache"
    if cache_key not in st_module.session_state:
        st_module.session_state[cache_key] = {}

    all_quiz_items = st_module.session_state[cache_key].copy()

    current_quiz = st_module.session_state.get(quiz_state_key, []) if quiz_state_key else []
    for q in current_quiz:
        qid = q.get("id")
        if qid:
            all_quiz_items[qid] = q
            st_module.session_state[cache_key][qid] = q

    missing_card_ids = [cid for cid in card_ids if cid not in all_quiz_items]
    if not missing_card_ids:
        return all_quiz_items

    disk_cache_key = "srs_disk_quiz_items_cache"
    if disk_cache_key not in st_module.session_state:
        disk_quiz_items = load_all_quiz_items_wrapper()
        filtered = {}
        for card_id, item in disk_quiz_items.items():
            question = item.get("question", "")
            if question and "placeholder" not in question.lower() and len(question) > 20:
                filtered[card_id] = item
        st_module.session_state[disk_cache_key] = filtered
    else:
        filtered = st_module.session_state[disk_cache_key]

    for card_id in missing_card_ids:
        if card_id in filtered:
            all_quiz_items[card_id] = filtered[card_id]
            st_module.session_state[cache_key][card_id] = filtered[card_id]
            continue

        quiz_item = load_quiz_item_by_id_wrapper(card_id)
        if quiz_item:
            all_quiz_items[card_id] = quiz_item
            st_module.session_state[cache_key][card_id] = quiz_item

    return all_quiz_items


def _render_review_stats(card_meta: Dict | None):
    if not card_meta:
        return
    review_count = card_meta.get("review_count", 0)
    next_due = card_meta.get("next_due", "")
    pretty = _pretty_date_delta(next_due)
    interval_label = _format_interval(card_meta)
    st.markdown("**Progress / Review stats**")
    st.caption(f"Reviewed {review_count} time(s) | Interval: {interval_label} | {pretty}")


def _render_review_card_fallback(
    st_module: Any,
    srs_mgr: SRSManager,
    card_id: str,
    card_index: int,
    card_meta: Dict | None,
    stored_question: str,
    reviewed_this_session: set,
):
    with st_module.container():
        st_module.markdown('<div class="la-card"></div>', unsafe_allow_html=True)
        card_title = _shorten(stored_question or card_id, 80) or card_id
        st_module.markdown(f"### Card {card_index}: {card_title}")

        st_module.markdown("**Question**")
        if stored_question:
            if "\n" in stored_question or len(stored_question) > 300:
                st_module.text_area(
                    label="",
                    value=stored_question,
                    height=120,
                    key=f"srs_qtext_{card_id}",
                    disabled=True,
                )
            else:
                st_module.write(stored_question)
        else:
            st_module.info("Question text is unavailable for this card.")

        st_module.markdown("**Review this question. Try to recall the answer before rating it.**")
        col_correct, col_incorrect = st_module.columns(2)
        with col_correct:
            if st_module.button(
                "Yes, I remember this",
                key=f"srs_correct_{card_id}",
                type="primary",
                use_container_width=True,
            ):
                srs_mgr.mark_review_with_rating(card_id, "good")
                reviewed_this_session.add(card_id)
                st_module.session_state["srs_reviewed_this_session"] = reviewed_this_session
                st_module.rerun()

        with col_incorrect:
            if st_module.button(
                "No, I need to review",
                key=f"srs_incorrect_{card_id}",
                use_container_width=True,
            ):
                srs_mgr.mark_review_with_rating(card_id, "hard")
                reviewed_this_session.add(card_id)
                st_module.session_state["srs_reviewed_this_session"] = reviewed_this_session
                st_module.rerun()

        _render_review_stats(card_meta)
        st_module.markdown("---")


def _render_review_card(
    st_module: Any,
    srs_mgr: SRSManager,
    card_id: str,
    card_index: int,
    card_meta: Dict | None,
    quiz_item: Dict,
    reviewed_this_session: set,
):
    q_text = quiz_item.get("question", "") or (card_meta or {}).get("question", "")
    choices = quiz_item.get("choices", {}) if isinstance(quiz_item.get("choices"), dict) else {}
    if not choices:
        choices = (card_meta or {}).get("choices", {}) if isinstance((card_meta or {}).get("choices"), dict) else {}
    answer_letter = quiz_item.get("answer", None) or (card_meta or {}).get("answer", None)
    explanation = quiz_item.get("explanation", "") or (card_meta or {}).get("explanation", "")
    is_mcq_card = _is_mcq_card(quiz_item, card_meta)

    show_answer_key = f"srs_show_{card_id}"
    if show_answer_key not in st_module.session_state:
        st_module.session_state[show_answer_key] = False

    with st_module.container():
        st_module.markdown('<div class="la-card"></div>', unsafe_allow_html=True)
        card_title = _shorten(q_text or card_id, 80) or card_id
        st_module.markdown(f"### Card {card_index}: {card_title}")

        st_module.markdown("**Question**")
        if q_text:
            if "\n" in q_text or len(q_text) > 300:
                st_module.text_area(
                    label="",
                    value=q_text,
                    height=120,
                    key=f"srs_qtext_{card_id}",
                    disabled=True,
                )
            else:
                st_module.write(q_text)
        else:
            st_module.info("Question text unavailable for this card.")

        if is_mcq_card and choices:
            st_module.markdown("**Answer Choices**")
            for label in ["A", "B", "C", "D"]:
                if label in choices:
                    st_module.write(f"**{label}.** {choices[label]}")

        st_module.markdown("---")

        if not st_module.session_state[show_answer_key]:
            st_module.caption("Type: MCQ card" if is_mcq_card else "Type: Concept card")

            if is_mcq_card:
                placeholder = "Select an answer"
                options = [placeholder] + [
                    f"{label}. {choices.get(label, '')}"
                    for label in ["A", "B", "C", "D"]
                    if choices.get(label, "")
                ]
                select_key = f"srs_select_{card_id}"
                selected = st_module.selectbox(
                    "Choose your answer",
                    options=options,
                    key=select_key,
                )
                check_disabled = selected == placeholder
                if st_module.button(
                    "Check answer",
                    key=f"srs_check_{card_id}",
                    disabled=check_disabled,
                    type="primary",
                    use_container_width=True,
                ):
                    st_module.session_state[show_answer_key] = True
                    st_module.session_state[f"srs_checked_choice_{card_id}"] = (
                        selected.split(".", 1)[0].strip() if "." in selected else ""
                    )
                    st_module.rerun()
            else:
                col_front_a, col_front_b = st_module.columns(2)
                with col_front_a:
                    if st_module.button(
                        "Show answer",
                        key=f"show_{card_id}",
                        type="primary",
                        use_container_width=True,
                    ):
                        st_module.session_state[show_answer_key] = True
                        st_module.rerun()
                with col_front_b:
                    if st_module.button(
                        "I don't remember",
                        key=f"forget_{card_id}",
                        use_container_width=True,
                    ):
                        srs_mgr.mark_review_with_rating(card_id, "hard")
                        reviewed_this_session.add(card_id)
                        st_module.session_state["srs_reviewed_this_session"] = reviewed_this_session
                        st_module.session_state[show_answer_key] = False
                        st_module.rerun()
        else:
            st_module.markdown("**Back**")
            if is_mcq_card:
                checked = st_module.session_state.get(f"srs_checked_choice_{card_id}", "")
                if checked and answer_letter:
                    if checked == str(answer_letter).strip().upper():
                        st_module.success(f"Correct: **{answer_letter}.** {choices.get(answer_letter, '')}")
                    else:
                        st_module.error(f"Incorrect. You chose **{checked}**.")
                        st_module.success(f"Correct answer: **{answer_letter}.** {choices.get(answer_letter, '')}")
                elif answer_letter and choices.get(answer_letter):
                    st_module.success(f"**{answer_letter}.** {choices[answer_letter]}")
                else:
                    st_module.info("Answer choices are not available for this card.")
            else:
                if explanation:
                    st_module.success(explanation)
                else:
                    st_module.info(q_text or "No answer available.")

            if explanation:
                st_module.markdown("**Generated explanation**")
                st_module.write(explanation)

            st_module.markdown("**How difficult was this problem?**")
            rating_cols = st_module.columns(3)
            if rating_cols[0].button("Hard", key=f"rate_hard_{card_id}", use_container_width=True):
                srs_mgr.mark_review_with_rating(card_id, "hard")
                reviewed_this_session.add(card_id)
                st_module.session_state["srs_reviewed_this_session"] = reviewed_this_session
                st_module.session_state[show_answer_key] = False
                st_module.rerun()

            if rating_cols[1].button(
                "Good",
                key=f"rate_good_{card_id}",
                type="primary",
                use_container_width=True,
            ):
                srs_mgr.mark_review_with_rating(card_id, "good")
                reviewed_this_session.add(card_id)
                st_module.session_state["srs_reviewed_this_session"] = reviewed_this_session
                st_module.session_state[show_answer_key] = False
                st_module.rerun()

            if rating_cols[2].button("Easy", key=f"rate_easy_{card_id}", use_container_width=True):
                srs_mgr.mark_review_with_rating(card_id, "easy")
                reviewed_this_session.add(card_id)
                st_module.session_state["srs_reviewed_this_session"] = reviewed_this_session
                st_module.session_state[show_answer_key] = False
                st_module.rerun()

            _render_review_stats(card_meta)

        st_module.markdown("---")


# ---------------------------------------------------------------------------
# Public: tabbed entry point (preferred)
# ---------------------------------------------------------------------------

def render_srs_section(st: Any, stem: str | None = None):
    """
    Single tabbed section replacing the two separate stacked sections.

    Tab 1 — Review (SRS): due cards, this lecture only, Hard/Good/Easy
    Tab 2 — Browse cards: all cards, this lecture only, view + delete only
    """
    stem = stem or _infer_stem_from_session(st.session_state)

    st.markdown("---")
    st.markdown('<a id="spaced-repetition-review"></a>', unsafe_allow_html=True)
    st.subheader("📌 Study mode")

    with st.expander("What is Spaced Repetition?", expanded=False):
        st.markdown(
            """
            **Spaced Repetition** helps you remember information long-term by reviewing it at increasing intervals.

            **How it works (quick):**
            - Hard: bring the card back sooner
            - Good: move it forward at a steady pace
            - Easy: stretch it much further out

            **To get started:**
            1. Generate a quiz from your PDF above
            2. Click **Add to SRS** on questions you want to review later
            3. Come back here to review due cards for this lecture
            """
        )

    tab_review, tab_browse = st.tabs(["Review (SRS)", "📚 Browse cards"])

    with tab_review:
        _render_review_tab(st, stem)

    with tab_browse:
        _render_browse_tab(st, stem)


def _render_review_tab(st_module: Any, stem: str | None):
    """Due cards only, lecture-scoped, Hard/Good/Easy rating."""
    quiz_state_key = f"quiz_items_{stem}" if stem else None

    if not stem:
        st_module.info("Open a lecture to review its due SRS cards.")
        return

    try:
        srs_mgr = SRSManager()
        lecture_cards = _get_lecture_card_ids(srs_mgr, stem)
        due_cards = [cid for cid in srs_mgr.get_due_cards() if cid in lecture_cards]

        current_quiz = st_module.session_state.get(quiz_state_key, []) if quiz_state_key else []
        if current_quiz:
            registered_ids = set(lecture_cards)
            unregistered = [q for q in current_quiz if q.get("id") not in registered_ids]
            if unregistered:
                st_module.info(
                    f"You have {len(unregistered)} quiz question(s) in this lecture not in SRS yet. "
                    "Use **Add to SRS** on any question you want to keep reviewing."
                )

        if not lecture_cards:
            st_module.info("No SRS cards found for this lecture yet. Add a quiz question or confused item to SRS first.")
            return

        col1, col2, col3 = st_module.columns(3)
        with col1:
            st_module.metric("Total cards", len(lecture_cards))
        with col2:
            st_module.metric(
                "Due now",
                len(due_cards),
                delta=None if len(due_cards) == 0 else f"{len(due_cards)} to review",
            )
        with col3:
            reviewed_count = sum(
                1 for cid in lecture_cards if srs_mgr.get_card_meta(cid).get("review_count", 0) > 0
            )
            st_module.metric("Reviewed", reviewed_count)

        if not due_cards:
            st_module.info("No cards are due for this lecture right now.")
            return

        st_module.success(f"You have {len(due_cards)} card(s) due for review in this lecture.")
        st_module.markdown("---")

        all_quiz_items = _load_quiz_items_for_cards(st_module, quiz_state_key, due_cards)
        reviewed_this_session = st_module.session_state.get("srs_reviewed_this_session", set())

        for idx, card_id in enumerate(due_cards, start=1):
            if card_id in reviewed_this_session:
                continue

            card_meta = srs_mgr.get_card_meta(card_id)
            stored_question = (card_meta or {}).get("question", "")
            quiz_item = _hydrate_card_quiz_item(card_id, all_quiz_items.get(card_id), card_meta)

            if not quiz_item:
                _render_review_card_fallback(
                    st_module=st_module,
                    srs_mgr=srs_mgr,
                    card_id=card_id,
                    card_index=idx,
                    card_meta=card_meta,
                    stored_question=stored_question,
                    reviewed_this_session=reviewed_this_session,
                )
            else:
                _render_review_card(
                    st_module=st_module,
                    srs_mgr=srs_mgr,
                    card_id=card_id,
                    card_index=idx,
                    card_meta=card_meta,
                    quiz_item=quiz_item,
                    reviewed_this_session=reviewed_this_session,
                )
    except Exception as e:
        st_module.error("Error loading SRS data.")
        st_module.exception(e)


def _render_browse_tab(st_module: Any, stem: str | None):
    """All lecture cards, view + delete only — no SRS actions."""
    quiz_state_key = f"quiz_items_{stem}" if stem else None

    st_module.caption("Scope: This lecture · All cards · View & delete only")

    if not stem:
        st_module.info("Open a lecture to browse its saved cards.")
        return

    try:
        srs_mgr = SRSManager()
        lecture_cards = _get_lecture_card_ids(srs_mgr, stem)

        if not lecture_cards:
            st_module.info("No SRS cards found for this lecture yet.")
            return

        all_quiz_items = _load_quiz_items_for_cards(st_module, quiz_state_key, lecture_cards)
        due_cards = set(cid for cid in srs_mgr.get_due_cards() if cid in lecture_cards)

        for card_id in lecture_cards:
            meta = srs_mgr.get_card_meta(card_id)
            quiz_item = _hydrate_card_quiz_item(card_id, all_quiz_items.get(card_id), meta)
            if not quiz_item:
                quiz_item = _hydrate_card_quiz_item(card_id, load_quiz_item_by_id_wrapper(card_id), meta)

            stored_question = meta.get("question", "") if meta else ""
            question_text = (quiz_item or {}).get("question", "") or stored_question or card_id
            choices = (quiz_item or {}).get("choices", {}) if isinstance((quiz_item or {}).get("choices"), dict) else {}
            if not choices and isinstance((meta or {}).get("choices"), dict):
                choices = (meta or {}).get("choices", {})
            answer_letter = (quiz_item or {}).get("answer", "") or (meta or {}).get("answer", "")
            explanation = (quiz_item or {}).get("explanation", "") or (meta or {}).get("explanation", "")
            is_mcq_card = _is_mcq_card(quiz_item, meta)

            with st_module.container():
                st_module.markdown('<div class="la-card"></div>', unsafe_allow_html=True)
                status_text = "Due now" if card_id in due_cards else _pretty_date_delta((meta or {}).get("next_due", ""))
                st_module.markdown(f"### {_shorten(question_text, 90)}")

                review_count = (meta or {}).get("review_count", 0)
                interval_label = _format_interval(meta)
                st_module.caption(f"Status: {status_text} | Reviewed {review_count} time(s) | Interval: {interval_label}")
                st_module.caption("Type: MCQ card" if is_mcq_card else "Type: Concept card")

                st_module.markdown("**Question**")
                if question_text:
                    if "\n" in question_text or len(question_text) > 300:
                        st_module.text_area(
                            label="",
                            value=question_text,
                            height=120,
                            key=f"browse_qtext_{card_id}",
                            disabled=True,
                        )
                    else:
                        st_module.write(question_text)

                if is_mcq_card and choices:
                    st_module.markdown("**Answer Choices**")
                    for label in ["A", "B", "C", "D"]:
                        if label in choices:
                            st_module.write(f"**{label}.** {choices[label]}")

                if answer_letter:
                    st_module.markdown("**Answer**")
                    if choices.get(answer_letter):
                        st_module.write(f"{answer_letter}. {choices.get(answer_letter, '')}")
                    else:
                        st_module.write(str(answer_letter))

                if explanation:
                    st_module.markdown("**Explanation**")
                    st_module.write(explanation)

                if st_module.button("🗑 Remove card", key=f"browse_remove_{card_id}", use_container_width=True):
                    try:
                        srs_mgr.delete_card(card_id)
                        reviewed_this_session = st_module.session_state.get("srs_reviewed_this_session", set())
                        if card_id in reviewed_this_session:
                            reviewed_this_session.discard(card_id)
                            st_module.session_state["srs_reviewed_this_session"] = reviewed_this_session
                        st_module.success("Card removed.")
                        st_module.rerun()
                    except Exception as e:
                        st_module.error(f"Failed to remove card: {e}")

                st_module.markdown("---")
    except Exception as e:
        st_module.error("Error loading SRS data.")
        st_module.exception(e)


# ---------------------------------------------------------------------------
# Legacy single-mode exports (kept for backward compat if needed elsewhere)
# ---------------------------------------------------------------------------

def render(st: Any, stem: str | None = None):
    """Legacy: renders review tab only."""
    stem = stem or _infer_stem_from_session(st.session_state)
    st.markdown("---")
    st.markdown('<a id="spaced-repetition-review"></a>', unsafe_allow_html=True)
    st.subheader("Spaced Repetition Review")
    _render_review_tab(st, stem)


def render_browse(st: Any, stem: str | None = None):
    """Legacy: renders browse tab only."""
    stem = stem or _infer_stem_from_session(st.session_state)
    st.markdown("---")
    st.markdown('<a id="browse-cards"></a>', unsafe_allow_html=True)
    st.subheader("📚 Browse cards")
    _render_browse_tab(st, stem)