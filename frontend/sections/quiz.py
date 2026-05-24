"""
Quiz UI — round-based batches (1–10 questions) with per-round summaries.

  - Only the **current (latest) round** is fully inline. Earlier rounds sit in **collapsed**
    expanders once you add another round from the summary “make more questions” actions.
  - After every completed round: score, weak topics, “Continue with 5 more questions”, and
    custom 1–10 + Generate (session flat list stays synced for Confused / SRS).
  - Overall progress (totals + weakest areas + recent rounds) appears only after 2+ rounds.
  - Per-question: brief explanation + optional “Show reasoning” expander (why_correct, why_wrong, source).
"""

import uuid
import hashlib
import time
import re
import json
import requests
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple
import os
import streamlit as st
import streamlit.components.v1 as components

from frontend.handlers import (
    generate_quiz,
    save_quiz_to_disk,
    record_quiz_result,
)
from backend.study_srs import SRSManager
from backend.quiz_session_log import append_quiz_session, recent_sessions, trend_vs_prior

_QUESTION_ANGLE_LABELS = {
    "definition": "Definition",
    "application": "Application",
    "misconception": "Misconception trap",
    "comparison": "Compare concepts",
    "mechanism": "Mechanism / reasoning",
    "not_true": "Which is false?",
    "consequence": "Implication",
    "property": "Property / requirement",
    "criticism": "Limitation / critique",
}


def _normalize_quiz_question_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _stable_quiz_item_id(q: Dict[str, Any]) -> str:
    """Public id from item, or one stable UUID stored on the dict (session survives reruns)."""
    raw = (q.get("id") or "").strip()
    if raw:
        return raw
    slot = "_la_quiz_item_id"
    if slot not in q or not str(q.get(slot) or "").strip():
        q[slot] = str(uuid.uuid4())
    return str(q[slot]).strip()


def _quiz_session_chunk_counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for q in items or []:
        cid = (q.get("chunk_id") or q.get("source_chunk_id") or "").strip().lower()
        if not cid:
            continue
        counts[cid] = counts.get(cid, 0) + 1
    return counts


def _legacy_wave_breaks_to_rounds(flat_items: List[Dict[str, Any]], wave_breaks_1based: set) -> List[Dict[str, Any]]:
    """Split a flat quiz list using legacy 1-based indices where each new 'wave' starts."""
    if not flat_items:
        return []
    breaks = sorted(wave_breaks_1based)
    rounds: List[Dict[str, Any]] = []
    start = 0
    for b in breaks:
        cut = max(0, int(b) - 1)
        if cut > start:
            rounds.append({"items": flat_items[start:cut]})
            start = cut
    rounds.append({"items": flat_items[start:]})
    return [r for r in rounds if r.get("items")]


def _flatten_round_items(rounds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rounds or []:
        items = r.get("items")
        if isinstance(items, list):
            out.extend(items)
    return out


def _ensure_quiz_rounds(
    st_module: Any,
    quiz_state_key: str,
    wave_breaks_key: str,
    rounds_key: str,
) -> List[Dict[str, Any]]:
    flat = st_module.session_state.get(quiz_state_key, []) or []
    if rounds_key not in st_module.session_state:
        wb_raw = st_module.session_state.get(wave_breaks_key, [])
        wb_set = set(wb_raw) if isinstance(wb_raw, list) else set()
        if flat and wb_set:
            st_module.session_state[rounds_key] = _legacy_wave_breaks_to_rounds(flat, wb_set)
        elif flat:
            st_module.session_state[rounds_key] = [{"items": list(flat)}]
        else:
            st_module.session_state[rounds_key] = []
    rounds = st_module.session_state.get(rounds_key, [])
    if not isinstance(rounds, list):
        rounds = []
        st_module.session_state[rounds_key] = rounds
    if not rounds and flat:
        st_module.session_state[rounds_key] = [{"items": list(flat)}]
        rounds = st_module.session_state[rounds_key]
    # Keep flat list in sync for Confused / SRS consumers
    merged = _flatten_round_items(rounds)
    if not merged and flat:
        st_module.session_state[rounds_key] = [{"items": list(flat)}]
        rounds = st_module.session_state[rounds_key]
        merged = flat
    if merged != flat:
        st_module.session_state[quiz_state_key] = merged
    return rounds


def _scroll_streamlit_to_anchor(anchor_id: str) -> None:
    """Best-effort scroll so the new round is in view (Streamlit iframe layout varies)."""
    aid = json.dumps(anchor_id)
    components.html(
        f"""
        <script>
        const ID = {aid};
        function findEl() {{
            const byId = (doc) => (doc && doc.getElementById ? doc.getElementById(ID) : null);
            let el = byId(window.parent.document);
            if (el) return el;
            const scanRoots = [window.parent.document, document];
            for (const root of scanRoots) {{
                if (!root || !root.querySelectorAll) continue;
                const frames = root.querySelectorAll("iframe");
                for (const fr of frames) {{
                    try {{
                        const d = fr.contentDocument;
                        el = byId(d);
                        if (el) return el;
                    }} catch (e) {{}}
                }}
            }}
            return byId(document);
        }}
        function go() {{
            const el = findEl();
            if (el) el.scrollIntoView({{ behavior: "instant", block: "start" }});
        }}
        setTimeout(go, 0);
        setTimeout(go, 120);
        setTimeout(go, 400);
        </script>
        """,
        height=0,
        width=0,
    )


def _weak_topics_from_items(render_items: List[Dict[str, Any]], st_module: Any) -> Dict[str, int]:
    wrong_topic_counts: Dict[str, int] = {}
    for render_item in render_items:
        sub = st_module.session_state.get(render_item["submit_key"])
        if not sub or sub.get("is_correct", False):
            continue
        q = render_item["question"]
        topic = (
            (q.get("concept_label") or "").strip()
            or (q.get("source_chunk_preview") or "").strip()
            or (q.get("source_chunk") or "").strip()
        )
        if topic:
            if len(topic) > 120:
                topic = topic[:120].rstrip() + "…"
            wrong_topic_counts[topic] = wrong_topic_counts.get(topic, 0) + 1
    return wrong_topic_counts


def _count_noun(count: int, singular: str) -> str:
    return f"{count} {singular}" if count == 1 else f"{count} {singular}s"


def _round_progress(render_items: List[Dict[str, Any]], st_module: Any) -> Tuple[int, int, int, int]:
    """answered, correct, wrong, total"""
    total = len(render_items)
    answered = correct = wrong = 0
    for render_item in render_items:
        sub = st_module.session_state.get(render_item["submit_key"])
        if not sub:
            continue
        answered += 1
        if sub.get("is_correct", False):
            correct += 1
        else:
            wrong += 1
    return answered, correct, wrong, total


def _render_single_question(
    st_module: Any,
    *,
    stem: str,
    doc_id: str,
    quiz_state_key: str,
    generation_token: int,
    q: Dict[str, Any],
    idx_in_round: int,
    round_idx: int,
    seen_qids: Dict[str, int],
    mark_selection_made_fn,
) -> Dict[str, Any]:
    """Render one MCQ card; returns metadata for aggregation / SRS batch actions."""
    qid = _stable_quiz_item_id(q)
    seen_qids[qid] = seen_qids.get(qid, 0) + 1
    key_suffix = f"{qid}_{seen_qids[qid]}" if seen_qids[qid] > 1 else qid
    q_text = q.get("question", "")
    choices = q.get("choices", {}) or {}
    answer_letter = q.get("answer", None)

    brief_explanation = (q.get("brief_explanation") or q.get("explanation") or "").strip()
    detailed_explanation: Dict = q.get("detailed_explanation") or {}

    rk = "" if round_idx == 0 else f"r{round_idx}_"
    submit_key = f"{quiz_state_key}_sub_{generation_token}_{rk}{key_suffix}"
    srs_key = f"{quiz_state_key}_srs_{generation_token}_{rk}{key_suffix}"
    exp_key = f"{quiz_state_key}_expander_{generation_token}_{rk}{key_suffix}"
    if exp_key not in st_module.session_state:
        st_module.session_state[exp_key] = False

    card_key = hashlib.md5(
        f"{quiz_state_key}|{generation_token}|{rk}{key_suffix}".encode("utf-8")
    ).hexdigest()[:16]

    already_submitted = bool(st_module.session_state.get(submit_key))

    status_icon = ""
    if already_submitted:
        submitted = st_module.session_state.get(submit_key, {})
        status_icon = " ✅" if submitted.get("is_correct") else " ❌"

    qtype_raw = (q.get("question_type") or "").strip().lower()
    angle = _QUESTION_ANGLE_LABELS.get(
        qtype_raw,
        qtype_raw.replace("_", " ").title() if qtype_raw else "",
    )

    with st_module.container(border=True, key=f"la_quiz_card_{card_key}"):
        st_module.markdown('<div class="la-quiz-card-start"></div>', unsafe_allow_html=True)
        st_module.markdown(f"### Q{idx_in_round}{status_icon}")
        if angle:
            st_module.markdown(
                f'<span style="display:inline-block;padding:0.2rem 0.65rem;border-radius:999px;'
                f"font-size:0.82rem;font-weight:600;background:#ecfeff;color:#115e59;"
                f'border:1px solid #99f6e4;">{angle}</span>',
                unsafe_allow_html=True,
            )
        if q_text:
            st_module.markdown(q_text)
        st_module.markdown("---")

        correct_letter_upper = (answer_letter or "").strip().upper()
        chosen_letter_post = ""
        if already_submitted:
            chosen_letter_post = (
                st_module.session_state.get(submit_key, {}).get("chosen", "") or ""
            ).strip().upper()

        for label in ["A", "B", "C", "D"]:
            opt = choices.get(label, "")
            marker = ""
            if already_submitted:
                if label == correct_letter_upper:
                    marker = "✅ "
                elif label == chosen_letter_post:
                    marker = "❌ "
            if opt:
                btn_label = f"{marker}{label}. {opt}"
            else:
                btn_label = f"{marker}{label}. (no option)"
            btn_key = f"la_quiz_ans_{card_key}_{label}"
            if st_module.button(
                btn_label,
                key=btn_key,
                disabled=already_submitted,
                use_container_width=True,
            ):
                chosen_letter = label
                is_correct = (
                    (chosen_letter == answer_letter)
                    if (chosen_letter and answer_letter)
                    else False
                )
                st_module.session_state[submit_key] = {
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

                st_module.session_state[exp_key] = True
                try:
                    st_module.rerun()
                except Exception:
                    try:
                        st_module.experimental_rerun()
                    except Exception:
                        pass

        with st_module.container():
            st_module.markdown('<div class="la-action-bar"></div>', unsafe_allow_html=True)
            st_module.markdown("**Actions**")
            if st_module.button("Add to SRS", key=srs_key, use_container_width=True):
                try:
                    mgr = SRSManager()
                    concept_label = (q.get("concept_label") or "").strip()
                    concept_id = (q.get("concept_id") or "").strip()
                    topic_title = concept_label or ""
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
                            "concept_label": concept_label,
                            "concept_id": concept_id,
                            "title": topic_title,
                            "concept": topic_title,
                        },
                    )
                    if "srs_quiz_items_cache" not in st_module.session_state:
                        st_module.session_state["srs_quiz_items_cache"] = {}
                    st_module.session_state["srs_quiz_items_cache"][qid] = {
                        "id": qid,
                        "question": q_text or "",
                        "choices": choices or {},
                        "answer": answer_letter,
                        "brief_explanation": brief_explanation,
                        "detailed_explanation": detailed_explanation,
                        "item_type": "mcq",
                        "origin": "quiz_mcq",
                        "concept_label": concept_label,
                        "concept_id": concept_id,
                    }
                    st_module.session_state[f"{srs_key}_done"] = True
                    st_module.info("Added to SRS.")
                    try:
                        st_module.rerun()
                    except Exception:
                        try:
                            st_module.experimental_rerun()
                        except Exception:
                            pass
                except Exception as e:
                    st_module.error("Failed to register SRS card.")
                    st_module.exception(e)

        submitted = st_module.session_state.get(submit_key)
        if submitted:
            chosen = submitted.get("chosen")
            is_correct = submitted.get("is_correct", False)

            if is_correct:
                st_module.success("✅ Correct — well done!")
            else:
                correct_display = "(not provided)"
                if answer_letter and choices.get(answer_letter):
                    correct_display = f"{answer_letter}. {choices.get(answer_letter)}"
                if chosen:
                    chosen_text = choices.get(chosen, "")
                    st_module.error(
                        f"❌ Incorrect — you chose **{chosen}**. {chosen_text}\n\n"
                        f"**Correct:** {correct_display}"
                    )
                else:
                    st_module.error(f"❌ Incorrect.\n\n**Correct:** {correct_display}")

            effective_brief = brief_explanation or _fallback_brief(q_text, choices, answer_letter)
            if effective_brief:
                st_module.markdown(effective_brief)

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
                with st_module.expander("🔽 Show reasoning"):
                    _render_detailed_explanation(
                        detailed_explanation,
                        choices,
                        answer_letter or "",
                    )

    return {
        "question": q,
        "qid": qid,
        "key_suffix": key_suffix,
        "submit_key": submit_key,
        "round_idx": round_idx,
    }


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


def _try_append_more_questions(
    st_module: Any,
    *,
    stem: str,
    text: str,
    llm,
    API_DEFAULT: str,
    doc_id: str,
    quiz_state_key: str,
    wave_breaks_key: str,
    rounds_key: str,
    n_more: int,
) -> None:
    """Generate up to n_more (1–10) new items and append as a new round."""
    n_more = max(1, min(10, int(n_more)))
    if not text:
        st_module.warning("No document text — cannot continue.")
        return
    with st_module.spinner("Generating more questions..."):
        try:
            flat = _flatten_round_items(st_module.session_state.get(rounds_key, []))
            if not flat:
                flat = st_module.session_state.get(quiz_state_key, []) or []
            chunk_counts = _quiz_session_chunk_counts(flat)
            exclude_q = [
                (item.get("question") or "").strip()
                for item in flat
                if (item.get("question") or "").strip()
            ][:20]
            if st_module.session_state.get("use_api_mode", False):
                more_items, _ = generate_quiz(
                    stem=stem,
                    context_text=text,
                    n=n_more,
                    use_api_mode=True,
                    api_base=os.getenv("API_BASE", API_DEFAULT),
                    token=st_module.session_state.get("api_token", "") or "",
                    llm_call=None,
                    session_chunk_counts=chunk_counts,
                    exclude_questions=exclude_q,
                )
            else:
                more_items, _ = generate_quiz(
                    stem=stem,
                    context_text=text,
                    n=n_more,
                    use_api_mode=False,
                    llm_call=llm,
                    session_chunk_counts=chunk_counts,
                    exclude_questions=exclude_q,
                )

            existing_ids = {(item.get("id") or "") for item in flat if item.get("id")}
            existing_qnorm = {_normalize_quiz_question_text(item.get("question") or "") for item in flat}
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

            if not appended:
                st_module.info(
                    "No new unique questions were generated — try again or pick a different count."
                )
                return

            rounds = st_module.session_state.get(rounds_key, [])
            if not isinstance(rounds, list):
                rounds = []
            rounds.append({"items": appended})
            st_module.session_state[rounds_key] = rounds

            combined = flat + appended
            st_module.session_state[quiz_state_key] = combined

            waves = st_module.session_state.get(wave_breaks_key, [])
            if not isinstance(waves, list):
                waves = []
            waves.append(len(flat) + 1)
            st_module.session_state[wave_breaks_key] = waves

            try:
                save_quiz_to_disk(stem, combined)
                if "srs_disk_quiz_items_cache" in st_module.session_state:
                    del st_module.session_state["srs_disk_quiz_items_cache"]
            except Exception as e:
                st_module.warning(f"Continued in session but failed to save to disk: {e}")
            st_module.success(f"Added {_count_noun(len(appended), 'new question')} in a new round.")
            st_module.session_state[f"la_scroll_quiz_latest_{stem}"] = True
            try:
                st_module.rerun()
            except Exception:
                try:
                    st_module.experimental_rerun()
                except Exception:
                    pass
        except requests.HTTPError as he:
            try:
                detail = he.response.json().get("detail", str(he))
            except Exception:
                detail = str(he)
            st_module.error(f"Continue failed: {detail}")
        except Exception as e:
            st_module.error("Continue failed.")
            st_module.exception(e)


def render(st: Any, stem: str, text: str, llm, hist_key: str):
    """
    Render the Quiz generation + Quiz item UI.
    """
    API_DEFAULT = os.getenv("API_BASE", "http://localhost:8000") if "os" in globals() else "http://localhost:8000"
    
    st.markdown('<a id="study-quiz"></a>', unsafe_allow_html=True)

    quiz_state_key = f"quiz_items_{stem}"
    quiz_generation_key = f"{quiz_state_key}_generation"
    wave_breaks_key = f"{quiz_state_key}_wave_breaks"
    rounds_key = f"{quiz_state_key}_rounds"
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
        n_q = st.number_input(
            "Number of quiz items (1–10)",
            min_value=1,
            max_value=10,
            value=5,
            help="Each batch is one round. Generate 1–10 questions at a time.",
        )
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

                            st.session_state[rounds_key] = [{"items": list(quiz_items)}]
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

    rounds = _ensure_quiz_rounds(st, quiz_state_key, wave_breaks_key, rounds_key)
    generation_token = int(st.session_state.get(quiz_generation_key, 0))

    def _mark_selection_made_local(sel_key: str):
        _mark_selection_made(sel_key)

    st.markdown("")

    quiz_items = st.session_state.get(quiz_state_key, []) or []
    if not quiz_items:
        st.info("No quiz items generated yet. Click 'Generate quiz' to create items.")
        return

    quiz_render_items: List[Dict[str, Any]] = []
    rounds_with_items = [r for r in rounds if (r.get("items") or [])]
    n_rounds = len(rounds_with_items)
    do_scroll_latest = bool(st.session_state.pop(f"la_scroll_quiz_latest_{stem}", False))
    latest_anchor_id = "la-quiz-latest-" + hashlib.md5(stem.encode("utf-8")).hexdigest()[:16]

    for ri, round_data in enumerate(rounds_with_items):
        items = round_data.get("items") or []
        if not items:
            continue
        round_label_num = ri + 1

        base_title = f"Round {round_label_num}"

        is_latest_round = ri == n_rounds - 1
        use_collapsible = not is_latest_round

        def _render_round_inner() -> None:
            if is_latest_round and do_scroll_latest:
                st.markdown(
                    f'<div id="{latest_anchor_id}" style="scroll-margin-top:5rem;"></div>',
                    unsafe_allow_html=True,
                )
            seen_qids_round: Dict[str, int] = {}
            round_render_items: List[Dict[str, Any]] = []
            for j, q in enumerate(items, start=1):
                ritem = _render_single_question(
                    st,
                    stem=stem,
                    doc_id=doc_id,
                    quiz_state_key=quiz_state_key,
                    generation_token=generation_token,
                    q=q,
                    idx_in_round=j,
                    round_idx=ri,
                    seen_qids=seen_qids_round,
                    mark_selection_made_fn=_mark_selection_made_local,
                )
                round_render_items.append(ritem)
            quiz_render_items.extend(round_render_items)

            answered, correct, _, total_r_live = _round_progress(round_render_items, st)
            if total_r_live and answered < total_r_live:
                st.caption(f"Progress: {answered}/{total_r_live} answered")

            is_round_complete = total_r_live > 0 and answered == total_r_live
            if is_round_complete:
                accuracy_pct = (correct / total_r_live) * 100.0 if total_r_live else 0.0
                if accuracy_pct >= 90:
                    encouragement = "🎯 Excellent work — you've got a strong grip on this lecture."
                elif accuracy_pct >= 70:
                    encouragement = "✨ Strong round — a couple more passes will lock this in."
                elif accuracy_pct >= 50:
                    encouragement = "💡 Keep practicing this lecture — you're getting there."
                else:
                    encouragement = "📚 Keep practicing — try reviewing the lecture and giving it another go."

                log_key = f"{quiz_state_key}_session_logged_{generation_token}_r{ri}_{total_r_live}"
                if not st.session_state.get(log_key):
                    try:
                        append_quiz_session(
                            stem=stem,
                            doc_id=doc_id or "",
                            correct=int(correct),
                            total=int(total_r_live),
                            ts_iso=datetime.now(timezone.utc).isoformat(),
                        )
                        st.session_state[log_key] = True
                    except Exception:
                        pass

                st.markdown("---")
                st.markdown(f"#### ✅ Round {round_label_num} Complete")
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Score (this round)", f"{correct} / {total_r_live}")
                with m2:
                    st.metric("Accuracy", f"{accuracy_pct:.0f}%")
                st.markdown(encouragement)

                weak = _weak_topics_from_items(round_render_items, st)
                if weak:
                    st.markdown("**Weak topics this round:**")
                    for t, c in sorted(weak.items(), key=lambda kv: -kv[1])[:8]:
                        st.markdown(f"- {t} ({c}× missed)")
                show_continue = ri == n_rounds - 1
                if show_continue:
                    st.markdown("")
                    st.caption("Keep going in small batches (max 10 questions per round).")
                    cq, cc = st.columns([1, 2])
                    with cq:
                        if st.button(
                            "Continue with 5 more questions",
                            key=f"{quiz_state_key}_q5_r{round_label_num}_{generation_token}",
                            type="primary",
                            use_container_width=True,
                        ):
                            _try_append_more_questions(
                                st,
                                stem=stem,
                                text=text,
                                llm=llm,
                                API_DEFAULT=API_DEFAULT,
                                doc_id=doc_id,
                                quiz_state_key=quiz_state_key,
                                wave_breaks_key=wave_breaks_key,
                                rounds_key=rounds_key,
                                n_more=5,
                            )
                    with cc:
                        st.markdown("**Custom batch (1–10)**")
                        ic1, ic2 = st.columns([2, 1])
                        with ic1:
                            cust_n = st.number_input(
                                "Questions",
                                min_value=1,
                                max_value=10,
                                value=5,
                                help="How many questions to add in the next round.",
                                key=f"{quiz_state_key}_custom_n_r{round_label_num}_{generation_token}",
                            )
                        with ic2:
                            st.write("")
                            st.write("")
                            if st.button(
                                "Generate",
                                key=f"{quiz_state_key}_custom_go_r{round_label_num}_{generation_token}",
                                use_container_width=True,
                            ):
                                _try_append_more_questions(
                                    st,
                                    stem=stem,
                                    text=text,
                                    llm=llm,
                                    API_DEFAULT=API_DEFAULT,
                                    doc_id=doc_id,
                                    quiz_state_key=quiz_state_key,
                                    wave_breaks_key=wave_breaks_key,
                                    rounds_key=rounds_key,
                                    n_more=int(cust_n),
                                )

        if use_collapsible:
            with st.expander(base_title, expanded=False):
                _render_round_inner()
        else:
            st.markdown(f"### {base_title}")
            st.markdown("")
            _render_round_inner()

    if do_scroll_latest:
        _scroll_streamlit_to_anchor(latest_anchor_id)

    if n_rounds >= 2:
        st.markdown("---")
        st.markdown("### Overall progress")
        tot_q = len(quiz_render_items)
        total_correct_all = 0
        total_answered_all = 0
        for render_item in quiz_render_items:
            sub = st.session_state.get(render_item["submit_key"])
            if not sub:
                continue
            total_answered_all += 1
            if sub.get("is_correct", False):
                total_correct_all += 1
        st.metric("Total (session so far)", f"{total_correct_all} / {tot_q}")
        trend_msg, delta = trend_vs_prior(stem)
        c_tr1, c_tr2 = st.columns(2)
        with c_tr1:
            st.caption(trend_msg)
        with c_tr2:
            if delta is None:
                st.caption("Trend vs prior sessions: —")
            else:
                st.caption(f"Trend vs prior sessions: {delta:+.0f} pts vs avg")

        all_weak = _weak_topics_from_items(quiz_render_items, st)
        if all_weak:
            top_lines = [f"{t} ({c}×)" for t, c in sorted(all_weak.items(), key=lambda kv: -kv[1])[:6]]
            st.markdown("**Weakest areas (session):** " + "; ".join(top_lines))
        else:
            st.markdown("**Weakest areas (session):** none flagged yet — nice work.")

        last3 = recent_sessions(stem, 3)
        if last3:
            lines = []
            for row in reversed(last3):
                try:
                    pct = float(row.get("accuracy_pct", 0))
                    tot = int(row.get("total", 0))
                    cor = int(row.get("correct", 0))
                except Exception:
                    pct, tot, cor = 0.0, 0, 0
                ts = (row.get("ts") or "")[:16].replace("T", " ")
                lines.append(f"- **{ts} UTC** — {cor}/{tot} · **{pct:.0f}%**")
            st.markdown("**Recent quiz rounds** (newest first)")
            st.markdown(chr(10).join(lines))

    total_questions = len(quiz_render_items)
    total_answered = 0
    total_correct = 0
    total_wrong = 0
    for render_item in quiz_render_items:
        submitted = st.session_state.get(render_item["submit_key"])
        if not submitted:
            continue
        total_answered += 1
        if submitted.get("is_correct", False):
            total_correct += 1
        else:
            total_wrong += 1

    session_fully_answered = total_questions > 0 and total_answered == total_questions

    # ── Next-step recommendations (preserved logic for missed items) ──────
    if total_answered and total_wrong > 0:
        st.markdown("### 🔁 Next step recommendations")
        st.markdown(f"You answered {_count_noun(total_answered, 'question')}. Missed: {total_wrong}.")
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
                    r_idx = int(render_item.get("round_idx", 0) or 0)
                    card_id = (
                        f"{base_qid}_r{r_idx}_{generation_token}_{key_suffix}"
                        if base_qid
                        else f"missed_r{r_idx}_{generation_token}_{key_suffix}"
                    )
                    if card_id in seen_card_ids:
                        continue
                    seen_card_ids.add(card_id)
                    concept_label_m = (q.get("concept_label") or "").strip()
                    concept_id_m = (q.get("concept_id") or "").strip()
                    topic_m = concept_label_m or ""
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
                            "concept_label": concept_label_m,
                            "concept_id": concept_id_m,
                            "title": topic_m,
                            "concept": topic_m,
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
                        "concept_label": concept_label_m,
                        "concept_id": concept_id_m,
                    }
                    added += 1
                st.success(f"Added {_count_noun(added, 'missed concept')} to SRS.")
                st.markdown("Jump to Confused section below.")
            except Exception as e:
                st.error(f"Could not add missed concepts: {e}")

    elif total_answered and total_wrong == 0 and not session_fully_answered:
        st.markdown("### 🔁 Next step recommendations")
        st.markdown(f"You answered {_count_noun(total_answered, 'question')}. Missed: 0.")
        st.markdown("Great progress so far — keep going.")