# frontend/sections/confused.py
"""
Confused? Quick prioritized list section extracted from frontend/app.py

Expose a single function:
    render(st, stem, embeddings_path, index_path, use_faiss_search, llm)

This module mirrors the UI and behavior originally in app.py without changes.
"""

import html as html_mod
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import streamlit as st

# handlers and backend helpers (same as in app.py)
from frontend.handlers import perform_confusion_analysis, perform_query, delete_confusion_entries
from backend.study_srs import SRSManager
from backend.quiz_storage import load_quiz_item_by_id


API_DEFAULT = os.getenv("API_BASE", "http://localhost:8000")

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


def _stable_card_key(stem: str, item: dict, idx: int) -> str:
    raw = (
        item.get("store_key")
        or item.get("card_id")
        or item.get("concept_bucket_key")
        or item.get("chunk_id")
        or f"{stem}_{idx}"
    )
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", str(raw)).strip("_")
    return safe[:80] or f"{stem}_{idx}"


def _resolve_mcq_question_text(
    qid: str,
    question: str,
    *,
    item: Optional[dict] = None,
    meta: Optional[dict] = None,
) -> str:
    resolved = " ".join(str(question or "").split()).strip()
    if resolved:
        return resolved
    if isinstance(meta, dict):
        resolved = " ".join(str(meta.get("question") or "").split()).strip()
        if resolved:
            return resolved
    if isinstance(item, dict):
        card_qid = str(item.get("quiz_question_id") or item.get("last_mcq_id") or "").strip()
        if qid and qid == card_qid:
            resolved = " ".join(
                str(
                    item.get("original_question")
                    or item.get("question")
                    or item.get("last_question")
                    or ""
                ).split()
            ).strip()
            if resolved:
                return resolved
    if qid:
        quiz_item = load_quiz_item_by_id(qid) or {}
        resolved = " ".join(str(quiz_item.get("question") or "").split()).strip()
    return resolved


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


def _question_angle_label(question_type: str) -> str:
    qtype_raw = str(question_type or "").strip().lower()
    if not qtype_raw:
        return "Quiz"
    return _QUESTION_ANGLE_LABELS.get(
        qtype_raw,
        qtype_raw.replace("_", " ").title(),
    )


def _choice_label(letter: str, choices: dict) -> str:
    clean_letter = str(letter or "").strip().upper()
    if not clean_letter or not isinstance(choices, dict):
        return ""
    return " ".join(str(choices.get(clean_letter, "") or "").split())


def _chosen_mistake_text(letter: str, choices: dict, why_wrong: dict) -> str:
    clean_letter = str(letter or "").strip().upper()
    if not clean_letter:
        return ""
    if isinstance(why_wrong, dict):
        reason = " ".join(str(why_wrong.get(clean_letter, "") or "").split())
        if reason:
            return reason
    return _choice_label(clean_letter, choices)


def _enforce_word_limit(text: str, max_words: int = 8) -> str:
    words = " ".join(str(text or "").split()).split()
    if not words:
        return ""
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _sanitize_confusion_blurb(text: str) -> str:
    clean = " ".join(str(text or "").replace('"', "").replace("'", "").split())
    clean = clean.strip(" .-—")
    return _enforce_word_limit(clean, 8)


def _blurb_cache() -> dict:
    key = "confusion_blurb_cache"
    if key not in st.session_state:
        st.session_state[key] = {}
    return st.session_state[key]


def _blurb_request_from_evidence(
    evidence_entry: dict,
    concept_name: str,
    last_qid: str,
    cache_key: str,
) -> dict | None:
    """Build a batch-blurb request dict, or None if cached / not eligible."""
    if _blurb_cache().get(cache_key):
        return None

    meta = evidence_entry.get("meta", {}) or {}
    angle = _question_angle_label(meta.get("question_type") or "")
    choices = meta.get("choices") if isinstance(meta.get("choices"), dict) else {}
    why_wrong = meta.get("why_wrong") if isinstance(meta.get("why_wrong"), dict) else {}
    correct = str(meta.get("answer") or "").strip().upper()
    correct_text = _choice_with_text(correct, choices) or correct or "the correct answer"

    qid = str(meta.get("qid") or evidence_entry.get("qid") or "").strip()
    chosen = str(meta.get("last_chosen_answer") or "").strip().upper()
    use_chosen = bool(chosen and correct and chosen != correct and qid and qid == last_qid)
    wrong_text = _choice_with_text(chosen, choices) if use_chosen else ""
    if not wrong_text and not correct_text:
        return None

    return {
        "cache_key": cache_key,
        "concept_name": " ".join(str(concept_name or "this concept").split()),
        "angle": angle,
        "wrong_answer": wrong_text or "their answer",
        "correct_answer": correct_text,
    }


def _parse_batch_blurb_response(raw: str, expected_keys: list[str]) -> dict[str, str]:
    text = str(raw or "").strip()
    if not text:
        return {}
    payload = None
    try:
        payload = json.loads(text)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                payload = json.loads(match.group(0))
            except Exception:
                payload = None
    if not isinstance(payload, dict):
        return {}

    out: dict[str, str] = {}
    for key in expected_keys:
        val = payload.get(key)
        if val is None:
            continue
        blurb = _sanitize_confusion_blurb(str(val))
        if blurb:
            out[key] = blurb
    return out


def _generate_confusion_blurbs_batch(llm_call, requests: list[dict]) -> dict[str, str]:
    """One LLM call for many confusion blurbs. Updates session cache."""
    if not llm_call or not requests:
        return {}

    pending = [r for r in requests if r.get("cache_key") and not _blurb_cache().get(r["cache_key"])]
    if not pending:
        return {}

    lines = []
    for i, req in enumerate(pending, start=1):
        lines.append(
            f'{i}. key="{req["cache_key"]}": concept="{req["concept_name"]}", '
            f'question_type="{req["angle"]}", wrong="{req["wrong_answer"]}", '
            f'correct="{req["correct_answer"]}"'
        )

    prompt = (
        "For each numbered item, write ONE short confusion description (max 8 words). "
        "Describe the conceptual mistake, not the answer text. "
        "Never quote or paraphrase the actual answer options.\n\n"
        + "\n".join(lines)
        + '\n\nReply with ONLY a JSON object mapping each key string to its description. '
        'Example: {"stem_1_abc": "confused definition with example"}'
    )
    expected_keys = [str(r["cache_key"]) for r in pending]
    try:
        raw = llm_call(prompt)
    except Exception:
        return {}

    parsed = _parse_batch_blurb_response(raw, expected_keys)
    cache = _blurb_cache()
    for key, blurb in parsed.items():
        cache[key] = blurb
    return parsed


def _collect_pending_blurb_requests(
    confusion_items: list[dict],
    stem: str,
) -> list[dict]:
    """Gather uncached blurb requests across all visible confusion cards."""
    requests: list[dict] = []
    seen_keys: set[str] = set()
    preview_limit = min(len(confusion_items), 5)

    for idx, item in enumerate(confusion_items[:preview_limit], start=1):
        evidence = item.get("evidence", []) or []
        extracted_concept = item.get("concept_label") or item.get("concept") or item.get("title") or ""
        last_qid = str(item.get("quiz_question_id") or item.get("last_mcq_id") or "").strip()
        seen_q: set[tuple[str, str]] = set()

        for e in evidence:
            if isinstance(e, str):
                continue
            if e.get("type") not in ("quiz", "persisted"):
                continue
            meta = e.get("meta", {}) or {}
            qtext = " ".join(str(meta.get("question") or e.get("question") or "").split())
            qid = str(meta.get("qid") or e.get("qid") or "").strip()
            if not qtext:
                continue
            dedupe_key = (qid, qtext.strip().lower())
            if dedupe_key in seen_q:
                continue
            seen_q.add(dedupe_key)
            if len(seen_q) > 3:
                break

            cache_key = f"{stem}_{idx}_{qid or dedupe_key}"
            if cache_key in seen_keys:
                continue
            seen_keys.add(cache_key)

            req = _blurb_request_from_evidence(
                e,
                concept_name=extracted_concept,
                last_qid=last_qid,
                cache_key=cache_key,
            )
            if req:
                requests.append(req)
    return requests


def _ensure_confusion_blurbs(
    llm_call,
    stem: str,
    confusion_items: list[dict],
) -> None:
    """Prefetch all pending blurbs for this lecture in one batched LLM call."""
    if not llm_call:
        return
    pending = _collect_pending_blurb_requests(confusion_items, stem)
    if not pending:
        return
    batch_sig = "|".join(sorted(r["cache_key"] for r in pending))
    done_key = f"conf_blurb_batch_done_{stem}"
    if st.session_state.get(done_key) == batch_sig:
        return
    with st.spinner("Analyzing your mistakes…"):
        _generate_confusion_blurbs_batch(llm_call, pending)
    st.session_state[done_key] = batch_sig


def _fallback_confusion_blurb(meta: dict, last_qid: str) -> str:
    choices = meta.get("choices") if isinstance(meta.get("choices"), dict) else {}
    why_wrong = meta.get("why_wrong") if isinstance(meta.get("why_wrong"), dict) else {}
    correct = str(meta.get("answer") or "").strip().upper()
    qid = str(meta.get("qid") or "").strip()
    chosen = str(meta.get("last_chosen_answer") or "").strip().upper()
    use_chosen = bool(chosen and correct and chosen != correct and qid and qid == last_qid)
    if use_chosen:
        reason = _chosen_mistake_text(chosen, choices, why_wrong)
        if reason and not reason.strip().upper().startswith(chosen):
            blurb = _sanitize_confusion_blurb(reason)
            if blurb:
                return blurb
    return "repeated confusion on this concept"


def _format_flagged_confusion_line(
    evidence_entry: dict,
    last_qid: str,
    cache_key: str,
) -> str:
    """Compact one-liner: [Question type] — [conceptual confusion, max 8 words]."""
    meta = evidence_entry.get("meta", {}) or {}
    angle = _question_angle_label(meta.get("question_type") or "")

    cached = _blurb_cache().get(cache_key)
    if isinstance(cached, str) and cached.strip():
        blurb = cached.strip()
    else:
        blurb = _fallback_confusion_blurb(meta, last_qid=last_qid)

    return f"{angle} — {blurb}"


def _build_reason_rows(
    evidence: list,
    last_qid: str,
    stem: str,
    card_idx: int,
) -> list[str]:
    """Build up to 3 reason lines from evidence (no LLM; uses cache or fallback)."""
    reason_rows: list[str] = []
    seen_q: set[tuple[str, str]] = set()
    for e in evidence:
        if isinstance(e, str):
            continue
        if e.get("type") in ("quiz", "persisted"):
            meta = e.get("meta", {}) or {}
            qtext = " ".join(str(meta.get("question") or e.get("question") or "").split())
            qid = str(meta.get("qid") or e.get("qid") or "").strip()
            if not qtext:
                continue
            dedupe_key = (qid, qtext.strip().lower())
            if dedupe_key in seen_q:
                continue
            seen_q.add(dedupe_key)
            blurb_cache_key = f"{stem}_{card_idx}_{qid or dedupe_key}"
            reason_rows.append(
                _format_flagged_confusion_line(
                    e,
                    last_qid=last_qid,
                    cache_key=blurb_cache_key,
                )
            )
        else:
            extra = _shorten(" ".join(str(e).split()), limit=120)
            if extra:
                reason_rows.append(extra)
    return reason_rows


def _render_followup_loading(message: str) -> None:
    """Compact circle spinner and a short (~3 word) label for under-button feedback."""
    words = (message or "Drafting chat reply").split()
    label = html_mod.escape(" ".join(words[:3]))
    st.markdown(
        f"""
        <div class="la-followup-loading" role="status" aria-live="polite">
          <span class="la-followup-loading__ring" aria-hidden="true"></span>
          <span class="la-followup-loading__text">{label}</span>
        </div>
        <style>
        .la-followup-loading {{
          display: flex;
          align-items: center;
          gap: 0.55rem;
          margin-top: 0.45rem;
          font-size: 0.88rem;
          color: #4a5568;
        }}
        .la-followup-loading__ring {{
          width: 16px;
          height: 16px;
          border: 2px solid #d8dee9;
          border-top-color: #1a7f64;
          border-radius: 50%;
          animation: la-followup-spin 0.75s linear infinite;
          flex-shrink: 0;
        }}
        @keyframes la-followup-spin {{
          to {{ transform: rotate(360deg); }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _process_confusion_explain(
    st: Any,
    stem: str,
    embeddings_path: Path,
    index_path: Path,
    use_faiss_search: bool,
    llm,
) -> None:
    """Run a queued 'Explain simply' request from a confusion card (after cards render)."""
    pending_key = f"conf_explain_pending_{stem}"
    loading_key = f"conf_explain_loading_{stem}"
    card_loading_key = f"conf_explain_card_{stem}"
    pending = st.session_state.pop(pending_key, None)
    if not pending:
        return

    card_key = pending.get("card_key") or ""
    result_key = f"conf_explain_result_{stem}_{card_key}"
    try:
        concept = pending.get("concept") or "this concept"
        evidence_rows = pending.get("evidence_rows") or []
        compact_evidence = []
        for row in evidence_rows[:2]:
            compact_evidence.append(f"- [{row['qid']}] {_shorten(row['question'], 120)}")
        explain_q = (
            "You are tutoring a learner on a confusion point. "
            "Explain simply and provide one short concrete example.\n\n"
            f"Confused concept: {concept}\n"
            "Compact evidence snippets:\n"
            f"{chr(10).join(compact_evidence) if compact_evidence else '- None available'}"
        )
        candidate_index_path = (
            str(index_path) if index_path and Path(index_path).exists() else None
        )
        st.session_state[result_key] = perform_query(
            question=explain_q,
            embeddings_path=str(embeddings_path),
            top_k=3,
            use_faiss=bool(use_faiss_search),
            faiss_index_path=candidate_index_path,
            llm_call=llm,
        )
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()
            except Exception:
                pass
    except Exception as e:
        st.error("Failed to generate explanation.")
        st.exception(e)
    finally:
        st.session_state.pop(loading_key, None)
        st.session_state.pop(card_loading_key, None)


def _process_confusion_chat_followup(
    st: Any,
    stem: str,
    embeddings_path: Path,
    index_path: Path,
    llm,
    embeddings_ready: bool,
) -> None:
    """Auto-send a follow-up queued from a confusion card (after cards render)."""
    pending_input_key = f"chat_pending_input_{stem}"
    auto_send_key = f"chat_auto_send_{stem}"
    pending_input = st.session_state.pop(pending_input_key, None)
    if not pending_input:
        return
    if not st.session_state.pop(auto_send_key, False):
        st.session_state[pending_input_key] = pending_input
        return

    from frontend.sections.chat import _submit_chat_message

    hist_key = f"chat_history_{stem}"
    mod_reset_key = f"chat_mod_reset_{stem}"
    clear_input_key = f"chat_clear_input_{stem}"
    chat_k = int(st.session_state.get(f"chat_k_{stem}", 3))
    style = st.session_state.get(f"chat_style_{stem}", "Standard")
    explain_new = style in ("Beginner friendly", "Beginner friendly + knowledge check")
    include_example = style == "Include a practical example"
    turn_quiz = style == "Beginner friendly + knowledge check"

    st.session_state[f"chat_force_question_{stem}"] = True
    _submit_chat_message(
        st,
        user_msg=pending_input,
        stem=stem,
        hist_key=hist_key,
        embeddings_path=embeddings_path,
        index_path=index_path,
        embeddings_ready=bool(embeddings_ready),
        chat_k=chat_k,
        explain_new=explain_new,
        include_example=include_example,
        turn_quiz=turn_quiz,
        llm=llm,
        mod_reset_key=mod_reset_key,
        clear_input_key=clear_input_key,
        from_confusion_followup=True,
        show_spinner=False,
    )


def _format_time_12h(dt: datetime) -> str:
    hour = dt.hour % 12 or 12
    am_pm = "AM" if dt.hour < 12 else "PM"
    return f"{hour}:{dt.minute:02d} {am_pm}"


def _format_timestamp(value: str) -> str:
    raw = " ".join(str(value or "").split()).strip()
    if not raw:
        return ""
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return raw
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local_dt = dt.astimezone()
    local_date = local_dt.date()
    today = datetime.now().astimezone().date()
    days_ago = (today - local_date).days
    if days_ago == 0:
        return f"Today at {_format_time_12h(local_dt)}"
    if days_ago == 1:
        return f"Yesterday at {_format_time_12h(local_dt)}"
    if days_ago > 1:
        return f"{days_ago} days ago"
    return raw


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
        st.markdown(
            """
            <style>
            section[data-testid="stMain"] .la-concept-card-wrap .la-concept-srs-label,
            section[data-testid="stMain"] [class*="st-key-la_concept_card_"] .la-concept-srs-label {
              font-size: 0.875rem;
              color: rgb(49, 51, 63);
              margin: 0 0 0.35rem;
              line-height: 1.4;
            }
            section[data-testid="stMain"] .la-concept-card-wrap .la-concept-srs-helper,
            section[data-testid="stMain"] [class*="st-key-la_concept_card_"] .la-concept-srs-helper {
              font-size: 0.75rem;
              font-weight: 400;
              color: #6b7280;
              margin: -0.15rem 0 0.4rem;
              line-height: 1.35;
            }
            section[data-testid="stMain"] .la-concept-card-wrap:has(.la-concept-srs-picker) [data-testid="stPopover"] > button,
            section[data-testid="stMain"] [class*="st-key-la_concept_card_"]:has(.la-concept-srs-picker) [data-testid="stPopover"] > button {
              white-space: normal !important;
              text-align: left !important;
              height: auto !important;
              min-height: 2.75rem;
              line-height: 1.4 !important;
              word-break: break-word;
              overflow-wrap: anywhere;
            }
            section[data-testid="stMain"] [data-testid="stPopoverBody"] [data-testid="stRadio"] [role="radiogroup"] label,
            section[data-testid="stMain"] [data-testid="stPopoverBody"] [data-testid="stRadio"] [role="radiogroup"] label p,
            section[data-testid="stMain"] [data-testid="stPopoverBody"] [data-testid="stRadio"] [role="radiogroup"] label div {
              white-space: normal !important;
              overflow: visible !important;
              text-overflow: unset !important;
              word-break: break-word;
              overflow-wrap: anywhere;
              line-height: 1.4 !important;
              max-width: 100% !important;
            }
            section[data-testid="stMain"] [data-testid="stPopoverBody"] [data-testid="stRadio"] [role="radiogroup"] label {
              align-items: flex-start !important;
              height: auto !important;
              min-height: 2.75rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        preview_limit = min(len(real_confusions), 5)
        for idx, item in enumerate(real_confusions[:preview_limit], start=1):
            card_key = _stable_card_key(stem, item, idx)
            with st.container(border=True, key=f"la_concept_card_{card_key}"):
                st.markdown('<div class="la-concept-card-start"></div>', unsafe_allow_html=True)
                concept = item.get("title") or item.get("concept") or item.get("concept_label") or "Unlabeled concept"
                extracted_concept = item.get("concept_label") or item.get("concept") or concept
                strength = int(item.get("signal_strength", 0))
                total_attempts = int(item.get("total_attempts", 0) or 0)
                error_rate = float(item.get("error_rate", 0.0) or 0.0)
                last_seen = _format_timestamp(item.get("last_seen", ""))
                header_cols = st.columns([2.4, 1, 1])
                with header_cols[0]:
                    st.markdown(
                        f'<p class="la-concept-title">{idx}. {html_mod.escape(concept)}</p>',
                        unsafe_allow_html=True,
                    )
                    if last_seen:
                        st.caption(f"Last seen: {last_seen}")
                with header_cols[1]:
                    st.metric("Error rate", f"{error_rate:.0%}")
                with header_cols[2]:
                    st.metric("Attempts", total_attempts)
                st.markdown('<hr class="la-concept-divider" />', unsafe_allow_html=True)
                evidence = item.get("evidence", []) or []
                evidence_quiz_rows = []
                item_type = (item.get("item_type") or ("mcq" if item.get("is_mcq") else "concept")).strip().lower()
                is_mcq_item = item_type == "mcq"

                # Build a deduplicated list of WRONG-only candidate MCQs from evidence.
                # `evidence` is sourced from the card's mcq_history, which the
                # confusion store only appends to on incorrect attempts, so every
                # entry here represents a question the user actually got wrong.
                mcq_candidates = []
                seen_mcq_ids: set[str] = set()
                for e in evidence:
                    if isinstance(e, str):
                        qid = e
                        question = ""
                        meta = {}
                    else:
                        meta = e.get("meta", {}) or {}
                        qid = meta.get("qid") or e.get("qid") or ""
                        question = e.get("question") or meta.get("question") or ""
                    stable_id = str(qid).strip() if qid else ""
                    question = _resolve_mcq_question_text(
                        stable_id,
                        question,
                        item=item,
                        meta=meta if isinstance(meta, dict) else {},
                    )
                    if stable_id:
                        if stable_id in seen_mcq_ids:
                            for row in mcq_candidates:
                                if row.get("id") == stable_id and not row.get("question") and question:
                                    row["question"] = question
                                    row["label"] = question
                            continue
                        seen_mcq_ids.add(stable_id)
                    elif not question:
                        continue
                    if not question:
                        continue
                    mcq_candidates.append(
                        {"id": stable_id or None, "question": question, "label": question}
                    )

                # Only fall back to the card's latest question if it really was a
                # wrong attempt. The persisted `last_question`/`original_question`
                # field is updated on every recorded answer (correct or wrong), so
                # we must gate this behind `last_is_correct == False` to avoid
                # leaking a correctly-answered question into the SRS dropdown.
                if is_mcq_item and not bool(item.get("last_is_correct", False)):
                    mcq_payload = _build_mcq_from_item(item, {"id": item.get("quiz_question_id"), "question": item.get("original_question")})
                    fallback_id = str(mcq_payload.get("id") or "").strip()
                    fallback_question = str(mcq_payload.get("question") or "").strip()
                    if fallback_question:
                        if fallback_id and fallback_id in seen_mcq_ids:
                            for row in mcq_candidates:
                                if row.get("id") == fallback_id and not row.get("question"):
                                    row["question"] = fallback_question
                                    row["label"] = fallback_question
                        else:
                            if fallback_id:
                                seen_mcq_ids.add(fallback_id)
                            mcq_candidates.insert(
                                0,
                                {
                                    "id": fallback_id or None,
                                    "question": fallback_question,
                                    "label": fallback_question,
                                },
                            )

                fallback_label = "No wrong-answered question available"
                mcq_candidates = mcq_candidates[:5]

                if not mcq_candidates:
                    mcq_candidates = [{"id": None, "question": "", "label": fallback_label}]

                selected_mcq_key = f"conf_selected_mcq_{stem}_{card_key}"
                mcq_option_indices = list(range(len(mcq_candidates)))

                def _mcq_option_label(option_index: int) -> str:
                    row = mcq_candidates[option_index]
                    return (row.get("question") or row.get("label") or "").strip()

                for e in evidence:
                    if e.get("type") in ("quiz", "persisted"):
                        meta = e.get("meta", {}) or {}
                        qtext = " ".join(str(meta.get("question") or e.get("question") or "").split())
                        qid = str(meta.get("qid") or e.get("qid") or "").strip()
                        if qid and qtext:
                            evidence_quiz_rows.append({"qid": qid, "question": qtext})

                # Primary key: use the item's own store_key/card_id first (most reliable)
                _primary_delete_key = (
                    item.get("store_key")
                    or item.get("card_id")
                    or item.get("chunk_id")
                    or ""
                ).strip()
                delete_keys = [_primary_delete_key] if _primary_delete_key else []
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
                delete_keys = list(dict.fromkeys(delete_keys))  # dedupe, primary key first

                # Centralized action input resolution for this card
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

                st.markdown('<div class="la-concept-footer-marker"></div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="la-concept-actions"><p class="la-concept-next-step">Next step</p></div>',
                    unsafe_allow_html=True,
                )
                primary_action_cols = st.columns(2)
                secondary_action_cols = st.columns(2)
                explain_result_key = f"conf_explain_result_{stem}_{card_key}"
                explain_result = st.session_state.get(explain_result_key)
                with primary_action_cols[0]:
                    # Explain simply (kept — student-first)
                    explain_btn_key = f"conf_explain_{stem}_{card_key}"
                    if st.button("Explain simply", key=explain_btn_key, type="primary", use_container_width=True):
                        st.session_state.pop(explain_result_key, None)
                        st.session_state[f"conf_explain_pending_{stem}"] = {
                            "card_key": card_key,
                            "concept": extracted_concept,
                            "evidence_rows": list(evidence_quiz_rows[:2]),
                        }
                        st.session_state[f"conf_explain_loading_{stem}"] = True
                        st.session_state[f"conf_explain_card_{stem}"] = card_key
                        try:
                            st.rerun()
                        except Exception:
                            try:
                                st.experimental_rerun()
                            except Exception:
                                pass
                    explain_loading_key = f"conf_explain_loading_{stem}"
                    explain_card_key = f"conf_explain_card_{stem}"
                    if (
                        st.session_state.get(explain_loading_key)
                        and st.session_state.get(explain_card_key) == card_key
                    ):
                        _render_followup_loading("Drafting simple explanation")

                with primary_action_cols[1]:
                    follow_key = f"conf_follow_{stem}_{card_key}"
                    if st.button("Ask follow-up in Chat", key=follow_key, use_container_width=True):
                        follow_prompt = (
                            f"Concept: {extracted_concept}\n"
                            "The user struggled with this concept in recent quizzes. Explain it clearly, give simple examples, and highlight key points that might cause confusion."
                        )
                        st.session_state[f"chat_pending_input_{stem}"] = follow_prompt
                        st.session_state[f"chat_auto_send_{stem}"] = True
                        st.session_state[f"chat_followup_loading_{stem}"] = True
                        st.session_state[f"chat_followup_card_{stem}"] = card_key
                        st.session_state[f"open_chat_tab_{stem}"] = True
                        try:
                            st.rerun()
                        except Exception:
                            try:
                                st.experimental_rerun()
                            except Exception:
                                pass
                    loading_key = f"chat_followup_loading_{stem}"
                    card_loading_key = f"chat_followup_card_{stem}"
                    if (
                        st.session_state.get(loading_key)
                        and st.session_state.get(card_loading_key) == card_key
                    ):
                        _render_followup_loading("Drafting chat reply")
                with secondary_action_cols[0]:
                    pass
                with secondary_action_cols[1]:
                    confirm_key = f"conf_reset_confirm_{stem}_{card_key}"
                    if st.session_state.get(confirm_key):
                        st.warning("Reset this confusion card?")
                        yes_col, no_col = st.columns(2)
                        if yes_col.button("Yes, reset", key=f"{confirm_key}_yes", use_container_width=True, type="primary"):
                            delete_confusion_entries(delete_keys)
                            st.session_state.pop(confirm_key, None)
                            st.rerun()
                        if no_col.button("Cancel", key=f"{confirm_key}_no", use_container_width=True):
                            st.session_state.pop(confirm_key, None)
                            st.rerun()
                    else:
                        delete_key = f"conf_delete_{stem}_{card_key}"
                        if st.button("Reset", key=delete_key, use_container_width=True):
                            if not delete_keys:
                                st.warning("No persisted entries found to delete.")
                            else:
                                st.session_state[confirm_key] = True
                                st.rerun()

                st.markdown('<div class="la-concept-srs-picker"></div>', unsafe_allow_html=True)
                st.markdown(
                    '<p class="la-concept-srs-label">Question to add to SRS</p>'
                    '<p class="la-concept-srs-helper">Showing your most recent wrong attempts</p>',
                    unsafe_allow_html=True,
                )
                _srs_help = (
                    "Explain simply and Ask follow-up in Chat use the concept directly. "
                    "Pick a question here only for Add to SRS."
                )
                _saved_mcq_index = st.session_state.get(selected_mcq_key, mcq_option_indices[0])
                if _saved_mcq_index not in mcq_option_indices:
                    _saved_mcq_index = mcq_option_indices[0]
                _srs_trigger_label = (_mcq_option_label(_saved_mcq_index) or "").strip()
                if not _srs_trigger_label or _srs_trigger_label == fallback_label:
                    _srs_trigger_label = "Choose a question"
                with st.popover(_srs_trigger_label, use_container_width=True, help=_srs_help):
                    selected_mcq_index = st.radio(
                        "Question to add to SRS",
                        options=mcq_option_indices,
                        format_func=_mcq_option_label,
                        key=selected_mcq_key,
                        label_visibility="collapsed",
                    )
                selected_mcq = mcq_candidates[selected_mcq_index]
                selected_qid = selected_mcq.get("id")
                selected_question = selected_mcq.get("question") or ""
                if not is_mcq_item:
                    st.caption("Type: Concept confusion item")

                add_srs_key = f"conf_add_srs_{stem}_{card_key}"
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

                if explain_result is not None:
                    st.markdown("**Explanation**")
                    st.write(explain_result.get("answer", "(no answer returned)"))
                    if explain_result.get("retrieved"):
                        with st.expander("Show retrieved chunks used for this explanation"):
                            for rc in explain_result.get("retrieved", []):
                                st.write(rc.get("text", "")[:1000] + ("..." if len(rc.get("text", "")) > 1000 else ""))
        if notice_rendered:
            st.session_state.pop(followup_notice_key, None)

    embeddings_ready = bool(embeddings_path and Path(embeddings_path).exists())
    _process_confusion_explain(
        st,
        stem=stem,
        embeddings_path=Path(embeddings_path),
        index_path=Path(index_path),
        use_faiss_search=use_faiss_search,
        llm=llm,
    )
    _process_confusion_chat_followup(
        st,
        stem=stem,
        embeddings_path=Path(embeddings_path),
        index_path=Path(index_path),
        llm=llm,
        embeddings_ready=embeddings_ready,
    )
    st.session_state.pop(f"chat_followup_card_{stem}", None)
    st.markdown("---")
