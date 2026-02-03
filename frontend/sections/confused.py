# frontend/sections/confused.py
"""
Confused? Quick prioritized list section extracted from frontend/app.py

Expose a single function:
    render(st, stem, embeddings_path, index_path, use_faiss_search, llm)

This module mirrors the UI and behavior originally in app.py but improves the
presentation and layout for readability and scanning. **No logic, keys, or
persistence have been changed** — only the UI/UX.
"""

import os
import re
from pathlib import Path
from typing import Any, List, Dict

import streamlit as st

# handlers and backend helpers (same as in app.py)
from frontend.handlers import perform_confusion_analysis, perform_query
from backend.study_srs import SRSManager


API_DEFAULT = os.getenv("API_BASE", "http://localhost:8000")


def _render_evidence_block(evidence: List[Dict[str, Any]]):
    """Render a compact evidence list (used inside expanders)."""
    if not evidence:
        st.write("_No evidence available._")
        return

    for e in evidence:
        try:
            etype = e.get("type", "")
            if etype in ("quiz", "persisted"):
                meta = e.get("meta", {}) or {}
                qid = meta.get("qid") or e.get("qid") or "<no-id>"
                qtext = (meta.get("question") or e.get("question") or "")[:800]
                # Use markdown with a visible line break
                st.markdown(f"- **Quiz** — id=`{qid}`  \n  {qtext}")
            else:
                txt = str(e)[:800]
                st.markdown(f"- {txt}")
        except Exception:
            st.markdown(f"- {str(e)[:400]}")


def render(
    st: Any,
    stem: str,
    embeddings_path: Path,
    index_path: Path,
    use_faiss_search: bool,
    llm,
):
    """
    Render the 'Confused? Quick prioritized list' UI for the given document (stem).
    Keep behavior identical to previous inline implementation, but improve layout.

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

    st.markdown("## Confused? Quick prioritized list")
    st.write("These are concepts you repeatedly missed in quizzes — review them before moving on.")

    # gather session-local quiz context (not authoritative)
    history = st.session_state.get(hist_key, []) or []
    quiz_items_session = st.session_state.get(quiz_state_key, []) or []

    # Build quiz_submissions from session (same as original)
    quiz_submissions = []
    for q in quiz_items_session:
        qid = q.get("id")
        submit_key = f"{quiz_state_key}_sub_{qid}"
        sub = st.session_state.get(submit_key)
        if sub:
            quiz_submissions.append(
                {
                    "id": qid,
                    "question": q.get("question", "") or "",
                    "is_correct": bool(sub.get("is_correct", False)),
                }
            )

    # fetch persisted top confusions (handlers returns only real quiz-based confusions)
    results = []
    try:
        top_limit = 5  # cap to 3-5 as requested (handlers also enforces)
        results = (
            perform_confusion_analysis(
                history=history,
                quiz_submissions=quiz_submissions,
                retrieved_chunks=[],
                top_n=top_limit,
                llm_call=llm,
            )
            or []
        )
    except Exception as e:
        st.error("Failed to compute prioritized confusions.")
        st.exception(e)
        results = []

    # require positive signal_strength (wrong_count) — safety filter
    real_confusions = [r for r in results if int(r.get("signal_strength", 0)) > 0]

    # When empty, show one calm line only (no cards/icons)
    if not real_confusions:
        st.write("No repeated quiz mistakes yet 👍")
        st.markdown("---")
        return

    # If there are confusions, show a short summary and allow expanding the list
    st.info(f"Found {len(real_confusions)} prioritized confusion(s). Focus on the highest-signal items first.")
    st.markdown("")  # spacing

    # If results length equals the cap, indicate that this is a prioritized subset
    if len(results) >= top_limit:
        st.caption(f"Showing up to {top_limit} highest-priority confusions.")

    # allow user to show all (though backend already caps); keep UI compact for long lists
    show_all_key = f"conf_show_all_{stem}"
    if show_all_key not in st.session_state:
        st.session_state[show_all_key] = False

    # A simple toggle to show the full set if user wants it
    cols_toggle = st.columns([4, 1])
    with cols_toggle[0]:
        st.markdown("### Prioritized confusions")
    with cols_toggle[1]:
        # Use a toggle behavior - one button toggles between show all / show top
        if st.button("Show all", key=f"{show_all_key}_btn") and not st.session_state[show_all_key]:
            st.session_state[show_all_key] = True
        elif st.button("Show top", key=f"{show_all_key}_btn2") and st.session_state[show_all_key]:
            st.session_state[show_all_key] = False

    # Choose which results to display based on the toggle (keeps original ordering)
    display_results = real_confusions if st.session_state[show_all_key] else real_confusions[:min(len(real_confusions), 5)]

    # Render each confusion as a nicely spaced block inside an expander so long text won't overwhelm
    for idx, item in enumerate(display_results, start=1):
        concept = item.get("concept", "(no concept)")
        strength = int(item.get("signal_strength", 0))
        reason = item.get("reason", "")
        evidence = item.get("evidence", []) or []

        # Expander title: concept + missed count for quick scanning
        title = f"{idx}. {concept} — missed {strength} time{'s' if strength != 1 else ''}"
        with st.expander(title, expanded=False):
            # Reason (if present) shown subtly above evidence
            if reason:
                st.caption(reason)

            # Evidence: collapsible to avoid clutter
            if evidence:
                with st.expander("🔎 Evidence (examples)", expanded=False):
                    _render_evidence_block(evidence)
            else:
                st.write("_No evidence available_")

            st.markdown("")  # spacing

            # Action row: Explain simply | Add to SRS
            # Keep keys identical to original to preserve behavior
            explain_btn_key = f"conf_explain_{stem}_{idx}"
            add_srs_key = f"conf_add_srs_{stem}_{idx}"

            col_a, col_b, col_c = st.columns([2, 2, 3])
            with col_a:
                if st.button("🧠 Explain simply", key=explain_btn_key):
                    try:
                        candidate_index_path = str(index_path) if index_path and Path(index_path).exists() else None
                        raw_concept = item.get("concept", "")
                        explain_q = f"Explain this simply and give 1 short example: {raw_concept}"
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
                                    txt = rc.get("text", "") or ""
                                    preview = txt[:1000] + ("..." if len(txt) > 1000 else "")
                                    st.write(preview)
                    except Exception as e:
                        st.error("Failed to generate explanation.")
                        st.exception(e)

            with col_b:
                # Add to SRS (idempotent) — keep same logic and key naming
                if st.button("➕ Add to SRS", key=add_srs_key):
                    try:
                        mgr = SRSManager()
                        # try to extract canonical quiz id from evidence metadata
                        card_id = None
                        for e in evidence:
                            meta = e.get("meta", {}) or {}
                            if meta.get("qid"):
                                card_id = meta.get("qid")
                                break
                            if e.get("qid"):
                                card_id = e.get("qid")
                                break

                        # fallback deterministic id derived from concept+stem (same sanitization)
                        if not card_id:
                            safe = re.sub(r"[^a-zA-Z0-9_]+", "_", concept).strip("_")[:40] or "conf"
                            card_id = f"{stem}_{safe}"

                        mgr.ensure_card(card_id)
                        st.success(f"Added to SRS: {card_id}")
                    except Exception as e:
                        st.error("Failed to add to SRS.")
                        st.exception(e)

            with col_c:
                # Show a short preview of the first pieces of evidence inline for context,
                # this gives users quick context without expanding.
                if evidence:
                    preview_lines: List[str] = []
                    for e in evidence[:2]:
                        if e.get("type") in ("quiz", "persisted"):
                            meta = e.get("meta", {}) or {}
                            qtext = (meta.get("question") or e.get("question") or "")[:120].replace("\n", " ")
                            preview_lines.append(f"• Quiz: {qtext}")
                        else:
                            # compute replacement outside f-string to avoid backslash-in-expression syntax issues
                            raw = str(e)[:120].replace("\n", " ")
                            preview_lines.append(f"• {raw}")
                    # show preview lines joined with line breaks
                    st.write("\n".join(preview_lines))
                else:
                    st.write("_No quick preview available_")

    st.markdown("---")
