# frontend/app.py
import os
import sys
import re
import json
import html as html_mod
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables once at startup
load_dotenv()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import streamlit.components.v1 as components
import pdfplumber
import numpy as np
import requests

# ------------------------
# New imports (refactored)
# ------------------------
from frontend.ui_helpers import (
    strip_key_concepts_from_answer,
    clean_summary_text,
    trim_history_to_max_turns,
    render_assistant_html,
)
from frontend.handlers import (
    init_llm,
    create_embeddings_if_needed,
    load_embeddings_wrapper,
    perform_query,
    perform_summary,
    perform_chat,
    generate_quiz,
    build_index,
    save_quiz_to_disk,
    load_all_quiz_items_wrapper,
    load_quiz_item_by_id_wrapper,
)

# Backend helpers (unchanged)
from backend.create_embeddings import EMBED_DIM, create_embeddings_for_text, load_embeddings
from backend.vectorstore.faiss_store import build_faiss_index  # optional; handlers uses it too
from backend.generate_quiz import generate_mcq_from_context
from backend.study_srs import SRSManager, INTERVALS
from backend.quiz_storage import save_quiz_items, load_quiz_item_by_id, load_all_quiz_items

# initialize LLM (this mirrors existing behaviour)
llm = init_llm()
if llm is None:
    st.warning("LLM not available — using placeholders")
else:
    st.info("LLM ready — will generate real MCQs")

# Check FAISS builder availability (handlers also has this; keep for early error messages)
try:
    from backend.vectorstore.faiss_store import build_faiss_index as _maybe_build_faiss
    _faiss_builder_available = True
except Exception:
    _faiss_builder_available = False

st.set_page_config(page_title="Learning Assistant", layout="centered")
st.title("Learning Assistant")
st.markdown(
"""
Upload a lecture PDF and study it interactively. Controls:

- Toggle SAFE vs REAL embeddings.
- (Re)create embeddings file.
- Build FAISS index (optional).
- Toggle FAISS vs NumPy search.

Primary study modes:

1. **Ask a Question (RAG)** — targeted answers using retrieved chunks.
2. **Generate Summary** — quick lecture overview.
3. **Chat** — conversational mode with history.

Learning Enhancements:

- **Quiz Generation (MCQs)** — self-test knowledge.
- **Spaced Repetition (SRS)** — review cards efficiently.
- **Transparent Retrieval** — see chunks used for answers.

A simple, interactive tool for effective lecture study.
"""
)

# ----------------------------
# API mode toggle + token UI
# ----------------------------
API_DEFAULT = os.getenv("API_BASE", "http://localhost:8000")
if "use_api_mode" not in st.session_state:
    st.session_state.use_api_mode = False

# read default token from env var, allow override in UI
default_token = os.getenv("API_TOKEN", "") or ""
if "api_token" not in st.session_state:
    st.session_state.api_token = default_token

col_api, col_token = st.columns([1, 2])
with col_api:
    st.session_state.use_api_mode = st.checkbox("Use API mode", value=st.session_state.use_api_mode)
with col_token:
    st.session_state.api_token = st.text_input(
        "API Token (override)", value=st.session_state.api_token, type="password",
        help="Reads default from environment; typing here overrides for this session",
    )


# ----------------------------
# Upload + controls
# ----------------------------
uploaded = st.file_uploader("Upload a lecture PDF", type=["pdf"])
if uploaded:
    # Save uploaded file
    tmp_pdf = Path("data/raw") / uploaded.name
    tmp_pdf.parent.mkdir(parents=True, exist_ok=True)
    tmp_pdf.write_bytes(uploaded.read())

    # Extract text preview
    with pdfplumber.open(tmp_pdf) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    text = "\n\n".join(pages)

    st.subheader("Extracted preview")
    st.write(text[:1000] + ("..." if len(text) > 1000 else ""))

    # derive the stem and paths
    stem = tmp_pdf.stem
    embeddings_path = Path(f"data/processed/{stem}_embeddings.json")
    index_path = Path(f"data/processed/{stem}_embeddings.index")

    # UI controls
    col1, col2, col3 = st.columns(3)
    with col1:
        use_safe_toggle = st.checkbox(
            "Use SAFE embeddings (filter sensitive or risky outputs)",
            value=os.getenv("USE_SAFE_EMBEDDINGS", "1").lower() in ("1", "true", "yes"),
        )
    with col2:
        recreate_btn = st.button("Recreate embeddings")
    with col3:
        build_index_btn = st.button("Build FAISS index")
    use_faiss_search = st.checkbox("Use FAISS for retrieval (faster retrieval for large documents)",
                                   value=False)

    os.environ["USE_SAFE_EMBEDDINGS"] = "1" if use_safe_toggle else "0"

    # Create embeddings when missing or when they are requested
    if (not embeddings_path.exists()) or recreate_btn:
        try:
            with st.spinner("Creating embeddings..."):
                rows = create_embeddings_for_text(text, str(embeddings_path), dim=EMBED_DIM)
                st.success(f"Embeddings created: {embeddings_path} ({len(rows)} chunks)")
        except Exception as e:
            st.error("Failed to create embeddings.")
            st.exception(e)

    # Show embeddings status
    try:
        ids, texts, vecs = load_embeddings(str(embeddings_path))
        st.info(
            f"Embeddings file: {embeddings_path} — {len(ids)} chunks — Mode: {'SAFE' if os.environ.get('USE_SAFE_EMBEDDINGS','1') in ('1','true') else 'REAL'}"
        )
    except Exception:
        ids, texts, vecs = [], [], np.array([])
        st.warning("Embeddings file not found. Click (Re)create embeddings.")

    # Build FAISS index if requested
    if build_index_btn:
        if not embeddings_path.exists():
            st.error("Cannot build FAISS index — embeddings file missing. Create embeddings first.")
        elif not _faiss_builder_available:
            st.error("FAISS builder not available (faiss-cpu not installed). Install: pip install faiss-cpu")
        else:
            try:
                with st.spinner("Building FAISS index..."):
                    build_index(str(embeddings_path), str(index_path))
                    st.success(f"FAISS index built: {index_path}")
            except Exception as e:
                st.error("FAISS index build failed.")
                st.exception(e)

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Ask a Question", "Generate Summary", "Chat"])

    # -------------------
    # Tab: Ask a Question
    # -------------------
    with tab1:
        st.subheader("Ask a question (Retrieval-Augmented)")
        question = st.text_input("Your question (based on uploaded lecture)", key="qa_question")
        k = st.number_input("Top-k chunks to retrieve", min_value=1, max_value=10, value=3, step=1, key="qa_k")
        if st.button("Ask (RAG)"):
            if len(ids) == 0:
                st.error("Embeddings not loaded. Create embeddings first.")
            elif not question or not question.strip():
                st.warning("Please enter a question.")
            else:
                candidate_index_path = str(index_path) if index_path.exists() else None
                with st.spinner("Retrieving relevant chunks and asking Gemini..."):
                    try:
                        if st.session_state.use_api_mode:
                            resp = perform_query(
                                question=question,
                                embeddings_path=str(embeddings_path),
                                top_k=int(k),
                                use_faiss=bool(use_faiss_search),
                                faiss_index_path=candidate_index_path,
                                use_api_mode=True,
                                api_base=os.getenv("API_BASE", API_DEFAULT),
                                token=st.session_state.api_token or "",
                                llm_call=None,
                            )
                        else:
                            resp = perform_query(
                                question=question,
                                embeddings_path=str(embeddings_path),
                                top_k=int(k),
                                use_faiss=bool(use_faiss_search),
                                faiss_index_path=candidate_index_path,
                                use_api_mode=False,
                                llm_call=llm,
                            )

                        ans = resp.get("answer")
                        retrieved_chunks = resp.get("retrieved", [])
                        prompt_used = resp.get("prompt")
                        provenance = resp.get("provenance")
                        latency = resp.get("latency", None)

                        display_answer = strip_key_concepts_from_answer(ans or "")
                        st.subheader("Top retrieved chunks (expand to read)")
                        if not retrieved_chunks:
                            st.info("No chunks retrieved.")
                        else:
                            for i, chunk in enumerate(retrieved_chunks, start=1):
                                score = chunk.get("score", 0.0)
                                cid = chunk.get("id")
                                pos = chunk.get("pos")
                                with st.expander(f"{i}. chunk (id={cid} pos={pos} score={score:.4f})"):
                                    st.write(chunk.get("text", "")[:2000] + ("..." if len(chunk.get("text", "")) > 2000 else ""))
                        st.subheader("Answer")
                        st.write(display_answer or "")
                        if latency is not None:
                            st.caption(f"Latency: {latency:.3f}s")
                        st.subheader("Provenance (sentence -> chunk)")
                        if provenance and provenance.get("sentences"):
                            for s in provenance.get("sentences", []):
                                st.write(f"- \"{s['sentence']}\" → chunk_id={s.get('chunk_id')} (score={s.get('score'):.3f})")
                        else:
                            st.info("No provenance available.")
                        with st.expander("Prompt used (debug)"):
                            if prompt_used:
                                st.code(prompt_used[:4000])
                            else:
                                st.info("No prompt captured.")
                    except requests.HTTPError as he:
                        try:
                            err_json = he.response.json()
                            detail = err_json.get("detail", str(he))
                        except Exception:
                            detail = str(he)
                        st.error(f"API request failed: {detail}")
                        st.exception(he)
                    except FileNotFoundError as fe:
                        st.error("File missing for retrieval.")
                        st.exception(fe)
                    except Exception as e:
                        st.error("RAG failed — see details.")
                        st.exception(e)

    # -----------------------
    # Tab: Generate Summary
    # -----------------------
    with tab2:
        st.subheader("Generate document-level summary (explicit)")
        summary_type = st.selectbox("Summary detail level", ["brief", "detailed"], index=0)
        summary_top_k = st.number_input("Limit to first N chunks (leave blank / 0 to use all)", min_value=0, max_value=10000, value=0, step=1)
        if st.button("Generate Summary"):
            if len(ids) == 0:
                st.error("Embeddings not loaded. Create embeddings first.")
            else:
                candidate_index_path = str(index_path) if index_path.exists() else None
                with st.spinner("Generating summary..."):
                    try:
                        if st.session_state.use_api_mode:
                            resp = perform_summary(
                                embeddings_path=str(embeddings_path),
                                summary_type=summary_type,
                                top_k=(int(summary_top_k) if summary_top_k > 0 else None),
                                use_api_mode=True,
                                api_base=os.getenv("API_BASE", API_DEFAULT),
                                token=st.session_state.api_token or "",
                            )
                        else:
                            resp = perform_summary(
                                embeddings_path=str(embeddings_path),
                                summary_type=summary_type,
                                top_k=(int(summary_top_k) if summary_top_k > 0 else None),
                                use_api_mode=False,
                                llm_call=None,
                            )

                        summary = resp.get("summary", "")
                        key_concepts = resp.get("key_concepts", []) or []
                        used_chunks = resp.get("used_chunks", []) or []
                        summary_display = clean_summary_text(summary)
                        st.subheader("Summary")
                        st.write(summary_display or "")
                        st.subheader("Key concepts / highlights")
                        if key_concepts:
                            st.write(f"{len(key_concepts)} items — " + ", ".join(key_concepts))
                        else:
                            st.info("No key concepts extracted.")
                        with st.expander("Show used chunks (preview)"):
                            if not used_chunks:
                                st.info("No chunks available.")
                            else:
                                for c in used_chunks:
                                    if isinstance(c, dict):
                                        cid = c.get("id", "<no-id>")
                                        pos = c.get("pos", "<no-pos>")
                                        txt = c.get("text", "")
                                        st.write(f"- id={cid} pos={pos}: {txt[:300]}{'...' if len(txt) > 300 else ''}")
                                    elif isinstance(c, str):
                                        st.write(f"- chunk text preview: {c[:300]}{'...' if len(c) > 300 else ''}")
                                    else:
                                        s = str(c)
                                        st.write(f"- {s[:300]}{'...' if len(s) > 300 else ''}")
                    except requests.HTTPError as he:
                        try:
                            detail = he.response.json().get("detail", str(he))
                        except Exception:
                            detail = str(he)
                        st.error(f"API summarize failed: {detail}")
                        st.exception(he)
                    except Exception as e:
                        st.error("Summary generation failed.")
                        st.exception(e)

    # -----------------------
    # Tab: Chat
    # -----------------------
    with tab3:
        st.subheader("Chat with the lecture (conversational)")

        hist_key = f"chat_history_{stem}"
        if hist_key not in st.session_state:
            st.session_state[hist_key] = []

        # Chat controls
        chat_k = st.number_input("Top-k chunks to retrieve for each turn", min_value=1, max_value=10, value=3, step=1, key="chat_k")

        if st.button("Clear chat"):
            # Reset only this document's chat history
            st.session_state[hist_key] = []
            st.success("Chat history cleared for this document.")

        with st.form(key=f"chat_form_{stem}"):
            user_msg = st.text_input("Message to assistant", key=f"chat_input_{stem}")
            send_pressed = st.form_submit_button("Send")

        # If the user submitted, process and update session_state before rendering the history
        if send_pressed:
            if len(ids) == 0:
                st.error("Embeddings not loaded. Create embeddings first.")
            elif not user_msg or not user_msg.strip():
                st.warning("Please enter a message.")
            else:
                candidate_index_path = str(index_path) if index_path.exists() else None
                payload_history = [{"role": h.get("role"), "content": h.get("content")} for h in st.session_state[hist_key]]
                with st.spinner("Running conversational RAG..."):
                    try:
                        if st.session_state.use_api_mode:
                            resp = perform_chat(
                                question=user_msg,
                                embeddings_path=str(embeddings_path),
                                history=payload_history,
                                top_k=int(chat_k),
                                use_faiss=bool(use_faiss_search),
                                faiss_index_path=candidate_index_path,
                                use_api_mode=True,
                                api_base=os.getenv("API_BASE", API_DEFAULT),
                                token=st.session_state.api_token or "",
                                llm_call=None,
                            )
                        else:
                            resp = perform_chat(
                                question=user_msg,
                                embeddings_path=str(embeddings_path),
                                history=payload_history,
                                top_k=int(chat_k),
                                use_faiss=bool(use_faiss_search),
                                faiss_index_path=candidate_index_path,
                                use_api_mode=False,
                                llm_call=llm,
                            )

                        ans = resp.get("answer")
                        updated_history = resp.get("history", None)
                        retrieved = resp.get("retrieved", []) or []
                        prompt_used = resp.get("prompt")
                        provenance = resp.get("provenance")

                        display_answer = strip_key_concepts_from_answer(ans or "")

                        if updated_history:
                            new_hist: List[Dict[str, Any]] = []
                            for h in updated_history:
                                role = h.get("role", "user")
                                content_raw = h.get("content", "") or ""
                                if role and role.lower().startswith("assistant"):
                                    content = strip_key_concepts_from_answer(content_raw)
                                else:
                                    content = content_raw
                                new_hist.append({"role": role, "content": content})
                            if new_hist and new_hist[-1].get("role") and new_hist[-1]["role"].lower().startswith("assistant"):
                                new_hist[-1]["meta"] = {"retrieved": retrieved or [], "prompt": prompt_used, "provenance": provenance}
                            st.session_state[hist_key] = trim_history_to_max_turns(new_hist, max_turns=6)
                        else:
                            st.session_state[hist_key].append({"role": "user", "content": user_msg})
                            st.session_state[hist_key].append(
                                {"role": "assistant", "content": display_answer, "meta": {"retrieved": retrieved or [], "prompt": prompt_used, "provenance": provenance}}
                            )
                            st.session_state[hist_key] = trim_history_to_max_turns(st.session_state[hist_key], max_turns=6)

                        st.success("Assistant replied — see chat above.")

                    except Exception as e:
                        st.error("Conversational RAG failed.")
                        st.exception(e)

        st.markdown("----")
        chat_history: List[Dict[str, Any]] = st.session_state[hist_key]
        if not chat_history:
            st.info("No messages yet. Start by asking a question below.")
        else:
            for idx, turn in enumerate(chat_history):
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role and role.lower().startswith("assistant"):
                    content = strip_key_concepts_from_answer(content or "")
                meta = turn.get("meta", {}) or {}

                if role == "user":
                    st.markdown(f"**You:**\n\n{content}", unsafe_allow_html=True)
                else:
                    # use helper to build assistant HTML
                    rendered = render_assistant_html(content)
                    components.html(rendered["html"], height=rendered["height"], scrolling=True)

                # show metadata expanders after the message render
                if meta:
                    retrieved = meta.get("retrieved", []) or []
                    prov = meta.get("provenance")
                    prompt_used = meta.get("prompt")
                    if retrieved:
                        with st.expander("Retrieved chunks (turn)"):
                            for c in retrieved:
                                try:
                                    st.write(f"- id={c.get('id')} pos={c.get('pos')} score={c.get('score'):.4f}")
                                except Exception:
                                    st.write(f"- {c}")
                    if prov and prov.get("sentences"):
                        with st.expander("Provenance (this turn)"):
                            for s in prov.get("sentences", []):
                                st.write(f"- \"{s.get('sentence')}\" → chunk={s.get('chunk_id')} (score={s.get('score'):.3f})")
                    if prompt_used:
                        with st.expander("Prompt used (debug)"):
                            st.code(str(prompt_used)[:4000])

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
                    if st.session_state.use_api_mode:
                        quiz_items, latency = generate_quiz(
                            stem=stem,
                            context_text=text,
                            n=int(n_q),
                            use_api_mode=True,
                            api_base=os.getenv("API_BASE", API_DEFAULT),
                            token=st.session_state.api_token or "",
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

    def _mark_selection_made(sel_key: str):
        st.session_state[f"{sel_key}_made"] = True

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
                        on_change=_mark_selection_made,
                        args=(selection_key,),
                    )
                except Exception:
                    st.selectbox(
                        "Choose an answer",
                        dropdown_with_placeholder,
                        key=selection_key,
                        disabled=already_submitted,
                        on_change=_mark_selection_made,
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

    st.markdown("---")
    
    # -----------------------
    # SRS Review Section
    # -----------------------
    st.subheader("📚 Spaced Repetition Review")
    
    with st.expander("ℹ️ What is Spaced Repetition?", expanded=False):
        st.markdown("""
        **Spaced Repetition** is a study technique that helps you remember information long-term by reviewing 
        it at increasing intervals. The more you remember something correctly, the longer you wait before reviewing it again.
        
        **How it works:**
        1. When you answer a quiz question correctly → review again in **1 day**
        2. Get it right again → review in **3 days**
        3. Keep getting it right → intervals increase to **7, 14, then 30 days**
        4. If you get it wrong → interval resets to **1 day** to strengthen memory
        
        **To get started:**
        1. Generate a quiz from your PDF above
        2. Click **"Start SRS"** on questions you want to review later
        3. Come back here to review cards when they're due!
        """)

    try:
        srs_mgr = SRSManager()
        all_cards = list(srs_mgr._data.keys())
        due_cards = srs_mgr.get_due_cards()
        
        if not all_cards:
            st.info("📝 **No cards registered yet.**\n\nTo start using spaced repetition:\n1. Generate a quiz from your PDF above\n2. Click **'Start SRS'** on any question you want to review later\n3. Come back here to review when cards are due!")
        else:
            current_quiz = st.session_state.get(quiz_state_key, [])
            if current_quiz:
                registered_ids = set(all_cards)
                unregistered = [q for q in current_quiz if q.get("id") not in registered_ids]
                if unregistered:
                    st.info(f"💡 **Tip:** You have {len(unregistered)} quiz question(s) generated above. "
                           f"Click **'Start SRS'** on any question to add it to your spaced repetition review!")
            
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                st.metric("Total Cards", len(all_cards))
            with col_stats2:
                st.metric("Due Now", len(due_cards), delta=None if len(due_cards) == 0 else f"{len(due_cards)} to review")
            with col_stats3:
                reviewed_count = sum(1 for cid in all_cards if srs_mgr.get_card_meta(cid).get("review_count", 0) > 0)
                st.metric("Reviewed", reviewed_count)
            
            if due_cards:
                st.success(f"🎯 **You have {len(due_cards)} card(s) due for review!**")
                st.markdown("---")
                
                cache_key = "srs_quiz_items_cache"
                if cache_key not in st.session_state:
                    st.session_state[cache_key] = {}
                
                all_quiz_items = st.session_state[cache_key].copy()
                
                current_quiz = st.session_state.get(quiz_state_key, [])
                for q in current_quiz:
                    all_quiz_items[q.get("id")] = q
                
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
                
                for idx, card_id in enumerate(due_cards, 1):
                    if card_id in reviewed_this_session:
                        continue
                    
                    card_meta = srs_mgr.get_card_meta(card_id)
                    quiz_item = all_quiz_items.get(card_id)
                    doc_name = card_id.rsplit("_", 1)[0] if "_" in card_id else "Unknown"
                    
                    if not quiz_item:
                        st.markdown(f"### Card {idx}: {card_id}")
                        is_placeholder = "placeholder" in card_id.lower() or (card_meta and card_meta.get("review_count", 0) == 0)
                        if doc_name != "Unknown":
                            st.warning(f"📄 **Card from: {doc_name}**")
                            if is_placeholder:
                                st.info(f"💡 This appears to be a placeholder card from an old session. "
                                       f"You can delete it and register new questions from the quiz section above.")
                            else:
                                st.info(f"💡 **Tip:** Upload the PDF '{doc_name}.pdf' and generate a quiz to see the full question. "
                                       f"For now, you can review based on your memory of this topic.")
                        else:
                            st.info("💡 **Tip:** This card was registered from a previous session. "
                                   "Upload the same PDF and generate a quiz to see the full question.")
                        
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
                        
                        col_correct, col_incorrect = st.columns(2)
                        with col_correct:
                            if st.button(f"✅ Yes, I remember this", key=f"srs_correct_{card_id}", type="primary", use_container_width=True):
                                srs_mgr.mark_review(card_id, correct=True)
                                reviewed_this_session.add(card_id)
                                st.session_state["srs_reviewed_this_session"] = reviewed_this_session
                                st.success("✅ Great! Card scheduled for next review.")
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
                        st.markdown(f"### Card {idx}: Review Question")
                        
                        q_text = quiz_item.get("question", "")
                        choices = quiz_item.get("choices", {}) or {}
                        answer_letter = quiz_item.get("answer", None)
                        explanation = quiz_item.get("explanation", "")
                        
                        st.markdown("**📝 Question:**")
                        st.write(q_text)
                        
                        st.markdown("**🔤 Answer Choices:**")
                        for label in ["A", "B", "C", "D"]:
                            if label in choices:
                                st.write(f"**{label}.** {choices[label]}")
                        
                        show_answer_key = f"srs_show_{card_id}"
                        
                        if show_answer_key not in st.session_state:
                            st.session_state[show_answer_key] = False
                        
                        st.markdown("---")
                        
                        if not st.session_state[show_answer_key]:
                            st.markdown("**💭 Think about your answer, then click below to reveal the correct answer:**")
                            if st.button("🔍 Show Answer", key=f"show_{card_id}", type="primary", use_container_width=True):
                                st.session_state[show_answer_key] = True
                                st.rerun()
                        else:
                            st.markdown("**✅ Correct Answer:**")
                            if answer_letter and choices.get(answer_letter):
                                st.success(f"**{answer_letter}.** {choices[answer_letter]}")
                            
                            if explanation:
                                st.markdown("**📖 Explanation:**")
                                st.info(explanation)
                            
                            st.markdown("---")
                            st.markdown("**🎯 Did you get it right?**")
                            
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
                            
                            if card_meta:
                                review_count = card_meta.get("review_count", 0)
                                interval_idx = card_meta.get("interval_index", 0)
                                next_due = card_meta.get("next_due", "")
                                if next_due:
                                    try:
                                        next_due_dt = datetime.fromisoformat(next_due)
                                        days_until = (next_due_dt - datetime.utcnow()).days
                                        st.caption(f"📊 Progress: Reviewed {review_count} time(s) | Current interval: {INTERVALS[interval_idx]} days | Next review in: {days_until} days")
                                    except Exception:
                                        st.caption(f"📊 Progress: Reviewed {review_count} time(s) | Current interval: {INTERVALS[interval_idx]} days")
                        
                        st.markdown("---")
            else:
                st.info("✅ **No cards due for review right now!** Great job staying on top of your studies. 🎉")
            
            st.markdown("---")
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
                        status_text = "**Due now**" if is_due else "Not due"
                        
                        next_due = meta.get("next_due", "") if meta else ""
                        review_count = meta.get("review_count", 0) if meta else 0
                        interval_idx = meta.get("interval_index", 0) if meta else 0
                        
                        st.markdown(f"{status_icon} **{card_id}** - {status_text}")
                        st.write(f"   {question_preview}")
                        if meta:
                            try:
                                if next_due:
                                    next_due_dt = datetime.fromisoformat(next_due)
                                    days_until = (next_due_dt - datetime.utcnow()).days
                                    if days_until <= 0:
                                        due_text = "Due now"
                                    else:
                                        due_text = f"Due in {days_until} day(s)"
                                else:
                                    due_text = "N/A"
                            except Exception:
                                due_text = next_due[:10] if next_due else "N/A"
                            
                            st.caption(f"   📊 Reviewed {review_count} time(s) | Interval: {INTERVALS[interval_idx]} days | {due_text}")
                        st.markdown("")
    
    except Exception as e:
        st.error("Error loading SRS data.")
        st.exception(e)

    st.markdown("---")
    st.caption("Tip: For reproducible tests set USE_SAFE_EMBEDDINGS=1 and build FAISS index to compare results with NumPy search.")
